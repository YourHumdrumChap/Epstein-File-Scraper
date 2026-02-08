from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from doj_disclosures.core.ai_flagger import LinearFlaggerModel, load_ai_flagger_model
from doj_disclosures.core.config import CrawlSettings
from doj_disclosures.core.db import Database
from doj_disclosures.core.embedding_index import build_embeddings_for_text
from doj_disclosures.core.embeddings import EmbeddingProvider, blob_to_vector, get_default_provider, vector_to_blob
from doj_disclosures.core.matching import KeywordMatcher, MatchHit
from doj_disclosures.core.ner import extract_entities
from doj_disclosures.core.parser import DocumentParser, ParsedDocument
from doj_disclosures.core.redactions import analyze_pdf_redactions
from doj_disclosures.core.relevance import TopicVector, build_topic_vector, compute_entity_density, compute_relevance, hostname
from doj_disclosures.core.storage_gating import StoragePlan, compute_flagged_path, move_to
from doj_disclosures.core.tables import extract_tables_from_pdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticContext:
    provider: EmbeddingProvider | None
    topic: TopicVector | None
    model_name: str
    hv_centroid: tuple[list[float], float] | None
    ir_centroid: tuple[list[float], float] | None
    ai_flagger: LinearFlaggerModel | None


@dataclass(frozen=True)
class PipelineDeps:
    settings: CrawlSettings
    db: Database
    storage: StoragePlan
    parser: DocumentParser
    matcher: KeywordMatcher
    penalties: dict[str, float]
    semantic: SemanticContext


@dataclass(frozen=True)
class PipelineInput:
    url: str
    final_url: str
    local_path: Path
    content_type: str
    file_size: int | None
    sha256: str
    fetched_at: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class PipelineMetrics:
    topic_similarity: float
    relevance_score: float
    entity_density: float
    url_penalty: float
    feedback_boost: float


@dataclass(frozen=True)
class PipelineOutput:
    doc_id: int
    parsed: ParsedDocument
    hits: list[MatchHit]
    passes_relevance: bool
    final_path: Path
    metrics: PipelineMetrics
async def load_feedback_centroids(*, db: Database, model_name: str) -> tuple[tuple[list[float], float] | None, tuple[list[float], float] | None]:
    try:
        hv = await db.get_feedback_centroid(label="high_value", model_name=model_name)
        ir = await db.get_feedback_centroid(label="irrelevant", model_name=model_name)
        hv_out = (hv.vec, hv.norm) if hv is not None and hv.norm > 0 else None
        ir_out = (ir.vec, ir.norm) if ir is not None and ir.norm > 0 else None
        return hv_out, ir_out
    except Exception:
        return None, None


async def build_semantic_context_async(*, settings: CrawlSettings, db: Database) -> SemanticContext:
    model_name = str(getattr(settings, "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    provider = get_default_provider(model_name)
    topic = build_topic_vector(provider) if provider is not None else None
    hv_centroid = None
    ir_centroid = None
    ai_flagger = None
    if provider is not None:
        hv_centroid, ir_centroid = await load_feedback_centroids(db=db, model_name=model_name)
        ai_flagger = await load_ai_flagger_model(db=db, model_name=model_name)

    return SemanticContext(
        provider=provider,
        topic=topic,
        model_name=model_name,
        hv_centroid=hv_centroid,
        ir_centroid=ir_centroid,
        ai_flagger=ai_flagger,
    )


async def process_document(
    *,
    deps: PipelineDeps,
    inp: PipelineInput,
    now: str | None = None,
    allow_move: bool = True,
    reprocess_existing: bool = False,
    log: Callable[[str], None] | None = None,
) -> PipelineOutput:
    s = deps.settings
    now = now or datetime.now(timezone.utc).isoformat()
    log = log or (lambda _m: None)

    parsed = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: deps.parser.parse(inp.local_path, inp.content_type, fallback_title=inp.local_path.name),
    )

    hits = await asyncio.get_running_loop().run_in_executor(None, lambda: deps.matcher.match(parsed.text))

    total_words = max(1, len([w for w in parsed.text.replace("[PAGE", " ").split() if w.strip()]))
    entity_mentions = 0
    try:
        dens_ents = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: extract_entities(
                parsed.text,
                enabled=True,
                engine="regex",
                spacy_model=str(getattr(s, "ner_spacy_model", "en_core_web_sm")),
            ),
        )
        for e in dens_ents or []:
            entity_mentions += int(e.get("count") or 0)
    except Exception:
        entity_mentions = 0
    entity_density = compute_entity_density(total_entity_mentions=entity_mentions, total_words=total_words)

    topic_sim = 0.0
    relevance_score = 0.0
    feedback_boost = 0.0
    ai_high_value_prob: float | None = None
    host = hostname(inp.url)
    url_penalty = float(deps.penalties.get(host, 0.0) or 0.0)

    provider = deps.semantic.provider
    topic = deps.semantic.topic

    if provider is not None and topic is not None:
        try:
            vec = provider.embed([parsed.text[:12000]])[0]
            blob, doc_norm = vector_to_blob(vec)
            doc_vec = blob_to_vector(blob)
            rel = compute_relevance(
                doc_vec=doc_vec,
                doc_norm=doc_norm,
                topic=topic,
                hv_centroid=deps.semantic.hv_centroid,
                ir_centroid=deps.semantic.ir_centroid,
                url_penalty=url_penalty,
                entity_density=entity_density,
            )
            topic_sim = rel.topic_similarity
            relevance_score = rel.relevance_score
            feedback_boost = float(getattr(rel, "feedback_similarity_boost", 0.0) or 0.0)

            if bool(getattr(s, "ai_flagger_enabled", True)) and deps.semantic.ai_flagger is not None:
                try:
                    base_rel = compute_relevance(
                        doc_vec=doc_vec,
                        doc_norm=doc_norm,
                        topic=topic,
                        hv_centroid=None,
                        ir_centroid=None,
                        url_penalty=0.0,
                        entity_density=entity_density,
                    )
                    ai_high_value_prob = deps.semantic.ai_flagger.predict_high_value_prob(
                        embedding=doc_vec,
                        relevance_score=float(base_rel.relevance_score),
                        topic_similarity=float(base_rel.topic_similarity),
                        entity_density=float(entity_density),
                    )
                except Exception:
                    ai_high_value_prob = None
        except Exception:
            topic_sim = 0.0
            relevance_score = 0.0
            feedback_boost = 0.0
            ai_high_value_prob = None

    topic_thr = 0.20
    rel_thr = 0.18
    density_thr = 0.001

    if provider is not None and topic is not None:
        if hits and len(hits) == 1 and topic_sim < topic_thr and relevance_score < rel_thr:
            hits = []

    if provider is None or topic is None:
        passes_relevance = bool(hits)
    else:
        strong_keyword_evidence = bool(hits) and len(hits) >= 2
        passes_relevance = (
            (topic_sim >= topic_thr and entity_density >= density_thr)
            or (relevance_score >= rel_thr and bool(hits) and len(hits) >= 2)
            or (relevance_score >= (rel_thr + 0.08))
        )

        if bool(getattr(s, "feedback_auto_flag_enabled", True)):
            flag_thr = float(getattr(s, "feedback_auto_flag_threshold", 0.22))
            triage_thr = float(getattr(s, "feedback_auto_triage_threshold", -0.22))

            if feedback_boost >= flag_thr:
                if not passes_relevance:
                    log(f"Auto-flagged by feedback model (boost={feedback_boost:.3f} >= {flag_thr:.3f})")
                passes_relevance = True
            elif feedback_boost <= triage_thr and not strong_keyword_evidence:
                if passes_relevance:
                    log(f"Auto-triaged by feedback model (boost={feedback_boost:.3f} <= {triage_thr:.3f})")
                passes_relevance = False

        if ai_high_value_prob is not None:
            p_flag = float(getattr(s, "ai_flagger_flag_threshold", 0.80))
            p_triage = float(getattr(s, "ai_flagger_triage_threshold", 0.20))

            if ai_high_value_prob >= p_flag:
                if not passes_relevance:
                    log(f"Auto-flagged by AI flagger (p={ai_high_value_prob:.3f} >= {p_flag:.3f})")
                passes_relevance = True
            elif ai_high_value_prob <= p_triage and not strong_keyword_evidence:
                if passes_relevance:
                    log(f"Auto-triaged by AI flagger (p={ai_high_value_prob:.3f} <= {p_triage:.3f})")
                passes_relevance = False

    src_path = inp.local_path
    suffix = src_path.suffix

    final_path = src_path
    if allow_move:
        if passes_relevance:
            dst_path = compute_flagged_path(
                flagged_dir=deps.storage.flagged_dir,
                sha256=inp.sha256,
                suffix=suffix,
                storage_layout=str(getattr(s, "storage_layout", "flat")),
                display_name=str(parsed.title or src_path.stem),
            )
            try:
                final_path = move_to(dst_path, src_path)
            except Exception:
                final_path = src_path
        else:
            dst_path = deps.storage.triaged_dir / f"{inp.sha256}{suffix}"
            try:
                final_path = move_to(dst_path, src_path)
            except Exception:
                final_path = src_path

    doc_id = await deps.db.add_document(
        url=inp.url,
        final_url=inp.final_url,
        title=parsed.title,
        content_type=inp.content_type,
        file_size=inp.file_size,
        sha256=inp.sha256,
        local_path=str(final_path),
        fetched_at=inp.fetched_at,
    )

    if reprocess_existing:
        await deps.db.update_document_storage(doc_id=doc_id, local_path=str(final_path), title=parsed.title, content_type=inp.content_type)
        await deps.db.purge_derived_for_doc(doc_id=doc_id)

    await deps.db.update_document_metrics(
        doc_id=doc_id,
        relevance_score=float(relevance_score),
        topic_similarity=float(topic_sim),
        entity_density=float(entity_density),
        url_penalty=float(url_penalty),
    )
    await deps.db.add_fts_content(doc_id=doc_id, url=inp.url, title=parsed.title, content=parsed.text)

    # Redaction flags (best-effort)
    if bool(getattr(s, "redaction_detection_enabled", True)) and (
        "pdf" in (inp.content_type or "").lower() or str(final_path).lower().endswith(".pdf")
    ):
        try:
            findings = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: analyze_pdf_redactions(Path(str(final_path)), extracted_text=parsed.text),
            )
            thr = float(getattr(s, "redaction_page_score_threshold", 0.25) or 0.25)
            flags = [
                {
                    "page_no": int(f.get("page_no") or 0),
                    "flag": "redaction",
                    "score": float(f.get("score") or 0.0),
                    "details": f.get("details"),
                }
                for f in (findings or [])
                if float(f.get("score") or 0.0) >= thr
            ]
            if flags:
                await deps.db.add_page_flags(doc_id=doc_id, flags=flags, created_at=now)
        except Exception as e:
            log(f"WARN: redaction detection failed: {e}")

    # Build and store embeddings for hybrid search (best-effort).
    if bool(getattr(s, "embedding_index_enabled", False)):
        try:
            model_name = str(getattr(s, "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"))
            idx_provider = get_default_provider(model_name)
            if idx_provider is not None:
                embeddings = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: build_embeddings_for_text(parsed.text, provider=idx_provider),
                )
                if embeddings:
                    await deps.db.add_embeddings(doc_id=doc_id, embeddings=embeddings, created_at=now)
        except Exception as e:
            log(f"WARN: embedding index failed: {e}")

    # Extract and store entities (best-effort).
    if bool(getattr(s, "ner_enabled", True)):
        try:
            entities = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: extract_entities(
                    parsed.text,
                    enabled=True,
                    engine=str(getattr(s, "ner_engine", "spacy")),
                    spacy_model=str(getattr(s, "ner_spacy_model", "en_core_web_sm")),
                ),
            )
            if entities:
                await deps.db.add_entities(doc_id=doc_id, entities=entities, created_at=now)
        except Exception as e:
            log(f"WARN: entity extraction failed: {e}")

    # Extract and store tables for PDFs (best-effort).
    if "pdf" in (inp.content_type or "").lower() or str(final_path).lower().endswith(".pdf"):
        try:
            tables = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: extract_tables_from_pdf(Path(str(final_path))),
            )
            if tables:
                await deps.db.add_tables(doc_id=doc_id, tables=tables, created_at=now)
        except Exception as e:
            log(f"WARN: table extraction failed: {e}")

    if hits:
        await deps.db.add_matches(
            doc_id=doc_id,
            matches=[(h.method, h.pattern, h.score, h.snippet) for h in hits],
            created_at=now,
        )

    metrics = PipelineMetrics(
        topic_similarity=float(topic_sim),
        relevance_score=float(relevance_score),
        entity_density=float(entity_density),
        url_penalty=float(url_penalty),
        feedback_boost=float(feedback_boost),
    )

    return PipelineOutput(
        doc_id=doc_id,
        parsed=parsed,
        hits=list(hits),
        passes_relevance=bool(passes_relevance),
        final_path=Path(str(final_path)),
        metrics=metrics,
    )
