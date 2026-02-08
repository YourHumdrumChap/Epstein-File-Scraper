from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from doj_disclosures.core.embeddings import EmbeddingProvider, blob_to_vector, vector_to_blob
from doj_disclosures.core.feedback import Centroid
from doj_disclosures.core.parser import DocumentParser

logger = logging.getLogger(__name__)


AI_FLAGGER_MODEL_VERSION = 1
AI_FLAGGER_KV_PREFIX = "ai_flagger_model_v1:"


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _guess_content_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return "application/pdf"
    if suf in {".htm", ".html"}:
        return "text/html"
    if suf == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "text/plain"


def _normalize(vec: list[float]) -> tuple[list[float], float]:
    blob, norm = vector_to_blob(vec)
    return blob_to_vector(blob), float(norm)


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = min(len(v) for v in vectors if v)
    if dim <= 0:
        return []
    acc = [0.0] * dim
    n = 0
    for v in vectors:
        if not v:
            continue
        n += 1
        for i in range(dim):
            acc[i] += float(v[i])
    if n <= 0:
        return []
    return [x / n for x in acc]


def embed_text_robust(
    provider: EmbeddingProvider,
    text: str,
    *,
    max_chars: int = 60000,
    chunk_chars: int = 4500,
    overlap: int = 300,
    max_chunks: int = 12,
) -> list[float]:
    """Embed long documents by chunking and mean-pooling embeddings.

    Uses sentence-transformers normalized embeddings; we average and then re-normalize.
    """

    t = (text or "").strip()
    if not t:
        return []

    t = t[: max(0, int(max_chars))]
    chunk_chars = max(800, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))

    chunks: list[str] = []
    start = 0
    while start < len(t) and len(chunks) < int(max_chunks):
        end = min(len(t), start + chunk_chars)
        chunks.append(t[start:end])
        if end >= len(t):
            break
        start = max(0, end - overlap)

    vecs = provider.embed(chunks)
    pooled = _mean_vectors(vecs)
    pooled2, _ = _normalize(pooled)
    return pooled2


@dataclass(frozen=True)
class TrainingRow:
    label: str  # "high_value" or "irrelevant"
    local_path: str
    url: str
    title: str
    relevance_score: float | None
    topic_similarity: float | None
    entity_density: float | None


def load_semantic_sorted_tsv(*, path: Path, expected_label: str | None = None) -> list[TrainingRow]:
    if not path.exists():
        return []

    rows: list[TrainingRow] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return []

    header = lines[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    required = ["review_status", "local_path", "url", "title"]
    if any(k not in idx for k in required):
        return []

    def _f(name: str, parts: list[str]) -> float | None:
        i = idx.get(name)
        if i is None or i >= len(parts):
            return None
        s = str(parts[i]).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        status = str(parts[idx["review_status"]]).strip().lower() if idx["review_status"] < len(parts) else ""
        if expected_label and status and status != expected_label:
            continue
        if expected_label and not status:
            status = expected_label

        if status not in {"high_value", "irrelevant"}:
            continue

        local_path = str(parts[idx["local_path"]]).strip() if idx["local_path"] < len(parts) else ""
        url = str(parts[idx["url"]]).strip() if idx["url"] < len(parts) else ""
        title = str(parts[idx["title"]]).strip() if idx["title"] < len(parts) else ""
        if not local_path:
            continue

        rows.append(
            TrainingRow(
                label=status,
                local_path=local_path,
                url=url,
                title=title,
                relevance_score=_f("relevance_score", parts),
                topic_similarity=_f("topic_similarity", parts),
                entity_density=_f("entity_density", parts),
            )
        )

    return rows


@dataclass(frozen=True)
class LinearFlaggerModel:
    """A small, fast high_value-vs-irrelevant classifier.

    It predicts P(high_value) using a single linear layer over:
    - normalized document embedding
    - standardized scalar features (relevance_score, topic_similarity, entity_density)

    Stored in DB KV as JSON; inference requires no sklearn.
    """

    version: int
    model_name: str
    embedding_dim: int
    scalar_feature_names: tuple[str, ...]
    scalar_mean: tuple[float, ...]
    scalar_scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    trained_at: str
    n_examples: int
    metrics: dict[str, float]

    def predict_high_value_prob(
        self,
        *,
        embedding: list[float],
        relevance_score: float,
        topic_similarity: float,
        entity_density: float,
    ) -> float:
        if not embedding or len(embedding) < self.embedding_dim:
            return 0.5

        scalars = [float(relevance_score), float(topic_similarity), float(entity_density)]
        zs: list[float] = []
        for i, x in enumerate(scalars):
            mu = float(self.scalar_mean[i]) if i < len(self.scalar_mean) else 0.0
            sc = float(self.scalar_scale[i]) if i < len(self.scalar_scale) else 1.0
            if sc <= 1e-8:
                sc = 1.0
            zs.append((x - mu) / sc)

        w = self.weights
        # dot([emb, zs], w) + b
        acc = float(self.bias)
        dim = int(self.embedding_dim)
        for i in range(dim):
            acc += float(w[i]) * float(embedding[i])
        for j in range(len(zs)):
            acc += float(w[dim + j]) * float(zs[j])
        return float(_sigmoid(acc))

    def to_json(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "model_name": str(self.model_name),
            "embedding_dim": int(self.embedding_dim),
            "scalar_feature_names": list(self.scalar_feature_names),
            "scalar_mean": list(self.scalar_mean),
            "scalar_scale": list(self.scalar_scale),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "trained_at": str(self.trained_at),
            "n_examples": int(self.n_examples),
            "metrics": dict(self.metrics or {}),
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "LinearFlaggerModel" | None:
        try:
            version = int(data.get("version") or 0)
            if version != AI_FLAGGER_MODEL_VERSION:
                return None
            model_name = str(data.get("model_name") or "")
            embedding_dim = int(data.get("embedding_dim") or 0)
            weights = [float(x) for x in (data.get("weights") or [])]
            scalar_mean = [float(x) for x in (data.get("scalar_mean") or [])]
            scalar_scale = [float(x) for x in (data.get("scalar_scale") or [])]
            scalar_feature_names = tuple(str(x) for x in (data.get("scalar_feature_names") or []))
            bias = float(data.get("bias") or 0.0)
            trained_at = str(data.get("trained_at") or "")
            n_examples = int(data.get("n_examples") or 0)
            metrics = data.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            if embedding_dim <= 0:
                return None
            expected_len = embedding_dim + len(scalar_feature_names)
            if len(weights) != expected_len:
                return None
            if len(scalar_mean) != len(scalar_feature_names) or len(scalar_scale) != len(scalar_feature_names):
                return None
            return LinearFlaggerModel(
                version=version,
                model_name=model_name,
                embedding_dim=embedding_dim,
                scalar_feature_names=scalar_feature_names,
                scalar_mean=tuple(float(x) for x in scalar_mean),
                scalar_scale=tuple(float(x) for x in scalar_scale),
                weights=tuple(float(x) for x in weights),
                bias=bias,
                trained_at=trained_at,
                n_examples=n_examples,
                metrics={str(k): float(v) for k, v in metrics.items() if isinstance(k, str)},
            )
        except Exception:
            return None


def model_kv_key(*, model_name: str) -> str:
    return f"{AI_FLAGGER_KV_PREFIX}{model_name}"


async def load_ai_flagger_model(*, db, model_name: str) -> LinearFlaggerModel | None:
    try:
        raw = await db.kv_get(model_kv_key(model_name=model_name))
        if not raw:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return LinearFlaggerModel.from_json(data)
    except Exception:
        return None


async def save_ai_flagger_model(*, db, model: LinearFlaggerModel) -> None:
    await db.kv_set(model_kv_key(model_name=model.model_name), json.dumps(model.to_json()))


@dataclass(frozen=True)
class TrainingResult:
    model: LinearFlaggerModel | None
    hv_centroid: Centroid | None
    ir_centroid: Centroid | None
    n_used: int
    skipped: int


def _standardize_columns(x_cols: list[list[float]]) -> tuple[list[float], list[float]]:
    """Return mean and scale (stddev) for each column."""
    if not x_cols:
        return [], []
    import numpy as np  # type: ignore

    x = np.asarray(x_cols, dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [float(v) for v in mu.tolist()], [float(v) for v in sd.tolist()]


def train_flagger_from_rows(
    *,
    rows: list[TrainingRow],
    provider: EmbeddingProvider,
    parser: DocumentParser,
    model_name: str,
    max_examples: int | None = None,
) -> TrainingResult:
    """Train a linear model from labeled docs listed in semantic_sorted.txt.

    This is intentionally fast: it reads the local PDFs, embeds their text, trains
    a simple linear classifier, and also rebuilds the feedback centroids.
    """

    if max_examples is not None:
        rows = rows[: max(1, int(max_examples))]

    # Lazy imports to keep base runtime light.
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import SGDClassifier  # type: ignore
        from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
    except Exception as e:
        logger.info("AI flagger training unavailable (missing sklearn/numpy): %s", e)
        return TrainingResult(model=None, hv_centroid=None, ir_centroid=None, n_used=0, skipped=len(rows))

    from doj_disclosures.core.ner import extract_entities
    from doj_disclosures.core.relevance import build_topic_vector, compute_entity_density, compute_relevance

    topic = build_topic_vector(provider)
    embedding_dim_guess = None

    X_emb: list[list[float]] = []
    X_scal: list[list[float]] = []
    y: list[int] = []
    hv_vecs: list[list[float]] = []
    ir_vecs: list[list[float]] = []

    skipped = 0
    for r in rows:
        p = Path(str(r.local_path))
        if not p.exists():
            skipped += 1
            continue

        try:
            parsed = parser.parse(p, _guess_content_type(p), fallback_title=r.title or p.name)
            vec = embed_text_robust(provider, parsed.text)
            if not vec:
                skipped += 1
                continue

            if embedding_dim_guess is None:
                embedding_dim_guess = len(vec)
            if len(vec) != embedding_dim_guess:
                # Dimension mismatch (shouldn't happen). Skip for safety.
                skipped += 1
                continue

            # Recompute scalar metrics so they're consistent across labels.
            # (semantic_sorted.txt may be missing for one class, and storing those
            # values would make the model learn artifacts.)
            total_words = max(1, len([w for w in parsed.text.replace("[PAGE", " ").split() if w.strip()]))
            entity_mentions = 0
            try:
                ents = extract_entities(parsed.text, enabled=True, engine="regex", spacy_model="en_core_web_sm")
                for e in ents or []:
                    entity_mentions += int(e.get("count") or 0)
            except Exception:
                entity_mentions = 0
            entity_density = compute_entity_density(total_entity_mentions=entity_mentions, total_words=total_words)

            doc_vec, doc_norm = _normalize(vec)
            rel = compute_relevance(
                doc_vec=doc_vec,
                doc_norm=doc_norm,
                topic=topic,
                hv_centroid=None,
                ir_centroid=None,
                url_penalty=0.0,
                entity_density=float(entity_density),
            )

            X_emb.append(doc_vec)
            X_scal.append([float(rel.relevance_score), float(rel.topic_similarity), float(entity_density)])
            if r.label == "high_value":
                y.append(1)
                hv_vecs.append(doc_vec)
            else:
                y.append(0)
                ir_vecs.append(doc_vec)
        except Exception:
            skipped += 1

    if embedding_dim_guess is None or len(y) < 6 or len(set(y)) < 2:
        # Not enough labeled data; still rebuild centroids.
        hv_mean = _mean_vectors(hv_vecs)
        ir_mean = _mean_vectors(ir_vecs)
        hv_normed, hv_norm = _normalize(hv_mean) if hv_mean else ([], 0.0)
        ir_normed, ir_norm = _normalize(ir_mean) if ir_mean else ([], 0.0)
        hv_cent = Centroid(vec=hv_normed, norm=float(hv_norm), count=len(hv_vecs)) if hv_mean else None
        ir_cent = Centroid(vec=ir_normed, norm=float(ir_norm), count=len(ir_vecs)) if ir_mean else None
        return TrainingResult(model=None, hv_centroid=hv_cent, ir_centroid=ir_cent, n_used=len(y), skipped=skipped)

    X_emb_np = np.asarray(X_emb, dtype=float)
    X_scal_np = np.asarray(X_scal, dtype=float)
    y_np = np.asarray(y, dtype=int)

    # Standardize scalar features; keep embedding as-is (already normalized).
    mu, sd = _standardize_columns(X_scal)
    mu_np = np.asarray(mu, dtype=float)
    sd_np = np.asarray(sd, dtype=float)
    X_scal_z = (X_scal_np - mu_np) / sd_np

    X = np.concatenate([X_emb_np, X_scal_z], axis=1)

    # Train/test split for basic quality sanity-check.
    test_size = 0.25 if len(y) >= 20 else 0.33
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_np, test_size=test_size, random_state=1337, stratify=y_np)
    except Exception:
        X_train, X_test, y_train, y_test = X, X, y_np, y_np

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=2000, tol=1e-4, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Metrics (best-effort)
    metrics: dict[str, float] = {}
    try:
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
        if len(set(y_test.tolist())) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
    except Exception:
        pass

    coef = clf.coef_.reshape(-1)
    bias = float(clf.intercept_.reshape(-1)[0])

    scalar_feature_names = ("relevance_score", "topic_similarity", "entity_density")
    model = LinearFlaggerModel(
        version=AI_FLAGGER_MODEL_VERSION,
        model_name=str(model_name),
        embedding_dim=int(embedding_dim_guess),
        scalar_feature_names=scalar_feature_names,
        scalar_mean=tuple(mu),
        scalar_scale=tuple(sd),
        weights=tuple(float(x) for x in coef.tolist()),
        bias=bias,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_examples=int(len(y)),
        metrics=metrics,
    )

    # Rebuild centroids from used examples.
    hv_mean = _mean_vectors(hv_vecs)
    ir_mean = _mean_vectors(ir_vecs)
    hv_normed, hv_norm = _normalize(hv_mean) if hv_mean else ([], 0.0)
    ir_normed, ir_norm = _normalize(ir_mean) if ir_mean else ([], 0.0)

    hv_cent = Centroid(vec=hv_normed, norm=float(hv_norm), count=len(hv_vecs)) if hv_mean else None
    ir_cent = Centroid(vec=ir_normed, norm=float(ir_norm), count=len(ir_vecs)) if ir_mean else None

    return TrainingResult(model=model, hv_centroid=hv_cent, ir_centroid=ir_cent, n_used=int(len(y)), skipped=skipped)


def load_training_rows_from_flagged_dir(*, flagged_dir: Path) -> list[TrainingRow]:
    hv_path = flagged_dir / "high_value" / "semantic_sorted.txt"
    ir_path = flagged_dir / "irrelevant" / "semantic_sorted.txt"

    rows: list[TrainingRow] = []

    hv_rows = load_semantic_sorted_tsv(path=hv_path, expected_label="high_value")
    ir_rows = load_semantic_sorted_tsv(path=ir_path, expected_label="irrelevant")

    def _scan(label: str) -> list[TrainingRow]:
        base = flagged_dir / label
        if not base.exists():
            return []
        exts = {".pdf", ".docx", ".txt", ".html", ".htm"}
        out: list[TrainingRow] = []
        for p in sorted(base.glob("**/*")):
            if not p.is_file():
                continue
            if p.name.lower() == "semantic_sorted.txt":
                continue
            if p.suffix.lower() not in exts:
                continue
            out.append(
                TrainingRow(
                    label=label,
                    local_path=str(p),
                    url="",
                    title=p.name,
                    relevance_score=None,
                    topic_similarity=None,
                    entity_density=None,
                )
            )
        return out

    rows.extend(hv_rows if hv_rows else _scan("high_value"))
    rows.extend(ir_rows if ir_rows else _scan("irrelevant"))
    return rows
