from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from doj_disclosures.core.ai_flagger import load_training_rows_from_flagged_dir, save_ai_flagger_model, train_flagger_from_rows
from doj_disclosures.core.config import AppConfig, CrawlSettings
from doj_disclosures.core.crawler import Crawler, looks_downloadable
from doj_disclosures.core.db import Database
from doj_disclosures.core.downloader import Downloader, NotModifiedError
from doj_disclosures.core.matching import KeywordMatcher
from doj_disclosures.core.parser import DocumentParser
from doj_disclosures.core.pipeline import PipelineDeps, PipelineInput, build_semantic_context_async, process_document
from doj_disclosures.core.relevance import load_url_penalties
from doj_disclosures.core.storage_gating import plan_storage
from doj_disclosures.core.feedback import PHRASE_BLACKLIST_KEY, URL_PENALTIES_KEY
from doj_disclosures.core.release_monitor import store_snapshot_and_diff
from doj_disclosures.core.triage_index import write_semantic_sorted_index


logger = logging.getLogger(__name__)


async def train_flagger_cli(*, config: AppConfig, flagged_dir: Path, model_name: str, max_examples: int | None = None) -> int:
    """Train the AI flagger from your flagged folders and save it to the DB.

    Uses semantic_sorted.txt + the referenced local files for labels and features.
    """

    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    db = Database(config.paths.db_path)
    db.initialize_sync()

    s = config.crawl
    semantic = await build_semantic_context_async(settings=s, db=db)
    if semantic.provider is None:
        logger.error("Semantic provider unavailable; install semantic extra and try again")
        return 2

    rows = load_training_rows_from_flagged_dir(flagged_dir=flagged_dir)
    if not rows:
        logger.error("No labeled rows found under %s (expected high_value/semantic_sorted.txt and irrelevant/semantic_sorted.txt)", flagged_dir)
        return 2

    parser = DocumentParser(
        ocr_enabled=s.ocr_enabled,
        ocr_engine=getattr(s, "ocr_engine", "tesseract"),
        ocr_dpi=int(getattr(s, "ocr_dpi", 200)),
        ocr_preprocess=bool(getattr(s, "ocr_preprocess", True)),
        ocr_median_filter=bool(getattr(s, "ocr_median_filter", True)),
        ocr_threshold=getattr(s, "ocr_threshold", None),
    )

    res = train_flagger_from_rows(
        rows=rows,
        provider=semantic.provider,
        parser=parser,
        model_name=model_name,
        max_examples=max_examples,
    )

    if res.hv_centroid is not None:
        await db.set_feedback_centroid(label="high_value", model_name=model_name, centroid=res.hv_centroid)
    if res.ir_centroid is not None:
        await db.set_feedback_centroid(label="irrelevant", model_name=model_name, centroid=res.ir_centroid)

    if res.model is not None:
        await save_ai_flagger_model(db=db, model=res.model)
        logger.info(
            "trained AI flagger model_name=%s used=%s skipped=%s metrics=%s",
            model_name,
            res.n_used,
            res.skipped,
            json.dumps(res.model.metrics or {}),
        )
        return 0

    logger.warning("centroids rebuilt but classifier not trained (need more labeled examples). used=%s skipped=%s", res.n_used, res.skipped)
    return 1


def _load_keywords_sync(path: Path) -> list[str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "seed_keywords" in data:
                return [str(x) for x in (data.get("seed_keywords") or [])]
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            return []
    return []


async def run_headless(*, config: AppConfig, seed_urls: list[str]) -> int:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    db = Database(config.paths.db_path)
    db.initialize_sync()

    s = config.crawl

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=s.max_concurrency)
    pause = asyncio.Event(); pause.set()
    stop = asyncio.Event()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        crawler = Crawler(db=db, settings=s, session=session, pause_event=pause, stop_event=stop)
        await db.clear_pending_urls()
        await crawler.initialize(seed_urls=seed_urls)

        storage = plan_storage(config.paths.output_dir)
        downloader = Downloader(settings=s, session=session, output_dir=storage.raw_dir, pause_event=pause, stop_event=stop)

        penalties = load_url_penalties(await db.kv_get(URL_PENALTIES_KEY))
        semantic = await build_semantic_context_async(settings=s, db=db)

        keywords = _load_keywords_sync(config.paths.keywords_path)
        # Phrase blacklist learned from feedback.
        blacklist: set[str] = set()
        raw_bl = await db.kv_get(PHRASE_BLACKLIST_KEY)
        try:
            data = json.loads(raw_bl) if raw_bl else []
            if isinstance(data, list):
                blacklist = {str(x).strip() for x in data if str(x).strip()}
        except Exception:
            blacklist = set()
        if blacklist:
            keywords = [k for k in keywords if str(k).strip() and str(k).strip() not in blacklist]
        matcher = KeywordMatcher(
            keywords=keywords,
            query=s.query,
            fuzzy_enabled=True,
            semantic_enabled=s.semantic_enabled,
            semantic_threshold=s.semantic_threshold,
            stopwords={w.strip().lower() for w in s.stopwords.split(",") if w.strip()},
        )
        parser = DocumentParser(
            ocr_enabled=s.ocr_enabled,
            ocr_engine=getattr(s, "ocr_engine", "tesseract"),
            ocr_dpi=int(getattr(s, "ocr_dpi", 200)),
            ocr_preprocess=bool(getattr(s, "ocr_preprocess", True)),
            ocr_median_filter=bool(getattr(s, "ocr_median_filter", True)),
            ocr_threshold=getattr(s, "ocr_threshold", None),
        )

        pipeline_deps = PipelineDeps(
            settings=s,
            db=db,
            storage=storage,
            parser=parser,
            matcher=matcher,
            penalties=penalties,
            semantic=semantic,
        )

        processed = 0
        downloaded = 0
        flagged = 0

        async for item in crawler.iter_discovered():
            item_url = item.url
            kind = "document" if looks_downloadable(item_url) else "page"
            now = datetime.now(timezone.utc).isoformat()
            await db.update_url_attempt(url=item_url, status="processing", last_attempt_at=now, http_status=None, error=None)
            try:
                if kind == "page":
                    await crawler.process_page(item_url)
                else:
                    etag, last_modified = await db.get_url_cache_headers(url=item_url)
                    cache_headers: dict[str, str] = {}
                    if etag:
                        cache_headers["If-None-Match"] = etag
                    if last_modified:
                        cache_headers["If-Modified-Since"] = last_modified

                    dl = await downloader.download(item_url, cache_headers=(cache_headers or None))
                    downloaded += 1

                    out = await process_document(
                        deps=pipeline_deps,
                        inp=PipelineInput(
                            url=item_url,
                            final_url=dl.final_url,
                            local_path=Path(str(dl.local_path)),
                            content_type=dl.content_type,
                            file_size=dl.file_size,
                            sha256=dl.sha256,
                            fetched_at=dl.fetched_at,
                            etag=dl.etag,
                            last_modified=dl.last_modified,
                        ),
                        now=now,
                        allow_move=True,
                        reprocess_existing=False,
                        log=lambda m: logger.info("%s", m),
                    )

                    if out.hits:
                        flagged += 1

                    await db.update_url_attempt(
                        url=item_url,
                        status="done",
                        last_attempt_at=now,
                        http_status=200,
                        error=None,
                        content_type=dl.content_type,
                        title=out.parsed.title,
                        final_url=dl.final_url,
                        local_path=str(out.final_path),
                        sha256=dl.sha256,
                        etag=dl.etag,
                        last_modified=dl.last_modified,
                    )

                processed += 1
                if processed % 25 == 0:
                    logger.info("processed=%s downloaded=%s flagged=%s", processed, downloaded, flagged)
            except NotModifiedError:
                processed += 1
                await db.update_url_attempt(url=item_url, status="done", last_attempt_at=now, http_status=304, error=None)
            except Exception as e:
                processed += 1
                await db.update_url_attempt(url=item_url, status="retry", last_attempt_at=now, http_status=None, error=str(e))

        # Release snapshot + diff (best-effort)
        try:
            diff = await store_snapshot_and_diff(db)
            logger.info(
                "release_diff added=%s changed=%s removed=%s",
                len(diff.added),
                len(diff.changed),
                len(diff.removed),
            )
        except Exception:
            pass

        # Write semantic index files
        try:
            rows = await db.query_flagged_with_metrics(limit=100000)
            write_semantic_sorted_index(out_dir=storage.flagged_dir, rows=rows)
            hv = [r for r in rows if str(r.get("review_status") or "").lower() == "high_value"]
            ir = [r for r in rows if str(r.get("review_status") or "").lower() == "irrelevant"]
            write_semantic_sorted_index(out_dir=storage.flagged_dir / "high_value", rows=hv)
            write_semantic_sorted_index(out_dir=storage.flagged_dir / "irrelevant", rows=ir)
        except Exception:
            pass

    logger.info("done processed=%s downloaded=%s flagged=%s", processed, downloaded, flagged)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="doj-disclosures-cli", description="Headless crawl mode for DOJ Disclosures crawler")
    p.add_argument("--config", type=str, default="", help="Path to config.json (defaults to app data config)")
    p.add_argument("--db", type=str, default="", help="Path to SQLite DB (overrides config)")
    p.add_argument("--output", type=str, default="", help="Output directory for downloads (overrides config)")
    p.add_argument("--keywords", type=str, default="", help="Keywords JSON file (overrides config)")
    p.add_argument("--seed", action="append", default=[], help="Seed URL (repeatable)")
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--rps", type=float, default=None, help="Base requests per second")
    p.add_argument("--follow-pages", action="store_true", help="Follow discovered pages (not just enqueue downloadable docs)")
    p.add_argument("--allow-offsite", action="store_true")
    p.add_argument("--age-verify-opt-in", action="store_true")
    p.add_argument("--storage-layout", choices=["flat", "hashed"], default=None)

    # Training mode for AI flagger
    p.add_argument("--train-flagger", action="store_true", help="Train AI flagger from flagged folders and save model to DB")
    p.add_argument("--flagged-dir", type=str, default="", help="Path to Flagged directory (defaults to <output>/Flagged)")
    p.add_argument("--model-name", type=str, default="", help="Embedding model name override")
    p.add_argument("--max-train-examples", type=int, default=0, help="Cap number of labeled docs used for training (0 = no cap)")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    cfg = AppConfig.load()
    if args.config:
        p = Path(args.config)
        if p.exists():
            try:
                cfg = AppConfig.from_json(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                cfg = AppConfig.load()

    paths = cfg.paths
    if args.db:
        paths = replace(paths, db_path=Path(args.db))
    if args.output:
        paths = replace(paths, output_dir=Path(args.output))
    if args.keywords:
        paths = replace(paths, keywords_path=Path(args.keywords))

    crawl = cfg.crawl
    overrides: dict[str, object] = {}
    if args.max_concurrency is not None:
        overrides["max_concurrency"] = int(args.max_concurrency)
    if args.rps is not None:
        overrides["requests_per_second"] = float(args.rps)
    if args.follow_pages:
        overrides["follow_discovered_pages"] = True
    if args.allow_offsite:
        overrides["allow_offsite"] = True
    if args.age_verify_opt_in:
        overrides["age_verify_opt_in"] = True
    if args.storage_layout:
        overrides["storage_layout"] = str(args.storage_layout)

    if overrides:
        crawl = replace(crawl, **overrides)

    cfg = replace(cfg, paths=paths, crawl=crawl)

    if bool(getattr(args, "train_flagger", False)):
        storage = plan_storage(cfg.paths.output_dir)
        flagged_dir = Path(args.flagged_dir) if str(getattr(args, "flagged_dir", "") or "").strip() else storage.flagged_dir
        model_name = str(args.model_name).strip() if str(getattr(args, "model_name", "") or "").strip() else str(crawl.embedding_model_name)
        max_examples = int(args.max_train_examples) if int(getattr(args, "max_train_examples", 0) or 0) > 0 else None
        raise SystemExit(asyncio.run(train_flagger_cli(config=cfg, flagged_dir=flagged_dir, model_name=model_name, max_examples=max_examples)))

    seed_urls = [str(x).strip() for x in (args.seed or []) if str(x).strip()]
    if not seed_urls:
        seed_urls = list(cfg.last_seed_urls) if cfg.last_seed_urls else [cfg.crawl.start_url]

    raise SystemExit(asyncio.run(run_headless(config=cfg, seed_urls=seed_urls)))
