from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir


@dataclass(frozen=True)
class Paths:
    app_dir: Path
    db_path: Path
    log_path: Path
    output_dir: Path
    keywords_path: Path


@dataclass(frozen=True)
class CrawlSettings:
    start_url: str = "https://www.justice.gov/epstein/doj-disclosures"
    allow_offsite: bool = False
    follow_discovered_pages: bool = False
    max_concurrency: int = 6
    requests_per_second: float = 1.0
    user_agent: str = "DOJDisclosuresCrawler/0.1 (+respect-robots)"
    max_retries: int = 4
    backoff_base_seconds: float = 0.75
    ocr_enabled: bool = False
    # OCR settings
    # `ocr_engine`: currently supports "tesseract" or "none".
    ocr_engine: str = "tesseract"
    ocr_dpi: int = 200
    ocr_preprocess: bool = True
    ocr_median_filter: bool = True
    # If None, use an automatic (Otsu) threshold.
    ocr_threshold: int | None = None
    # Named Entity Recognition (NER)
    ner_enabled: bool = True
    ner_engine: str = "spacy"  # "spacy" or "regex"
    ner_spacy_model: str = "en_core_web_sm"
    # Embedding index + hybrid search (optional; requires semantic extra)
    embedding_index_enabled: bool = False
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Storage layout
    storage_layout: str = "flat"  # "flat" or "hashed"
    # Redaction detection
    redaction_detection_enabled: bool = True
    redaction_page_score_threshold: float = 0.25
    semantic_enabled: bool = False
    semantic_threshold: float = 0.62
    auto_download: bool = True
    manual_review_only: bool = False
    age_verify_opt_in: bool = False
    # When enabled, documents that return HTTP 304 (Not Modified) will still be re-parsed
    # and re-indexed from the cached local file. Useful for testing new scoring/matching.
    reprocess_cached_on_not_modified: bool = False
    # Some DOJ listing pages (notably dataset pagination ?page>=1) may be protected by Akamai
    # and return 403 to non-browser HTTP clients. When enabled, the crawler will attempt
    # to fetch HTML using Playwright as a fallback.
    use_browser_for_blocked_pages: bool = False
    # Optional raw Cookie header value to send with requests.
    # Useful if a CDN/WAF requires a session cookie obtained via a browser.
    cookie_header: str = ""
    stopwords: str = ""
    query: str = ""  # optional boolean/proximity query

    # Feedback learning: when semantic embeddings are available, use the user's
    # high_value/irrelevant labels to bias automatic flagging decisions.
    feedback_auto_flag_enabled: bool = True
    # If (sim_to_high_value - sim_to_irrelevant) >= threshold, auto-flag.
    feedback_auto_flag_threshold: float = 0.22
    # If (sim_to_high_value - sim_to_irrelevant) <= threshold, auto-triage unless
    # there is strong keyword evidence.
    feedback_auto_triage_threshold: float = -0.22

    # AI flagger: a lightweight learned classifier trained from your
    # Flagged/high_value vs Flagged/irrelevant folders (see semantic_sorted.txt).
    # If no trained model exists, this has no effect.
    ai_flagger_enabled: bool = True
    # If P(high_value) >= threshold, auto-flag.
    ai_flagger_flag_threshold: float = 0.80
    # If P(high_value) <= threshold, auto-triage unless there is strong keyword evidence.
    ai_flagger_triage_threshold: float = 0.20


@dataclass(frozen=True)
class AppConfig:
    paths: Paths
    crawl: CrawlSettings
    first_run_acknowledged: bool = False
    last_seed_urls: tuple[str, ...] = ()

    @staticmethod
    def default() -> "AppConfig":
        base = Path(user_data_dir(appname="DOJDisclosuresCrawler", appauthor=False))
        base.mkdir(parents=True, exist_ok=True)
        output_dir = base / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        keywords_path = base / "keywords.json"
        crawl = CrawlSettings()
        return AppConfig(
            paths=Paths(
                app_dir=base,
                db_path=base / "state.sqlite3",
                log_path=base / "app.log",
                output_dir=output_dir,
                keywords_path=keywords_path,
            ),
            crawl=crawl,
            first_run_acknowledged=False,
            last_seed_urls=(crawl.start_url,),
        )

    @property
    def config_path(self) -> Path:
        return self.paths.app_dir / "config.json"

    def to_json(self) -> dict[str, Any]:
        return {
            "first_run_acknowledged": self.first_run_acknowledged,
            "paths": {
                "output_dir": str(self.paths.output_dir),
                "keywords_path": str(self.paths.keywords_path),
            },
            "crawl": {**self.crawl.__dict__},
            "ui": {
                "last_seed_urls": list(self.last_seed_urls),
            },
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "AppConfig":
        cfg = AppConfig.default()
        crawl = replace(cfg.crawl, **(data.get("crawl") or {}))
        output_dir = Path((data.get("paths") or {}).get("output_dir", str(cfg.paths.output_dir)))
        keywords_path = Path((data.get("paths") or {}).get("keywords_path", str(cfg.paths.keywords_path)))
        output_dir.mkdir(parents=True, exist_ok=True)
        keywords_path.parent.mkdir(parents=True, exist_ok=True)
        paths = replace(cfg.paths, output_dir=output_dir, keywords_path=keywords_path)

        ui = data.get("ui") or {}
        raw_seeds = ui.get("last_seed_urls")
        seeds: list[str] = []
        if isinstance(raw_seeds, list):
            for x in raw_seeds:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    seeds.append(s)
        if not seeds:
            seeds = [crawl.start_url]

        return replace(
            cfg,
            paths=paths,
            crawl=crawl,
            first_run_acknowledged=bool(data.get("first_run_acknowledged", False)),
            last_seed_urls=tuple(seeds),
        )

    @classmethod
    def load(cls) -> "AppConfig":
        cfg = cls.default()
        if cfg.config_path.exists():
            try:
                data = json.loads(cfg.config_path.read_text(encoding="utf-8"))
                cfg = cls.from_json(data)
            except Exception:
                cfg = cls.default()
        return cfg

    def save(self) -> None:
        self.config_path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
