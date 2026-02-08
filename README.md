# DOJ Disclosures Crawler (GUI)

Cross-platform Python application that crawls the public DOJ Epstein disclosures site, discovers downloadable documents, downloads and parses them efficiently, indexes extracted text into SQLite FTS for fast search, and flags matches using keyword/regex/wildcard/fuzzy matching (with optional local semantic embeddings).

## Legal / Ethics Notice

On first launch, the app shows a plain-English notice reminding you to confirm you have the right to crawl/download the site’s content, and that the crawler obeys `robots.txt` and rate limits.

## Install

### Requirements

- Python 3.10+

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Tip: if you have multiple Python installs, you can skip activation and run everything through the venv interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

### Optional extras

OCR (requires installing Tesseract separately):

```powershell
pip install -e ".[ocr]"
```

Semantic matching / keyword suggestions:

```powershell
pip install -e ".[semantic]"
```

## Run

```powershell
doj-disclosures-gui
```

Or:

```powershell
python -m doj_disclosures.app
```

If you see `ModuleNotFoundError: No module named 'PySide6'`, you’re almost always running with the wrong interpreter.
Use the venv explicitly:

```powershell
.\.venv\Scripts\python.exe -m doj_disclosures.app
```

## Tests

```powershell
pip install -e ".[dev]"
pytest
```

## Packaging (PyInstaller)

```powershell
pip install pyinstaller
pyinstaller -y doj_disclosures.spec
```

The executable will be in `dist/`.
