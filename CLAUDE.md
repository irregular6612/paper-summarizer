# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python pipeline that converts research paper PDFs into Obsidian-compatible Markdown notes with LLM-generated Korean summaries, extracted figures/tables, and rich metadata (frontmatter, TOC, internal links).

## Commands

```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Run with uv
uv run python main.py process "path/to/paper.pdf"    # Single PDF
uv run python main.py process-all -f target-pdf       # Batch process folder
uv run python main.py watch --process-existing         # Watch mode
uv run python main.py init                             # Generate default config.yaml
```

## Architecture

The pipeline flows: **PDF → extractor → (parallel: image_export + summarizer) → markdown_gen → output**

- **`main.py`** — CLI entry point (argparse-based commands: `process`, `process-all`, `watch`, `init`)
- **`src/pipeline.py`** — Orchestrator. Runs image export and LLM summarization in parallel via `ThreadPoolExecutor`, then assembles final markdown
- **`src/extractor.py`** — PDF parsing via `marker-pdf`. Uses singleton pattern for model caching (~5 deep learning models, ~750MB). MPS acceleration on Apple Silicon with CPU fallback for table recognition
- **`src/summarizer.py`** — LLM summarization via `litellm`. Supports Ollama/OpenAI/Anthropic. Parallel section summarization (max_workers=3). Includes Qwen3 thinking mode optimization and `<think>` tag cleanup
- **`src/markdown_gen.py`** — Generates Obsidian-compatible markdown with frontmatter, TOC, embedded images, and summary callouts/blockquotes
- **`src/image_export.py`** — Extracts figures/tables as PNGs with abbreviation-based naming (e.g., `CLIP_fig1.png`)
- **`src/watcher.py`** — Directory monitoring via `watchdog` with 2-second debounce for large file copies
- **`src/config.py`** — Dataclass-based config with env var expansion (`${VAR}` syntax), multi-path config file search

## Key Design Decisions

- **Custom surya-ocr fork** (`irregular6612/surya`, `mps-version` branch) for Apple Silicon MPS support
- **Korean output** with English technical terms preserved — prompts are in English but instruct the LLM to respond in Korean
- Summaries default to **blockquote style** but support callout style via config
- Output goes to `output/<Paper Title>/` with an `assets/` subfolder for images
- Config loaded from `config.yaml` (local) or `~/.paper-summary/config.yaml` (home)

## Configuration

`config.yaml` controls LLM provider/model, paths, markdown options, and output settings. Default LLM is Ollama with `qwen3:14b` at temperature 0.6. API keys use env var expansion: `api_key: "${OPENAI_API_KEY}"`.
