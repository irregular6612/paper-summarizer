# Project Context: Paper Summary Pipeline

## Overview
This project is a Python-based pipeline designed to convert research paper PDFs into Obsidian-compatible Markdown notes. It automates the extraction of structure, text, and images (figures/tables) and uses LLMs to generate summaries.

## Tech Stack
*   **Language:** Python >= 3.12
*   **Dependency Manager:** `uv` (recommended) or `pip`
*   **Key Libraries:**
    *   `marker-pdf`: For PDF to Markdown/JSON conversion and structure extraction.
    *   `litellm`: For interfacing with various LLM providers (Ollama, OpenAI, Anthropic).
    *   `watchdog`: For monitoring directories for new PDF files.
    *   `pandas`, `pyyaml`: For data handling and configuration.

## Directory Structure
*   `src/`: Contains the core logic.
    *   `extractor.py`: Handles PDF parsing using `marker-pdf`.
    *   `summarizer.py`: Manages LLM interactions for summarization via `litellm`.
    *   `pipeline.py`: Orchestrates the extraction, summarization, and output generation.
    *   `watcher.py`: Implements the folder watching functionality.
    *   `config.py`: Configuration management.
    *   `markdown_gen.py`, `image_export.py`: Helper modules for output generation.
*   `target-pdf/`: Default directory for input PDF files (watched folder).
*   `output/`: Directory where processed Markdown and assets are saved.
*   `main.py`: CLI entry point.
*   `config.yaml`: Configuration file for LLM settings and paths.

## Setup and Usage

### Installation
```bash
uv sync  # Recommended
# OR
pip install -e .
```

### Configuration
Ensure `config.yaml` exists (generate with `python main.py init` if needed). Configure LLM provider (Ollama, OpenAI, Anthropic) and API keys.

### Running
*   **Process a single file:**
    ```bash
    python main.py process path/to/paper.pdf
    ```
*   **Process all files in a folder:**
    ```bash
    python main.py process-all --folder target-pdf
    ```
*   **Watch for new files:**
    ```bash
    python main.py watch
    # To process existing files on start:
    python main.py watch --process-existing
    ```

## Development Conventions
*   **Type Hinting:** Extensive use of Python type hints (`typing` module).
*   **Logging:** Uses the standard `logging` module.
*   **Error Handling:** Try-except blocks are used in the main CLI to catch processing failures.
*   **Code Style:** Follows standard Python conventions (PEP 8).

## How to run
uv run python main.py process "YOUR PDF"
