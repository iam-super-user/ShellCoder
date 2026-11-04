# ShellCoder

A personal, fully local LLM assistant for exploit development, malware analysis, and
reverse engineering. It runs entirely on your own machine through [Ollama](https://ollama.com),
pairs the model with a small set of agent tools, and keeps a searchable knowledge vault
of your own research notes, source, and PDFs.

Nothing leaves the host. No API keys, no cloud calls.

## What's inside

- **`shellcoder.py`** — interactive agent loop. Talks to the local model, supports both
  Ollama's native tool-calling and a text-based `<tool_call>` fallback for models that
  don't expose tools.
- **`tools.py`** — the agent's tools: `read_file`, `write_file`, `run_bash`
  (asks for confirmation before running anything), and `search_vault`.
- **`data_vault.py`** — the RAG pipeline. Ingests code, text, PDFs (with OCR fallback for
  scanned pages), images, and `.docx` into a local Chroma vector DB. Watches folders for
  changes and re-indexes automatically.
- **`rag_engine.py`** — thin query helper the agent uses to hit the vault.
- **`Shellcoder.Modelfile`** — the Ollama model definition. Edit the identity, parameters,
  and purpose here, then build it (see below).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- [Tesseract](https://github.com/tesseract-ocr/tesseract) and
  [Poppler](https://poppler.freedesktop.org/) on your PATH (only needed for PDF/image OCR)

Install the Python deps:

```bash
pip install -r requirements.txt
```

## Setup

Build the model from the Modelfile (the name must stay `Shellcoder` unless you also change
`MODEL_NAME` in `shellcoder.py`):

```bash
ollama create Shellcoder -f Shellcoder.Modelfile
```

## Usage

Start the agent:

```bash
python shellcoder.py
```

Feed the knowledge vault and query it directly:

```bash
# Ingest a single file
python data_vault.py --ingest ./shellcode.c

# Bulk-ingest a folder (recursive)
python data_vault.py --ingest-dir ./research/

# Watch a folder and auto-ingest on save
python data_vault.py --watch ./reverse_engineering/

# Ask a question against everything you've ingested
python data_vault.py --query "What is the SEH offset mentioned in the PDF?"

# Drop chunks for files that no longer exist on disk
python data_vault.py --purge
```

Tune the model's behavior, temperature, context window, and persona in
`Shellcoder.Modelfile` — the comments in that file explain each knob.

## Disclaimer

ShellCoder is built for authorized security research, education, and testing on systems
you own or have explicit permission to assess. It is dual-use tooling: the same techniques
that help you understand and defend systems can cause harm if misused.

You are responsible for how you use it and for complying with all applicable laws and
regulations in your jurisdiction. The author accepts no liability for misuse or for any
damage resulting from use of this project. Don't point it at anything you aren't allowed to.
