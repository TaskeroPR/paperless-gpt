# paperless-gpt

A Python script that automatically classifies documents in your [Paperless-ngx](https://docs.paperless-ngx.com/) inbox using a local LLM via [Ollama](https://ollama.com/). Supports multiple Paperless-ngx instances.

For each inbox document, the LLM analyzes the OCR text and assigns:

- **Title** -- concise, descriptive title with key identifiers
- **Date** -- the document's creation/issue date
- **Correspondent** -- person or organization associated with the document
- **Document Type** -- category like Invoice, Receipt, Letter, Contract, etc.
- **Tags** -- 1-5 relevant tags

The LLM prefers existing correspondents, document types, and tags from your Paperless instance. It only creates new ones when no existing entry matches.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- A Paperless-ngx instance with API access enabled
- An API token for each instance (found in *My Profile* in the Paperless-ngx web UI)

## Setup

```bash
git clone https://github.com/TaskeroPR/paperless-gpt.git
cd paperless-gpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example config and fill in your details:

```bash
cp config.example.yaml config.yaml
```

## Configuration

Edit `config.yaml`:

```yaml
ollama:
  url: "http://localhost:11434"
  model: "qwen3:8b"

paperless_instances:
  - name: "Home"
    url: "https://paperless.example.com"
    token: "your-api-token-here"
    inbox_tag: "Inbox"

  - name: "Work"
    url: "https://paperless-work.example.com"
    token: "another-api-token"
    inbox_tag: "Inbox"

processing:
  max_content_length: 4000  # Max characters of document text sent to LLM
  dry_run: false             # Log classifications without modifying documents
  log_level: "INFO"          # DEBUG, INFO, WARNING, ERROR
```

| Option | Description | Default |
|---|---|---|
| `ollama.url` | Ollama API endpoint | `http://localhost:11434` |
| `ollama.model` | Ollama model to use | `qwen3:8b` |
| `paperless_instances[].name` | Display name for the instance | required |
| `paperless_instances[].url` | Paperless-ngx base URL | required |
| `paperless_instances[].token` | API token | required |
| `paperless_instances[].inbox_tag` | Tag name that marks unprocessed documents | `Inbox` |
| `processing.max_content_length` | Truncate document text to this many characters | `4000` |
| `processing.dry_run` | If `true`, only log what would be changed | `false` |
| `processing.log_level` | Logging verbosity | `INFO` |

## Usage

```bash
python process.py
```

### What it does

1. Checks that the configured Ollama model is available (auto-pulls if not)
2. For each Paperless-ngx instance:
   - Fetches all documents tagged with the inbox tag
   - Skips documents that were already processed in a previous run
   - Fetches existing correspondents, document types, and tags
   - Sends each document's text to the LLM for classification
   - Creates new correspondents, document types, or tags if the LLM suggests ones that don't exist yet
   - Updates the document metadata via the API
   - Keeps the inbox tag so you can review the results manually

### Dry run

Set `dry_run: true` in your config to see what the script would do without making any changes:

```bash
# Or temporarily edit config.yaml
python process.py
```

The log output shows the classification results and what PATCH requests would be made.

### Re-processing documents

The script tracks processed document IDs in `processed.json` (one entry per instance). To re-process a document, remove its ID from the file, or delete `processed.json` entirely to re-process everything.

## LLM Model

The default model is `qwen3:8b`, which provides a good balance of accuracy and speed for document classification. It:

- Fits comfortably in 16GB RAM (~5.2GB at Q4 quantization)
- Runs well on Apple Silicon
- Supports structured JSON output
- Handles both German and English documents

If the model isn't available locally, the script pulls it automatically on first run.

Other models that work well (set in `ollama.model`):

| Model | Size | Notes |
|---|---|---|
| `qwen3:8b` | ~5GB | Recommended default |
| `qwen2.5:7b` | ~4.7GB | Good alternative |
| `llama3.1:8b` | ~4.7GB | Strong general-purpose |
| `mistral:7b` | ~4.1GB | Fast, efficient |

## How it works

```
Paperless-ngx                    Ollama (local)
     |                                |
     |-- GET /api/tags/ ------------->|
     |-- GET /api/documents/ ------->|
     |-- GET /api/correspondents/ -->|
     |-- GET /api/document_types/ -->|
     |                                |
     |   For each inbox document:     |
     |                                |
     |          document text ------->| POST /api/chat
     |          + existing metadata   | (structured JSON output)
     |                                |
     |          classification <------|
     |                                |
     |-- POST /api/tags/ (if new) -->|
     |-- PATCH /api/documents/{id}/ >|
```

## License

MIT
