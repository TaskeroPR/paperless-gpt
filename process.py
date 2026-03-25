#!/usr/bin/env python3
"""Classify Paperless-ngx inbox documents using a local Ollama LLM."""

import json
import logging
import os
import sys
import time

import requests
import yaml

PROCESSED_FILE = "processed.json"

logger = logging.getLogger("paperless-gpt")

SYSTEM_PROMPT = """\
You are a document classification assistant for a paperless document management system.
Analyze the OCR text of a document and assign appropriate metadata.
Documents may be in German or English — respond with metadata in the document's language.

Rules:
- correspondent: The person, company, or organization that sent or is primarily associated \
with this document. Prefer an existing correspondent from the provided list. \
Only suggest a new name if none of the existing ones match.
- document_type: Classify what kind of document this is (e.g., Invoice, Receipt, Letter, \
Contract, Bank Statement, Tax Document, Insurance Document). \
Prefer an existing type from the provided list. Only suggest a new name if none match.
- tags: Assign 1-5 relevant tags that describe the document's topic or category. \
Prefer existing tags from the provided list. You may suggest new tags only if truly needed.
- title: Write a concise, descriptive title (max 80 characters). Include key identifiers \
like invoice numbers, dates, or reference numbers when present.
- date: Extract the primary document date in YYYY-MM-DD format. This is usually the date \
the document was created or issued, NOT a due date or expiry date. Use null if undeterminable."""

OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "date": {"type": ["string", "null"]},
        "correspondent": {"type": ["string", "null"]},
        "document_type": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "date", "correspondent", "document_type", "tags"],
}


class PaperlessClient:
    """Communicates with a single Paperless-ngx instance."""

    def __init__(self, name: str, url: str, token: str, inbox_tag: str):
        self.name = name
        self.base_url = url.rstrip("/")
        self.inbox_tag = inbox_tag
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Token {token}"

    def _rewrite_url(self, url: str) -> str:
        """Replace scheme/host in a URL with the configured base_url.

        Paperless may return http:// in pagination URLs even when accessed
        via https:// through a reverse proxy.
        """
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        base = urlparse(self.base_url)
        return urlunparse(parsed._replace(scheme=base.scheme, netloc=base.netloc))

    def _get_paginated(self, endpoint: str, params: dict | None = None) -> list[dict]:
        url = f"{self.base_url}{endpoint}"
        all_results = []
        while url:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get("results", []))
            next_url = data.get("next")
            url = self._rewrite_url(next_url) if next_url else None
            params = None  # params already encoded in next URL
        return all_results

    def get_inbox_tag_id(self) -> int | None:
        tags = self._get_paginated("/api/tags/", {"name__iexact": self.inbox_tag})
        for tag in tags:
            if tag["name"].lower() == self.inbox_tag.lower():
                return tag["id"]
        return None

    def get_inbox_documents(self, inbox_tag_id: int) -> list[dict]:
        return self._get_paginated("/api/documents/", {"tags__id__all": inbox_tag_id})

    def get_correspondents(self) -> dict[int, str]:
        return {c["id"]: c["name"] for c in self._get_paginated("/api/correspondents/")}

    def get_document_types(self) -> dict[int, str]:
        return {d["id"]: d["name"] for d in self._get_paginated("/api/document_types/")}

    def get_tags(self) -> dict[int, str]:
        return {t["id"]: t["name"] for t in self._get_paginated("/api/tags/")}

    def _find_or_create(self, name: str, existing: dict[int, str], endpoint: str) -> int:
        name_lower = name.lower()
        for id_, existing_name in existing.items():
            if existing_name.lower() == name_lower:
                return id_
        resp = self.session.post(
            f"{self.base_url}{endpoint}",
            json={"name": name},
            timeout=30,
        )
        resp.raise_for_status()
        new_id = resp.json()["id"]
        existing[new_id] = name
        logger.info("[%s] Created new entry '%s' (id=%d) at %s", self.name, name, new_id, endpoint)
        return new_id

    def find_or_create_correspondent(self, name: str, existing: dict[int, str]) -> int:
        return self._find_or_create(name, existing, "/api/correspondents/")

    def find_or_create_document_type(self, name: str, existing: dict[int, str]) -> int:
        return self._find_or_create(name, existing, "/api/document_types/")

    def find_or_create_tag(self, name: str, existing: dict[int, str]) -> int:
        return self._find_or_create(name, existing, "/api/tags/")

    def update_document(self, doc_id: int, patch_data: dict) -> dict:
        resp = self.session.patch(
            f"{self.base_url}/api/documents/{doc_id}/",
            json=patch_data,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def ensure_model(ollama_url: str, model: str) -> None:
    """Check if the model is available locally, pull it if not."""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Cannot connect to Ollama at {ollama_url}: {e}") from e

    available = [m["name"] for m in resp.json().get("models", [])]
    # Check with and without :latest suffix
    if model in available or f"{model}:latest" in available:
        logger.info("Model '%s' is available", model)
        return

    # Also check if the model name without tag matches any available model
    model_base = model.split(":")[0]
    for name in available:
        if name.split(":")[0] == model_base:
            logger.info("Model '%s' is available (as '%s')", model, name)
            return

    logger.info("Model '%s' not found locally, pulling...", model)
    resp = requests.post(
        f"{ollama_url}/api/pull",
        json={"name": model, "stream": True},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            status = json.loads(line)
            if "status" in status:
                logger.info("Pull: %s", status["status"])
            if status.get("error"):
                raise RuntimeError(f"Failed to pull model '{model}': {status['error']}")
    logger.info("Model '%s' pulled successfully", model)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text at a word boundary."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated + "\n[... truncated]"


def classify_document(
    ollama_url: str,
    model: str,
    document_text: str,
    correspondents: list[str],
    document_types: list[str],
    tags: list[str],
    max_content_length: int = 4000,
) -> dict:
    """Send document text to Ollama LLM and get structured classification back."""
    text = truncate_text(document_text, max_content_length)

    user_prompt = f"""Classify the following document.

EXISTING CORRESPONDENTS:
{', '.join(correspondents) if correspondents else '(none yet)'}

EXISTING DOCUMENT TYPES:
{', '.join(document_types) if document_types else '(none yet)'}

EXISTING TAGS:
{', '.join(tags) if tags else '(none yet)'}

DOCUMENT TEXT:
---
{text}
---"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": OLLAMA_SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        },
    }

    resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    logger.debug("LLM raw response: %s", content)
    return json.loads(content)


def load_processed(instance_name: str) -> set[int]:
    """Load the set of already-processed document IDs for an instance."""
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE) as f:
        data = json.load(f)
    return set(data.get(instance_name, []))


def save_processed(instance_name: str, doc_ids: set[int]) -> None:
    """Save the set of processed document IDs for an instance."""
    data = {}
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f:
            data = json.load(f)
    data[instance_name] = sorted(doc_ids)
    with open(PROCESSED_FILE, "w") as f:
        json.dump(data, f, indent=2)


def process_instance(client: PaperlessClient, ollama_url: str, model: str, config: dict) -> None:
    """Process all inbox documents for one Paperless-ngx instance."""
    processing = config.get("processing", {})
    dry_run = processing.get("dry_run", False)
    max_content_length = processing.get("max_content_length", 4000)

    processed_ids = load_processed(client.name)

    inbox_tag_id = client.get_inbox_tag_id()
    if inbox_tag_id is None:
        logger.warning("[%s] Inbox tag '%s' not found, skipping", client.name, client.inbox_tag)
        return

    documents = client.get_inbox_documents(inbox_tag_id)
    if not documents:
        logger.info("[%s] No inbox documents found", client.name)
        return

    # Filter out already-processed documents
    unprocessed = [d for d in documents if d["id"] not in processed_ids]
    logger.info("[%s] Found %d inbox document(s), %d new", client.name, len(documents), len(unprocessed))

    if not unprocessed:
        return

    correspondents = client.get_correspondents()
    document_types = client.get_document_types()
    tags = client.get_tags()

    total = len(unprocessed)
    for idx, doc in enumerate(unprocessed, 1):
        doc_id = doc["id"]
        doc_title = doc.get("title", f"Document {doc_id}")
        logger.info("[%s] [%d/%d] Processing: %s (id=%d)", client.name, idx, total, doc_title, doc_id)

        doc_start = time.monotonic()
        try:
            content = doc.get("content", "")
            if not content.strip():
                logger.warning("[%s] Document %d has no text content, skipping", client.name, doc_id)
                continue

            result = classify_document(
                ollama_url=ollama_url,
                model=model,
                document_text=content,
                correspondents=list(correspondents.values()),
                document_types=list(document_types.values()),
                tags=list(tags.values()),
                max_content_length=max_content_length,
            )

            logger.info("[%s] Classification for doc %d: %s", client.name, doc_id, json.dumps(result, ensure_ascii=False))

            patch = {}

            # Title
            if result.get("title"):
                patch["title"] = result["title"]

            # Date
            if result.get("date"):
                patch["created"] = result["date"]

            # Correspondent
            if result.get("correspondent"):
                try:
                    patch["correspondent"] = client.find_or_create_correspondent(
                        result["correspondent"], correspondents
                    )
                except requests.RequestException as e:
                    logger.error("[%s] Failed to resolve correspondent '%s': %s", client.name, result["correspondent"], e)

            # Document type
            if result.get("document_type"):
                try:
                    patch["document_type"] = client.find_or_create_document_type(
                        result["document_type"], document_types
                    )
                except requests.RequestException as e:
                    logger.error("[%s] Failed to resolve document type '%s': %s", client.name, result["document_type"], e)

            # Tags — merge LLM suggestions with existing document tags, keep inbox tag
            existing_doc_tag_ids = set(doc.get("tags", []))
            new_tag_ids = set()
            for tag_name in result.get("tags", []):
                try:
                    new_tag_ids.add(client.find_or_create_tag(tag_name, tags))
                except requests.RequestException as e:
                    logger.error("[%s] Failed to resolve tag '%s': %s", client.name, tag_name, e)
            merged_tags = existing_doc_tag_ids | new_tag_ids
            merged_tags.add(inbox_tag_id)  # Always keep inbox tag
            patch["tags"] = list(merged_tags)

            if dry_run:
                logger.info("[%s] DRY RUN — would update doc %d with: %s", client.name, doc_id, json.dumps(patch, ensure_ascii=False))
            else:
                client.update_document(doc_id, patch)
                logger.info("[%s] Updated document %d", client.name, doc_id)

            # Mark as processed so it won't be re-classified on next run
            processed_ids.add(doc_id)
            save_processed(client.name, processed_ids)

            elapsed = time.monotonic() - doc_start
            logger.info("[%s] Document %d processed in %.1fs", client.name, doc_id, elapsed)

        except Exception as e:
            elapsed = time.monotonic() - doc_start
            logger.error("[%s] Failed to process document %d after %.1fs: %s", client.name, doc_id, elapsed, e)
            continue


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate the YAML config file."""
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {path}", file=sys.stderr)
        print("Copy config.example.yaml to config.yaml and fill in your settings.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Invalid YAML in {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not config.get("ollama", {}).get("url"):
        print("Config error: ollama.url is required", file=sys.stderr)
        sys.exit(1)
    if not config.get("ollama", {}).get("model"):
        print("Config error: ollama.model is required", file=sys.stderr)
        sys.exit(1)
    if not config.get("paperless_instances"):
        print("Config error: at least one paperless instance is required", file=sys.stderr)
        sys.exit(1)

    for i, instance in enumerate(config["paperless_instances"]):
        for field in ("name", "url", "token"):
            if not instance.get(field):
                print(f"Config error: paperless_instances[{i}].{field} is required", file=sys.stderr)
                sys.exit(1)

    return config


def main() -> None:
    config = load_config()

    log_level = config.get("processing", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ollama_url = config["ollama"]["url"].rstrip("/")
    model = config["ollama"]["model"]

    try:
        ensure_model(ollama_url, model)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    for instance_cfg in config["paperless_instances"]:
        client = PaperlessClient(
            name=instance_cfg["name"],
            url=instance_cfg["url"],
            token=instance_cfg["token"],
            inbox_tag=instance_cfg.get("inbox_tag", "Inbox"),
        )
        logger.info("Processing instance: %s (%s)", client.name, client.base_url)
        try:
            process_instance(client, ollama_url, model, config)
        except requests.RequestException as e:
            logger.error("Instance '%s' unreachable: %s", client.name, e)
            continue

    logger.info("Done.")


if __name__ == "__main__":
    main()
