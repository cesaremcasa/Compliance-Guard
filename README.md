# Compliance Guard v3.2.0

Compliance Guard is a small FastAPI service that returns structured,
NIST-oriented analysis guidance. The v3.2.0 API has one canonical entrypoint:
`src.api.main:app`.

The default backend is a deterministic fixture backend. It makes the API
usable and testable without CUDA, model downloads, or credentials. The real
Mistral + LoRA backend remains available as an explicit, lazy-loaded option;
its manual canary is still pending for this release.

## What is and is not verified

The fake backend verifies request/response contracts, caching, limits, and
fixture plumbing. It is not a compliance evaluator and its output must not be
used as evidence of control implementation.

The real backend can produce model-generated guidance, but this repository
does not claim accuracy, coverage, latency, security, or production readiness.
Responses are advisory and require review against the applicable NIST source,
organizational policy, and local evidence. See
[`docs/GPU_CANARY.md`](docs/GPU_CANARY.md) for the separate manual gate.

## Quickstart (CPU-only)

Python 3.9+ is supported. In a clean environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
COMPLIANCE_GUARD_BACKEND=fake uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"text":"Describe the access control requirements in AC-2."}'
```

`/health` is a liveness endpoint and does not load a model or require a GPU.
The first fake analysis loads only a small JSON fixture.

The older FAISS/LangChain ingestion adapters are not needed by the canonical
API. Install `requirements-rag.txt` only when using those legacy tools.

## API

### `POST /analyze`

Request:

```json
{"text":"Describe the access control requirements in AC-2."}
```

Response fields:

```json
{
  "request_id": "…",
  "text": "…",
  "framework": "NIST SP 800-53 Rev. 5",
  "findings": [
    {"control_id": "AC-2", "status": "not_evaluated", "summary": "…"}
  ],
  "citations": [
    {"source": "NIST SP 800-53 Rev. 5", "control_id": "AC-2", "locator": "AC-2"}
  ],
  "backend": "fake",
  "cached": false,
  "latency_seconds": 0.0001
}
```

`findings` and `citations` are structured metadata, not independently
verified evidence. `framework` currently identifies the intended NIST source;
callers must verify the source and locator.

### `POST /generate` (deprecated compatibility adapter)

`/generate` remains available for one release. It accepts the former
`{"text":"…"}` request and keeps `generated_text`, `request_id`, `cached`, and
`latency_seconds`, while also returning the `/analyze` fields. The response
contains `Deprecation: true` and `X-Compliance-Guard-Deprecated: true`. New
clients should use `/analyze`.

The former `POST /compliance` (`query` request, `answer`/`source` response) is
also retained as a deprecated adapter for existing callers.

`GET /metrics`, `POST /feedback`, and `GET /stats` remain available. Feedback
is a bounded local JSONL store and is not a training pipeline.

## Runtime limits

These are implementation limits, not a complete security boundary:

- request bodies on analysis/generation/feedback paths are capped at 16 KiB;
- text input is capped at 500 characters;
- analysis/generation paths allow 10 requests per client per minute;
- feedback allows 5 requests per client per minute;
- exact-match responses are cached under `/tmp/compliance_cache` (override with
  `COMPLIANCE_GUARD_CACHE_DIR`); entries are capped at 1,000 / 64 MiB by
  default (override with `COMPLIANCE_GUARD_CACHE_MAX_ENTRIES` and
  `COMPLIANCE_GUARD_CACHE_MAX_BYTES`) and evicted deterministically;
- there is no authentication, authorization, tenant isolation, durable audit
  trail, or network-facing deployment hardening in this repository.

The service should run behind an appropriately configured gateway and should
not be treated as an internet-facing compliance authority. No keyword
blacklist is presented as a prompt-injection or security defense.

## Backend selection

Set `COMPLIANCE_GUARD_BACKEND=fake` (the default) for local work. To use the
real backend, install the optional dependencies and set
`COMPLIANCE_GUARD_BACKEND=real`:

```bash
python -m pip install -r requirements-gpu.txt
COMPLIANCE_GUARD_BACKEND=real uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The real backend requires CUDA by default, downloads/loads the configured base
model lazily on the first request, and applies the LoRA adapter at
`models/checkpoints` by default. Configure `COMPLIANCE_GUARD_BASE_MODEL`,
`COMPLIANCE_GUARD_LORA_ADAPTER`, and `COMPLIANCE_GUARD_TOKENIZER` as needed.
Model-provider credentials are supplied through the operator's environment;
none are stored here.

## Golden set and tests

Run the deterministic golden set without starting a server:

```bash
python3 scripts/run_golden_set.py
```

This reports deterministic fixture/contract checks for the checked-in
`tests/golden_set.json`; it is not a semantic accuracy score. Install the
development test tools and run the unit tests with `pytest -q`:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

The older `scripts/run_batch_validation.py` remains as a server-based
`/generate` compatibility check.

## Docker

The Docker image still includes the optional GPU stack for operators who need
the real backend, but starts the canonical API and defaults to the fake
backend. Set `COMPLIANCE_GUARD_BACKEND=real` explicitly for a model-serving
deployment. The Compose monitoring services are optional and require the
non-default Grafana credentials in `.env`.

## Repository adapters

Older server/launcher files remain in `src/api/` and at the repository root for
callers that still reference them. They are not additional supported API
entrypoints; deployments should use `src.api.main:app`.

## License

MIT. See [`LICENSE`](LICENSE).
