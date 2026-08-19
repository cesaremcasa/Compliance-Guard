"""Canonical Compliance Guard API entrypoint.

The process is usable on a CPU-only machine with the deterministic fake
backend. A real LoRA/GPU backend is opt-in and loaded only on the first
analysis request, so liveness does not depend on model downloads or CUDA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .backends import BackendManager, BackendUnavailable

try:  # Prometheus is part of the runtime requirements, but health stays importable.
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover - only used in a deliberately minimal environment.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _Metric:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def inc(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def dec(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def observe(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def labels(self, **_kwargs: Any) -> "_Metric":
            return self

    Counter = Histogram = Gauge = _Metric  # type: ignore[misc,assignment]

    def generate_latest() -> bytes:
        return b""


MAX_INPUT_LENGTH = int(os.getenv("COMPLIANCE_GUARD_MAX_INPUT_LENGTH", "500"))
MAX_API_BODY_BYTES = int(os.getenv("COMPLIANCE_GUARD_MAX_BODY_BYTES", str(16 * 1024)))
RATE_LIMIT = int(os.getenv("COMPLIANCE_GUARD_RATE_LIMIT", "10"))
FEEDBACK_RATE_LIMIT = int(os.getenv("COMPLIANCE_GUARD_FEEDBACK_RATE_LIMIT", "5"))
RATE_WINDOW_SECONDS = 60.0
CACHE_DIR = os.getenv("COMPLIANCE_GUARD_CACHE_DIR", "/tmp/compliance_cache")
CACHE_MAX_ENTRIES = int(os.getenv("COMPLIANCE_GUARD_CACHE_MAX_ENTRIES", "1000"))
CACHE_MAX_BYTES = int(
    os.getenv("COMPLIANCE_GUARD_CACHE_MAX_BYTES", str(64 * 1024 * 1024))
)
FEEDBACK_DIR = os.getenv("COMPLIANCE_GUARD_FEEDBACK_DIR", "/tmp/feedback")
MAX_FEEDBACK_STORAGE_BYTES = 5 * 1024 * 1024

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("compliance-guard")

REQUEST_COUNT = Counter(
    "compliance_request_total", "Total requests", ["method", "endpoint", "status"]
)
LATENCY_HISTOGRAM = Histogram("compliance_latency_seconds", "Analysis latency")
CACHE_HIT_COUNTER = Counter("cache_hit_total", "Cache hits")
CACHE_MISS_COUNTER = Counter("cache_miss_total", "Cache misses")
FEEDBACK_COUNT = Counter("user_feedback_total", "User feedback", ["rating"])
ACTIVE_REQUESTS = Gauge("active_requests", "Active requests")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    framework: Optional[str] = Field(default=None, min_length=1, max_length=128)

    class Config:
        extra = "forbid"


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)

    class Config:
        extra = "forbid"


class ComplianceRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)

    class Config:
        extra = "forbid"


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=128)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=1000)

    class Config:
        extra = "forbid"


class SlidingWindowRateLimiter:
    """Small in-process limiter retaining the former 10/min and 5/min limits."""

    def __init__(self) -> None:
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, scope: str, identity: str, limit: int) -> Tuple[bool, int]:
        now = time.monotonic()
        key = (scope, identity)
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - RATE_WINDOW_SECONDS:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(events[0] + RATE_WINDOW_SECONDS - now))
                return False, retry_after
            events.append(now)
            return True, 0

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class SimpleCache:
    """Bounded-input, disk-backed exact-match cache with atomic writes.

    Eviction is deterministic (oldest mtime, then filename) and only considers
    cache files matching the generated 32-character MD5 ``.json`` name.
    """

    def __init__(
        self,
        cache_dir: str,
        max_entries: int = CACHE_MAX_ENTRIES,
        max_bytes: int = CACHE_MAX_BYTES,
    ) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("cache limits must be positive")
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _cache_key(text: str) -> str:
        # Keep the MD5 key shape used by the v3 adapter for cache compatibility.
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_cache_file(path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix == ".json"
            and len(path.stem) == 32
            and all(char in string.hexdigits.lower() for char in path.stem)
        )

    def _cache_files_locked(self) -> List[Tuple[int, str, Path, int]]:
        files: List[Tuple[int, str, Path, int]] = []
        for path in self.cache_dir.glob("*.json"):
            if not self._is_cache_file(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime_ns, path.name, path, stat.st_size))
        files.sort(key=lambda item: (item[0], item[1]))
        return files

    def _evict_locked(self) -> None:
        files = self._cache_files_locked()
        total_bytes = sum(item[3] for item in files)
        while files and (
            len(files) > self.max_entries or total_bytes > self.max_bytes
        ):
            _mtime, _name, victim, size = files.pop(0)
            try:
                victim.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Could not evict cache entry %s: %s", victim, exc)
                continue
            total_bytes -= size

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        path = self.cache_dir / f"{self._cache_key(text)}.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable cache entry %s: %s", path, exc)
            return None
        return value if isinstance(value, dict) else None

    def set(self, text: str, value: Dict[str, Any]) -> None:
        path = self.cache_dir / f"{self._cache_key(text)}.json"
        temp_path = path.with_suffix(".json.tmp")
        with self._lock:
            self._evict_locked()
            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False)
                os.replace(temp_path, path)
                self._evict_locked()
            except (OSError, TypeError) as exc:
                logger.warning("Could not persist cache entry %s: %s", path, exc)
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


class FeedbackStore:
    def __init__(self, feedback_dir: str, max_bytes: int = MAX_FEEDBACK_STORAGE_BYTES) -> None:
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.jsonl"
        self.max_bytes = max_bytes
        self._lock = Lock()

    def save(self, value: Dict[str, Any]) -> None:
        encoded = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            current_size = self.feedback_file.stat().st_size if self.feedback_file.exists() else 0
            if current_size + len(encoded) > self.max_bytes:
                raise RuntimeError("feedback storage capacity reached")
            with self.feedback_file.open("ab") as handle:
                handle.write(encoded)


app = FastAPI(title="Compliance Guard API", version="3.2.0")
backend_manager = BackendManager()
cache = SimpleCache(CACHE_DIR)
feedback_store = FeedbackStore(FEEDBACK_DIR)
rate_limiter = SlidingWindowRateLimiter()

BODY_LIMITED_PATHS = {"/analyze", "/generate", "/compliance", "/feedback"}


@app.middleware("http")
async def request_guards(request: Request, call_next: Any) -> Response:
    """Apply body bounds and per-client rate limits before Pydantic parsing."""

    if request.method == "POST" and request.url.path in BODY_LIMITED_PATHS:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_API_BODY_BYTES:
                    return JSONResponse(
                        status_code=413, content={"detail": "Request body is too large"}
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})

        identity = request.client.host if request.client else "unknown"
        scope = "feedback" if request.url.path == "/feedback" else "analysis"
        limit = FEEDBACK_RATE_LIMIT if scope == "feedback" else RATE_LIMIT
        allowed, retry_after = rate_limiter.check(scope, identity, limit)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        chunks: List[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_API_BODY_BYTES:
                return JSONResponse(
                    status_code=413, content={"detail": "Request body is too large"}
                )
            chunks.append(chunk)
        request._body = b"".join(chunks)

    return await call_next(request)


def _request_id() -> str:
    return uuid.uuid4().hex[:16]


def _response_from_result(
    result: Dict[str, Any], request_id: str, cached: bool, latency_seconds: float
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "text": result["text"],
        "framework": result["framework"],
        "findings": result["findings"],
        "citations": result["citations"],
        "backend": result.get("backend"),
        "cached": cached,
        "latency_seconds": round(latency_seconds, 4),
    }


def _analyze_text(text: str) -> Tuple[Dict[str, Any], bool, float]:
    cached_result = cache.get(text)
    expected_backend = backend_manager.health["backend"]
    # Old v3 cache entries only have ``generated_text`` and must not bypass the
    # structured contract. Also avoid serving a fake result after switching to
    # the real backend (or vice versa) with a persistent cache volume.
    if (
        cached_result
        and "text" in cached_result
        and cached_result.get("backend") == expected_backend
    ):
        CACHE_HIT_COUNTER.inc()
        return cached_result, True, 0.0
    CACHE_MISS_COUNTER.inc()

    start = time.perf_counter()
    try:
        result = backend_manager.analyze(text).as_dict()
    except BackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed = time.perf_counter() - start
    cache.set(text, result)
    return result, False, elapsed


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "service": "compliance-guard", "version": "3.2.0"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness endpoint; it does not load or require a GPU model."""

    return {
        "status": "healthy",
        "model_loaded": backend_manager.health["backend_loaded"],
        **backend_manager.health,
        "version": "3.2.0",
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/analyze")
async def analyze(request: Request, body: AnalyzeRequest) -> Dict[str, Any]:
    request_id = _request_id()
    ACTIVE_REQUESTS.inc()
    try:
        result, cached, elapsed = _analyze_text(body.text)
        REQUEST_COUNT.labels(method="POST", endpoint="/analyze", status="200").inc()
        LATENCY_HISTOGRAM.observe(elapsed)
        response = _response_from_result(result, request_id, cached, elapsed)
        response["requested_framework"] = body.framework
        return response
    except HTTPException as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/analyze", status=str(exc.status_code)).inc()
        raise
    except Exception:
        logger.exception("Unexpected analysis error", extra={"request_id": request_id})
        REQUEST_COUNT.labels(method="POST", endpoint="/analyze", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/generate", deprecated=True)
async def generate(request: Request, body: GenerateRequest) -> JSONResponse:
    """Compatibility adapter retained for one release; use ``/analyze``."""

    request_id = _request_id()
    ACTIVE_REQUESTS.inc()
    try:
        result, cached, elapsed = _analyze_text(body.text)
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="200").inc()
        LATENCY_HISTOGRAM.observe(elapsed)
        response = _response_from_result(result, request_id, cached, elapsed)
        # Preserve the v3 response key while exposing the new structured fields.
        response["generated_text"] = response["text"]
        return JSONResponse(
            content=response,
            headers={"Deprecation": "true", "X-Compliance-Guard-Deprecated": "true"},
        )
    except HTTPException as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status=str(exc.status_code)).inc()
        raise
    except Exception:
        logger.exception("Unexpected generation error", extra={"request_id": request_id})
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/compliance", deprecated=True)
async def compliance(request: Request, body: ComplianceRequest) -> Dict[str, Any]:
    """Legacy adapter for the former ``query``/``answer`` contract."""

    request_id = _request_id()
    ACTIVE_REQUESTS.inc()
    try:
        result, cached, elapsed = _analyze_text(body.query)
        REQUEST_COUNT.labels(method="POST", endpoint="/compliance", status="200").inc()
        LATENCY_HISTOGRAM.observe(elapsed)
        return {
            "request_id": request_id,
            "answer": result["text"],
            "source": result["framework"],
            "text": result["text"],
            "framework": result["framework"],
            "findings": result["findings"],
            "citations": result["citations"],
            "backend": result.get("backend"),
            "cached": cached,
            "latency_seconds": round(elapsed, 4),
        }
    except HTTPException as exc:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/compliance", status=str(exc.status_code)
        ).inc()
        raise
    except Exception:
        logger.exception(
            "Unexpected compatibility analysis error", extra={"request_id": request_id}
        )
        REQUEST_COUNT.labels(method="POST", endpoint="/compliance", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/feedback")
async def feedback(body: FeedbackRequest) -> Dict[str, str]:
    feedback_data = {
        "request_id": body.request_id,
        "rating": body.rating,
        "comment": body.comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        feedback_store.save(feedback_data)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Feedback storage is temporarily unavailable"
        ) from exc
    FEEDBACK_COUNT.labels(rating=str(body.rating)).inc()
    return {"status": "success", "message": "Feedback received. Thank you!"}


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    cache_entries = (
        sum(1 for path in Path(CACHE_DIR).glob("*.json")) if Path(CACHE_DIR).exists() else 0
    )
    return {
        "cache_entries": cache_entries,
        "model_loaded": backend_manager.health["backend_loaded"],
        "backend": backend_manager.health["backend"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
