"""Inference backends for the Compliance Guard API.

The API imports this module without importing the GPU stack.  The real
Transformers/PEFT backend is imported and initialized only when it is
explicitly selected and the first analysis request arrives.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_FRAMEWORK = "NIST SP 800-53 Rev. 5"
CONTROL_ID_RE = re.compile(r"\b[A-Z]{2}-\d{1,3}(?:\(\d+\))?\b")


class BackendUnavailable(RuntimeError):
    """Raised when a selected backend cannot be used in this environment."""


@dataclass(frozen=True)
class BackendResult:
    """Stable result contract shared by fake and real inference backends."""

    text: str
    framework: str
    findings: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    backend: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "framework": self.framework,
            "findings": self.findings,
            "citations": self.citations,
            "backend": self.backend,
        }


class AnalysisBackend(Protocol):
    """Minimal interface required by the API."""

    name: str

    def analyze(self, text: str) -> BackendResult:
        """Analyze one bounded input and return a structured result."""


def _control_id(text: str) -> Optional[str]:
    match = CONTROL_ID_RE.search(text.upper())
    return match.group(0) if match else None


def _default_citation(control_id: Optional[str]) -> Dict[str, Any]:
    citation: Dict[str, Any] = {"source": DEFAULT_FRAMEWORK}
    if control_id:
        citation["control_id"] = control_id
        citation["locator"] = control_id
    return citation


class FakeBackend:
    """Deterministic fixture-backed backend for local development and tests.

    A fixture is used when the request matches one of its exact queries.  For
    other bounded inputs, the backend returns a deterministic, explicitly
    non-semantic response.  This makes the API contract testable without
    implying that the fake backend is a compliance evaluator.
    """

    name = "fake"

    def __init__(self, fixture_path: Optional[str] = None) -> None:
        self.fixture_path = Path(
            fixture_path
            or os.getenv("COMPLIANCE_GUARD_FAKE_FIXTURES", "tests/fixtures/fake_backend.json")
        )
        self._cases: Dict[str, Dict[str, Any]] = {}
        self._framework = DEFAULT_FRAMEWORK
        self._load_fixtures()

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split()).strip().casefold()

    def _load_fixtures(self) -> None:
        if not self.fixture_path.exists():
            logger.warning("Fake backend fixture file not found: %s", self.fixture_path)
            return

        try:
            payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackendUnavailable(f"Could not read fake fixtures: {exc}") from exc

        self._framework = payload.get("framework", DEFAULT_FRAMEWORK)
        cases = payload.get("cases", [])
        if not isinstance(cases, list):
            raise BackendUnavailable("Fake fixture 'cases' must be a list")

        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("query"), str):
                raise BackendUnavailable("Each fake fixture case needs a string query")
            self._cases[self._normalize(case["query"])] = case

    def analyze(self, text: str) -> BackendResult:
        case = self._cases.get(self._normalize(text))
        if case:
            return BackendResult(
                text=str(case["text"]),
                framework=str(case.get("framework", self._framework)),
                findings=list(case.get("findings", [])),
                citations=list(case.get("citations", [_default_citation(_control_id(text))])),
                backend=self.name,
            )

        control_id = _control_id(text)
        finding: Dict[str, Any] = {
            "status": "not_evaluated",
            "summary": "Deterministic fake backend response; no model-backed assessment was run.",
        }
        if control_id:
            finding["control_id"] = control_id

        return BackendResult(
            text=f"Fake backend response for: {text}",
            framework=self._framework,
            findings=[finding],
            citations=[_default_citation(control_id)],
            backend=self.name,
        )


class RealLoRABackend:
    """Lazy, GPU-backed Transformers + PEFT inference backend.

    The backend intentionally requires CUDA by default.  A real model run is
    a separate manual canary because it needs model access, GPU memory and
    potentially a Hugging Face credential outside this repository.
    """

    name = "real-lora"

    def __init__(
        self,
        base_model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        allow_cpu: Optional[bool] = None,
    ) -> None:
        self.base_model_id = base_model_id or os.getenv(
            "COMPLIANCE_GUARD_BASE_MODEL", "mistralai/Mistral-7B-v0.1"
        )
        self.adapter_path = adapter_path or os.getenv(
            "COMPLIANCE_GUARD_LORA_ADAPTER", "models/checkpoints"
        )
        self.tokenizer_path = tokenizer_path or os.getenv(
            "COMPLIANCE_GUARD_TOKENIZER", self.adapter_path
        )
        self.allow_cpu = (
            allow_cpu
            if allow_cpu is not None
            else os.getenv("COMPLIANCE_GUARD_ALLOW_CPU", "0").casefold() in {"1", "true", "yes"}
        )
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cuda"

    def _load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise BackendUnavailable(
                "real-lora backend requires the optional GPU dependencies "
                "(torch, transformers, accelerate, peft, bitsandbytes)"
            ) from exc

        self._torch = torch
        has_cuda = bool(torch.cuda.is_available())
        if not has_cuda and not self.allow_cpu:
            raise BackendUnavailable(
                "real-lora backend requires CUDA; use COMPLIANCE_GUARD_BACKEND=fake "
                "for a CPU-only quickstart"
            )

        self._device = "cuda" if has_cuda else "cpu"
        try:
            if has_cuda:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch.float32,
                )
            tokenizer_source = (
                self.tokenizer_path if Path(self.tokenizer_path).exists() else self.base_model_id
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self._model.eval()
        except Exception as exc:
            raise BackendUnavailable(f"could not load LoRA model or tokenizer: {exc}") from exc

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            self._load()

    def analyze(self, text: str) -> BackendResult:
        self._ensure_loaded()
        control_id = _control_id(text)
        prompt = (
            "You are assisting with a compliance review. Provide a concise answer "
            f"grounded in {DEFAULT_FRAMEWORK}. State uncertainty when evidence is missing.\n\n"
            f"Question: {text}\n\nAnswer:"
        )
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self._device == "cuda":
                inputs = inputs.to("cuda")
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = (
                generated[len(prompt) :].strip()
                if generated.startswith(prompt)
                else generated.strip()
            )
        except Exception as exc:
            raise BackendUnavailable(f"real-lora inference failed: {exc}") from exc

        finding: Dict[str, Any] = {
            "status": "requires_review",
            "summary": "Model-generated guidance; validate against the cited NIST control and local evidence.",
        }
        if control_id:
            finding["control_id"] = control_id
        return BackendResult(
            text=response_text,
            framework=DEFAULT_FRAMEWORK,
            findings=[finding],
            citations=[_default_citation(control_id)],
            backend=self.name,
        )


class BackendManager:
    """Thread-safe lazy backend factory used by the API process."""

    def __init__(
        self, backend_name: Optional[str] = None, fixture_path: Optional[str] = None
    ) -> None:
        self.backend_name = (
            backend_name or os.getenv("COMPLIANCE_GUARD_BACKEND", "fake")
        ).casefold()
        if self.backend_name not in {"fake", "real", "real-lora"}:
            raise ValueError("COMPLIANCE_GUARD_BACKEND must be 'fake' or 'real'")
        self.fixture_path = fixture_path
        self._backend: Optional[AnalysisBackend] = None
        self._lock = Lock()
        self.last_error: Optional[str] = None

    @property
    def health(self) -> Dict[str, Any]:
        backend_label = "real-lora" if self.backend_name in {"real", "real-lora"} else "fake"
        return {
            "backend": backend_label,
            "backend_loaded": self._backend is not None,
            "backend_error": self.last_error,
        }

    def _build_backend(self) -> AnalysisBackend:
        if self.backend_name == "fake":
            return FakeBackend(self.fixture_path)
        return RealLoRABackend()

    def _get_backend(self) -> AnalysisBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                try:
                    self._backend = self._build_backend()
                except BackendUnavailable as exc:
                    self.last_error = str(exc)
                    raise
        return self._backend

    def analyze(self, text: str) -> BackendResult:
        try:
            return self._get_backend().analyze(text)
        except BackendUnavailable as exc:
            self.last_error = str(exc)
            raise
