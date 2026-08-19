#!/usr/bin/env python3
"""Run one explicit, manual real LoRA/GPU inference canary.

The canary is intentionally opt-in and never runs in CI or as part of a
release command. It requires the caller to provide model access/configuration
through environment variables; no credential is stored in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.backends import BackendUnavailable, RealLoRABackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="Describe the access control requirements in AC-2.",
        help="One bounded question to use for the manual canary.",
    )
    args = parser.parse_args()

    if os.getenv("COMPLIANCE_GUARD_RUN_GPU_CANARY") != "1":
        print(
            "GPU canary is manual and pending: set COMPLIANCE_GUARD_RUN_GPU_CANARY=1 "
            "after confirming GPU/model access."
        )
        return 2
    if os.getenv("COMPLIANCE_GUARD_BACKEND") != "real":
        print("Set COMPLIANCE_GUARD_BACKEND=real for the real LoRA canary.")
        return 2

    backend = RealLoRABackend()
    try:
        result = backend.analyze(args.question).as_dict()
    except BackendUnavailable as exc:
        print(f"GPU/LoRA canary blocked: {exc}")
        return 2

    print(json.dumps({"status": "passed", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
