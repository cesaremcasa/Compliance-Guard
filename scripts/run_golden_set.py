#!/usr/bin/env python3
"""Run the checked-in golden set against the deterministic fake backend.

This is a contract/fixture check, not a claim about semantic model quality.
It does not download a model, require CUDA, or call a network service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.backends import FakeBackend  # noqa: E402


def run(golden_path: Path, fixture_path: Path) -> Dict[str, Any]:
    backend = FakeBackend(str(fixture_path))
    cases: List[Dict[str, Any]] = json.loads(golden_path.read_text(encoding="utf-8"))
    details: List[Dict[str, Any]] = []
    failures = 0

    for case in cases:
        question = case["question"]
        first = backend.analyze(question).as_dict()
        second = backend.analyze(question).as_dict()
        deterministic = first == second
        schema_ok = all(
            first.get(field) is not None for field in ("text", "framework", "findings", "citations")
        )
        expected_keywords = [
            str(keyword).casefold()
            for keyword in case.get("expected_keywords", case.get("keywords", []))
        ]
        keyword_observed = any(keyword in first["text"].casefold() for keyword in expected_keywords)
        passed = deterministic and schema_ok and keyword_observed
        if not passed:
            failures += 1
        details.append(
            {
                "id": case.get("id"),
                "passed": passed,
                "deterministic": deterministic,
                "schema_ok": schema_ok,
                "keyword_observed": keyword_observed,
            }
        )

    total = len(cases)
    return {
        "backend": "fake",
        "fixture": str(fixture_path),
        "contract_passed": failures == 0,
        "total": total,
        "passed": total - failures,
        "failed": failures,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=ROOT / "tests/golden_set.json")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests/fixtures/fake_backend.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/golden_set_fake.json")
    args = parser.parse_args()

    report = run(args.golden, args.fixtures)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Fake golden set: {report['passed']}/{report['total']} contract cases passed")
    print(f"Report: {args.report}")
    return 0 if report["contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
