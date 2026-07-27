#!/usr/bin/env python3
"""Validate the small public experiment bridge without external dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments" / "catalog.json"
REQUIRED_FIELDS = {
    "experiment_id",
    "title",
    "family",
    "status",
    "claim_level",
    "historical_commit",
    "environment",
    "benchmarks",
    "methods",
    "metrics",
    "evidence",
    "public_core_binding",
    "limitations",
}
ALLOWED_STATUS = {"completed", "partial", "negative", "engineering_only"}
ALLOWED_CLAIM_LEVEL = {"historical_pre_v03", "v03_contract_smoke"}


def _relative_file(path_text: Any) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("evidence path must be a non-empty string")
    pure = PurePosixPath(path_text)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path_text:
        raise ValueError(f"evidence path is not canonical and relative: {path_text!r}")
    path = (ROOT / Path(*pure.parts)).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"evidence file is missing: {path_text}")
    return path


def validate(catalog_path: Path = CATALOG) -> list[str]:
    issues: list[str] = []
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"cannot load catalog: {exc}"]
    if catalog.get("schema_version") != "hlsgraph.experiment-catalog.v1":
        issues.append("catalog schema_version is not supported")
    binding = catalog.get("public_core_binding")
    if not isinstance(binding, dict):
        issues.append("catalog public_core_binding is missing")
        binding = {}
    if binding.get("commit") != "e893196f6cf59faa3dcb6cc9372d121a5db7a7ff":
        issues.append("catalog is not bound to the published 0.3 commit")
    wheel_hash = binding.get("wheel_sha256")
    if (
        not isinstance(wheel_hash, str)
        or len(wheel_hash) != 64
        or any(char not in "0123456789abcdef" for char in wheel_hash)
    ):
        issues.append("catalog wheel_sha256 is invalid")
    smoke = catalog.get("contract_smoke_evidence")
    if not isinstance(smoke, dict):
        issues.append("catalog contract_smoke_evidence is missing")
    else:
        try:
            smoke_path = _relative_file(smoke.get("path"))
        except ValueError as exc:
            issues.append(f"contract smoke: {exc}")
        else:
            smoke_data = smoke_path.read_bytes()
            if smoke.get("size") != len(smoke_data):
                issues.append("contract smoke evidence size is stale")
            if smoke.get("sha256") != hashlib.sha256(smoke_data).hexdigest():
                issues.append("contract smoke evidence hash is stale")
    release_support = catalog.get("release_support_evidence")
    if not isinstance(release_support, list) or not release_support:
        issues.append("catalog release_support_evidence is missing")
    else:
        for index, item in enumerate(release_support):
            if not isinstance(item, dict):
                issues.append(f"release support evidence {index} is malformed")
                continue
            try:
                support_path = _relative_file(item.get("path"))
            except ValueError as exc:
                issues.append(f"release support evidence {index}: {exc}")
                continue
            support_data = support_path.read_bytes()
            if item.get("size") != len(support_data):
                issues.append(f"release support evidence {index} size is stale")
            if item.get("sha256") != hashlib.sha256(support_data).hexdigest():
                issues.append(f"release support evidence {index} hash is stale")
    rows = catalog.get("experiments")
    if not isinstance(rows, list):
        return issues + ["catalog experiments must be an array"]
    ids: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"experiments[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{prefix} is not an object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            issues.append(f"{prefix} is missing fields: {missing}")
        experiment_id = row.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            issues.append(f"{prefix} has an invalid experiment_id")
        elif experiment_id in ids:
            issues.append(f"{prefix} duplicates experiment_id {experiment_id}")
        else:
            ids.add(experiment_id)
        if row.get("status") not in ALLOWED_STATUS:
            issues.append(f"{prefix} has an invalid status")
        if row.get("claim_level") not in ALLOWED_CLAIM_LEVEL:
            issues.append(f"{prefix} has an invalid claim_level")
        evidence = row.get("evidence")
        metrics = row.get("metrics")
        if not isinstance(evidence, list) or not evidence:
            issues.append(f"{prefix} has no evidence")
            continue
        evidence_paths: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict):
                issues.append(f"{prefix} has malformed evidence")
                continue
            path_text = item.get("path")
            try:
                path = _relative_file(path_text)
            except ValueError as exc:
                issues.append(f"{prefix}: {exc}")
                continue
            evidence_paths.add(path_text)
            data = path.read_bytes()
            if item.get("size") != len(data):
                issues.append(f"{prefix} has a stale evidence size for {path_text}")
            if item.get("sha256") != hashlib.sha256(data).hexdigest():
                issues.append(f"{prefix} has a stale evidence hash for {path_text}")
        if not isinstance(metrics, list) or not metrics:
            issues.append(f"{prefix} has no metric references")
        else:
            for metric in metrics:
                if (
                    not isinstance(metric, dict)
                    or metric.get("source") not in evidence_paths
                    or not isinstance(metric.get("json_pointer"), str)
                ):
                    issues.append(f"{prefix} has an unbound metric reference")
    if len(rows) != 13:
        issues.append(f"catalog must contain 13 experiment families, found {len(rows)}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args(argv)
    issues = validate(args.catalog)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1
    print("experiment catalog: 13 families, hashes and metric bindings OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
