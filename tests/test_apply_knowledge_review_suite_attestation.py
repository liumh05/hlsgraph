from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from hlsgraph.knowledge import load_pack
from tools import apply_knowledge_review_suite_attestation as attestation
from tools import audit_release
from tools import knowledge_review_shards as shards
from tools import run_knowledge_review as review
from tools import seal_knowledge_review_suite as seal
from tests.test_knowledge_review_suite_seal import _pair


ROOT = Path(__file__).resolve().parents[1]


def _pretty_json(value: object) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def _artifact_hash(value: object) -> str:
    return hashlib.sha256(_pretty_json(value)).hexdigest()


def _copy_review_root(tmp_path: Path) -> Path:
    target = tmp_path / "review-root"
    target.mkdir()
    paths = set(attestation.SUITE_REVIEW_SOURCE_PATHS)
    for protocol_id in (
        shards.SEMANTIC_PROTOCOL_ID, shards.ADVERSARIAL_PROTOCOL_ID,
    ):
        paths.update(review.required_read_paths(ROOT, protocol_id))
    for relative in sorted(paths):
        source = ROOT / PurePosixPath(relative)
        destination = target / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return target


def _formal_inputs(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = _copy_review_root(tmp_path)
    semantic_artifact, adversarial_artifact = _pair()
    semantic_snapshot = review.freeze_review_snapshot(
        root, shards.SEMANTIC_PROTOCOL_ID,
    )
    adversarial_snapshot = review.freeze_review_snapshot(
        root, shards.ADVERSARIAL_PROTOCOL_ID,
    )

    artifacts: dict[str, tuple[dict[str, object], object]] = {}
    for protocol_id, source, snapshot in (
        (shards.SEMANTIC_PROTOCOL_ID, semantic_artifact, semantic_snapshot),
        (shards.ADVERSARIAL_PROTOCOL_ID, adversarial_artifact, adversarial_snapshot),
    ):
        result = copy.deepcopy(source["aggregate"])
        result["review_surface_sha256"] = snapshot.surfaces
        result["implementation_surface_sha256"] = (
            snapshot.implementation_surface_sha256
        )
        result["citation_audit_sha256"] = snapshot.citation_audit_sha256
        receipt = copy.deepcopy(source["receipt"])
        receipt["review_snapshot_sha256"] = snapshot.sha256
        receipt["citation_evidence_sha256"] = snapshot.citation_evidence_sha256
        receipt["output_schema_sha256"] = snapshot.output_schema_sha256
        receipt["result_sha256"] = _artifact_hash(result)
        artifacts[protocol_id] = (receipt, result)

        files = review.PROTOCOL_FILES[protocol_id]
        for key, payload in (
            ("result", _pretty_json(result)),
            ("trace", source["trace"]),
            ("receipt", _pretty_json(receipt)),
        ):
            path = root / PurePosixPath(files[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    semantic_receipt, semantic_result = artifacts[
        shards.SEMANTIC_PROTOCOL_ID
    ]
    adversarial_receipt, adversarial_result = artifacts[
        shards.ADVERSARIAL_PROTOCOL_ID
    ]
    pair_seal = seal.validate_suite_pair(
        semantic_receipt=semantic_receipt,
        adversarial_receipt=adversarial_receipt,
        semantic_result=semantic_result,
        adversarial_result=adversarial_result,
        plan=semantic_artifact["plan"],
        citation_audit=semantic_artifact["audit"],
    )
    values: dict[str, object] = {
        "semantic_receipt": semantic_receipt,
        "adversarial_receipt": adversarial_receipt,
        "semantic_result": semantic_result,
        "adversarial_result": adversarial_result,
        "suite_pair_seal": pair_seal,
        "plan": semantic_artifact["plan"],
        "citation_audit": semantic_artifact["audit"],
        "semantic_snapshot": semantic_snapshot,
        "adversarial_snapshot": adversarial_snapshot,
    }
    return root, values


def _pack_bytes(root: Path) -> dict[str, bytes]:
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    return {path.name: path.read_bytes() for path in sorted(pack_root.glob("*.json"))}


def test_build_updates_is_pure_path_free_and_review_ready(tmp_path: Path) -> None:
    root, values = _formal_inputs(tmp_path)
    before = _pack_bytes(root)
    semantic_snapshot = values["semantic_snapshot"]
    adversarial_snapshot = values["adversarial_snapshot"]
    assert isinstance(semantic_snapshot, review.ReviewSnapshot)
    assert isinstance(adversarial_snapshot, review.ReviewSnapshot)

    updates = attestation.build_updates(root, **values)
    assert _pack_bytes(root) == before
    assert len(updates) == attestation.EXPECTED_PACK_COUNT == 3
    for path, payload in updates.items():
        loaded = load_pack(json.loads(payload))
        assert loaded.review_ready
        assert loaded.coverage is not None
        assert loaded.coverage.review_status == "machine_repeated_reviewed"
        before_projection = review._semantic_pack_projection(
            before[path.name], label=path.name,
        )[:2]
        after_projection = review._semantic_pack_projection(
            payload, label=path.name,
        )[:2]
        assert after_projection == before_projection

    material = attestation.build_attestation_material(root, **values)
    assert set(material) == {"reviewers", "source_hashes", "review_evidence"}
    assert len(material["reviewers"]) == len(set(material["reviewers"])) == 6
    evidence = material["review_evidence"]
    assert evidence["suite_pair_seal"] == values["suite_pair_seal"]
    assert evidence["suite_pair_seal_sha256"] == _artifact_hash(
        values["suite_pair_seal"]
    )
    assert sorted(evidence["protocol_receipt_sha256s"].values()) == values[
        "suite_pair_seal"
    ]["receipt_sha256s"]
    invocations = evidence[attestation.REVIEW_INVOCATIONS_KEY]
    assert len(invocations) == 6
    assert all("event_stream_path" not in item for item in invocations)
    assert all(
        not any(
            isinstance(value, str)
            and (value.startswith(("/", "\\\\")) or ":\\" in value)
            for value in item.values()
        )
        for item in invocations
    )
    assert attestation.SUITE_REVIEW_SOURCE_PATHS <= set(
        material["source_hashes"]
    )
    assert review.freeze_review_snapshot(
        root, shards.SEMANTIC_PROTOCOL_ID,
    ) == semantic_snapshot
    assert review.freeze_review_snapshot(
        root, shards.ADVERSARIAL_PROTOCOL_ID,
    ) == adversarial_snapshot


@pytest.mark.parametrize("tamper", ["pair", "receipt", "result"])
def test_tampered_suite_inputs_never_write_packs(
    tmp_path: Path, tamper: str,
) -> None:
    root, values = _formal_inputs(tmp_path)
    before = _pack_bytes(root)
    changed = copy.deepcopy(values)
    if tamper == "pair":
        changed["suite_pair_seal"]["invocation_count"] = 5
    elif tamper == "receipt":
        changed["semantic_receipt"]["shard_invocations"][0][
            "raw_output_sha256"
        ] = "f" * 64
    else:
        changed["semantic_result"]["citation_results"][0]["verdict"] = "rejected"
    with pytest.raises(attestation.SuiteAttestationError):
        attestation.apply_attestation(root, **changed)
    assert _pack_bytes(root) == before


def test_operational_finalizer_never_writes_after_failed_full_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _values = _formal_inputs(tmp_path)
    before = _pack_bytes(root)
    monkeypatch.setattr(
        audit_release,
        "replay_knowledge_review_suite_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(
            verified=False,
            issues=("raw output hash differs from manifest",),
        ),
    )

    with pytest.raises(
        attestation.SuiteAttestationError,
        match="suite evidence audit failed",
    ):
        attestation.finalize_attestation(
            root, suite_evidence=tmp_path / "private-suite-evidence",
        )
    assert _pack_bytes(root) == before


def test_changed_frozen_snapshot_aborts_before_pack_write(tmp_path: Path) -> None:
    root, values = _formal_inputs(tmp_path)
    before = _pack_bytes(root)
    source = root / "src" / "hlsgraph" / "bundle.py"
    source.write_bytes(source.read_bytes() + b"\n# snapshot tamper\n")
    with pytest.raises(
        attestation.SuiteAttestationError, match="ReviewSnapshot changed",
    ):
        attestation.apply_attestation(root, **values)
    assert _pack_bytes(root) == before


def test_atomic_replace_failure_rolls_back_every_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, values = _formal_inputs(tmp_path)
    before = _pack_bytes(root)
    real_replace = os.replace
    staged_replaces = 0

    def fail_second_stage(source, destination):
        nonlocal staged_replaces
        source_path = Path(source)
        if (source_path.name[:2].isdigit()
                and source_path.suffix == ".json"):
            staged_replaces += 1
            if staged_replaces == 2:
                raise OSError("injected second-pack replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(review.os, "replace", fail_second_stage)
    with pytest.raises(OSError, match="injected second-pack"):
        attestation.apply_attestation(root, **values)
    assert _pack_bytes(root) == before


def test_apply_keeps_both_snapshots_and_activates_all_packs(
    tmp_path: Path,
) -> None:
    root, values = _formal_inputs(tmp_path)
    semantic_snapshot = values["semantic_snapshot"]
    adversarial_snapshot = values["adversarial_snapshot"]
    updates = attestation.apply_attestation(root, **values)
    assert _pack_bytes(root) == {
        path.name: payload for path, payload in updates.items()
    }
    assert all(load_pack(path).review_ready for path in updates)
    assert review.freeze_review_snapshot(
        root, shards.SEMANTIC_PROTOCOL_ID,
    ) == semantic_snapshot
    assert review.freeze_review_snapshot(
        root, shards.ADVERSARIAL_PROTOCOL_ID,
    ) == adversarial_snapshot
    # Rebuilding after activation is deterministic and idempotent.
    assert attestation.build_updates(root, **values) == updates
