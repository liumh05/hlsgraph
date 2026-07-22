from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest

from eval.agent_ab.audit import (
    FIXTURE_MANIFEST_PATH,
    _fixture_report_errors,
    _license_errors,
    audit_public_artifacts,
)
from eval.agent_ab.common import load_corpus_lock
from tools import audit_release as release_audit
from tools.audit_release import (
    REQUIRED_SDIST,
    _audit_sdist,
    _audit_wheel,
    _duplicate_archive_names,
    _expected_sdist_installable,
    _forbidden,
    _knowledge_payload_issues,
    _scan,
    _strict_file_bytes,
    _unsafe_archive_name,
)

try:  # pragma: no cover - Python 3.10 compatibility
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def _review_contract_digest(value: dict[str, object], hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_review_boundary_contract(cache_sha256: str = "a" * 64) -> dict[str, object]:
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    runtime: dict[str, object] = {
        "schema_version": "hlsgraph.knowledge-review.runtime-manifest.v2",
        "ownership_policy": "caller_owned_frozen_0500_no_links_v1",
        "executable_relative_path": "codex",
        "executable_sha256": release_audit.REVIEW_OFFICIAL_CODEX_ELF_SHA256,
        "entries": [
            {
                "relative_path": ".", "kind": "dir", "size": 0,
                "mode": "0500", "sha256": empty_sha256,
            },
            {
                "relative_path": "codex", "kind": "file", "size": 5,
                "mode": "0500", "sha256": release_audit.REVIEW_OFFICIAL_CODEX_ELF_SHA256,
            },
        ],
    }
    runtime["sha256"] = _review_contract_digest(runtime, "sha256")
    contract: dict[str, object] = {
        "schema_version": "hlsgraph.knowledge-review.boundary-contract.v2",
        "policy": "default_deny_minimal_allowlist_v1",
        "filesystem_allowlist": [
            {"token": ":minimal", "access": "read"},
            {"token": "$CACHE", "access": "read"},
            {"token": "$CODEX_RUNTIME", "access": "read"},
        ],
        "network_enabled": False,
        "runtime_manifest": runtime,
        "cache_manifest_sha256": cache_sha256,
        "cache_parent_policy": "caller_owned_0700_single_cache_v1",
        "evidence_parent_policy": "caller_owned_0700_dedicated_evidence_v1",
        "canary_results": {
            "cache_read": True,
            "runtime_read": True,
            "checkout_denied": True,
            "auth_denied": True,
            "external_denied": True,
            "peer_sibling_denied": True,
            "evidence_denied": True,
            "cache_write_denied": True,
        },
    }
    contract["contract_sha256"] = _review_contract_digest(
        contract, "contract_sha256",
    )
    return contract


def test_review_receipt_v4_schema_is_exactly_closed() -> None:
    value = json.loads(
        (ROOT / "tools" / "knowledge_review_receipt.schema.json").read_text(
            encoding="utf-8",
        )
    )
    assert release_audit._receipt_schema_contract_issues(value) == []
    weakened = copy.deepcopy(value)
    weakened["properties"]["boundary_contract"]["properties"][
        "canary_results"
    ]["additionalProperties"] = True
    assert any(
        "weakens boundary_contract" in item
        for item in release_audit._receipt_schema_contract_issues(weakened)
    )


def test_review_boundary_contract_validates_hashes_and_projection() -> None:
    cache_sha256 = "a" * 64
    contract = _valid_review_boundary_contract(cache_sha256)
    assert release_audit._boundary_contract_issues(
        contract, label="receipt",
        expected_cache_manifest_sha256=cache_sha256,
    ) == []
    projection = release_audit._receipt_invocation_projection(
        {"boundary_contract": contract}, b'{"boundary_contract":{}}\n',
    )
    assert projection["boundary_contract_sha256"] == contract["contract_sha256"]
    runtime = contract["runtime_manifest"]
    assert isinstance(runtime, dict)
    assert projection["runtime_manifest_sha256"] == runtime["sha256"]


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("extra-field", "not a closed contract"),
        ("cache-mismatch", "invalid cache_manifest_sha256"),
        ("runtime-hash", "runtime_manifest has an invalid sha256"),
        ("contract-hash", "invalid contract_sha256"),
        ("false-canary", "invalid canary_results"),
        ("unsafe-runtime-path", "unsafe relative_path"),
        ("missing-executable", "lacks its exact executable identity"),
        ("orphan-runtime-file", "incomplete directory tree"),
        ("group-writable-runtime", "writable"),
    ],
)
def test_review_boundary_contract_rejects_tampering(
    mutation: str, expected_issue: str,
) -> None:
    cache_sha256 = "a" * 64
    contract = _valid_review_boundary_contract(cache_sha256)
    runtime = contract["runtime_manifest"]
    assert isinstance(runtime, dict)
    entries = runtime["entries"]
    assert isinstance(entries, list)
    if mutation == "extra-field":
        contract["unexpected"] = True
    elif mutation == "cache-mismatch":
        contract["cache_manifest_sha256"] = "b" * 64
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    elif mutation == "runtime-hash":
        runtime["sha256"] = "b" * 64
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    elif mutation == "contract-hash":
        contract["contract_sha256"] = "b" * 64
    elif mutation == "false-canary":
        canaries = contract["canary_results"]
        assert isinstance(canaries, dict)
        canaries["auth_denied"] = False
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    elif mutation == "unsafe-runtime-path":
        entry = entries[-1]
        assert isinstance(entry, dict)
        entry["relative_path"] = "../codex"
        runtime["sha256"] = _review_contract_digest(runtime, "sha256")
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    elif mutation == "missing-executable":
        runtime["executable_relative_path"] = "missing-codex"
        runtime["sha256"] = _review_contract_digest(runtime, "sha256")
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    elif mutation == "orphan-runtime-file":
        entry = entries[-1]
        assert isinstance(entry, dict)
        entry["relative_path"] = "orphan/codex"
        runtime["executable_relative_path"] = "orphan/codex"
        runtime["sha256"] = _review_contract_digest(runtime, "sha256")
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    elif mutation == "group-writable-runtime":
        entry = entries[-1]
        assert isinstance(entry, dict)
        entry["mode"] = "0775"
        runtime["sha256"] = _review_contract_digest(runtime, "sha256")
        contract["contract_sha256"] = _review_contract_digest(
            contract, "contract_sha256",
        )
    issues = release_audit._boundary_contract_issues(
        contract, label="receipt",
        expected_cache_manifest_sha256=cache_sha256,
    )
    assert any(expected_issue in item for item in issues), issues


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "root/../escape",
        "/absolute/file",
        "C" + ":/absolute/file",
        "root\\windows-path",
        "root//double",
        "./root/file",
    ],
)
def test_archive_names_fail_closed(name: str) -> None:
    assert _unsafe_archive_name(name) is not None
    assert _unsafe_archive_name("hlsgraph-0.3.0/src/hlsgraph/model.py") is None


def test_archive_duplicate_detection_includes_case_collisions() -> None:
    assert _duplicate_archive_names(["a", "a"]) == ["a"]
    assert _duplicate_archive_names(["A/file", "a/file"])


def test_release_scan_rejects_pdf_private_endpoints_and_knowledge_bodies() -> None:
    assert _forbidden("docs/vendor-manual.PDF") == ".pdf"
    assert _forbidden("hlsgraph/knowledge/packs/manual.txt") == (
        "non-JSON knowledge-pack payload"
    )
    assert any(
        "PDF document magic" in item
        for item in _scan("docs/renamed.bin", b"%" + b"PDF-1.7\n")
    )
    private_endpoint = b"10" + b".23.45.67"
    assert any("RFC1918" in item for item in _scan("endpoint.txt", private_endpoint))
    assert not any("RFC1918" in item for item in _scan("loopback.txt", b"127.0.0.1"))

    host = b"fpga" + b"5090"
    user = b"srtp" + b"-agent"
    alias = b"s" + b"sh h" + b"ls"
    assert any("laboratory host" in item for item in _scan("host.txt", host))
    assert any("laboratory user" in item for item in _scan("user.txt", user))
    assert any("SSH alias" in item for item in _scan("alias.txt", alias))

    body_key = "raw" + "_text"
    payload = json.dumps({body_key: "vendor body"}).encode()
    assert _knowledge_payload_issues(
        "hlsgraph/knowledge/packs/bad.json", payload,
    )
    oversized = json.dumps({"summary": "x" * 513}).encode()
    assert _knowledge_payload_issues(
        "src/hlsgraph/knowledge/packs/bad.json", oversized,
    )
    assert _knowledge_payload_issues(
        "src/hlsgraph/knowledge/packs/bad.json", b"{not-json",
    ) == ["invalid knowledge-pack JSON in src/hlsgraph/knowledge/packs/bad.json"]


def test_wheel_rejects_duplicate_traversal_and_symlink_members(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("hlsgraph/model.py", b"one")
            archive.writestr("hlsgraph/model.py", b"two")
    assert any("duplicate member" in item for item in _audit_wheel(duplicate, ROOT, b"{}"))

    unsafe = tmp_path / "unsafe.whl"
    link = zipfile.ZipInfo("hlsgraph/link.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.py", b"escape")
        archive.writestr(link, b"target.py")
    issues = _audit_wheel(unsafe, ROOT, b"{}")
    assert any("unsafe wheel member" in item for item in issues)
    assert any("linked member" in item for item in issues)


def test_wheel_requires_exact_source_package_paths_and_bytes(tmp_path: Path) -> None:
    wheel = tmp_path / "payload.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("hlsgraph/__init__.py", b"changed")
        archive.writestr("hlsgraph/not-in-source.py", b"extra")
        archive.writestr(
            "hlsgraph-0.3.0.dist-info/METADATA",
            b"Name: hlsgraph\nVersion: 0.3.0\n",
        )
    issues = _audit_wheel(wheel, ROOT, b"{}")
    assert any("missing source package files" in item for item in issues)
    assert any("extra source package files" in item for item in issues)
    assert any("package bytes differ" in item for item in issues)


def _tar_member(name: str, data: bytes = b"x", *, kind: bytes | None = None) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    if kind is not None:
        member.type = kind
        member.size = 0
    else:
        member.size = len(data)
    return member


def test_sdist_rejects_duplicates_traversal_links_specials_and_multiple_roots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        first = _tar_member("hlsgraph-0.3.0/LICENSE")
        archive.addfile(first, io.BytesIO(b"x"))
        duplicate = _tar_member("hlsgraph-0.3.0/LICENSE")
        archive.addfile(duplicate, io.BytesIO(b"x"))
        traversal = _tar_member("hlsgraph-0.3.0/../escape")
        archive.addfile(traversal, io.BytesIO(b"x"))
        archive.addfile(_tar_member("hlsgraph-0.3.0/link", kind=tarfile.SYMTYPE))
        archive.addfile(_tar_member("hlsgraph-0.3.0/device", kind=tarfile.CHRTYPE))
        second_root = _tar_member("other-root/file")
        archive.addfile(second_root, io.BytesIO(b"x"))
    issues = _audit_sdist(path, b"{}")
    assert any("duplicate member" in item for item in issues)
    assert any("unsafe sdist member" in item for item in issues)
    assert sum("linked or special member" in item for item in issues) == 2
    assert any("exactly the root" in item for item in issues)


def test_sdist_requires_support_migration_and_review_inputs(tmp_path: Path) -> None:
    path = tmp_path / "minimal.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = _tar_member("hlsgraph-0.3.0/LICENSE")
        archive.addfile(member, io.BytesIO(b"x"))
    issues = _audit_sdist(path, b"{}")
    missing = next(item for item in issues if item.startswith("sdist is missing:"))
    for required in (
        "tests/attested_run_support.py",
        "tests/typed_report_support.py",
        "tests/fixtures/v02_minimal_bundle.json",
        "tools/knowledge_review.schema.json",
        "tools/knowledge_review_prompts/adversarial.md",
        "tools/knowledge_review_prompts/semantic.md",
        "tools/audit_release.py",
    ):
        assert required in REQUIRED_SDIST
        assert required in missing


def test_sdist_rejects_extra_or_changed_installable_source(
    tmp_path: Path,
) -> None:
    expected = _expected_sdist_installable(ROOT)
    path = tmp_path / "tampered.tar.gz"
    prefix = "hlsgraph-0.3.0/"
    with tarfile.open(path, "w:gz") as archive:
        for relative, data in sorted(expected.items()):
            if relative == "src/hlsgraph/version.py":
                data += b"# changed in sdist\n"
            member = _tar_member(prefix + relative, data)
            archive.addfile(member, io.BytesIO(data))
        extra_name = "src/hlsgraph/knowledge/packs/unreviewed_extra.json"
        extra_data = b'{"pack_id":"unreviewed.extra"}\n'
        extra = _tar_member(prefix + extra_name, extra_data)
        archive.addfile(extra, io.BytesIO(extra_data))
    issues = _audit_sdist(path, b"{}", root=ROOT)
    assert any("extra installable source/build inputs" in item for item in issues)
    assert any(
        "installable source/build bytes differ from source: src/hlsgraph/version.py"
        in item for item in issues
    )


def test_strict_release_read_rejects_file_changed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review.json"
    path.write_bytes(b"x" * 4096)
    real_read = os.read
    changed = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        data = real_read(descriptor, size)
        if data and not changed:
            before = path.stat()
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            changed = True
        return data

    monkeypatch.setattr(release_audit.os, "read", racing_read)
    with pytest.raises(ValueError, match="changed while it was read"):
        _strict_file_bytes(path, "racing review input", root=tmp_path)


def test_strict_external_read_rejects_hard_link_alias(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    path = parent / "review.raw.jsonl"
    path.write_bytes(b"{}\n")
    alias = parent / "review.alias.jsonl"
    os.link(path, alias)
    if os.name != "nt":
        path.chmod(0o600)
        parent.chmod(0o700)
    with pytest.raises(ValueError, match="hard-link aliases"):
        _strict_file_bytes(
            path, "raw review stream", required_file_mode=0o600,
            required_parent_mode=0o700, require_current_owner=True,
            require_single_link=True, max_bytes=1024,
        )


def test_strict_external_read_rejects_parent_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    path = parent / "review.raw.jsonl"
    path.write_bytes(b"{}\n")
    if os.name != "nt":
        path.chmod(0o600)
        parent.chmod(0o700)
    real_identity = release_audit._stat_identity
    identity_calls = 0

    def replaced_parent_identity(value):
        nonlocal identity_calls
        identity_calls += 1
        identity = real_identity(value)
        # The fifth/sixth identities are the immediate parent before/after.
        if identity_calls == 6:
            return (identity[0], identity[1] + 1, *identity[2:])
        return identity

    monkeypatch.setattr(release_audit, "_stat_identity", replaced_parent_identity)
    with pytest.raises(ValueError, match="changed while it was read"):
        _strict_file_bytes(
            path, "raw review stream", required_file_mode=0o600,
            required_parent_mode=0o700, require_current_owner=True,
            require_single_link=True, max_bytes=1024,
        )


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_public_eval_audit_normalizes_wide_character_artifacts(
    tmp_path: Path, encoding: str,
) -> None:
    private_path = "C" + ":" + "\\Us" + "ers\\person\\kernel.cpp"
    path = tmp_path / f"wide-{encoding}.json"
    path.write_bytes(json.dumps({"path": private_path}).encode(encoding))
    report = audit_public_artifacts([path])
    assert report["passed"] is False
    assert any(item["kind"] == "windows-absolute-path" for item in report["findings"])


def test_frozen_license_hashes_and_report_authority_are_enforced() -> None:
    lock = load_corpus_lock()
    assert _license_errors(lock) == []
    changed_lock = copy.deepcopy(lock)
    stream = next(item for item in changed_lock["corpora"] if item["id"] == "stream_blocks")
    license_entry = next(
        item for item in stream["files"] if item["destination"] == "UPSTREAM_LICENSE.txt"
    )
    license_entry["sha256"] = "0" * 64
    assert any("license byte hash drift" in item for item in _license_errors(changed_lock))

    fixture = tomllib.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert _fixture_report_errors(lock, fixture) == []
    changed_fixture = copy.deepcopy(fixture)
    report = next(
        item for item in changed_fixture["artifact_paths"]
        if item["path"] == "reports/dut_cosim.rpt"
    )
    report["metadata"].pop("fixture_authority")
    assert any(
        "dut_cosim.rpt" in item and "synthetic" in item
        for item in _fixture_report_errors(lock, changed_fixture)
    )
    outside_reports = copy.deepcopy(fixture)
    outside_reports["artifact_paths"].append({
        "path": "unexpected/tool-output.txt",
        "kind": "amd.vitis.csynth_report",
        "role": "tool_output",
        "metadata": {},
    })
    assert any(
        "unexpected/tool-output.txt" in item
        for item in _fixture_report_errors(lock, outside_reports)
    )
