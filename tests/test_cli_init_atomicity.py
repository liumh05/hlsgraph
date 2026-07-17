from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import hlsgraph.cli as cli_module
from hlsgraph.bundle import GraphBundle
from hlsgraph.cli import main


def _invoke(capsys, root: Path, *extra: str) -> tuple[int, dict]:
    code = main([
        "init", "--project", str(root), "--project-id", "test.atomic_init",
        "--name", "atomic init", "--top", "dut", "--source", "kernel.cpp",
        *extra,
    ])
    captured = capsys.readouterr()
    return code, json.loads(captured.out or captured.err)


def test_init_no_clobber_publish_preserves_manifest_won_by_peer(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    real_create = GraphBundle.create.__func__
    peer_text = "# manifest published by a concurrent peer\n"

    def peer_publishes(cls, project_root, manifest, **kwargs):
        bundle = real_create(cls, project_root, manifest, **kwargs)
        (Path(project_root) / "hlsgraph.toml").write_text(peer_text, encoding="utf-8")
        return bundle

    monkeypatch.setattr(GraphBundle, "create", classmethod(peer_publishes))
    code, result = _invoke(capsys, tmp_path)

    assert code == 1
    assert result["type"] == "CliError"
    assert (tmp_path / "hlsgraph.toml").read_text(encoding="utf-8") == peer_text
    assert not (tmp_path / ".hlsgraph").exists()
    assert not (tmp_path / ".hlsgraph-init.lock").exists()


def test_init_force_detects_in_place_peer_edit_before_replacement(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    manifest_path = tmp_path / "hlsgraph.toml"
    manifest_path.write_text("# original\n", encoding="utf-8")
    real_create = GraphBundle.create.__func__
    peer_text = "# peer changed the observed inode\n"

    def peer_edits(cls, project_root, manifest, **kwargs):
        bundle = real_create(cls, project_root, manifest, **kwargs)
        manifest_path.write_text(peer_text, encoding="utf-8")
        return bundle

    monkeypatch.setattr(GraphBundle, "create", classmethod(peer_edits))
    code, result = _invoke(capsys, tmp_path, "--force")

    assert code == 1
    assert result["type"] == "CliError"
    assert manifest_path.read_text(encoding="utf-8") == peer_text
    assert not (tmp_path / ".hlsgraph").exists()


def test_init_force_replaces_only_observed_manifest_and_cleans_guard(
    tmp_path: Path, capsys,
) -> None:
    manifest_path = tmp_path / "hlsgraph.toml"
    manifest_path.write_text("# explicitly replace me\n", encoding="utf-8")

    code, result = _invoke(capsys, tmp_path, "--force")

    assert code == 0, result
    assert "project_id = \"test.atomic_init\"" in manifest_path.read_text(
        encoding="utf-8",
    )
    assert not list(tmp_path.glob(".hlsgraph-init-parent.*.lock"))
    assert not list(tmp_path.glob(".hlsgraph.toml.*.tmp"))


def test_init_keeps_owner_token_until_manifest_publication(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    def fail_publish(path, text, **kwargs):
        marker = tmp_path / ".hlsgraph" / ".init-owner"
        assert marker.is_file()
        assert marker.read_text(encoding="ascii").strip()
        raise OSError("simulated publication failure")

    monkeypatch.setattr(cli_module, "_atomic_write_text", fail_publish)
    code, result = _invoke(capsys, tmp_path)

    assert code == 1
    assert result["type"] == "CliError"
    assert not (tmp_path / ".hlsgraph").exists()
    assert not (tmp_path / "hlsgraph.toml").exists()


def test_init_refuses_dangling_manifest_symlink_even_with_force(
    tmp_path: Path, capsys,
) -> None:
    link = tmp_path / "hlsgraph.toml"
    try:
        os.symlink("missing-target.toml", link)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    code, result = _invoke(capsys, tmp_path, "--force")

    assert code == 1
    assert "symlink" in result["error"]
    assert link.is_symlink()
    assert os.readlink(link) == "missing-target.toml"
    assert not (tmp_path / ".hlsgraph").exists()


def test_init_lock_is_no_clobber_and_peer_owned(tmp_path: Path, capsys) -> None:
    lock = tmp_path / ".hlsgraph-init.lock"
    lock.write_text("peer-token\n", encoding="ascii")

    code, result = _invoke(capsys, tmp_path)

    assert code == 1
    assert "another project initialization" in result["error"]
    assert lock.read_text(encoding="ascii") == "peer-token\n"
    assert not (tmp_path / ".hlsgraph").exists()
    assert not (tmp_path / "hlsgraph.toml").exists()


def test_nested_manifest_parent_swap_is_revalidated_before_publish(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    probe = tmp_path / "symlink-probe"
    try:
        os.symlink(attacker, probe, target_is_directory=True)
        probe.unlink()
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    real_create = GraphBundle.create.__func__
    original_parent = tmp_path / "nested-original"

    def swap_parent(cls, project_root, manifest, **kwargs):
        bundle = real_create(cls, project_root, manifest, **kwargs)
        parent = tmp_path / "nested"
        parent.rename(original_parent)
        os.symlink(attacker, parent, target_is_directory=True)
        return bundle

    monkeypatch.setattr(GraphBundle, "create", classmethod(swap_parent))
    code, result = _invoke(
        capsys, tmp_path, "--manifest", "nested/hlsgraph.toml",
    )

    assert code == 1
    assert result["type"] == "CliError"
    assert "partial state could not be removed" not in result["error"]
    assert (tmp_path / "nested").is_symlink()
    assert not (attacker / "hlsgraph.toml").exists()
    assert not (original_parent / "hlsgraph.toml").exists()
    assert not (tmp_path / ".hlsgraph").exists()


def test_nested_manifest_plain_directory_identity_swap_is_fail_closed(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    real_create = GraphBundle.create.__func__
    original_parent = tmp_path / "nested-original"

    def swap_parent(cls, project_root, manifest, **kwargs):
        bundle = real_create(cls, project_root, manifest, **kwargs)
        parent = tmp_path / "nested"
        parent.rename(original_parent)
        parent.mkdir()
        (parent / "peer-owned").write_text("keep\n", encoding="utf-8")
        return bundle

    monkeypatch.setattr(GraphBundle, "create", classmethod(swap_parent))
    code, result = _invoke(
        capsys, tmp_path, "--manifest", "nested/hlsgraph.toml",
    )

    assert code == 1
    assert result["type"] == "CliError"
    assert (tmp_path / "nested/peer-owned").read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / "nested/hlsgraph.toml").exists()
    assert not (original_parent / "hlsgraph.toml").exists()
    assert not (tmp_path / ".hlsgraph").exists()


def test_nested_parent_swap_during_publish_guard_claim_is_fail_closed(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    original_open = cli_module._open_owner_file
    original_parent = tmp_path / "config-original"
    swapped = False

    def swap_before_guard(path, token):
        nonlocal swapped
        if path.name.startswith(".hlsgraph-init-parent.") and not swapped:
            swapped = True
            parent = tmp_path / "config"
            parent.rename(original_parent)
            parent.mkdir()
            (parent / "peer-owned").write_text("keep\n", encoding="utf-8")
        return original_open(path, token)

    monkeypatch.setattr(cli_module, "_open_owner_file", swap_before_guard)
    code, result = _invoke(
        capsys, tmp_path, "--manifest", "config/hlsgraph.toml",
    )

    assert code == 1
    assert result["type"] == "CliError"
    assert swapped is True
    assert (tmp_path / "config/peer-owned").read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / "config/hlsgraph.toml").exists()
    assert not list((tmp_path / "config").glob(".hlsgraph-init-parent.*.lock"))
    assert not (tmp_path / ".hlsgraph").exists()


def test_nested_manifest_uses_identity_guard_and_cleans_coordination_files(
    tmp_path: Path, capsys,
) -> None:
    code, result = _invoke(
        capsys, tmp_path, "--manifest", "config/hlsgraph.toml",
    )

    assert code == 0, result
    assert (tmp_path / "config/hlsgraph.toml").is_file()
    assert not list((tmp_path / "config").glob(".hlsgraph-init-parent.*.lock"))
    assert not list((tmp_path / "config").glob(".*.tmp"))
    assert not (tmp_path / ".hlsgraph/.init-owner").exists()


def test_owner_marker_claim_is_no_clobber_and_preserves_peer_file(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    real_claim = cli_module._claim_owner_file
    peer_token = "peer-marker\n"

    def peer_wins(path, token):
        if path.name == ".init-owner":
            path.write_text(peer_token, encoding="ascii")
        return real_claim(path, token)

    monkeypatch.setattr(cli_module, "_claim_owner_file", peer_wins)
    code, result = _invoke(capsys, tmp_path)

    assert code == 1
    assert result["type"] == "CliError"
    assert "could not be removed" in result["error"]
    marker = tmp_path / ".hlsgraph/.init-owner"
    assert marker.read_text(encoding="ascii") == peer_token
    assert not (tmp_path / "hlsgraph.toml").exists()
