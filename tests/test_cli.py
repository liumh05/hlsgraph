from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import hlsgraph.cli as cli_module
from hlsgraph import Project
from hlsgraph.bundle import GraphBundle
from hlsgraph.cli import build_parser, main
from hlsgraph.manifest import load_manifest
from hlsgraph.runner import FakeRunner


def _invoke(capsys, *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    captured = capsys.readouterr()
    stream = captured.out if captured.out else captured.err
    return code, json.loads(stream)


def _indexed_project(tmp_path: Path, capsys) -> Path:
    source = tmp_path / "kernel.cpp"
    source.write_text(
        """#include <stdint.h>
void dut(int in[16], int out[16]) {
#pragma HLS PIPELINE II=1
  for (int i = 0; i < 16; ++i) out[i] = in[i] + 1;
}
""",
        encoding="utf-8",
    )
    code, result = _invoke(
        capsys, "init", "--project", str(tmp_path), "--project-id", "test.cli",
        "--name", "CLI fixture", "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 0
    assert result["private_source_embedded"] is False
    code, result = _invoke(capsys, "index", "--project", str(tmp_path), "--degraded")
    assert code == 0, result
    assert result["success"] is True
    return tmp_path


def test_parser_exposes_public_commands() -> None:
    parser = build_parser()
    choices = next(action for action in parser._actions
                   if action.dest == "command").choices
    assert {"init", "index", "status", "query", "explore", "retrieve", "run", "render",
            "export", "doctor", "knowledge", "serve"}.issubset(choices)
    args = parser.parse_args(["run", "--backend", "fake"])
    assert args.backend == "fake"
    assert args.allow_execution is False


def test_init_toml_escapes_user_strings_and_round_trips(tmp_path: Path, capsys) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    name = 'Quoted "demo" \\ laboratory'
    top = 'dut"variant'
    code, result = _invoke(
        capsys, "init", "--project", str(tmp_path), "--project-id", "test.quoted_init",
        "--name", name, "--top", top, "--source", "kernel.cpp",
    )
    assert code == 0, result
    manifest = load_manifest(tmp_path / "hlsgraph.toml")
    assert manifest.name == name
    assert manifest.build.top == top
    assert Project.open(tmp_path).bundle.manifest.name == name


def test_init_validates_before_atomic_manifest_replacement(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "hlsgraph.toml"
    original = "# existing user manifest\n"
    manifest_path.write_text(original, encoding="utf-8")
    code, result = _invoke(
        capsys, "init", "--project", str(tmp_path), "--force",
        "--project-id", "invalid project id", "--name", "invalid",
        "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 1
    assert result["type"] in {"ManifestError", "ValueError"}
    assert manifest_path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".hlsgraph").exists()


def test_init_failure_rolls_back_new_bundle_and_preserves_manifest(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    manifest_path = tmp_path / "hlsgraph.toml"
    original = "# existing user manifest\n"
    manifest_path.write_text(original, encoding="utf-8")

    def broken_create(cls, project_root, manifest, **kwargs):
        partial = Path(project_root) / ".hlsgraph"
        (partial / "partial-state").write_text("incomplete", encoding="utf-8")
        raise sqlite3.OperationalError("simulated database initialization failure")

    monkeypatch.setattr(GraphBundle, "create", classmethod(broken_create))
    code, result = _invoke(
        capsys, "init", "--project", str(tmp_path), "--force",
        "--project-id", "test.rollback", "--name", "rollback",
        "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 1
    assert result["type"] == "CliError"
    assert manifest_path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".hlsgraph").exists()


def test_init_refuses_existing_ledger_without_leaving_alternate_manifest(
    tmp_path: Path, capsys,
) -> None:
    root = _indexed_project(tmp_path, capsys)
    snapshot_id = Project.open(root).bundle.latest_snapshot().id
    alternate = root / "alternate.toml"
    code, result = _invoke(
        capsys, "init", "--project", str(root), "--force",
        "--manifest", alternate.name, "--project-id", "test.alternate",
        "--name", "alternate", "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 1
    assert "will not replace" in result["error"]
    assert not alternate.exists()
    assert Project.open(root).bundle.latest_snapshot().id == snapshot_id


def test_init_rollback_never_deletes_a_peer_replacement(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    def peer_replaces_claim(cls, project_root, manifest, **kwargs):
        bundle = Path(project_root) / ".hlsgraph"
        shutil.rmtree(bundle)
        bundle.mkdir()
        (bundle / "peer-owned").write_text("keep", encoding="utf-8")
        raise sqlite3.OperationalError("simulated concurrent replacement")

    monkeypatch.setattr(GraphBundle, "create", classmethod(peer_replaces_claim))
    code, result = _invoke(
        capsys, "init", "--project", str(tmp_path),
        "--project-id", "test.peer", "--name", "peer",
        "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 1
    assert "could not be removed" in result["error"]
    assert (tmp_path / ".hlsgraph/peer-owned").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "hlsgraph.toml").exists()


def test_init_owner_marker_failure_removes_only_the_claimed_empty_directory(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    original_claim = cli_module._claim_owner_file

    def fail_owner_marker(path, token):
        if path.name == ".init-owner":
            raise PermissionError("simulated owner marker failure")
        return original_claim(path, token)

    monkeypatch.setattr(cli_module, "_claim_owner_file", fail_owner_marker)
    code, result = _invoke(
        capsys, "init", "--project", str(tmp_path),
        "--project-id", "test.marker", "--name", "marker",
        "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 1
    assert result["type"] == "CliError"
    assert not (tmp_path / ".hlsgraph").exists()
    assert not (tmp_path / "hlsgraph.toml").exists()


def test_cli_init_index_query_status_render_and_export(tmp_path: Path, capsys) -> None:
    root = _indexed_project(tmp_path, capsys)

    code, result = _invoke(capsys, "status", "--project", str(root))
    assert code == 0
    assert result["snapshot_id"].startswith("snapshot_")
    assert result["stale"] is False

    code, result = _invoke(capsys, "query", "--project", str(root), "dut")
    assert code == 0
    assert any(item["name"] == "dut" for item in result["items"])

    code, result = _invoke(capsys, "explore", "--project", str(root), "dut")
    assert code == 0
    assert result["focus"]
    assert result["entities"]

    code, result = _invoke(
        capsys, "render", "--project", str(root), "graph.mmd", "--format", "mermaid",
    )
    assert code == 0
    assert Path(result["output"]).read_text(encoding="utf-8").startswith("flowchart LR")

    code, result = _invoke(
        capsys, "export", "--project", str(root), "graph.json", "--kind", "graph",
    )
    assert code == 0
    exported = json.loads(Path(result["output"]).read_text(encoding="utf-8"))
    assert "graph" in exported or "entities" in exported


def test_default_cli_and_sdk_index_share_canonical_identity(tmp_path: Path, capsys) -> None:
    root = _indexed_project(tmp_path, capsys)
    sdk_result = Project.open(root).index(degraded=True)
    assert sdk_result.success

    code, cli_result = _invoke(
        capsys, "index", "--project", str(root), "--degraded", "--force",
    )
    assert code == 0
    assert cli_result["snapshot_id"] == sdk_result.snapshot_id
    assert cli_result["graph_hash"] == sdk_result.graph_hash


def test_run_requires_explicit_backend_and_execution_acknowledgement(
    tmp_path: Path, capsys,
) -> None:
    root = _indexed_project(tmp_path, capsys)

    code, result = _invoke(
        capsys, "run", "--project", str(root), "--backend", "local",
    )
    assert code == 1
    assert "--allow-execution" in result["error"]

    code, result = _invoke(
        capsys, "run", "--project", str(root), "--backend", "fake",
    )
    assert code == 0
    assert result["backend"] == "fake"
    assert result["tool_truth"] is False


def test_run_loads_only_explicit_runner_v2_plugin(tmp_path: Path, capsys, monkeypatch) -> None:
    root = _indexed_project(tmp_path, capsys)
    selected = {}

    def load(names, configs):
        selected["names"] = names
        selected["configs"] = configs
        return [FakeRunner()]

    monkeypatch.setattr(cli_module, "load_runners", load)
    code, result = _invoke(
        capsys, "run", "--project", str(root), "--backend", "plugin",
        "--runner-plugin", "fixture", "--runner-config", '{"mode":"test"}',
    )
    assert code == 1
    assert "--allow-execution" in result["error"]

    code, result = _invoke(
        capsys, "run", "--project", str(root), "--backend", "plugin",
        "--runner-plugin", "fixture", "--runner-config", '{"mode":"test"}',
        "--allow-execution",
    )
    assert code == 0
    assert result["backend"] == "plugin"
    assert selected == {
        "names": ["fixture"], "configs": {"fixture": {"mode": "test"}},
    }


def test_doctor_and_knowledge_are_read_only_json_commands(tmp_path: Path, capsys) -> None:
    code, result = _invoke(capsys, "doctor")
    assert code == 0
    assert result["healthy"] is True
    assert "no vendor tool or SSH command was run" in result["notes"][0]
    assert not (tmp_path / ".hlsgraph").exists()

    code, result = _invoke(capsys, "knowledge")
    assert code == 0
    assert isinstance(result["packs"], list)

    guide = tmp_path / "UG-local.pdf"
    guide.write_bytes(b"private local guide bytes")
    local_index = tmp_path / "knowledge-index.json"
    code, result = _invoke(
        capsys, "knowledge", "index", "--project", str(tmp_path),
        "--path", str(guide), "--document-id", "local.ug",
        "--document-version", "1.0", "--output", str(local_index),
    )
    assert code == 0
    assert result["content_copied"] is False
    payload = local_index.read_text(encoding="utf-8")
    assert "private local guide bytes" not in payload
    assert json.loads(payload)["documents"][0]["sha256"] == result["document"]["sha256"]


def test_knowledge_index_default_path_is_in_private_sidecar(
    tmp_path: Path, capsys,
) -> None:
    guide = tmp_path / "local-guide.md"
    guide.write_text("private local guidance\n", encoding="utf-8")
    code, result = _invoke(
        capsys, "knowledge", "index", "--project", str(tmp_path),
        "--path", str(guide), "--document-id", "local.guide",
        "--document-version", "1",
    )
    assert code == 0
    index_path = tmp_path / ".hlsgraph/private/knowledge/documents.json"
    assert index_path.is_file()
    assert Path(result["output"]) == index_path
    assert "private local guidance" not in index_path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert index_path.parent.stat().st_mode & 0o777 == 0o700
        assert index_path.stat().st_mode & 0o777 == 0o600


def test_knowledge_v03_cli_coverage_install_sync_and_private_sidecar(
    tmp_path: Path, capsys,
) -> None:
    root = _indexed_project(tmp_path, capsys)
    code, coverage = _invoke(
        capsys, "knowledge", "coverage", "--project", str(root),
    )
    assert code == 0
    assert coverage["complete"] is True
    assert coverage["classification_complete"] is coverage["complete"]
    assert coverage["review_ready"] is all(
        item["review_status"] != "unreviewed" for item in coverage["items"]
    )
    assert coverage["items"] and coverage["installed_packs"]
    pack_id = coverage["installed_packs"][0]["pack_id"]

    code, installed = _invoke(
        capsys, "knowledge", "install", "--project", str(root),
        "--pack-id", pack_id,
    )
    assert code == 0
    assert installed["results"][0]["status"] == "unchanged"
    assert installed["mutated_design_facts"] is False

    code, synced = _invoke(
        capsys, "knowledge", "sync", "--project", str(root),
    )
    assert code == 1
    assert "not review_ready" in synced["error"]

    sentinel = "PRIVATE_CLI_KNOWLEDGE_SENTINEL"
    guide = root / "local-guide.md"
    guide.write_text(f"# Pipeline\n{sentinel} achieved II.\n", encoding="utf-8")
    metadata_index = root / "local-index.json"
    code, _ = _invoke(
        capsys, "knowledge", "index", "--project", str(root),
        "--path", str(guide), "--document-id", "local.guide",
        "--document-version", "1", "--output", str(metadata_index),
    )
    assert code == 0
    code, built = _invoke(
        capsys, "knowledge", "build-local-index", "--project", str(root),
        "--metadata-index", str(metadata_index),
    )
    assert code == 0
    assert built["content_embedded_in_canonical"] is False
    assert built["manifest"]["chunk_count"] == 1
    assert sentinel not in json.dumps(built, ensure_ascii=False)
    assert (root / ".hlsgraph/private/knowledge/chunks.sqlite").is_file()


def test_knowledge_local_index_parser_is_explicit_and_receives_key_value_config(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    import hlsgraph.plugins as plugins

    root = _indexed_project(tmp_path, capsys)
    guide = root / "local-guide.md"
    guide.write_text("# Pipeline\nachieved II evidence.\n", encoding="utf-8")
    metadata_index = root / "local-index.json"
    code, _ = _invoke(
        capsys, "knowledge", "index", "--project", str(root),
        "--path", str(guide), "--document-id", "local.guide",
        "--document-version", "1", "--output", str(metadata_index),
    )
    assert code == 0

    loaded = []
    monkeypatch.setattr(
        plugins, "load_knowledge_parser",
        lambda name, config: loaded.append((name, config)) or None,
    )
    code, _ = _invoke(
        capsys, "knowledge", "build-local-index", "--project", str(root),
        "--metadata-index", str(metadata_index), "--parser", "test.pdf",
        "--parser-config", "pages=12", "--parser-config", "language=en",
    )
    assert code == 0
    assert loaded == [("test.pdf", {"pages": 12, "language": "en"})]

    code, error = _invoke(
        capsys, "knowledge", "build-local-index", "--project", str(root),
        "--metadata-index", str(metadata_index), "--parser-config", "pages=12",
    )
    assert code == 1
    assert error["error"] == "--parser-config requires --parser"


def test_status_before_first_index_is_supported(tmp_path: Path, capsys) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    code, _ = _invoke(
        capsys, "init", "--project", str(tmp_path), "--project-id", "test.unindexed",
        "--top", "dut", "--source", "kernel.cpp",
    )
    assert code == 0
    code, result = _invoke(capsys, "status", "--project", str(tmp_path))
    assert code == 0
    assert result["snapshot_id"] is None
    assert result["stale"] is True


def test_serve_delegates_to_lazy_rest_entrypoint(tmp_path: Path, monkeypatch) -> None:
    import hlsgraph.api

    calls = []
    monkeypatch.setattr(hlsgraph.api, "serve", lambda *args, **kwargs: calls.append((args, kwargs)))
    code = main([
        "serve", "--project", str(tmp_path), "--host", "127.0.0.1", "--port", "8765",
        "--snapshot", "snapshot_test",
    ])
    assert code == 0
    assert calls == [((tmp_path.resolve(),), {
        "host": "127.0.0.1", "port": 8765,
        "snapshot_id": "snapshot_test", "allow_remote": False,
    })]
