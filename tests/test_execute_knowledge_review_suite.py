from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import jsonschema

from tools import knowledge_review_shards as shards
from tools import run_knowledge_review as review
from tools import execute_knowledge_review_suite as executor


ROOT = Path(__file__).resolve().parents[1]


def _chunk(path: str, payload: bytes) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "index": 0,
        "path": path,
        "sha256": digest,
        "size": len(payload),
        "byte_start": 0,
        "byte_end": len(payload),
        "original_sha256": digest,
        "original_size": len(payload),
    }


def _projected_cache(tmp_path: Path) -> review.ReviewCache:
    root = tmp_path / "cache"
    source_path = "chunks/source/00.utf8"
    citation_path = "chunks/citation/00.utf8"
    source = b"alpha\n"
    citation = b"private citation text\n"
    for relative, payload in ((source_path, source), (citation_path, citation)):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    manifest = {
        "chunk_contract": {"sha256": "1" * 64},
        "files": [{
            "path": "src/example.py",
            "hash_kind": "raw_sha256",
            "sha256": hashlib.sha256(source).hexdigest(),
            "cache_sha256": hashlib.sha256(source).hexdigest(),
            "model_inspection_required": True,
            "chunks": [_chunk(source_path, source)],
        }],
        "citations": [{
            "requested_url": "https://example.invalid/rule",
            "evidence_url": "https://example.invalid/rule",
            "resolver_id": "direct.sha256.v1",
            "reference_ids": ["2" * 64],
            "body_sha256": "3" * 64,
            "inspection_sha256": hashlib.sha256(citation).hexdigest(),
            "parser_id": "utf8.v1",
            "parser_command_sha256": "4" * 64,
            "inspection_chunks": [_chunk(citation_path, citation)],
        }],
    }
    review._harden_private_tree(root)
    return review.ReviewCache(root, manifest, review._canonical_json(manifest))


def _completed(command: str, output: str, call_id: str) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "id": call_id,
            "command": command,
            "status": "completed",
            "exit_code": 0,
            "aggregated_output": output,
        },
    }


def test_canonical_command_is_path_neutral_and_pinned() -> None:
    argv = executor.canonical_shard_command_argv()
    joined = "\n".join(argv)
    assert "$ROOT/" + executor.SHARD_OUTPUT_SCHEMA_PATH in argv
    assert "$CACHE" in argv
    assert "$CODEX_RUNTIME" in joined
    assert "gpt-5.6-sol" in argv
    assert 'model_reasoning_effort="medium"' in argv
    assert "model_context_window=372000" in argv
    assert "model_auto_compact_token_limit=300000" in argv
    assert 'model_auto_compact_token_limit_scope="total"' in argv
    assert 'model_provider="hlsgraph_review_http"' in argv
    assert any(
        value.startswith("model_providers.hlsgraph_review_http=")
        and 'wire_api="responses"' in value
        and "requires_openai_auth=true" in value
        and "supports_websockets=false" in value
        for value in argv
    )
    assert all(
        feature in argv
        for feature in ("code_mode", "code_mode_host", "code_mode_only")
    )
    assert "D:\\" not in joined and "/root/" not in joined
    assert executor.canonical_shard_command_sha256() == (
        "2637881d66731c150ef54968f851785c9ed09421be1c88a81c58cbf4b9786a61"
    )


def test_actual_command_must_normalize_exactly_to_declared_argv(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    codex = runtime / "codex"
    codex.write_bytes(b"fixture")
    cache = tmp_path / "cache"
    cache.mkdir()
    filesystem = {
        ":minimal": "read", str(cache.resolve()): "read",
        str(runtime.resolve()): "read",
    }
    profile = [
        f"default_permissions={review._toml(review.PERMISSION_PROFILE)}",
        f"permissions.{review.PERMISSION_PROFILE}.network.enabled=false",
        f"permissions.{review.PERMISSION_PROFILE}.filesystem="
        + review._toml_inline_table(filesystem),
        'web_search="disabled"',
    ]
    actual = executor._actual_shard_command(
        root=ROOT, cache_root=cache, codex=str(codex.resolve()),
        profile_values=profile,
    )
    assert executor.normalize_actual_shard_command(
        actual, root=ROOT, cache_root=cache, codex=codex,
    ) == executor.canonical_shard_command_argv()
    assert executor.validate_actual_shard_command(
        actual, root=ROOT, cache_root=cache, codex=codex,
    ) == executor.canonical_shard_command_sha256()
    changed = list(actual)
    changed.insert(-1, "--unexpected")
    with pytest.raises(executor.SuiteExecutionError, match="actual Codex argv"):
        executor.validate_actual_shard_command(
            changed, root=ROOT, cache_root=cache, codex=codex,
        )
    changed_provider = [
        value.replace("supports_websockets=false", "supports_websockets=true")
        for value in actual
    ]
    with pytest.raises(executor.SuiteExecutionError, match="actual Codex argv"):
        executor.validate_actual_shard_command(
            changed_provider, root=ROOT, cache_root=cache, codex=codex,
        )


def test_assigned_material_is_one_exact_command_per_chunk(tmp_path: Path) -> None:
    cache = _projected_cache(tmp_path)
    material = executor.assigned_chunk_material(cache)
    assert material.paths == tuple(sorted(material.paths))
    assert material.commands == tuple(
        review._codex_shell_event_command(f"head -n 100000000 {path}")
        for path in material.paths
    )
    assert len(material.payloads) == len(material.commands) == 2
    assert len(material.inventory_sha256) == 64


def test_codex_shell_event_wrapper_is_exact_and_fail_closed() -> None:
    inner = "head -n 100000000 chunks/source/a/000000-deadbeef.utf8"
    wrapped = review._codex_shell_event_command(inner)
    assert wrapped == f"/bin/bash -lc '{inner}'"
    assert review._unwrap_codex_shell_event_command(wrapped) == inner
    for changed in (
        inner,
        f"/bin/sh -lc '{inner}'",
        f"/bin/bash -lc \"{inner}\"",
        f"/bin/bash -lc '{inner}; true'",
    ):
        with pytest.raises(ValueError):
            review._unwrap_codex_shell_event_command(changed)


def test_budget_uses_prompt_chunks_commands_and_event_allowance(
    tmp_path: Path,
) -> None:
    material = executor.assigned_chunk_material(_projected_cache(tmp_path))

    class CountingTokenizer:
        @staticmethod
        def encode(text: str) -> list[str]:
            return list(text)

    budget = executor.enforce_shard_token_budget(
        prompt=b"prompt", material=material, tokenizer=CountingTokenizer(),
    )
    assert budget.prompt_tokens == len("prompt")
    assert budget.chunk_tokens == sum(len(value) for value in material.payloads)
    assert budget.command_tokens == sum(len(value) for value in material.commands)
    assert budget.tool_event_count == 2
    assert budget.tool_event_overhead_tokens == (
        2 * shards.TOOL_EVENT_OVERHEAD_TOKENS
    )
    assert budget.within_budget is True


def test_sanitizer_redacts_citation_and_exact_inventory_rejects_extra(
    tmp_path: Path,
) -> None:
    cache = _projected_cache(tmp_path)
    material = executor.assigned_chunk_material(cache)
    rows = []
    for index, command in enumerate(material.commands):
        inner = review._unwrap_codex_shell_event_command(command)
        output, _operations, _citation = review._expected_command(cache, inner)
        rows.append(_completed(command, output, f"call-{index}"))
    raw = review._canonical_jsonl(rows)
    sanitized = executor.sanitize_shard_raw_stream(raw, cache)
    assert b"private citation text" not in sanitized
    assert b"HLSGRAPH_REVIEW_CACHE_OUTPUT:" in sanitized
    executor.require_exact_command_inventory(sanitized, material.commands)
    with pytest.raises(executor.SuiteExecutionError, match="inventory differs"):
        executor.require_exact_command_inventory(
            review._canonical_jsonl(rows + [rows[0]]), material.commands,
        )


def test_sanitizer_rejects_non_deterministic_command_output(tmp_path: Path) -> None:
    cache = _projected_cache(tmp_path)
    command = executor.assigned_chunk_material(cache).commands[0]
    raw = review._canonical_jsonl([_completed(command, "wrong", "call-0")])
    with pytest.raises(executor.SuiteExecutionError, match="differs"):
        executor.sanitize_shard_raw_stream(raw, cache)


def test_injected_process_runner_receives_exact_inputs(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_runner(command, prompt, cwd, environment, timeout):
        seen.update({
            "command": list(command), "prompt": prompt, "cwd": cwd,
            "environment": dict(environment), "timeout": timeout,
        })
        return executor.ProcessOutcome(0, b"{}\n", b"")

    result = executor.invoke_process(
        runner=fake_runner, command=["codex", "exec"], prompt=b"prompt",
        cwd=tmp_path, environment={"LANG": "C.UTF-8"}, timeout_seconds=9,
    )
    assert result.returncode == 0
    assert seen == {
        "command": ["codex", "exec"], "prompt": b"prompt",
        "cwd": tmp_path, "environment": {"LANG": "C.UTF-8"},
        "timeout": 9,
    }


def test_nonzero_process_persists_exact_and_redacted_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _projected_cache(tmp_path / "projected")
    private = b"private citation text\n"
    stdout = review._canonical_jsonl([{
        "type": "error", "message": private.decode("utf-8"),
    }])
    stderr = b"failure: " + private
    monkeypatch.setattr(review, "MAX_RAW_REVIEW_BYTES", 1)
    outcome = executor.invoke_process(
        runner=lambda *_args: executor.ProcessOutcome(7, stdout, stderr),
        command=["codex", "exec"], prompt=b"prompt", cwd=cache.root,
        environment={"LANG": "C.UTF-8"}, timeout_seconds=9,
    )
    evidence = tmp_path / "evidence"
    paths = {
        "raw_path": evidence / executor.RAW_STREAM_PATH,
        "sanitized_path": evidence / executor.SANITIZED_STREAM_PATH,
        "raw_stderr_path": evidence / executor.RAW_STDERR_PATH,
        "stderr_path": evidence / executor.STDERR_PATH,
        "process_path": evidence / executor.PROCESS_EVIDENCE_PATH,
    }

    with pytest.raises(executor.SuiteExecutionError, match="exit code 7"):
        executor._persist_process_outcome(
            outcome=outcome, command=["codex", "exec"], cwd=cache.root,
            prompt=b"prompt", command_sha256="a" * 64, cache=cache,
            **paths,
        )

    assert paths["raw_path"].read_bytes() == stdout
    assert paths["raw_stderr_path"].read_bytes() == stderr
    assert private not in paths["sanitized_path"].read_bytes()
    assert private not in paths["stderr_path"].read_bytes()
    assert b"private citation text" not in paths["sanitized_path"].read_bytes()
    process = json.loads(paths["process_path"].read_bytes())
    assert paths["process_path"].read_bytes() == review._canonical_json(process)
    assert process["returncode"] == 7
    assert process["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()
    assert process["stderr_sha256"] == hashlib.sha256(stderr).hexdigest()
    assert set(path.name for path in evidence.iterdir()) == {
        executor.RAW_STREAM_PATH, executor.SANITIZED_STREAM_PATH,
        executor.RAW_STDERR_PATH, executor.STDERR_PATH,
        executor.PROCESS_EVIDENCE_PATH,
    }


def test_zero_process_persists_safe_diagnostics_before_contract_validation(
    tmp_path: Path,
) -> None:
    cache = _projected_cache(tmp_path / "projected")
    private = b"private citation text\n"
    stdout = review._canonical_jsonl([{
        "type": "error", "message": "transport warning",
    }])
    stderr = b"transport warning: " + private
    outcome = executor.ProcessOutcome(0, stdout, stderr)
    evidence = tmp_path / "evidence"
    paths = {
        "raw_path": evidence / executor.RAW_STREAM_PATH,
        "sanitized_path": evidence / executor.SANITIZED_STREAM_PATH,
        "raw_stderr_path": evidence / executor.RAW_STDERR_PATH,
        "stderr_path": evidence / executor.STDERR_PATH,
        "process_path": evidence / executor.PROCESS_EVIDENCE_PATH,
    }

    executor._persist_process_outcome(
        outcome=outcome, command=["codex", "exec"], cwd=cache.root,
        prompt=b"prompt", command_sha256="a" * 64, cache=cache,
        **paths,
    )
    assert not paths["sanitized_path"].exists()
    assert not paths["stderr_path"].exists()
    executor._persist_success_derivatives(
        outcome=outcome, cache=cache,
        sanitized_path=paths["sanitized_path"],
        stderr_path=paths["stderr_path"],
    )
    with pytest.raises(executor.SuiteExecutionError, match="inventory differs"):
        executor.require_exact_command_inventory(
            stdout, executor.assigned_chunk_material(cache).commands,
        )

    assert paths["raw_path"].read_bytes() == stdout
    assert paths["raw_stderr_path"].read_bytes() == stderr
    assert private not in paths["sanitized_path"].read_bytes()
    assert private not in paths["stderr_path"].read_bytes()
    process = json.loads(paths["process_path"].read_bytes())
    assert process["returncode"] == 0
    assert set(path.name for path in evidence.iterdir()) == {
        executor.RAW_STREAM_PATH, executor.SANITIZED_STREAM_PATH,
        executor.RAW_STDERR_PATH, executor.STDERR_PATH,
        executor.PROCESS_EVIDENCE_PATH,
    }


def test_zero_process_sanitizer_failure_still_persists_redacted_derivatives(
    tmp_path: Path,
) -> None:
    cache = _projected_cache(tmp_path / "projected")
    private = b"private citation text\n"
    outcome = executor.ProcessOutcome(
        0, b"not-json " + private, b"bad stream " + private,
    )
    evidence = tmp_path / "evidence"
    paths = {
        "raw_path": evidence / executor.RAW_STREAM_PATH,
        "sanitized_path": evidence / executor.SANITIZED_STREAM_PATH,
        "raw_stderr_path": evidence / executor.RAW_STDERR_PATH,
        "stderr_path": evidence / executor.STDERR_PATH,
        "process_path": evidence / executor.PROCESS_EVIDENCE_PATH,
    }
    executor._persist_process_outcome(
        outcome=outcome, command=["codex", "exec"], cwd=cache.root,
        prompt=b"prompt", command_sha256="a" * 64, cache=cache,
        **paths,
    )

    with pytest.raises(ValueError, match="cannot parse strict"):
        executor._persist_success_derivatives(
            outcome=outcome, cache=cache,
            sanitized_path=paths["sanitized_path"],
            stderr_path=paths["stderr_path"],
        )

    assert private not in paths["sanitized_path"].read_bytes()
    assert private not in paths["stderr_path"].read_bytes()
    assert paths["raw_path"].read_bytes() == outcome.stdout
    assert paths["raw_stderr_path"].read_bytes() == outcome.stderr


def test_successful_oversized_process_stream_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review, "MAX_RAW_REVIEW_BYTES", 1)
    with pytest.raises(executor.SuiteExecutionError, match="fixed limit"):
        executor.invoke_process(
            runner=lambda *_args: executor.ProcessOutcome(0, b"{}\n", b""),
            command=["codex", "exec"], prompt=b"prompt", cwd=tmp_path,
            environment={"LANG": "C.UTF-8"}, timeout_seconds=9,
        )


def test_formal_tokenizer_requires_one_frozen_offline_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "tokenizer"
    cache.mkdir()
    payload = b"pinned tokenizer table"
    table = cache / executor.TOKENIZER_BPE_CACHE_KEY
    table.write_bytes(payload)
    if executor.os.name != "nt":
        table.chmod(0o400)
        cache.chmod(0o500)
    sentinel = object()
    monkeypatch.setenv(executor.TOKENIZER_CACHE_ENV, str(cache.resolve()))
    monkeypatch.setattr(executor, "TOKENIZER_BPE_SIZE", len(payload))
    monkeypatch.setattr(
        executor, "TOKENIZER_BPE_SHA256", hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(shards, "load_verified_tokenizer", lambda: sentinel)
    if executor.os.name != "nt":
        monkeypatch.setattr(review, "_mount_fstype", lambda _path: "ext4")

    loaded, resolved = executor._load_offline_formal_tokenizer()

    assert loaded is sentinel
    assert resolved == cache.resolve()


def test_formal_tokenizer_rejects_missing_extra_and_changed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(executor.TOKENIZER_CACHE_ENV, raising=False)
    with pytest.raises(executor.SuiteExecutionError, match="absolute"):
        executor._load_offline_formal_tokenizer()

    cache = tmp_path / "tokenizer"
    cache.mkdir()
    table = cache / executor.TOKENIZER_BPE_CACHE_KEY
    table.write_bytes(b"wrong")
    extra = cache / "extra"
    extra.write_bytes(b"unexpected")
    if executor.os.name != "nt":
        table.chmod(0o400)
        extra.chmod(0o400)
        cache.chmod(0o500)
        monkeypatch.setattr(review, "_mount_fstype", lambda _path: "ext4")
    monkeypatch.setenv(executor.TOKENIZER_CACHE_ENV, str(cache.resolve()))
    with pytest.raises(executor.SuiteExecutionError, match="exactly"):
        executor._load_offline_formal_tokenizer()

    if executor.os.name != "nt":
        cache.chmod(0o700)
        extra.unlink()
        cache.chmod(0o500)
    else:
        extra.unlink()
    monkeypatch.setattr(executor, "TOKENIZER_BPE_SIZE", 5)
    monkeypatch.setattr(executor, "TOKENIZER_BPE_SHA256", "0" * 64)
    with pytest.raises(executor.SuiteExecutionError, match="identity differs"):
        executor._load_offline_formal_tokenizer()


def test_formal_publication_rejects_injected_runner_before_host_mutation(
    tmp_path: Path,
) -> None:
    def fake_runner(command, prompt, cwd, environment, timeout):
        return executor.ProcessOutcome(0, b"", b"")

    with pytest.raises(executor.SuiteExecutionError, match="built-in"):
        executor.execute_knowledge_review_suite(
            tmp_path, tmp_path / "work", codex_command="codex",
            timeout_seconds=1, process_runner=fake_runner, publish=True,
        )
    assert not (tmp_path / "work").exists()


def test_formal_publication_rejects_injected_fetcher_before_host_mutation(
    tmp_path: Path,
) -> None:
    def fake_fetcher(url, timeout, maximum):  # pragma: no cover - never called
        raise AssertionError((url, timeout, maximum))

    with pytest.raises(executor.SuiteExecutionError, match="online citation"):
        executor.execute_knowledge_review_suite(
            tmp_path, tmp_path / "work", codex_command="codex",
            timeout_seconds=1, fetcher=fake_fetcher, publish=True,
        )
    assert not (tmp_path / "work").exists()


def test_control_surface_includes_executor_and_suite_contracts() -> None:
    surface = executor.freeze_control_surface(
        Path(__file__).resolve().parents[1],
    )
    assert "tools/execute_knowledge_review_suite.py" in surface
    assert "tools/knowledge_review_shard.schema.json" in surface
    assert set(surface) == set(executor.CONTROL_SURFACE_PATHS)
    assert all(len(value) == 64 for value in surface.values())


def test_suite_evidence_manifest_has_fixed_protocol_and_shard_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {"sha256": "a" * 64}
    monkeypatch.setattr(
        review, "_validate_runtime_manifest", lambda value: dict(value),
    )
    monkeypatch.setattr(
        review, "_validate_boundary_contract",
        lambda value, expected_cache_sha256: dict(value),
    )
    protocols = {}
    boundaries = {}
    process_hashes = {}
    for protocol_index, protocol_id in enumerate(executor.PROTOCOL_ORDER):
        invocations = []
        protocol_boundaries = []
        protocol_process_hashes = []
        for shard_index, shard_id in enumerate(shards.SHARD_ORDER):
            cache_hash = hashlib.sha256(
                f"{protocol_id}:{shard_id}:cache".encode()
            ).hexdigest()
            contract_hash = hashlib.sha256(
                f"{protocol_id}:{shard_id}:boundary".encode()
            ).hexdigest()
            invocations.append({
                "shard_manifest": {"shard_id": shard_id},
                "cache_manifest_sha256": cache_hash,
                "raw_output_sha256": hashlib.sha256(
                    f"{protocol_id}:{shard_id}:raw".encode()
                ).hexdigest(),
                "sanitized_output_sha256": hashlib.sha256(
                    f"{protocol_id}:{shard_id}:sanitized".encode()
                ).hexdigest(),
                "boundary_contract_sha256": contract_hash,
            })
            protocol_boundaries.append({
                "runtime_manifest": runtime,
                "contract_sha256": contract_hash,
            })
            protocol_process_hashes.append(hashlib.sha256(
                f"{protocol_id}:{shard_id}:process".encode()
            ).hexdigest())
        protocols[protocol_id] = SimpleNamespace(
            protocol_id=protocol_id,
            invocations=tuple(invocations),
            snapshot=SimpleNamespace(sha256=f"{protocol_index + 1}" * 64),
            full_cache=SimpleNamespace(sha256=f"{protocol_index + 3}" * 64),
            receipt={"full_evidence_surface_sha256": "9" * 64},
        )
        boundaries[protocol_id] = protocol_boundaries
        process_hashes[protocol_id] = protocol_process_hashes
    manifest = executor.build_suite_evidence_manifest(
        runtime_manifest=runtime, protocols=protocols,
        boundary_contracts=boundaries,
        process_evidence_sha256s=process_hashes,
    )
    evidence_schema = json.loads(
        (ROOT / "tools/knowledge_review_suite_evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(evidence_schema)
    jsonschema.Draft202012Validator(evidence_schema).validate(manifest)
    assert set(manifest) == {"schema_version", "runtime_manifest", "protocols"}
    assert [row["protocol_id"] for row in manifest["protocols"]] == list(
        executor.PROTOCOL_ORDER
    )
    for row in manifest["protocols"]:
        assert set(row) == {
            "protocol_id", "acquisition_mode",
            "replay_source_manifest_sha256", "review_snapshot_sha256",
            "full_cache_manifest_sha256",
            "full_citation_evidence_surface_sha256", "shards",
        }
        assert [item["shard_id"] for item in row["shards"]] == list(
            shards.SHARD_ORDER
        )
        assert all(set(item) == {
            "shard_id", "projected_cache_manifest_sha256",
            "raw_output_sha256", "sanitized_output_sha256",
            "process_evidence_sha256", "boundary_contract",
        } for item in row["shards"])
    assert manifest["protocols"][0]["acquisition_mode"] == (
        "online_pinned_identity"
    )
    assert manifest["protocols"][0]["replay_source_manifest_sha256"] is None
    assert manifest["protocols"][1]["acquisition_mode"] == (
        "offline_replay_from_semantic"
    )
    assert manifest["protocols"][1]["replay_source_manifest_sha256"] == (
        protocols[executor.PROTOCOL_ORDER[0]].full_cache.sha256
    )
