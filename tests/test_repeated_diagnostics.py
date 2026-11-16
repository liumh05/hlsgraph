from __future__ import annotations

from dataclasses import replace

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import LibClangExtractor
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    Diagnostic, DiagnosticSeverity, SourceAnchor, TranslationUnit, VariantAction, stable_id,
)
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError


SOURCE = """int dut(int a[8]) {
  int s = 0;
  for (int i = 0; i < COLS; i++) s += a[i];
  for (int j = 0; j < COLS; j++) s += a[j];
  return s + COLS;
}
"""
requires_clang = pytest.mark.skipif(
    not LibClangExtractor.available(), reason="requires libclang",
)


def test_diagnostic_identity_includes_location_and_preserves_unlocated_ids():
    diagnostic = Diagnostic(
        snapshot_id="snapshot.test", code="clang.diagnostic",
        severity=DiagnosticSeverity.ERROR, message="undeclared identifier",
        stage="ast", subject_id="entity.test", run_id="run.test",
    )
    assert diagnostic.id == stable_id("diagnostic", {
        "snapshot": diagnostic.snapshot_id, "code": diagnostic.code,
        "message": diagnostic.message, "stage": diagnostic.stage,
        "run": diagnostic.run_id, "subject": diagnostic.subject_id,
    })
    locations = [
        SourceAnchor("artifact.first", start_line=3, start_column=23),
        SourceAnchor("artifact.first", start_line=4, start_column=23),
        SourceAnchor("artifact.first", start_line=3, start_column=24),
        SourceAnchor("artifact.second", start_line=3, start_column=23),
    ]
    located = [replace(diagnostic, id="", anchor=anchor) for anchor in locations]
    assert len({item.id for item in located}) == len(locations)
    assert replace(located[0], id="").id == located[0].id


@requires_clang
@pytest.mark.parametrize("variant", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_repeated_compile_errors_fail_cleanly_and_keep_ids(
    tmp_path, monkeypatch, variant, duplicate,
):
    source = tmp_path / "kernel.cpp"
    source.write_text("int dut(int a[8]) { return a[0]; }\n" if variant else SOURCE)
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest("dup", "dup", "dut", "kernel.cpp"),
    ))
    if variant:
        parent = project.index()
        assert parent.success
        action = VariantAction(
            parent_snapshot_id=parent.snapshot_id, kind="source.edit",
            scope_id=None,
            delta={"change": "undeclared loop bound"}, proposer="consumer.fixture",
        )
        project.record_variant_action(action)
        source.write_text(SOURCE)

    if duplicate:
        extract = LibClangExtractor.extract

        def repeated_extract(self, context):
            result = extract(self, context)
            result.diagnostics.extend([replace(item) for item in result.diagnostics])
            return result

        monkeypatch.setattr(LibClangExtractor, "extract", repeated_extract)

    first_ids = None
    for _ in range(2):
        project = Project(GraphBundle.open(tmp_path))
        result = project.index_variant(action.id) if variant else project.index()
        assert result.success is False
        assert not project.bundle.store.has_graph(result.snapshot_id)
        diagnostics = project.bundle.store.diagnostics(result.snapshot_id)
        errors = [item for item in diagnostics if item.code == "clang.diagnostic"
                  and item.message == "use of undeclared identifier 'COLS'"]
        assert len(errors) == 3
        artifact_id = next(item.id for item in project.bundle.store.artifacts(
            result.snapshot_id,
        ) if item.uri == "kernel.cpp")
        assert {(item.anchor.artifact_id, item.anchor.start_line, item.anchor.start_column)
                for item in errors} == {
            (artifact_id, line_number, line.index("COLS") + 1)
            for line_number, line in enumerate(SOURCE.splitlines(), 1) if "COLS" in line
        }
        assert result.diagnostics == len(diagnostics)
        ids = {item.id for item in diagnostics}
        if first_ids is None:
            first_ids = ids
        assert ids == first_ids
        runs = project.bundle.store.runs(result.snapshot_id)
        assert all(set(run.diagnostics) == ids for run in runs)
        if variant:
            assert result.materialization_status == "failed"
            assert project.bundle.latest_snapshot().id == parent.snapshot_id
            attempts = project.materializations(action.id)
            assert all(set(item["diagnostic_ids"]) == ids for item in attempts)
    assert len(runs) == 2
    if variant:
        assert len(attempts) == 2


@requires_clang
def test_same_header_error_from_multiple_translation_units_is_one_fact(tmp_path):
    (tmp_path / "common.h").write_text("inline int helper() { return COLS; }\n")
    (tmp_path / "kernel.cpp").write_text('#include "common.h"\nint dut() { return helper(); }\n')
    (tmp_path / "other.cpp").write_text('#include "common.h"\n')
    manifest = minimal_manifest("dup", "dup", "dut", "kernel.cpp")
    manifest.build.translation_units.append(TranslationUnit("other.cpp"))
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index()
    assert result.success is False
    errors = [item for item in project.bundle.store.diagnostics(result.snapshot_id)
              if item.code == "clang.diagnostic"]
    assert len(errors) == 1
    assert errors[0].message == "use of undeclared identifier 'COLS'"
    header = next(item for item in project.bundle.store.artifacts(result.snapshot_id)
                  if item.uri == "common.h")
    assert errors[0].anchor.artifact_id == header.id
    assert errors[0].anchor.start_line == 1
    ids = {item.id for item in project.bundle.store.diagnostics(result.snapshot_id)}
    retried = project.index()
    assert retried.success is False
    assert {item.id for item in project.bundle.store.diagnostics(retried.snapshot_id)} == ids


@requires_clang
def test_successful_retry_keeps_shared_diagnostics_active(tmp_path):
    (tmp_path / "kernel.cpp").write_text('#warning repeated warning\nvoid dut() {}\n')
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest("dup", "dup", "dut", "kernel.cpp"),
    ))
    first = project.index()
    assert first.success
    diagnostics = project.bundle.store.active_diagnostics(first.snapshot_id)
    assert any(item.code == "clang.diagnostic" for item in diagnostics)
    second = project.index()
    assert second.success
    assert second.snapshot_id == first.snapshot_id
    assert project.bundle.store.active_diagnostics(second.snapshot_id) == diagnostics
    assert project.bundle.store.diagnostics(second.snapshot_id) == diagnostics


def test_store_identical_diagnostics_are_idempotent_but_changes_still_rejected(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("dup", "dup", "dut", "kernel.cpp"),
    )
    snapshot = bundle.snapshot()
    diagnostic = Diagnostic(
        snapshot_id=snapshot.id, code="test.repeated",
        severity=DiagnosticSeverity.ERROR, message="same fact",
    )
    bundle.store.add_diagnostics([diagnostic, replace(diagnostic)])
    assert bundle.store.diagnostics(snapshot.id) == [diagnostic]
    with pytest.raises(StoreError, match="immutable diagnostics row changed"):
        bundle.store.add_diagnostics([replace(diagnostic, severity=DiagnosticSeverity.WARNING)])
