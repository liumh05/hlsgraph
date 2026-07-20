from __future__ import annotations

from dataclasses import dataclass

import pytest

import hlsgraph.plugins as plugins
import hlsgraph.sdk as sdk
from hlsgraph.bundle import GraphBundle
from hlsgraph.manifest import minimal_manifest
from hlsgraph.plugins import PluginError, load_extractors, load_runners
from hlsgraph.runner import FakeRunner, PROTOCOL_VERSION
from hlsgraph.sdk import Project


class GoodExtractor:
    name = "test.good"
    version = "1"

    def supports(self, context):
        return True

    def extract(self, context):
        return None


class UnorderedIdentityExtractor(GoodExtractor):
    name = "test.unordered_identity"

    @staticmethod
    def runtime_identity():
        return {"features": {"alpha", "beta"}}


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    loaded: object
    load_count: int = 0

    def load(self):
        self.load_count += 1
        return self.loaded


def test_empty_plugin_selection_does_not_discover_host_entry_points(monkeypatch) -> None:
    def forbidden(group: str):
        raise AssertionError(f"entry-point discovery was not expected for {group}")

    monkeypatch.setattr(plugins, "_entries", forbidden)
    assert load_extractors([]) == []


def test_empty_runner_selection_does_not_discover_host_entry_points(monkeypatch) -> None:
    def forbidden(group: str):
        raise AssertionError(f"entry-point discovery was not expected for {group}")

    monkeypatch.setattr(plugins, "_entries", forbidden)
    assert load_runners([]) == []


@pytest.mark.parametrize("names", [[""], [" plugin"], [1]])
def test_malformed_plugin_names_fail_before_discovery(monkeypatch, names) -> None:
    def forbidden(group: str):
        raise AssertionError(f"entry-point discovery was not expected for {group}")

    monkeypatch.setattr(plugins, "_entries", forbidden)
    with pytest.raises(PluginError, match="non-empty trimmed strings"):
        load_extractors(names)


@pytest.mark.parametrize("names", [{"a", "b"}, {"a": "b"}])
def test_unordered_plugin_collections_are_rejected_before_discovery(
    monkeypatch, names,
) -> None:
    def forbidden(group: str):
        raise AssertionError(f"entry-point discovery was not expected for {group}")

    monkeypatch.setattr(plugins, "_entries", forbidden)
    with pytest.raises(PluginError, match="ordered list or tuple"):
        load_extractors(names)


def test_sdk_rejects_unordered_plugin_profile_before_snapshot_creation(tmp_path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest("test.plugin_order", "plugins", "dut", "kernel.cpp"),
    ))
    with pytest.raises(ValueError, match="ordered list or tuple"):
        project.index(degraded=True, options={"extractor_plugins": {"a", "b"}})
    assert project.bundle.store.latest_candidate("test.plugin_order") is None


@pytest.mark.parametrize("value", [
    {"alpha", "beta"}, frozenset({"alpha", "beta"}), b"bytes", float("nan"),
])
def test_sdk_rejects_noncanonical_nested_options_before_snapshot_creation(
    tmp_path, value,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest("test.option_identity", "options", "dut", "kernel.cpp"),
    ))
    with pytest.raises(ValueError, match="canonical JSON|non-finite"):
        project.index(degraded=True, options={"plugin_config": {"value": value}})
    assert project.bundle.store.latest_candidate("test.option_identity") is None


def test_sdk_rejects_noncanonical_plugin_runtime_identity_before_snapshot(
    tmp_path, monkeypatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest(
            "test.plugin_runtime_identity", "plugin runtime", "dut", "kernel.cpp",
        ),
    ))
    monkeypatch.setattr(
        sdk, "load_extractors", lambda names: [UnorderedIdentityExtractor()],
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        project.index(degraded=True, options={"extractor_plugins": ["bad"]})
    assert project.bundle.store.latest_candidate("test.plugin_runtime_identity") is None


def test_unrequested_ambiguity_does_not_block_explicit_unique_plugin(monkeypatch) -> None:
    duplicate_a = FakeEntryPoint("duplicate", "pkg_a:plugin", GoodExtractor)
    duplicate_b = FakeEntryPoint("duplicate", "pkg_b:plugin", GoodExtractor)
    unique = FakeEntryPoint("unique", "pkg_unique:plugin", GoodExtractor)
    monkeypatch.setattr(
        plugins, "_entries", lambda group: [duplicate_a, duplicate_b, unique],
    )

    loaded = load_extractors(["unique"])
    assert len(loaded) == 1
    assert isinstance(loaded[0], GoodExtractor)
    assert unique.load_count == 1
    assert duplicate_a.load_count == duplicate_b.load_count == 0


def test_requested_ambiguous_or_unknown_plugin_fails_closed(monkeypatch) -> None:
    duplicate_a = FakeEntryPoint("duplicate", "pkg_a:plugin", GoodExtractor)
    duplicate_b = FakeEntryPoint("duplicate", "pkg_b:plugin", GoodExtractor)
    monkeypatch.setattr(plugins, "_entries", lambda group: [duplicate_a, duplicate_b])

    with pytest.raises(PluginError, match="ambiguous extractor plugin names: duplicate"):
        load_extractors(["duplicate"])
    with pytest.raises(PluginError, match="unknown extractor plugins: missing"):
        load_extractors(["missing"])
    assert duplicate_a.load_count == duplicate_b.load_count == 0


def test_explicit_plugin_contract_is_validated_after_unique_load(monkeypatch) -> None:
    incomplete = FakeEntryPoint("incomplete", "pkg:plugin", object())
    monkeypatch.setattr(plugins, "_entries", lambda group: [incomplete])

    with pytest.raises(PluginError, match="is missing 'name'"):
        load_extractors(["incomplete"])
    assert incomplete.load_count == 1


def test_runner_v2_plugin_is_explicit_configured_and_protocol_checked(monkeypatch) -> None:
    class ConfiguredRunner(FakeRunner):
        name = "runner.fixture"

        def __init__(self, marker: str):
            super().__init__()
            self.marker = marker

    entry = FakeEntryPoint("fixture", "pkg:runner", ConfiguredRunner)
    monkeypatch.setattr(plugins, "_entries", lambda group: [entry])
    loaded = load_runners(["fixture"], {"fixture": {"marker": "selected"}})
    assert len(loaded) == 1
    assert loaded[0].marker == "selected"
    assert loaded[0].capabilities()["protocol_version"] == PROTOCOL_VERSION
    assert entry.load_count == 1


def test_runner_plugin_rejects_legacy_protocol(monkeypatch) -> None:
    class LegacyRunner(FakeRunner):
        name = "runner.legacy"

        def capabilities(self):
            value = super().capabilities()
            value["protocol_version"] = "hlsgraph.runner.v1"
            return value

    entry = FakeEntryPoint("legacy", "pkg:runner", LegacyRunner)
    monkeypatch.setattr(plugins, "_entries", lambda group: [entry])
    with pytest.raises(PluginError, match="hlsgraph.runner.v2"):
        load_runners(["legacy"])


def test_runner_plugin_resource_guard_authority_is_explicit_boolean(monkeypatch) -> None:
    class GuardCapableRunner(FakeRunner):
        name = "runner.guard_capable"
        can_report_resource_guard = True

    entry = FakeEntryPoint("guard", "pkg:runner", GuardCapableRunner)
    monkeypatch.setattr(plugins, "_entries", lambda group: [entry])
    loaded = load_runners(["guard"])
    assert loaded[0].capabilities()["can_report_resource_guard"] is True

    class AmbiguousRunner(GuardCapableRunner):
        def capabilities(self):
            value = super().capabilities()
            value["can_report_resource_guard"] = "yes"
            return value

    ambiguous = FakeEntryPoint("ambiguous", "pkg:runner", AmbiguousRunner)
    monkeypatch.setattr(plugins, "_entries", lambda group: [ambiguous])
    with pytest.raises(PluginError, match="boolean can_report_resource_guard"):
        load_runners(["ambiguous"])


def test_runner_plugin_runtime_guard_authority_is_explicit_boolean(monkeypatch) -> None:
    class RuntimeCapableRunner(FakeRunner):
        name = "runner.runtime_guard_capable"
        can_report_runtime_resource_guard = True

    entry = FakeEntryPoint("runtime", "pkg:runner", RuntimeCapableRunner)
    monkeypatch.setattr(plugins, "_entries", lambda group: [entry])
    loaded = load_runners(["runtime"])
    assert loaded[0].capabilities()["can_report_runtime_resource_guard"] is True

    class AmbiguousRunner(RuntimeCapableRunner):
        def capabilities(self):
            value = super().capabilities()
            value["can_report_runtime_resource_guard"] = "yes"
            return value

    ambiguous = FakeEntryPoint("ambiguous", "pkg:runner", AmbiguousRunner)
    monkeypatch.setattr(plugins, "_entries", lambda group: [ambiguous])
    with pytest.raises(
            PluginError, match="boolean can_report_runtime_resource_guard"):
        load_runners(["ambiguous"])
