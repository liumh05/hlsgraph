from __future__ import annotations

from dataclasses import dataclass

import pytest

import hlsgraph.plugins as plugins
import hlsgraph.sdk as sdk
from hlsgraph.bundle import GraphBundle
from hlsgraph.manifest import minimal_manifest
from hlsgraph.plugins import PluginError, load_extractors
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
