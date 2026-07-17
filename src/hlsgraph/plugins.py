"""Explicit, deterministic plugin discovery for vendor and dialect adapters."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Sequence


EXTRACTOR_GROUP = "hlsgraph.extractors.v1"
RUNNER_GROUP = "hlsgraph.runners.v1"


class PluginError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    name: str
    group: str
    distribution: str | None
    value: str


def _entries(group: str) -> list[Any]:
    try:
        return list(metadata.entry_points().select(group=group))
    except AttributeError:  # pragma: no cover - Python 3.10 compatibility
        return list(metadata.entry_points().get(group, []))


def descriptors(group: str) -> list[PluginDescriptor]:
    entries = _entries(group)
    result = []
    for entry in entries:
        distribution = getattr(getattr(entry, "dist", None), "name", None)
        result.append(PluginDescriptor(entry.name, group, distribution, entry.value))
    return sorted(result, key=lambda item: (item.name, item.distribution or "", item.value))


def load_extractors(names: Sequence[str]) -> list[Any]:
    """Load only explicitly named extractor entry points.

    Installed plugins are never executed merely by opening a read-only bundle.
    """
    if not isinstance(names, (list, tuple)):
        raise PluginError("extractor plugin names must be an ordered list or tuple")
    raw_names = list(names)
    if any(not isinstance(item, str) or not item.strip() or item != item.strip()
           for item in raw_names):
        raise PluginError("extractor plugin names must be non-empty trimmed strings")
    requested = list(dict.fromkeys(raw_names))
    # Empty selection is the common, security-sensitive path.  Do not even
    # enumerate the host environment when no plugin was explicitly requested:
    # an unrelated broken/ambiguous installation must not perturb canonical
    # indexing or read-only bundle use.
    if not requested:
        return []

    entries = _entries(EXTRACTOR_GROUP)
    requested_set = set(requested)
    duplicates = sorted({item.name for item in entries
                         if item.name in requested_set
                         and sum(other.name == item.name for other in entries) > 1})
    if duplicates:
        raise PluginError(f"ambiguous extractor plugin names: {', '.join(duplicates)}")
    available = {item.name: item for item in entries if item.name in requested_set}
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise PluginError(f"unknown extractor plugins: {', '.join(unknown)}")
    result: list[Any] = []
    for name in requested:
        loaded = available[name].load()
        extractor = loaded() if isinstance(loaded, type) else loaded
        for attribute in ("name", "version", "supports", "extract"):
            if not hasattr(extractor, attribute):
                raise PluginError(f"extractor plugin {name!r} is missing {attribute!r}")
        result.append(extractor)
    return result


__all__ = ["EXTRACTOR_GROUP", "RUNNER_GROUP", "PluginDescriptor", "PluginError",
           "descriptors", "load_extractors"]
