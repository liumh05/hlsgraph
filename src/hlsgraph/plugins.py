"""Explicit, deterministic plugin discovery for vendor and dialect adapters."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Mapping, Sequence


EXTRACTOR_GROUP = "hlsgraph.extractors.v1"
RUNNER_GROUP = "hlsgraph.runners.v2"
EMBEDDER_GROUP = "hlsgraph.embedders.v1"
EMBEDDER_PROTOCOL_VERSION = "hlsgraph.embedder.v1"
KNOWLEDGE_PARSER_GROUP = "hlsgraph.knowledge_parsers.v1"
KNOWLEDGE_PARSER_PROTOCOL_VERSION = "hlsgraph.knowledge_parser.v1"


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


def load_runners(
    names: Sequence[str], configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    """Load only explicitly selected runner-v2 entry points.

    Runner plugins are executable code.  Empty selection therefore avoids even
    enumerating installed entry points, matching the extractor security model.
    A selected entry point may expose an already-configured instance, a class,
    or a factory accepting keyword configuration.
    """
    from .model import stable_hash
    from .runner import PROTOCOL_VERSION, Runner
    from .runner.core import _grant_trusted_plugin_runner

    if not isinstance(names, (list, tuple)):
        raise PluginError("runner plugin names must be an ordered list or tuple")
    raw_names = list(names)
    if any(not isinstance(item, str) or not item.strip() or item != item.strip()
           for item in raw_names):
        raise PluginError("runner plugin names must be non-empty trimmed strings")
    requested = list(dict.fromkeys(raw_names))
    if not requested:
        return []
    config_values = dict(configs or {})
    if any(name not in requested for name in config_values):
        unknown_configs = sorted(set(config_values) - set(requested))
        raise PluginError(
            "runner configuration supplied for unselected plugins: "
            + ", ".join(unknown_configs)
        )
    for name, value in config_values.items():
        if not isinstance(value, Mapping):
            raise PluginError(f"runner plugin {name!r} configuration must be an object")

    entries = _entries(RUNNER_GROUP)
    requested_set = set(requested)
    duplicates = sorted({item.name for item in entries
                         if item.name in requested_set
                         and sum(other.name == item.name for other in entries) > 1})
    if duplicates:
        raise PluginError(f"ambiguous runner plugin names: {', '.join(duplicates)}")
    available = {item.name: item for item in entries if item.name in requested_set}
    unknown = sorted(requested_set - set(available))
    if unknown:
        raise PluginError(f"unknown runner plugins: {', '.join(unknown)}")

    result: list[Any] = []
    for name in requested:
        loaded = available[name].load()
        config = dict(config_values.get(name, {}))
        if isinstance(loaded, type):
            runner = loaded(**config)
        elif isinstance(loaded, Runner) or (
                hasattr(loaded, "execute") and hasattr(loaded, "fingerprint")):
            if config:
                raise PluginError(
                    f"runner plugin {name!r} exposes an instance and cannot accept configuration"
                )
            runner = loaded
        elif callable(loaded):
            runner = loaded(**config)
        else:
            raise PluginError(f"runner plugin {name!r} is not a runner or factory")
        for attribute in ("name", "fingerprint", "capabilities", "execute"):
            if not hasattr(runner, attribute):
                raise PluginError(f"runner plugin {name!r} is missing {attribute!r}")
        if not isinstance(runner.name, str) or not runner.name:
            raise PluginError(f"runner plugin {name!r} has an invalid runner name")
        if not isinstance(runner.fingerprint, str) or not runner.fingerprint:
            raise PluginError(f"runner plugin {name!r} has an invalid fingerprint")
        try:
            capabilities = runner.capabilities()
        except Exception as exc:
            raise PluginError(f"runner plugin {name!r} capabilities failed: {exc}") from exc
        if (not isinstance(capabilities, Mapping)
                or capabilities.get("protocol_version") != PROTOCOL_VERSION):
            raise PluginError(
                f"runner plugin {name!r} does not implement {PROTOCOL_VERSION}"
            )
        if not isinstance(capabilities.get("can_report_resource_guard"), bool):
            raise PluginError(
                f"runner plugin {name!r} must declare boolean "
                "can_report_resource_guard capability"
            )
        if not isinstance(
                capabilities.get("can_report_runtime_resource_guard"), bool):
            raise PluginError(
                f"runner plugin {name!r} must declare boolean "
                "can_report_runtime_resource_guard capability"
            )
        # Explicitly selected entry points are executable trusted code.  This
        # private registration is what distinguishes them from arbitrary Runner
        # objects supplied through the public Python API.
        _grant_trusted_plugin_runner(runner, stable_hash({
            "group": RUNNER_GROUP,
            "entry_name": name,
            "distribution": getattr(getattr(available[name], "dist", None), "name", None),
            "entry_value": available[name].value,
            "runner_name": runner.name,
            "runner_fingerprint": runner.fingerprint,
            "protocol_version": PROTOCOL_VERSION,
        }))
        result.append(runner)
    return result


def load_embedder(
    name: str | None, config: Mapping[str, Any] | None = None,
) -> Any | None:
    """Load one explicitly selected, local-only embedding provider.

    No entry points are enumerated for the safe default ``None``.  Providers
    must declare that they neither use network access nor remote APIs; HLSGraph
    never downloads a model or forwards private chunks on their behalf.
    """
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise PluginError("embedder plugin name must be a non-empty trimmed string or None")
    if config is not None and not isinstance(config, Mapping):
        raise PluginError("embedder plugin configuration must be an object")
    entries = [item for item in _entries(EMBEDDER_GROUP) if item.name == name]
    if len(entries) > 1:
        raise PluginError(f"ambiguous embedder plugin name: {name}")
    if not entries:
        raise PluginError(f"unknown embedder plugin: {name}")
    loaded = entries[0].load()
    values = dict(config or {})
    if isinstance(loaded, type):
        embedder = loaded(**values)
    elif callable(loaded) and not hasattr(loaded, "embed"):
        embedder = loaded(**values)
    else:
        if values:
            raise PluginError(
                f"embedder plugin {name!r} exposes an instance and cannot accept configuration"
            )
        embedder = loaded
    for attribute in ("name", "version", "fingerprint", "capabilities", "embed"):
        if not hasattr(embedder, attribute):
            raise PluginError(f"embedder plugin {name!r} is missing {attribute!r}")
    capabilities = embedder.capabilities()
    if not isinstance(capabilities, Mapping):
        raise PluginError(f"embedder plugin {name!r} capabilities must be an object")
    if capabilities.get("protocol_version") != EMBEDDER_PROTOCOL_VERSION:
        raise PluginError(
            f"embedder plugin {name!r} does not implement {EMBEDDER_PROTOCOL_VERSION}"
        )
    if capabilities.get("local_only") is not True:
        raise PluginError(f"embedder plugin {name!r} must declare local_only=true")
    if capabilities.get("network_access") is not False:
        raise PluginError(f"embedder plugin {name!r} must declare network_access=false")
    import re
    if (not isinstance(embedder.name, str) or not embedder.name.strip()
            or not isinstance(embedder.version, str) or not embedder.version.strip()
            or not isinstance(embedder.fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", embedder.fingerprint)):
        raise PluginError(f"embedder plugin {name!r} has invalid immutable identity")
    return embedder


def load_knowledge_parser(
    name: str | None, config: Mapping[str, Any] | None = None,
) -> Any | None:
    """Load one explicitly selected, local-only knowledge document parser."""
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise PluginError(
            "knowledge parser plugin name must be a non-empty trimmed string or None"
        )
    if config is not None and not isinstance(config, Mapping):
        raise PluginError("knowledge parser plugin configuration must be an object")
    entries = [item for item in _entries(KNOWLEDGE_PARSER_GROUP) if item.name == name]
    if len(entries) > 1:
        raise PluginError(f"ambiguous knowledge parser plugin name: {name}")
    if not entries:
        raise PluginError(f"unknown knowledge parser plugin: {name}")
    loaded = entries[0].load()
    values = dict(config or {})
    if isinstance(loaded, type):
        parser = loaded(**values)
    elif callable(loaded) and not hasattr(loaded, "parse"):
        parser = loaded(**values)
    else:
        if values:
            raise PluginError(
                f"knowledge parser plugin {name!r} exposes an instance and "
                "cannot accept configuration"
            )
        parser = loaded
    for attribute in ("name", "version", "fingerprint", "capabilities", "parse"):
        if not hasattr(parser, attribute):
            raise PluginError(
                f"knowledge parser plugin {name!r} is missing {attribute!r}"
            )
    capabilities = parser.capabilities()
    if not isinstance(capabilities, Mapping):
        raise PluginError(
            f"knowledge parser plugin {name!r} capabilities must be an object"
        )
    if capabilities.get("protocol_version") != KNOWLEDGE_PARSER_PROTOCOL_VERSION:
        raise PluginError(
            f"knowledge parser plugin {name!r} does not implement "
            f"{KNOWLEDGE_PARSER_PROTOCOL_VERSION}"
        )
    if capabilities.get("local_only") is not True:
        raise PluginError(
            f"knowledge parser plugin {name!r} must declare local_only=true"
        )
    if capabilities.get("network_access") is not False:
        raise PluginError(
            f"knowledge parser plugin {name!r} must declare network_access=false"
        )
    media_types = capabilities.get("media_types")
    if (not isinstance(media_types, (list, tuple)) or not media_types
            or any(not isinstance(item, str) or not item.strip()
                   for item in media_types)):
        raise PluginError(
            f"knowledge parser plugin {name!r} must declare non-empty media_types"
        )
    import re
    if (not isinstance(parser.name, str) or not parser.name.strip()
            or not isinstance(parser.version, str) or not parser.version.strip()
            or not isinstance(parser.fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", parser.fingerprint)):
        raise PluginError(
            f"knowledge parser plugin {name!r} has invalid immutable identity"
        )
    return parser


__all__ = [
    "EMBEDDER_GROUP", "EMBEDDER_PROTOCOL_VERSION", "EXTRACTOR_GROUP",
    "KNOWLEDGE_PARSER_GROUP", "KNOWLEDGE_PARSER_PROTOCOL_VERSION", "RUNNER_GROUP",
    "PluginDescriptor", "PluginError", "descriptors", "load_embedder",
    "load_extractors", "load_knowledge_parser", "load_runners",
]
