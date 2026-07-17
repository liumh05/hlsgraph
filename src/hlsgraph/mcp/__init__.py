"""Read-only Model Context Protocol surface."""

from .service import DEFAULT_IMPACT_RELATIONS, ReadOnlyMcpService


def create_mcp(*args, **kwargs):
    from .server import create_mcp as implementation
    return implementation(*args, **kwargs)


def run_stdio(*args, **kwargs):
    from .server import run_stdio as implementation
    return implementation(*args, **kwargs)

__all__ = ["DEFAULT_IMPACT_RELATIONS", "ReadOnlyMcpService", "create_mcp", "run_stdio"]
