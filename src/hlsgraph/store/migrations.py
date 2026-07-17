"""Explicit ledger migration registry.

HLSGraph never updates a database's schema marker merely because newer code
opened it.  A breaking schema change must add a deterministic, reviewed step to
``MIGRATIONS`` and be invoked explicitly by a user-facing migration command.
"""
from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable

from ..version import SCHEMA_VERSION


MigrationApply = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True, slots=True)
class MigrationStep:
    from_version: str
    to_version: str
    description: str
    apply: MigrationApply


# v0.1 is the first public schema, so no historical transformation exists yet.
# Future releases append reviewed steps; they must never mutate observation
# meaning without preserving the old payload or recording a new observation.
MIGRATIONS: tuple[MigrationStep, ...] = ()


def migration_path(from_version: str, to_version: str = SCHEMA_VERSION) -> list[MigrationStep]:
    if from_version == to_version:
        return []
    by_source = {step.from_version: step for step in MIGRATIONS}
    result: list[MigrationStep] = []
    current = from_version
    visited: set[str] = set()
    while current != to_version:
        if current in visited or current not in by_source:
            raise ValueError(
                f"no explicit HLSGraph ledger migration from {from_version!r} to {to_version!r}"
            )
        visited.add(current)
        step = by_source[current]
        result.append(step)
        current = step.to_version
    return result


def apply_migrations(connection: sqlite3.Connection, from_version: str,
                     to_version: str = SCHEMA_VERSION) -> list[MigrationStep]:
    steps = migration_path(from_version, to_version)
    for step in steps:
        step.apply(connection)
        connection.execute(
            "UPDATE schema_info SET value=? WHERE key='schema_version'", (step.to_version,)
        )
    return steps
