"""Optional PyTorch Geometric adapter; the core package never imports Torch."""
from __future__ import annotations

from typing import Any

from ..bundle import GraphBundle
from ..model import DatasetManifest, Stage
from ..version import FEATURE_SCHEMA_VERSION
from .ml import _feature_schema_document, _static_features, _validated_dataset_manifest


def _feature_graph(bundle: GraphBundle, snapshot_id: str,
                   dataset: DatasetManifest | None = None) -> tuple[list[Any], list[Any], DatasetManifest]:
    if dataset is None:
        dataset = DatasetManifest(
            dataset_id=f"dataset.{snapshot_id}",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            snapshot_ids=[snapshot_id],
        )
    else:
        dataset = _validated_dataset_manifest(dataset)
    if dataset.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported PyG feature schema")
    if snapshot_id not in dataset.snapshot_ids:
        raise ValueError("dataset manifest does not include the requested snapshot")
    valid_stages = {item.value for item in Stage}
    if not dataset.feature_stages or not set(dataset.feature_stages).issubset(valid_stages):
        raise ValueError("dataset feature_stages are empty or unsupported")
    if len(set(dataset.feature_attribute_allowlist)) != len(
            dataset.feature_attribute_allowlist):
        raise ValueError("dataset feature_attribute_allowlist must be unique")
    if any(not isinstance(item, str) or not item.strip()
           for item in dataset.feature_attribute_allowlist):
        raise ValueError("feature attribute names must be non-empty strings")
    graph = bundle.store.load_graph(snapshot_id)
    allowed = set(dataset.feature_stages)
    nodes = sorted((item for item in graph.entities.values() if item.stage in allowed),
                   key=lambda item: item.id)
    node_ids = {item.id for item in nodes}
    edges = sorted((item for item in graph.relations.values()
                    if item.stage in allowed
                    and item.src in node_ids and item.dst in node_ids),
                   key=lambda item: item.id)
    return nodes, edges, dataset


def _feature_vocabulary(
    bundle: GraphBundle, dataset: DatasetManifest,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build one deterministic vocabulary across every declared dataset snapshot."""
    allowed = set(dataset.feature_stages)
    kinds = sorted({entity.kind for snapshot_id in sorted(dataset.snapshot_ids)
                    for entity in bundle.store.load_graph(snapshot_id).entities.values()
                    if entity.stage in allowed})
    kind_vocabulary = {name: index for index, name in enumerate(kinds)}
    # Stage codes are global and versioned by FEATURE_SCHEMA_VERSION, not local
    # to whichever stages happen to appear in one graph.
    stage_vocabulary = {item.value: index for index, item in enumerate(Stage)}
    node_ids_by_snapshot = {
        snapshot_id: {
            entity.id for entity in bundle.store.load_graph(snapshot_id).entities.values()
            if entity.stage in allowed
        }
        for snapshot_id in sorted(dataset.snapshot_ids)
    }
    edge_kinds = sorted({relation.kind for snapshot_id in sorted(dataset.snapshot_ids)
                         for relation in bundle.store.load_graph(snapshot_id).relations.values()
                         if relation.stage in allowed
                         and relation.src in node_ids_by_snapshot[snapshot_id]
                         and relation.dst in node_ids_by_snapshot[snapshot_id]})
    edge_kind_vocabulary = {name: index for index, name in enumerate(edge_kinds)}
    return kind_vocabulary, stage_vocabulary, edge_kind_vocabulary


def to_pyg_data(bundle: GraphBundle, snapshot_id: str,
                dataset: DatasetManifest | None = None) -> Any:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - depends on an optional heavy extra
        raise RuntimeError("PyG export requires hlsgraph[pyg]") from exc

    nodes, edges, dataset = _feature_graph(bundle, snapshot_id, dataset)
    node_index = {item.id: index for index, item in enumerate(nodes)}
    kinds, stages, edge_kinds = _feature_vocabulary(bundle, dataset)
    x = (torch.tensor([[kinds[item.kind], stages[item.stage]] for item in nodes],
                      dtype=torch.long)
         if nodes else torch.empty((0, 2), dtype=torch.long))
    edge_index = torch.tensor(
        [[node_index[item.src] for item in edges], [node_index[item.dst] for item in edges]],
        dtype=torch.long,
    ) if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(
        [[edge_kinds[item.kind], stages[item.stage]] for item in edges],
        dtype=torch.long,
    ) if edges else torch.empty((0, 2), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Python metadata is intentional: no QoR observation is smuggled into x.
    data.node_id = [item.id for item in nodes]
    data.node_kind = [item.kind for item in nodes]
    data.node_stage = [item.stage for item in nodes]
    data.node_kind_vocab = dict(kinds)
    data.node_stage_vocab = dict(stages)
    data.edge_id = [item.id for item in edges]
    data.edge_kind = [item.kind for item in edges]
    data.edge_stage = [item.stage for item in edges]
    data.edge_kind_vocab = dict(edge_kinds)
    feature_attributes = set(dataset.feature_attribute_allowlist)
    # Variable-shape static attributes remain auditable Python metadata rather
    # than being silently coerced into x.  They use the exact same sanitizer as
    # JSONL/Parquet; achieved observations, labels and predictions remain out.
    data.node_features = [
        _static_features(
            item.attrs, feature_attributes,
            entity_kind=item.kind, authority=item.authority,
        ) for item in nodes
    ]
    data.edge_features = [
        _static_features(item.attrs, feature_attributes) for item in edges
    ]
    data.snapshot_id = snapshot_id
    data.feature_schema_version = dataset.feature_schema_version
    data.feature_stages = list(dataset.feature_stages)
    data.feature_attribute_allowlist = list(dataset.feature_attribute_allowlist)
    data.static_feature_schema = _feature_schema_document(feature_attributes)
    data.feature_contract = (
        "node kind/stage indices and edge kind/stage indices; "
        "static attribute metadata uses the positive nested feature schema; "
        "DatasetManifest stage and attribute firewalls applied; "
        "observations, labels, and predictions excluded"
    )
    return data
