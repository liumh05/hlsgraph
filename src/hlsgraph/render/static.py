"""Deterministic Mermaid, DOT, and lightweight SVG presentation exports."""
from __future__ import annotations

import html
import math
import re
from collections import defaultdict, deque
from typing import Any


def _sid(value: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", value)


def to_mermaid(data: dict[str, Any]) -> str:
    lines = ["flowchart LR"]
    if data.get("notice"):
        lines.append(f'  notice["{data["notice"]}"]')
    parent_ids = {node["parent"] for node in data["nodes"] if node.get("parent")}
    parents = {node["id"]: node for node in data["nodes"] if node["id"] in parent_ids}
    node_lines: dict[str, str] = {}
    for node in data["nodes"]:
        if node["id"] in parents:
            continue
        label = node["name"]
        ii = node.get("metrics", {}).get("achieved_II")
        if ii is not None:
            label += f"\\nII {ii}"
        node_lines[node["id"]] = f'  {_sid(node["id"])}["{label.replace(chr(34), chr(39))}"]'
        if not node.get("parent"):
            lines.append(node_lines[node["id"]])
    for parent in parents.values():
        label = str(parent["name"]).replace('"', "'")
        lines.append(f'  subgraph {_sid(parent["id"])}["{label}"]')
        lines.extend(node_lines[node["id"]] for node in data["nodes"] if node.get("parent") == parent["id"])
        lines.append("  end")
    for index, edge in enumerate(data["edges"]):
        label = f'|FIFO {edge["fifo_depth"]}|' if edge.get("fifo_depth") is not None else ""
        lines.append(f'  {_sid(edge["source"])} -->{label} {_sid(edge["target"])}')
        dash = {"CONTAINS": "6 4", "BINDS": "2 4"}.get(edge["type"])
        width = max(1.0, min(6.0, 1.0 + math.log2((edge.get("fifo_depth") or 1) + 1)))
        if dash or edge["type"] == "CHANNEL":
            lines.append(f'  linkStyle {index} stroke-width:{width:.2f}px' + (f',stroke-dasharray:{dash}' if dash else ""))
    lines.extend(["  classDef bottleneck fill:#d23b35,color:#fff,stroke:#7d1713,stroke-width:3px",
                  "  classDef compute fill:#3979b9,color:#fff", "  classDef mem fill:#7b8794,color:#fff"])
    for node in data["nodes"]:
        cls = "bottleneck" if node.get("is_bottleneck") else node.get("category", "compute")
        lines.append(f'  class {_sid(node["id"])} {cls}')
    return "\n".join(lines) + "\n"


def to_dot(data: dict[str, Any]) -> str:
    lines = ["digraph hlsgraph {", "  rankdir=LR;", '  graph [bgcolor="transparent"];',
             '  node [shape=box,style="rounded,filled",fontcolor="white",width=1.5,height=0.55];']
    if data.get("notice"):
        lines.append(f'  label="{data["notice"]}"; labelloc=t;')
    parent_ids = {node["parent"] for node in data["nodes"] if node.get("parent")}
    for parent in data["nodes"]:
        children = [node for node in data["nodes"] if node.get("parent") == parent["id"]]
        if children:
            label = str(parent["name"]).replace('"', "'")
            lines.append(f'  subgraph cluster_{_sid(parent["id"])} {{ label="{label}";')
            lines.extend(f'    "{node["id"]}";' for node in children)
            lines.append('  }')
    for node in data["nodes"]:
        if node["id"] in parent_ids:
            continue
        color = "#d23b35" if node.get("is_bottleneck") else ("#3979b9" if node.get("category") == "compute" else "#7b8794")
        label = str(node["name"]).replace('"', "'")
        lines.append(f'  "{node["id"]}" [label="{label}",fillcolor="{color}"];')
    for edge in data["edges"]:
        depth = edge.get("fifo_depth")
        width = max(1.0, min(6.0, 1.0 + math.log2((depth or 1) + 1)))
        style = {"CONTAINS": "dashed", "BINDS": "dotted"}.get(edge["type"], "solid")
        lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [penwidth={width:.2f},style={style}];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _positions(data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    nodes = [item["id"] for item in data["nodes"]]
    incoming = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in data["edges"]:
        if edge["source"] in incoming and edge["target"] in incoming:
            incoming[edge["target"]] += 1
            outgoing[edge["source"]].append(edge["target"])
    queue = deque(sorted(node for node, degree in incoming.items() if degree == 0))
    rank = {node: 0 for node in queue}
    while queue:
        node = queue.popleft()
        for target in sorted(outgoing[node]):
            rank[target] = max(rank.get(target, 0), rank[node] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    for node in nodes:
        rank.setdefault(node, max(rank.values(), default=0) + 1)
    rows: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        rows[rank[node]].append(node)
    positions = {}
    parent_by_id = {item["id"]: item.get("parent", "") for item in data["nodes"]}
    groups = sorted(set(parent_by_id.values()))
    group_stride = len(nodes) * 100
    origin_x, origin_y = (100, 95) if data.get("notice") else (70, 65)
    for column in sorted(rows):
        for row, node in enumerate(sorted(rows[column])):
            group_offset = groups.index(parent_by_id[node]) * group_stride if len(groups) > 1 else 0
            positions[node] = (origin_x + column * 220, origin_y + row * 100 + group_offset)
    return positions


def to_svg(data: dict[str, Any]) -> str:
    positions = _positions(data)
    width = max((x for x, _ in positions.values()), default=100) + 180
    if data.get("notice"):
        width = max(width, 480)
    height = max((y for _, y in positions.values()), default=100) + 100
    node_by_id = {item["id"]: item for item in data["nodes"]}
    parent_ids = {node["parent"] for node in data["nodes"] if node.get("parent")}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#7e8da1"/></marker></defs>',
             '<rect width="100%" height="100%" fill="#f7f8fa"/>']
    if data.get("notice"):
        lines.append(f'<text x="16" y="24" font-family="system-ui,sans-serif" font-size="14">{html.escape(data["notice"])}</text>')
    for parent in data["nodes"]:
        children = [positions[node["id"]] for node in data["nodes"] if node.get("parent") == parent["id"]]
        if children:
            left, top = min(x for x, y in children) - 90, min(y for x, y in children) - 55
            right, bottom = max(x for x, y in children) + 90, max(y for x, y in children) + 45
            lines.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" rx="12" fill="#e8edf3" stroke="#7e8da1"/>')
            lines.append(f'<text x="{left+8}" y="{top+18}" font-size="12">{html.escape(parent["name"])}</text>')
    for edge in data["edges"]:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]; x2, y2 = positions[edge["target"]]
        depth = edge.get("fifo_depth")
        stroke = max(1.5, min(7.0, 1.2 + 1.4 * math.log2((depth or 1) + 1)))
        dash = {"CONTAINS": "6 4", "BINDS": "2 4"}.get(edge["type"], "none")
        lines.append(f'<path d="M{x1+70},{y1} C{x1+130},{y1} {x2-130},{y2} {x2-70},{y2}" fill="none" stroke="#7e8da1" stroke-width="{stroke:.2f}" stroke-dasharray="{dash}" marker-end="url(#arrow)"/>')
    for node_id, (x, y) in sorted(positions.items()):
        node = node_by_id[node_id]
        if node_id in parent_ids:
            continue
        color = "#d23b35" if node.get("is_bottleneck") else ("#3979b9" if node.get("category") == "compute" else "#7b8794")
        lines.append(f'<rect x="{x-70}" y="{y-28}" width="140" height="56" rx="9" fill="{color}" stroke="#344054" stroke-width="2"/>')
        label = html.escape(str(node["name"]))
        lines.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="white">{label}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"

