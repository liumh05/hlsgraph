"""Small real-parser fixtures shared by public evidence-boundary tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from hlsgraph.extract.base import ExtractionContext
from hlsgraph.extract.vitis import VitisReportExtractor
from hlsgraph.extract.vivado import VivadoReportExtractor


def write_csynth_xml(path: Path, *, latency: int = 42) -> None:
    path.write_text(
        "<profile>\n"
        "  <UserAssignments><TopModelName>dut</TopModelName>"
        "<TargetClockPeriod>5</TargetClockPeriod></UserAssignments>\n"
        "  <SummaryOfTimingAnalysis><EstimatedClockPeriod>4.5"
        "</EstimatedClockPeriod></SummaryOfTimingAnalysis>\n"
        "  <SummaryOfOverallLatency>"
        f"<Best-caseLatency>{latency}</Best-caseLatency>"
        f"<Worst-caseLatency>{latency}</Worst-caseLatency>"
        "<Interval-min>1</Interval-min><Interval-max>1</Interval-max>"
        "</SummaryOfOverallLatency>\n"
        "</profile>\n",
        encoding="utf-8",
    )


def write_csim_json(path: Path, *, passed: bool = True) -> None:
    status = "pass" if passed else "fail"
    value = 0 if passed else 1
    path.write_text(
        '{"schema_version":"hlsgraph.vitis.csim.v1",'
        f'"status":"{status}","exit_code":{value},'
        f'"mismatches":{value},"assertions_failed":{value}}}\n',
        encoding="utf-8",
    )


def write_cosim_report(path: Path, *, passed: bool = True) -> None:
    status = "Pass" if passed else "Fail"
    path.write_text(
        f"| Verilog | {status} | 1 | 1 | 1 | 1 | 1 | 1 |\n",
        encoding="utf-8",
    )


def write_vivado_utilization(path: Path, *, lut: int = 10, dsp: int = 1) -> None:
    path.write_text(f"LUT={lut}\nDSP={dsp}\n", encoding="utf-8")


def write_vivado_timing(path: Path, *, wns: float = 0.1) -> None:
    path.write_text(f"WNS: {wns}\nTNS: 0\n", encoding="utf-8")


def parsed_report_observation(
    bundle: Any,
    artifact: Any,
    *,
    predicate: str,
    run_id: str,
    value: Any | None = None,
    subject_id: str | None = None,
    snapshot_id: str | None = None,
) -> Any:
    """Return one exact built-in parser output rebound to its producing run."""

    parser = (VitisReportExtractor()
              if artifact.kind.startswith("amd.vitis.")
              else VivadoReportExtractor())
    # Most fixtures have one stable snapshot; multi-snapshot red-line tests
    # pass the historical identity explicitly.
    snapshot = (bundle.store.snapshot(snapshot_id)
                if snapshot_id is not None else bundle.snapshot())
    graph = bundle.store.load_graph(snapshot.id)
    result = parser.extract(ExtractionContext(
        project_root=bundle.project_root,
        manifest=bundle.store.snapshot_manifest(snapshot.id),
        snapshot=snapshot,
        artifacts={artifact.id: artifact},
        options={"existing_graph": graph},
    ))
    matches = [
        item for item in result.observations
        if (item.predicate == predicate
            and (value is None or item.value == value)
            and (subject_id is None or item.subject_id == subject_id))
    ]
    assert len(matches) == 1, [
        (item.subject_id, item.predicate, item.value, item.stage)
        for item in result.observations
    ]
    return replace(matches[0], id="", run_id=run_id)
