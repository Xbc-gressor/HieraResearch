from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate  # noqa: E402
from ib_support import hrow, write_checkpoint  # noqa: E402

CONT_HISTORY = [
    hrow(
        {"depth": d, "lr": 0.001 * (i + 1), "dropout": 0.1, "mode": "slow"},
        90.0 + i,
        origin="bout_0",
    )
    for i, d in enumerate([1, 2, 3, 5, 6, 7, 8, 4])
]


def _checkpoint_id(ckpt_dir) -> str:
    return json.loads((ckpt_dir / "checkpoint.json").read_text())["checkpoint_id"]


def _checkpoint_hash(ckpt_dir) -> str:
    return hashlib.sha256((ckpt_dir / "checkpoint.json").read_bytes()).hexdigest()


def write_cell(
    cells_root,
    name,
    ckpt_dir,
    arm,
    seed,
    *,
    auc,
    final,
    beat=True,
    status="ok",
    evaluations=10,
    hash_override=None,
):
    """Fabricate one cell dir (manifest.json + result.json) for ckpt_dir."""
    cell_dir = Path(cells_root) / name
    cell_dir.mkdir(parents=True)
    manifest = {
        "checkpoint_id": _checkpoint_id(ckpt_dir),
        "checkpoint_hash": hash_override or _checkpoint_hash(ckpt_dir),
        "arm": arm,
        "seed": seed,
    }
    result = {
        "status": status,
        "evaluations": evaluations,
        "auc": auc,
        "final_relative_improvement": final,
        "relative_improvement_at": {"2": final, "4": final},
        "beat_initial_incumbent": beat,
        "first_improvement_eval": 2 if beat else None,
        "llm_calls": 10,
        "llm_input_tokens": 100,
        "llm_output_tokens": 50,
        "ranker_fallback_count": 0,
    }
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))
    (cell_dir / "result.json").write_text(json.dumps(result))
    return cell_dir


@pytest.fixture()
def corpus(tmp_path):
    """first stratum: 3 checkpoints (eligible); cont_improved: 1 (ineligible)."""
    ckpts = tmp_path / "ckpts"
    first_a = write_checkpoint(ckpts, name="A", regime="first", stratum="first")
    first_c = write_checkpoint(ckpts, name="C", regime="first", stratum="first")
    first_d = write_checkpoint(ckpts, name="D", regime="first", stratum="first")
    cont_b = write_checkpoint(
        ckpts,
        name="B",
        regime="continuation",
        stratum="cont_improved",
        history=CONT_HISTORY,
    )
    cells = tmp_path / "cells"
    # first / checkpoint A: two seeds per arm
    write_cell(cells, "a-cur-1", first_a, "current", 1, auc=10.0, final=12.0)
    write_cell(cells, "a-cur-2", first_a, "current", 2, auc=14.0, final=16.0)
    write_cell(cells, "a-tpe-1", first_a, "pool_tpe", 1, auc=12.0, final=14.0)
    write_cell(cells, "a-tpe-2", first_a, "pool_tpe", 2, auc=18.0, final=20.0)
    # first / checkpoints C, D: one seed per arm
    write_cell(cells, "c-cur-1", first_c, "current", 1, auc=20.0, final=22.0)
    write_cell(cells, "c-tpe-1", first_c, "pool_tpe", 1, auc=21.0, final=25.0)
    write_cell(cells, "d-cur-1", first_d, "current", 1, auc=30.0, final=31.0)
    write_cell(cells, "d-tpe-1", first_d, "pool_tpe", 1, auc=33.0, final=35.0)
    # cont_improved / checkpoint B: one seed each (ineligible stratum)
    write_cell(cells, "b-cur-1", cont_b, "current", 1, auc=5.0, final=6.0)
    write_cell(cells, "b-tpe-1", cont_b, "pool_tpe", 1, auc=7.0, final=9.0)
    # zero-evaluation unsupported cell: counted, never summarized
    write_cell(
        cells, "a-spsa-1", first_a, "spsa", 1,
        auc=None, final=0.0, beat=False, status="unsupported", evaluations=0,
    )
    # hash-mismatched cell: flagged, parked under "unknown"
    write_cell(
        cells, "a-cur-bad", first_a, "current", 9, auc=99.0, final=99.0,
        hash_override="0" * 64,
    )
    return {"ckpts": ckpts, "cells": cells}


def build(corpus):
    cells = aggregate.load_cells(corpus["cells"])
    aggregate.attach_checkpoint_info(cells, corpus["ckpts"])
    return aggregate.summarize(cells)


def test_eligible_stratum_cross_checkpoint_stats(corpus) -> None:
    report = build(corpus)
    first = report["by_stratum"]["first"]
    assert first["checkpoint_count"] == 3
    assert first["ranking_eligible"] is True
    tpe = first["arms"]["pool_tpe"]
    # per-checkpoint medians over seeds first
    assert tpe["per_checkpoint"]["A"]["auc"] == 15.0
    # across checkpoints: medians A=15, C=21, D=33 -> median 21, mean 23
    assert tpe["across_checkpoints"]["auc"]["median"] == 21.0
    assert tpe["across_checkpoints"]["auc"]["mean"] == pytest.approx(23.0)
    # replicate spread: only A has >1 seed; range 18-12 = 6
    assert tpe["replicate_spread"]["auc"] == 6.0
    # pooled paired deltas: A:+2,+4  C:+1  D:+3
    pooled = tpe["paired_delta_vs_current"]["auc"]
    assert pooled["n"] == 4
    assert pooled["median"] == 2.5
    assert pooled["mean"] == 2.5
    # within-checkpoint paired delta on A
    assert tpe["per_checkpoint"]["A"]["paired_delta_vs_current"]["auc"]["median"] == 3.0
    # the baseline arm itself carries no paired blocks
    assert first["arms"]["current"]["paired_delta_vs_current"] is None
    assert "paired_delta_vs_current" not in first["arms"]["current"]["per_checkpoint"]["A"]
    # markdown renders the ranking table for the eligible stratum
    md = aggregate.render_markdown(report)
    assert "beat rate" in md


def test_ineligible_stratum_reports_per_checkpoint_only(corpus) -> None:
    report = build(corpus)
    cont = report["by_stratum"]["cont_improved"]
    # PLAN §七: one checkpoint < 3 -> no cross-checkpoint stats, no ranking
    assert cont["checkpoint_count"] == 1
    assert cont["ranking_eligible"] is False
    tpe = cont["arms"]["pool_tpe"]
    assert tpe["across_checkpoints"] is None
    assert tpe["replicate_spread"] is None
    assert tpe["paired_delta_vs_current"] is None
    # per-checkpoint results survive, including the within-checkpoint pair
    assert tpe["per_checkpoint"]["B"]["auc"] == 7.0
    assert (
        tpe["per_checkpoint"]["B"]["paired_delta_vs_current"]["auc"]["median"]
        == 2.0
    )
    md = aggregate.render_markdown(report)
    assert "per-checkpoint results only" in md
    assert "beat rate |" not in md.split("cont_improved")[1].split("##")[0]


def test_attrition_and_misconfiguration(corpus) -> None:
    report = build(corpus)
    spsa = report["by_stratum"]["first"]["arms"]["spsa"]
    # unsupported zero-eval cell: visible in statuses, absent from metrics
    assert spsa["statuses"] == {"unsupported": 1}
    assert spsa["per_checkpoint"] == {}
    # hash-mismatched cell: flagged, parked under "unknown" (0 verified
    # checkpoints -> ineligible), never under first
    flagged = [row for row in report["cells"] if not row["checkpoint_ok"]]
    assert len(flagged) == 1
    unknown = report["by_stratum"]["unknown"]
    assert unknown["ranking_eligible"] is False
    assert unknown["arms"]["current"]["cells"] == 1
    # the flagged cell is NOT part of first's checkpoint supply
    assert report["by_stratum"]["first"]["checkpoint_count"] == 3
    md = aggregate.render_markdown(report)
    assert "flagged cells" in md


def test_cli_writes_reports(corpus, tmp_path) -> None:
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    rc = aggregate.main(
        [
            "--cells", str(corpus["cells"]),
            "--checkpoints", str(corpus["ckpts"]),
            "--out", str(out),
            "--md", str(md),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert set(report["by_stratum"]) >= {"first", "cont_improved", "unknown"}
    assert "stratum: first" in md.read_text()
