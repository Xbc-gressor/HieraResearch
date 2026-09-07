"""Base-task construction for RGPE-style tuning-history transfer
(DESIGN-rgpe-history-transfer-v1 §1.1).

A *base task* is one already-tuned sibling candidate's (x, y) trajectory,
projected onto the recipient's parameter schema so the recipient's surrogate
can consume it as a prior task (Feurer et al., "Practical Transfer Learning
for Bayesian Optimization", RGPE). This module owns the projection; the
weighting lives in ``hebo_mace/suggest_rgpe.py``.

Alignment rule (the arm-2 rule of REPORT-gd-transfer-v2, lifted from single
points to whole dimensions):

1. a recipient dimension aligns to the donor dimension with the same name,
   else to the donor dimension with the same ``canonical`` role annotation
   (unique on the donor side), and only when the two kinds agree;
2. declared coupled groups (``couples_with``) are atomic on BOTH sides — a
   group aligns entirely or not at all;
3. the base model's inputs are the aligned recipient dimensions only
   (``shared_dims``). Recipient-only dimensions never enter the base model;
   donor-only dimensions are dropped, their effect on y absorbed by the base
   GP's noise term.

Per point: categorical values outside the recipient's options drop the point;
numeric values are clipped into the recipient's bounds (int dims rounded) so
the base GP is fitted inside the box the acquisition will query. Each base
task standardizes its own y (paper §4.2). Donors with fewer than
``min_points`` distinct projected points, or zero coverage, are not modelled.

Annotations are the extractor-time ``{key: {"canonical": str,
"couples_with": [str, ...]}}`` mapping per candidate; ``None`` means
name-only alignment with no coupling constraints.

Contracts are ``tools/inner_benchmark/space.CandidateContract`` objects
(``space.read_contract(train_py)``), shared by the offline harness and the
production tuner layer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inner_benchmark"))

DEFAULT_MIN_POINTS = 5


def align_dimensions(recipient, donor, recipient_ann=None, donor_ann=None) -> dict:
    """Recipient dimension name -> donor dimension name for every aligned pair."""
    recipient_ann = recipient_ann or {}
    donor_ann = donor_ann or {}
    donor_dims = {dim.name: dim for dim in donor.dimensions}

    canon_to_donor: dict = {}
    ambiguous: set = set()
    for key, ann in donor_ann.items():
        canonical = ann.get("canonical")
        if canonical is None or key not in donor_dims:
            continue
        if canonical in canon_to_donor:
            ambiguous.add(canonical)
        canon_to_donor[canonical] = key
    canon_to_donor = {c: k for c, k in canon_to_donor.items() if c not in ambiguous}

    align: dict = {}
    for dim in recipient.dimensions:
        donor_key = None
        if dim.name in donor_dims:
            donor_key = dim.name
        else:
            canonical = (recipient_ann.get(dim.name) or {}).get("canonical")
            if canonical is not None:
                donor_key = canon_to_donor.get(canonical)
        if donor_key is None or donor_dims[donor_key].kind != dim.kind:
            continue
        align[dim.name] = donor_key

    # Group atomicity, donor side then recipient side (project_v2 order).
    landed_donor = set(align.values())
    for donor_key, ann in donor_ann.items():
        group = {donor_key, *ann.get("couples_with", [])}
        if len(group) > 1 and not group.issubset(landed_donor):
            for rk in [rk for rk, dk in align.items() if dk in group]:
                del align[rk]
    for rk in list(align):
        group = {rk, *(recipient_ann.get(rk) or {}).get("couples_with", [])}
        if len(group) > 1 and not group.issubset(set(align)):
            del align[rk]
    return align


def project_point(donor_params: dict, align: dict, recipient) -> dict | None:
    """Donor params -> recipient-keyed params over the aligned dims only.

    None when a categorical value is not an option of the recipient dimension
    (the point carries no recipient-expressible information there).
    """
    dims = {dim.name: dim for dim in recipient.dimensions}
    out: dict = {}
    for rk, dk in align.items():
        if dk not in donor_params:
            return None
        value = donor_params[dk]
        dim = dims[rk]
        if dim.kind == "categorical":
            if not any(value is opt or value == opt for opt in dim.options):
                return None
            out[rk] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        number = min(max(number, dim.lo), dim.hi)
        out[rk] = int(round(number)) if dim.kind == "int" else number
    return out


def build_base_task(
    name: str,
    recipient,
    donor,
    rows,
    *,
    recipient_ann=None,
    donor_ann=None,
    min_points: int = DEFAULT_MIN_POINTS,
) -> dict | None:
    """One donor trajectory -> base task payload, or None when not modelled.

    ``rows`` is an iterable of (params, score) pairs; non-finite scores are
    skipped.
    """
    align = align_dimensions(recipient, donor, recipient_ann, donor_ann)
    if not align:
        return None
    shared_dims = [dim.name for dim in recipient.dimensions if dim.name in align]
    points: list = []
    for params, score in rows:
        if not isinstance(params, dict) or score is None:
            continue
        try:
            y = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y):
            continue
        projected = project_point(params, align, recipient)
        if projected is None:
            continue
        points.append((projected, y))
    distinct = {json.dumps(p, sort_keys=True, default=str) for p, _ in points}
    if len(distinct) < min_points:
        return None
    ys = [y for _, y in points]
    mean = sum(ys) / len(ys)
    std = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys))
    if std <= 0:
        return None
    return {
        "name": name,
        "shared_dims": shared_dims,
        "alignment": dict(align),
        "coverage": len(shared_dims) / max(1, len(recipient.dimensions)),
        "points": [
            {"params": params, "y_std": (y - mean) / std} for params, y in points
        ],
    }


def build_base_tasks(recipient, donors, *, recipient_ann=None,
                     min_points: int = DEFAULT_MIN_POINTS) -> list:
    """``donors``: iterable of {"name", "contract", "rows", "annotation"?}."""
    tasks = []
    for donor in donors:
        task = build_base_task(
            donor["name"],
            recipient,
            donor["contract"],
            donor["rows"],
            recipient_ann=recipient_ann,
            donor_ann=donor.get("annotation"),
            min_points=min_points,
        )
        if task is not None:
            tasks.append(task)
    return tasks


def rows_from_tune_report(report_path, donor_contract) -> list:
    """(params, score) rows of one production candidate's tune_report.json:
    phase_a warm rows plus every bout, bout boundaries ignored (§1.1 step 1).
    Reuses ``hebo_search._history_rows`` so the row semantics stay identical
    to the tuner's own history reader.
    """
    import hebo_search  # heavy tuner imports; only the production path pays

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = hebo_search._history_rows(report, "", donor_contract)
    return [(row.params, row.score) for row in rows if row.score is not None]


__all__ = [
    "DEFAULT_MIN_POINTS",
    "align_dimensions",
    "build_base_task",
    "build_base_tasks",
    "project_point",
    "rows_from_tune_report",
]
