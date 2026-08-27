"""`tools/slate.py replay`: a constructed generation replays clean, and
tampering with any link of the chain (pool, context, judge, manifest, ledger
seat) makes replay fail.  The generation fixture runs the real tools pipeline
(lanes -> propose -> construct -> scripted judges -> aggregate -> manifest ->
atomic admission); only the judge rankings are scripted.
"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import got_select  # noqa: E402
import semantic_search  # noqa: E402
import slate  # noqa: E402
from ledger_admission import SlateAdmissionRequest, admit_slate_atomic  # noqa: E402
from ledger_core import records_prefix_digest  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402
from semantic_space import complete_point, space_receipt  # noqa: E402
from tests.fixtures import background_text, fixture_registry, record  # noqa: E402


def _ledger_data(registry: dict) -> dict:
    """Five terminal fresh records; the judged generation starts above them."""
    base = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = []
    scores = [0.40, 0.50, 0.41, 0.52, 0.39]
    for index, score in enumerate(scores):
        point = filtered if index % 2 else base
        entry = record(f"00{index}", "fresh", [], point, score=score, status="keep")
        entry["best_warm_score"] = score + 0.05
        records.append(entry)
    return {
        "task": "hard-interactions",
        "tag": "slate-replay-test",
        "metric": "validation-loss",
        "records": records,
        "items": {},
        "lineage_snapshots": [],
        "dag_revision": 5,
        "search_space": space_receipt(registry),
        "search_space_state": empty_search_space_state(),
        "experience": {
            "schema_version": 3,
            "updated_at_run": "004",
            "generation": 1,
            "summary": "",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
            "dag_revision": 5,
        },
    }


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def build_admitted_generation(run_dir: Path) -> dict:
    """One full judged generation, admitted into the on-disk ledger."""
    registry = fixture_registry()
    (run_dir / "background.md").write_text(background_text(registry))
    data = _ledger_data(registry)
    (run_dir / "ledger.json").write_text(json.dumps(data, indent=2) + "\n")

    gen = run_dir / ".semantic" / "gen-0001"
    lanes_path = gen / "lanes.json"
    _quiet(
        got_select.cmd_decide,
        SimpleNamespace(
            ledger=str(run_dir / "ledger.json"),
            cfg=None,
            mode="lanes",
            output=str(lanes_path),
        ),
    )
    lanes_doc = json.loads(lanes_path.read_text())
    proposals_dir = gen / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    for lane in lanes_doc["lanes"]:
        _quiet(
            semantic_search.cmd_propose,
            SimpleNamespace(
                background=run_dir / "background.md",
                ledger=run_dir / "ledger.json",
                op=lane["op"],
                parents=",".join(lane["parents"]),
                max_points=128,
                baseline_only=False,
                output=proposals_dir / f"{lane['lane_id']}.json",
            ),
        )
    _quiet(
        slate.cmd_construct,
        SimpleNamespace(
            lanes=lanes_path,
            proposals_dir=proposals_dir,
            ledger=run_dir / "ledger.json",
            background=run_dir / "background.md",
            pool_size=6,
            pool_output=gen / "pool.json",
            context_output=gen / "context.json",
        ),
    )

    judgments = gen / "judgments"
    judgments.mkdir(exist_ok=True)
    for stage, top2 in {"regular-0": ["C1", "C3"], "regular-1": ["C3", "C1"]}.items():
        _quiet(
            slate.cmd_prepare_judge,
            SimpleNamespace(
                pool=gen / "pool.json",
                context=gen / "context.json",
                stage=stage,
                labels=None,
                task_brief=None,
                output=judgments / f"{stage}.input.json",
            ),
        )
        input_doc = json.loads((judgments / f"{stage}.input.json").read_text())
        order = input_doc["presented_order"]
        ranking = [label for label in order if label in top2] + [
            label for label in order if label not in top2
        ]
        receipt = judgments / f"{stage}.receipt.json"
        receipt.write_text(json.dumps({"ranking": ranking, "rationale": "t"}))
        code = _quiet(
            slate.cmd_validate_judge,
            SimpleNamespace(
                input=judgments / f"{stage}.input.json",
                receipt=receipt,
                session_id=f"sess-{stage}",
                model="grok/grok-4.6",
                output=judgments / f"{stage}.json",
            ),
        )
        assert code == 0
    _quiet(
        slate.cmd_aggregate,
        SimpleNamespace(
            pool=gen / "pool.json",
            context=gen / "context.json",
            judgments_dir=judgments,
            output=gen / "judge.json",
        ),
    )
    _quiet(
        slate.cmd_build_manifest,
        SimpleNamespace(
            lanes=lanes_path,
            pool=gen / "pool.json",
            context=gen / "context.json",
            judge=gen / "judge.json",
            reserved_run_ids="005,006",
            output=gen / "generation.json",
        ),
    )
    manifest = json.loads((gen / "generation.json").read_text())
    plans = gen / "plans"
    plans.mkdir(exist_ok=True)
    for slot in manifest["slate"]:
        (plans / f"slot-{slot['slot']}.json").write_text(
            json.dumps(
                {
                    "slot": slot["slot"],
                    "idea": f"Replay fixture idea for {slot['run_id']}.",
                    "change": f"Replay fixture change for slot {slot['slot']}.",
                    "candidate_name": f"seat_{slot['run_id']}",
                }
            )
        )
    admitted = admit_slate_atomic(
        data,
        SlateAdmissionRequest(
            background_path=run_dir / "background.md",
            catalog_path=None,
            manifest_path=gen / "generation.json",
            plans_dir=plans,
            run_dir=run_dir,
        ),
    )
    assert [entry["run_id"] for entry in admitted] == ["005", "006"]
    (run_dir / "ledger.json").write_text(json.dumps(data, indent=2) + "\n")
    return manifest


def run_replay(run_dir: Path):
    out = io.StringIO()
    with redirect_stdout(out):
        code = slate.cmd_replay(
            SimpleNamespace(
                ledger=run_dir / "ledger.json",
                semantic_dir=None,
                background=None,
                output=None,
                verbose=True,
            )
        )
    return code, json.loads(out.getvalue())


class ReplayTests(unittest.TestCase):
    def test_constructed_generation_replays_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            build_admitted_generation(run_dir)
            code, report = run_replay(run_dir)
            self.assertEqual(code, 0, json.dumps(report, indent=2))
            self.assertTrue(report["ok"])
            self.assertEqual(report["generations_checked"], 1)
            generation = report["generations"][0]
            self.assertFalse(generation["provisional"])
            self.assertEqual(generation["errors"], [])
            self.assertEqual(
                set(generation["checks"]),
                {
                    "lanes",
                    "pool",
                    "context",
                    "presented_orders",
                    "aggregation",
                    "manifest",
                    "ledger_binding",
                },
            )
            self.assertTrue(all(generation["checks"].values()))

    def test_tampering_any_link_fails_replay(self):
        cases = {
            "pool": lambda run_dir: self._edit(
                run_dir / ".semantic/gen-0001/pool.json",
                lambda doc: doc["pool"][0].update(coverage=0.0123),
            ),
            "context": lambda run_dir: self._edit(
                run_dir / ".semantic/gen-0001/context.json",
                lambda doc: doc.update(rendered_text="forged history"),
            ),
            "judge": lambda run_dir: self._edit(
                run_dir / ".semantic/gen-0001/judge.json",
                lambda doc: doc["aggregation"].update(
                    slate=list(reversed(doc["aggregation"]["slate"]))
                ),
            ),
            "manifest": lambda run_dir: self._edit(
                run_dir / ".semantic/gen-0001/generation.json",
                lambda doc: doc["slate"][0].update(run_id="099"),
            ),
            "ledger_seat": lambda run_dir: self._edit(
                run_dir / "ledger.json",
                lambda doc: doc["records"][5]["policy_receipt"]["judge"].update(
                    slate_index=1
                ),
            ),
        }
        for name, tamper in cases.items():
            with self.subTest(tamper=name), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                build_admitted_generation(run_dir)
                tamper(run_dir)
                code, report = run_replay(run_dir)
                self.assertEqual(code, 1, f"tamper {name} must fail replay")
                self.assertFalse(report["ok"])
                self.assertTrue(
                    report["generations"][0]["errors"],
                    f"tamper {name} produced no errors",
                )

    def test_manifest_bytes_digest_binds_the_seats(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = build_admitted_generation(run_dir)
            ledger = json.loads((run_dir / "ledger.json").read_text())
            manifest_digest = "sha256:" + hashlib.sha256(
                (run_dir / ".semantic/gen-0001/generation.json").read_bytes()
            ).hexdigest()
            for index, slot in enumerate(manifest["slate"]):
                record = ledger["records"][5 + index]
                judge = record["policy_receipt"]["judge"]
                self.assertEqual(judge["manifest_digest"], manifest_digest)
                self.assertEqual(judge["slate_index"], slot["slot"])
            # The A1 prefix digest still anchors the pre-admission ledger.
            context = json.loads(
                (run_dir / ".semantic/gen-0001/context.json").read_text()
            )
            self.assertEqual(
                context["prefix_digest"], records_prefix_digest(ledger["records"][:5])
            )

    @staticmethod
    def _edit(path: Path, mutate) -> None:
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
