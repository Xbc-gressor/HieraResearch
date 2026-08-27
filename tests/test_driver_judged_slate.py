"""Driver-level judged-slate tests (implementation design section 10, 6-9).

FakeSessionRunner scripts the role sessions; JudgedCmd dispatches the
judged-slate tools/ commands to their real in-process implementations, so the
driver speaks the genuine CLI contracts end to end (faking slate.py would
re-implement the deterministic kernel inside the test).  The task is named
``hard-interactions`` so the real ``ledger.py admit-slate`` resolves a real
task config from the repository.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import got_select  # noqa: E402
import ledger as ledger_cli  # noqa: E402
import semantic_search  # noqa: E402
import slate  # noqa: E402
from semantic_space import complete_point, digest, space_receipt  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402
from tests.fixtures import (  # noqa: E402
    background_text,
    fixture_registry,
    record as fixture_record,
)
from tests.test_driver_experiment import (  # noqa: E402
    ExperimentCmd,
    writer_effect,
)
from tests.test_slate_donor_binding import _donor_record  # noqa: E402
from tests.test_slate_replay import run_replay  # noqa: E402
from tools.scheduler.donor import build_donor_snapshot  # noqa: E402

from driver.events import EventsLog  # noqa: E402
from driver.loops.experiment import (  # noqa: E402
    _resolve_donor_extra,
    run_experiment,
)
from driver.session import FakeSessionRunner  # noqa: E402


TASK = "hard-interactions"
TAG = "t1"


def write_judged_task(repo: Path) -> None:
    task_dir = repo / "tasks" / TASK
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        """
[env]
project = "tasks/hard-interactions"
[result]
metric = "validation-loss"
[run]
working_dir = "tasks/hard-interactions"
[constraints]
editable_files = ["train.py"]
readonly_files = ["prepare.py"]
allow_dependencies = false
"""
    )
    (task_dir / "prepare.py").write_text("# fixed\n")
    (task_dir / "train.py").write_text("# baseline\n")


def _experience(dag_revision: int) -> dict:
    return {
        "schema_version": 3,
        "updated_at_run": "004",
        "generation": 1,
        "summary": "",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [],
        "dag_revision": dag_revision,
    }


def _ledger_data(registry: dict) -> dict:
    """Five terminal fresh records; the judged generation starts above them."""
    base = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = []
    scores = [0.40, 0.50, 0.41, 0.52, 0.39]
    for index, score in enumerate(scores):
        point = filtered if index % 2 else base
        entry = fixture_record(
            f"00{index}", "fresh", [], point, score=score, status="keep"
        )
        entry["best_warm_score"] = score + 0.05
        records.append(entry)
    return {
        "task": TASK,
        "tag": TAG,
        "metric": "validation-loss",
        "records": records,
        "items": {},
        "lineage_snapshots": [],
        "dag_revision": 5,
        "search_space": space_receipt(registry),
        "search_space_state": empty_search_space_state(),
        "experience": _experience(5),
    }


class JudgedCmd(ExperimentCmd):
    """ExperimentCmd plus the real judged-slate tools, run in-process."""

    def __init__(self, repo: Path):
        super().__init__(repo)
        # When set, lane-00's proposal set is truncated to this many entries
        # and every other lane's to zero — deterministic pool-size control.
        self.propose_keep: int | None = None

    @property
    def run_dir(self) -> Path:
        return self.repo / "runs" / TASK / TAG

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        args = [str(a) for a in args]
        script = args[1] if len(args) > 1 else ""
        handler = None
        if script == "tools/got_select.py":
            handler = lambda: self._decide_lanes(args)
        elif script == "tools/semantic_search.py":
            handler = lambda: self._propose(args)
        elif script == "tools/slate.py":
            handler = lambda: self._slate(args)
        elif script == "tools/ledger.py" and "admit-slate" in args:
            handler = lambda: self._admit_slate(args)
        if handler is None:
            return super().__call__(args, repo_root, check=check, capture=capture, **kw)
        self.calls.append(" ".join(args))
        return self._real(args, check, handler)

    @staticmethod
    def _real(args, check, fn):
        """Translate the tools' in-process contract back into a subprocess one."""
        out = io.StringIO()
        err = ""
        try:
            with contextlib.redirect_stdout(out):
                code = fn() or 0
        except SystemExit as exc:  # ledger.py CLI raises SystemExit on refusal
            code, err = 1, str(exc)
        except Exception as exc:  # ContractError & friends: tools exit 1
            code, err = 1, str(exc)
        if check and code:
            raise subprocess.CalledProcessError(code, args, out.getvalue(), err)
        return subprocess.CompletedProcess(args, code, out.getvalue(), err)

    @staticmethod
    def _opt(args, flag):
        return args[args.index(flag) + 1] if flag in args else None

    def _decide_lanes(self, args):
        return got_select.cmd_decide(
            SimpleNamespace(
                ledger=self._opt(args, "--ledger"),
                cfg=None,
                mode="lanes",
                output=self._opt(args, "--output"),
            )
        )

    def _propose(self, args):
        output = Path(self._opt(args, "--output"))
        code = semantic_search.cmd_propose(
            SimpleNamespace(
                background=Path(self._opt(args, "--background")),
                ledger=Path(self._opt(args, "--ledger")),
                op=self._opt(args, "--op"),
                parents=self._opt(args, "--parents"),
                max_points=int(self._opt(args, "--max-points")),
                baseline_only=False,
                output=output,
            )
        )
        if self.propose_keep is not None:
            if output.stem != "lane-00":
                # A lane without a proposal file is a lane without proposals;
                # an empty proposals list would fail validation instead.
                output.unlink()
                return code
            doc = json.loads(output.read_text())
            doc["proposals"] = doc["proposals"][: self.propose_keep]
            doc["proposal_set_revision"] = digest(
                {k: v for k, v in doc.items() if k != "proposal_set_revision"}
            )
            output.write_text(json.dumps(doc, indent=2) + "\n")
        return code

    def _slate(self, args):
        sub = args[2]
        if sub == "construct":
            return slate.cmd_construct(
                SimpleNamespace(
                    lanes=Path(self._opt(args, "--lanes")),
                    proposals_dir=Path(self._opt(args, "--proposals-dir")),
                    ledger=Path(self._opt(args, "--ledger")),
                    background=Path(self._opt(args, "--background")),
                    pool_size=None,
                    pool_output=Path(self._opt(args, "--pool-output")),
                    context_output=Path(self._opt(args, "--context-output")),
                )
            )
        if sub == "prepare-judge":
            return slate.cmd_prepare_judge(
                SimpleNamespace(
                    pool=Path(self._opt(args, "--pool")),
                    context=Path(self._opt(args, "--context")),
                    stage=self._opt(args, "--stage"),
                    labels=self._opt(args, "--labels"),
                    task_brief=(
                        Path(self._opt(args, "--task-brief"))
                        if "--task-brief" in args
                        else None
                    ),
                    output=Path(self._opt(args, "--output")),
                )
            )
        if sub == "validate-judge":
            return slate.cmd_validate_judge(
                SimpleNamespace(
                    input=Path(self._opt(args, "--input")),
                    receipt=(
                        Path(self._opt(args, "--receipt"))
                        if "--receipt" in args
                        else None
                    ),
                    session_id=self._opt(args, "--session-id"),
                    model=self._opt(args, "--model"),
                    output=Path(self._opt(args, "--output")),
                )
            )
        if sub == "aggregate":
            return slate.cmd_aggregate(
                SimpleNamespace(
                    pool=Path(self._opt(args, "--pool")),
                    context=Path(self._opt(args, "--context")),
                    judgments_dir=Path(self._opt(args, "--judgments-dir")),
                    output=Path(self._opt(args, "--output")),
                )
            )
        if sub == "build-manifest":
            return slate.cmd_build_manifest(
                SimpleNamespace(
                    lanes=Path(self._opt(args, "--lanes")),
                    pool=Path(self._opt(args, "--pool")),
                    context=Path(self._opt(args, "--context")),
                    judge=Path(self._opt(args, "--judge")),
                    reserved_run_ids=self._opt(args, "--reserved-run-ids"),
                    donor_snapshot=(
                        Path(self._opt(args, "--donor-snapshot"))
                        if "--donor-snapshot" in args
                        else None
                    ),
                    no_donor="--no-donor" in args,
                    output=Path(self._opt(args, "--output")),
                )
            )
        raise AssertionError(f"unexpected slate subcommand: {sub}")

    def _admit_slate(self, args):
        return ledger_cli.cmd_admit_slate(
            SimpleNamespace(
                ledger=self._opt(args, "--ledger"),
                task=None,
                background=self._opt(args, "--background"),
                catalog=None,
                manifest=self._opt(args, "--manifest"),
                plans_dir=self._opt(args, "--plans-dir"),
            )
        )


def judge_entry(*, invalid: bool = False):
    """A judge rollout ranking candidates in label order (consensus shape)."""
    entry: dict = {}

    def effect(ctx):
        input_path = (
            Path(ctx.extra["gen_dir"])
            / "judgments"
            / f"{ctx.extra['stage']}.input.json"
        )
        order = json.loads(input_path.read_text())["presented_order"]
        ranking = ["C1", "C1"] if invalid else sorted(order)
        entry["receipt"] = {"ranking": ranking, "rationale": "label order"}

    entry["side_effects"] = effect
    return entry


def plan_entry():
    """A per-seat plan receipt, compliant with the carrier's op."""
    entry: dict = {}

    def effect(ctx):
        manifest = json.loads(
            (Path(ctx.extra["gen_dir"]) / "generation.json").read_text()
        )
        slot = next(
            s for s in manifest["slate"] if s["slot"] == ctx.extra["slot"]
        )
        change = (
            f"from scratch at {slot['point_id']}"
            if slot["carrier"]["op"] == "fresh"
            else f"rework the {slot['carrier']['op']} carrier into the seat point"
        )
        entry["receipt"] = {
            "slot": ctx.extra["slot"],
            "idea": f"Concrete idea for {slot['run_id']}.",
            "change": change,
            "candidate_name": f"seat_{slot['run_id']}",
        }

    entry["side_effects"] = effect
    return entry


def writer_entry():
    return {
        "receipt": {"status": "written", "wrote": True, "candidate_dir": "x"},
        "side_effects": writer_effect,
    }


def extractor_entry(cmd: JudgedCmd):
    entry: dict = {}

    def effect(ctx):
        ledger = cmd._ledger()
        for record in ledger["records"]:
            if record["run_id"] == ctx.run_id:
                record["status"] = "keep"
        cmd._save_ledger(ledger)
        entry["receipt"] = {
            "run_id": ctx.run_id,
            "status": "keep",
            "ledger_updated": True,
        }

    entry["side_effects"] = effect
    return entry


def tuner_entry():
    return {"receipt": {"tuned_run_id": "none", "tuned": False,
                        "ledger_updated": False}}


class JudgedSlateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # ---------- fixtures ----------

    def _seed_run(self, *, max_evaluations: int = 100) -> None:
        """A resumed judged-slate run: cfg + background + 5 terminal records."""
        write_judged_task(self.repo)
        registry = fixture_registry()
        run_dir = self.repo / "runs" / TASK / TAG
        run_dir.mkdir(parents=True)
        (run_dir / "background.md").write_text(background_text(registry))
        (run_dir / "background_retrieval.json").write_text("{}")
        (run_dir / "framework_cfg.json").write_text(
            json.dumps(
                {
                    "max_evaluations": max_evaluations,
                    "dimension_strategy": "catalog_subset",
                    "semantic_search": {"policy": "judged_slate"},
                    "judged_slate": {"pool_size": 6},
                }
            )
        )
        (run_dir / "ledger.json").write_text(
            json.dumps(_ledger_data(registry), indent=2)
        )

    def _run(self, cmd: JudgedCmd, script: list) -> FakeSessionRunner:
        runner = FakeSessionRunner(script)
        run_experiment(TASK, TAG, runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        return runner

    def _gen_dir(self) -> Path:
        return self.repo / "runs" / TASK / TAG / ".semantic" / "gen-0001"

    def _events(self) -> list[str]:
        path = self.repo / "runs" / TASK / TAG / "driver_events.jsonl"
        return [
            json.loads(line)["kind"]
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def _new_records(self, cmd: JudgedCmd) -> list[dict]:
        return cmd._ledger()["records"][5:]

    # ---------- happy path (design 10.6) ----------

    def test_happy_path_judge_plan_admit_screen(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False, False, False, False, True]
        runner = self._run(cmd, [
            judge_entry(),
            judge_entry(),
            plan_entry(),
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        roles = [name for name, _ in runner.calls]
        self.assertEqual(
            roles,
            ["slate-judge", "slate-judge",
             "slate-plan-writer", "slate-plan-writer",
             "candidate-writer", "tunable-contract-extractor",
             "candidate-writer", "tunable-contract-extractor",
             "tuner-orchestrator"],
        )
        # judges start fresh sessions and receive the prepared payload inline
        for _, ctx in runner.calls[:2]:
            self.assertIsNone(ctx.resume_session_id)
            self.assertIn("Measured history", ctx.inline_payload)
            self.assertIn("Candidate C1", ctx.inline_payload)
        for _, ctx in runner.calls[2:4]:
            self.assertIn("Frozen slot assignment", ctx.inline_payload)
        # atomic admission: two schema-8 seats bound to the manifest
        manifest = json.loads((self._gen_dir() / "generation.json").read_text())
        self.assertEqual(
            [slot["run_id"] for slot in manifest["slate"]], ["005", "006"])
        seats = self._new_records(cmd)
        self.assertEqual([r["run_id"] for r in seats], ["005", "006"])
        self.assertTrue(all(r["status"] == "keep" for r in seats))
        for slot, record in zip(manifest["slate"], seats):
            receipt = record["policy_receipt"]
            self.assertEqual(receipt["schema_version"], 8)
            self.assertEqual(receipt["generation_id"],
                             manifest["generation_id"])
            self.assertEqual(receipt["judge"]["candidate_id"],
                             slot["candidate_id"])
        self.assertEqual(
            [k for k in self._events() if k.startswith("slate_")],
            ["slate_pool_built", "slate_judge_completed",
             "slate_manifested", "slate_admitted"],
        )
        self.assertEqual(cmd._ledger().get("phase"), "completed")

    def test_judge_correction_resumes_the_same_session(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False, False, False, False, True]
        runner = self._run(cmd, [
            judge_entry(invalid=True),   # permutation rejected by validate-judge
            judge_entry(),               # the single corrective resume
            judge_entry(),
            plan_entry(),
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles[:3], ["slate-judge"] * 3)
        first, corrective = runner.calls[0][1], runner.calls[1][1]
        self.assertEqual(
            corrective.resume_session_id,
            f"fake-sess-{first.invocation_id:04d}",
        )
        self.assertIn("correction_note", corrective.extra)
        self.assertEqual(len(self._new_records(cmd)), 2)

    # ---------- resume (design 10.7) ----------

    def test_resume_after_manifest_never_rejudges(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False]
        runner1 = self._run(cmd, [
            judge_entry(),
            judge_entry(),
            {"fail": ["killed before any plan"]},
        ])
        self.assertEqual(cmd._ledger().get("phase"), "blocked")
        manifest_bytes = (self._gen_dir() / "generation.json").read_bytes()
        self.assertEqual(len(cmd._ledger()["records"]), 5)

        cmd2 = JudgedCmd(self.repo)
        cmd2.reached = [False, False, False, False, True]
        runner2 = self._run(cmd2, [
            plan_entry(),
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd2),
            writer_entry(),
            extractor_entry(cmd2),
            tuner_entry(),
        ])
        roles = [name for name, _ in runner2.calls]
        self.assertNotIn("slate-judge", roles)
        self.assertEqual(
            roles,
            ["slate-plan-writer", "slate-plan-writer",
             "candidate-writer", "tunable-contract-extractor",
             "candidate-writer", "tunable-contract-extractor",
             "tuner-orchestrator"],
        )
        self.assertEqual(
            (self._gen_dir() / "generation.json").read_bytes(), manifest_bytes)
        self.assertEqual(
            [r["run_id"] for r in self._new_records(cmd2)], ["005", "006"])
        self.assertEqual(cmd2._ledger().get("phase"), "completed")

    def test_resume_fills_only_the_missing_plan(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False]
        self._run(cmd, [
            judge_entry(),
            judge_entry(),
            plan_entry(),                     # slot 0 lands
            {"fail": ["killed mid-plan"]},    # slot 1 does not
        ])
        plan0_bytes = (self._gen_dir() / "plans" / "slot-0.json").read_bytes()

        cmd2 = JudgedCmd(self.repo)
        cmd2.reached = [False, False, False, False, True]
        runner2 = self._run(cmd2, [
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd2),
            writer_entry(),
            extractor_entry(cmd2),
            tuner_entry(),
        ])
        plan_calls = [ctx for name, ctx in runner2.calls
                      if name == "slate-plan-writer"]
        self.assertEqual(len(plan_calls), 1)
        self.assertEqual(plan_calls[0].extra["slot"], 1)
        self.assertEqual(
            (self._gen_dir() / "plans" / "slot-0.json").read_bytes(),
            plan0_bytes,
        )
        self.assertEqual(
            [r["run_id"] for r in self._new_records(cmd2)], ["005", "006"])

    def test_resume_with_both_seats_pending_uses_step0_pipeline(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False, False]
        self._run(cmd, [
            judge_entry(),
            judge_entry(),
            plan_entry(),
            plan_entry(),
            {"fail": ["writer down"]},
            {"fail": ["writer still down"]},
        ])
        self.assertEqual(cmd._ledger().get("phase"), "blocked")
        self.assertEqual(
            [r["status"] for r in self._new_records(cmd)],
            ["pending", "pending"],
        )

        cmd2 = JudgedCmd(self.repo)
        cmd2.reached = [False, False, True]
        runner2 = self._run(cmd2, [
            writer_entry(),
            extractor_entry(cmd2),
            writer_entry(),
            extractor_entry(cmd2),
        ])
        roles = [name for name, _ in runner2.calls]
        # no re-judge, no re-plan, no new generation: pending seats resumed
        self.assertEqual(
            roles,
            ["candidate-writer", "tunable-contract-extractor",
             "candidate-writer", "tunable-contract-extractor"],
        )
        self.assertEqual(
            [r["status"] for r in self._new_records(cmd2)], ["keep", "keep"])
        self.assertEqual(cmd2._ledger().get("phase"), "completed")

    # ---------- degraded cardinality (design 10.8) ----------

    def test_admission_cap_zero_is_a_noop_generation(self) -> None:
        self._seed_run(max_evaluations=1)  # remaining 1 -> cap 0
        cmd = JudgedCmd(self.repo)
        runner = self._run(cmd, [tuner_entry()])
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["tuner-orchestrator"])
        self.assertFalse(
            any("slate.py" in call and "construct" in call
                for call in cmd.calls))
        self.assertFalse((self._gen_dir() / "generation.json").exists())
        self.assertEqual(len(cmd._ledger()["records"]), 5)
        self.assertEqual(cmd._ledger().get("phase"), "completed")

    def test_admission_cap_one_seats_coverage_leader_without_judges(self) -> None:
        self._seed_run(max_evaluations=2)  # terminal 2-row screen -> cap 1
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False, False, False, True]
        runner = self._run(cmd, [
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("slate-judge", roles)
        manifest = json.loads((self._gen_dir() / "generation.json").read_text())
        self.assertEqual(manifest["aggregation"]["path"],
                         "judge_skipped_cardinality")
        self.assertEqual(
            manifest["budget"]["candidate_objective_reservation"], 2
        )
        self.assertEqual(len(manifest["slate"]), 1)
        self.assertEqual(len(self._new_records(cmd)), 1)
        extractor_ctx = next(
            ctx for name, ctx in runner.calls
            if name == "tunable-contract-extractor"
        )
        self.assertEqual(extractor_ctx.extra["screening_k_eval"], 2)
        self.assertEqual(extractor_ctx.extra["screening_target_k_eval"], 3)

    def test_pool_of_one_skips_the_judge(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.propose_keep = 1
        cmd.reached = [False, False, False, True]
        runner = self._run(cmd, [
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("slate-judge", roles)
        manifest = json.loads((self._gen_dir() / "generation.json").read_text())
        self.assertEqual(manifest["cardinality"]["pool_actual"], 1)
        self.assertEqual(len(self._new_records(cmd)), 1)

    def test_pool_of_two_admits_both_without_judges(self) -> None:
        self._seed_run()
        cmd = JudgedCmd(self.repo)
        cmd.propose_keep = 2
        cmd.reached = [False, False, False, False, True]
        runner = self._run(cmd, [
            plan_entry(),
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("slate-judge", roles)
        manifest = json.loads((self._gen_dir() / "generation.json").read_text())
        self.assertEqual(manifest["aggregation"]["path"],
                         "judge_skipped_pool_le_B")
        self.assertEqual(
            [r["run_id"] for r in self._new_records(cmd)], ["005", "006"])


class JudgedSlateDonorBindingTests(unittest.TestCase):
    """Design §9.7: one generation binds one donor snapshot for all seats.

    The run freezes the transfer policy pair, so the driver builds the donor
    snapshot once at manifest commit and hands the manifest binding to every
    seat — even when the donor frontier moves between seats.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_transfer_run(self, *, with_donor: bool) -> None:
        """A resumed judged-slate run under the transfer policy pair."""
        write_judged_task(self.repo)
        registry = fixture_registry()
        run_dir = self.repo / "runs" / TASK / TAG
        run_dir.mkdir(parents=True)
        (run_dir / "background.md").write_text(background_text(registry))
        (run_dir / "background_retrieval.json").write_text("{}")
        (run_dir / "framework_cfg.json").write_text(
            json.dumps(
                {
                    "max_evaluations": 100,
                    "dimension_strategy": "catalog_subset",
                    "semantic_search": {"policy": "judged_slate"},
                    "judged_slate": {"pool_size": 6},
                    "tuner": {
                        "scheduler_policy": "anchor_transfer_challenger_v1",
                        "inner_policy": "hebo24-transfer10-hebo10",
                    },
                }
            )
        )
        data = _ledger_data(registry)
        if with_donor:
            data["records"].append(_donor_record(run_dir, "005", score=0.3))
        (run_dir / "ledger.json").write_text(json.dumps(data, indent=2))

    def _run_dir(self) -> Path:
        return self.repo / "runs" / TASK / TAG

    def _gen_dir(self) -> Path:
        return self._run_dir() / ".semantic" / "gen-0001"

    def _run(self, cmd: JudgedCmd, script: list) -> FakeSessionRunner:
        runner = FakeSessionRunner(script)
        run_experiment(TASK, TAG, runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        return runner

    def _extractor_contexts(self, runner: FakeSessionRunner) -> list:
        return [
            ctx
            for name, ctx in runner.calls
            if name == "tunable-contract-extractor"
        ]

    def test_generation_binds_one_snapshot_across_seats(self) -> None:
        self._seed_transfer_run(with_donor=True)
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False, False, False, False, True]
        run_dir = self._run_dir()

        def seat1_entry():
            """Seat 1 screens, then a strictly better donor finalizes."""
            entry: dict = {}

            def effect(ctx):
                ledger = cmd._ledger()
                for record in ledger["records"]:
                    if record["run_id"] == ctx.run_id:
                        record["status"] = "keep"
                better = _donor_record(run_dir, "009", score=0.1)
                ledger["records"].append(better)
                cmd._save_ledger(ledger)
                entry["receipt"] = {
                    "run_id": ctx.run_id,
                    "status": "keep",
                    "ledger_updated": True,
                }

            entry["side_effects"] = effect
            return entry

        runner = self._run(cmd, [
            judge_entry(),
            judge_entry(),
            plan_entry(),
            plan_entry(),
            writer_entry(),
            seat1_entry(),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        self.assertEqual(cmd._ledger().get("phase"), "completed")

        manifest = json.loads((self._gen_dir() / "generation.json").read_text())
        binding = manifest["donor_snapshot"]
        self.assertEqual(binding["status"], "bound")
        snapshot_path = run_dir / binding["path"]
        self.assertTrue(snapshot_path.is_file())

        # The donor frontier really moved between the two seats: rebuilding
        # now selects 009, not the snapshot the manifest bound.
        fresh = build_donor_snapshot(run_dir)
        self.assertEqual(fresh["status"], "ok")
        self.assertEqual(fresh["snapshot"]["selected"]["run_id"], "009")
        self.assertNotEqual(fresh["snapshot_id"], binding["snapshot_id"])

        # Both seats were handed the manifest binding, not the moved frontier.
        extractors = self._extractor_contexts(runner)
        self.assertEqual(len(extractors), 2)
        for ctx in extractors:
            self.assertEqual(ctx.extra["donor_binding"], "bound")
            self.assertEqual(ctx.extra["donor_snapshot"], str(snapshot_path))

        # Seat receipts citing the manifest-bound snapshot replay clean.
        for slot in manifest["slate"]:
            candidate = run_dir / "candidates" / slot["run_id"]
            candidate.mkdir(parents=True, exist_ok=True)
            (candidate / "_global_donor_transfer.json").write_text(
                json.dumps({"donor": {"snapshot_id": binding["snapshot_id"]}})
            )
        code, report = run_replay(run_dir)
        self.assertEqual(code, 0, json.dumps(report, indent=2))
        self.assertTrue(
            report["generations"][0]["checks"].get("donor_binding")
        )

    def test_pre_anchor_generation_binds_no_donor(self) -> None:
        self._seed_transfer_run(with_donor=False)
        cmd = JudgedCmd(self.repo)
        cmd.reached = [False, False, False, False, True]
        runner = self._run(cmd, [
            judge_entry(),
            judge_entry(),
            plan_entry(),
            plan_entry(),
            writer_entry(),
            extractor_entry(cmd),
            writer_entry(),
            extractor_entry(cmd),
            tuner_entry(),
        ])
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        manifest = json.loads((self._gen_dir() / "generation.json").read_text())
        self.assertEqual(
            manifest["donor_snapshot"],
            {
                "status": "no_donor",
                "snapshot_id": None,
                "path": None,
                "digest": None,
            },
        )
        for ctx in self._extractor_contexts(runner):
            self.assertEqual(ctx.extra["donor_binding"], "no_donor")
            self.assertNotIn("donor_snapshot", ctx.extra)


class CoverageArmDonorBindingTests(unittest.TestCase):
    """The coverage_attempt middle arm: no manifest, per-candidate binding."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_judged_task(self.repo)
        self.run_dir = self.repo / "runs" / TASK / "cov"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "framework_cfg.json").write_text(
            json.dumps(
                {
                    "max_evaluations": 100,
                    "semantic_search": {"policy": "coverage_attempt"},
                    "tuner": {
                        "scheduler_policy": "anchor_transfer_challenger_v1",
                        "inner_policy": "hebo24-transfer10-hebo10",
                    },
                }
            )
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _resolve(self, run_id: str) -> dict:
        return _resolve_donor_extra(
            self.run_dir, run_id, self.repo, None, EventsLog(self.run_dir)
        )

    def _ledger_with(self, records: list) -> None:
        (self.run_dir / "ledger.json").write_text(
            json.dumps({"records": records})
        )

    def test_binds_the_current_frontier_per_candidate(self) -> None:
        self._ledger_with([_donor_record(self.run_dir, "000", score=0.3)])
        extra = self._resolve("001")
        self.assertEqual(extra["donor_binding"], "bound")
        bound_path = Path(extra["donor_snapshot"])
        self.assertTrue(bound_path.is_file())
        self.assertEqual(
            json.loads(bound_path.read_text())["selected"]["run_id"], "000"
        )

    def test_existing_candidate_receipt_rebinds_its_frozen_snapshot(self) -> None:
        self._ledger_with([_donor_record(self.run_dir, "000", score=0.3)])
        first = self._resolve("001")
        bound_path = Path(first["donor_snapshot"])
        bound_id = json.loads(bound_path.read_text())["snapshot_id"]

        # The frontier moves; the candidate's existing receipt still wins (§8).
        candidate = self.run_dir / "candidates" / "001"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "_global_donor_transfer.json").write_text(
            json.dumps({"donor": {"snapshot_id": bound_id}})
        )
        records = json.loads((self.run_dir / "ledger.json").read_text())["records"]
        records.append(_donor_record(self.run_dir, "009", score=0.1))
        self._ledger_with(records)

        extra = self._resolve("001")
        self.assertEqual(extra["donor_binding"], "bound")
        self.assertEqual(extra["donor_snapshot"], str(bound_path))

    def test_no_eligible_donor_binds_no_donor(self) -> None:
        self._ledger_with([])
        self.assertEqual(self._resolve("001"), {"donor_binding": "no_donor"})


class CoverageArmRegressionTests(unittest.TestCase):
    """Design 10.9: a persisted coverage arm still drives plain _ideate()."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_coverage_attempt_arm_ignores_the_judged_path(self) -> None:
        from tests.test_driver_experiment import write_task

        write_task(self.repo)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps({
            "max_evaluations": 3,
            "dimension_strategy": "catalog_subset",
            "semantic_search": {"policy": "coverage_attempt"},
        }))
        (run_dir / "background.md").write_text("# bg\n")
        (run_dir / "background_retrieval.json").write_text("{}")
        (run_dir / "ledger.json").write_text(
            json.dumps({"records": [{"run_id": "000", "status": "keep"}]}))
        cmd = ExperimentCmd(self.repo)
        runner = FakeSessionRunner([
            {"receipt": {"actions": []}},
            tuner_entry(),
            {"receipt": {"actions": []}},
            tuner_entry(),
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        self.assertEqual(
            roles, ["idea-generator", "tuner-orchestrator"] * 2)
        self.assertFalse((run_dir / ".semantic").exists())
        self.assertNotIn("slate-judge", roles)
        self.assertEqual(cmd._ledger().get("phase"), "completed")


if __name__ == "__main__":
    unittest.main()
