"""Tests for tools/inner_benchmark/llm.py — the LLM bout-session layer.

No network. BoutSession tests use the production fakes: FakeSessionRunner
(driver/session.py) and, for real-runner usage capture, SDKSessionRunner
with a scripted client_factory (the tests/test_driver_session.py pattern).
Formatter and registry tests are SDK-free; the SDKSessionRunner test is
skipped when claude-agent-sdk is not installed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import checkpoint as checkpoint_mod  # noqa: E402
import llm  # noqa: E402
import space as space_mod  # noqa: E402
import state as state_mod  # noqa: E402
from driver.receipts import _TYPE_CHECKS, validate_receipt  # noqa: E402
from driver.roles import PROMPT_DIR, ROLES  # noqa: E402

try:  # driver.session needs anyio (stdlib-free envs may lack it)
    from driver.session import FakeSessionRunner, InvocationFailed

    HAS_DRIVER_SESSION = True
except ImportError:  # pragma: no cover
    FakeSessionRunner = None
    InvocationFailed = None
    HAS_DRIVER_SESSION = False

needs_driver_session = pytest.mark.skipif(
    not HAS_DRIVER_SESSION, reason="driver.session unavailable (no anyio)"
)


# --- fixtures ------------------------------------------------------------------


def make_contract() -> space_mod.CandidateContract:
    return space_mod.CandidateContract(
        path=Path("toy/train.py"),
        param_schema={
            "depth": "int",
            "lr": ("float", "log"),
            "mode": ("categorical", ["fast", "slow"]),
            "seed": "int",
        },
        search_space={
            "depth": ("int", 1, 8),
            "lr": ("float", 0.0001, 0.1, "log"),
            "mode": ("categorical", ["fast", "slow"]),
            "seed": ("int", 7, 7),  # degenerate -> FIXED
        },
        base_params={"depth": 4, "lr": 0.001, "mode": "fast", "seed": 7},
        dimensions=(
            space_mod.Dimension(name="depth", kind="int", lo=1, hi=8),
            space_mod.Dimension(name="lr", kind="float", log=True,
                                lo=0.0001, hi=0.1),
            space_mod.Dimension(name="mode", kind="categorical",
                                options=("fast", "slow")),
            space_mod.Dimension(name="seed", kind="int", lo=7, hi=7),
        ),
    )


def make_checkpoint(tmp_path: Path, source=None) -> checkpoint_mod.Checkpoint:
    return checkpoint_mod.Checkpoint(
        checkpoint_id="toy-c3-b1",
        regime="continuation",
        stratum="cont_improved",
        source={"candidate_id": "c3", "kind": "improve"}
        if source is None else source,
        checkpoint_dir=tmp_path,
        candidate_relpath="candidate",
        candidate_path=tmp_path / "candidate" / "train.py",
        task=checkpoint_mod.TaskSpec(score_fn="evaluate_config",
                                     preflight_fn="preflight_config",
                                     per_runtime_limit=None),
        incumbent=checkpoint_mod.Incumbent(
            params={"depth": 4, "lr": 0.001, "mode": "fast", "seed": 7},
            score=3.0,
        ),
        incumbent_is_inherited_control=False,
        history=(
            checkpoint_mod.HistoryRow(
                params={"depth": 2, "lr": 0.01, "mode": "fast", "seed": 7},
                score=5.0, status="ok", origin="phase_a"),
            checkpoint_mod.HistoryRow(
                params={"depth": 6, "lr": 0.001, "mode": "slow", "seed": 7},
                score=None, status="crash", origin="bout_0"),
        ),
    )


def make_bout(runner, run_dir: Path, role_name="bench-rewarm-proposer",
              first_extras=None) -> llm.BoutSession:
    return llm.BoutSession(
        role=llm.BENCH_ROLES[role_name],
        runner=runner,
        run_dir=run_dir,
        task="toytask",
        tag="toytag",
        first_extras=first_extras,
    )


# --- LLMConfig ----------------------------------------------------------------


def test_llm_config_manifest_round_trip() -> None:
    config = llm.LLMConfig(model="test-model", extra={"note": "x"})
    manifest = config.to_manifest()
    json.dumps(manifest)  # manifest must stay JSON-serializable
    assert manifest["model"] == "test-model"
    assert manifest["decoding"] is None  # production has no decoding knobs
    assert manifest["extra"] == {"note": "x"}
    assert isinstance(manifest["sdk_version"], str) and manifest["sdk_version"]
    assert llm.LLMConfig.from_manifest(manifest) == config


def test_llm_config_restores_recorded_sdk_version_verbatim() -> None:
    config = llm.LLMConfig.from_manifest({"model": "m", "sdk_version": "9.9.9-test"})
    assert config.sdk_version == "9.9.9-test"  # never re-detected
    assert config.extra == {}


def test_llm_config_model_is_caller_supplied() -> None:
    with pytest.raises(TypeError):  # no default model id anywhere
        llm.LLMConfig()


# --- role registry --------------------------------------------------------------

EXPECTED_SCHEMAS = {
    "bench-rewarm-proposer": {"configs": "list", "rationale": "?str"},
    "bench-active-set": {"parameter": "str", "step": "float", "rationale": "?str"},
    "bench-pool-proposer": {"configs": "list", "order": "list", "rationale": "?str"},
    "bench-pool-pairwise-judge": {
        "winner": ("enum", "A", "B"),
        "reasoning": "?str",
    },
    "bench-hillclimb-editor": {"edited": "bool", "summary": "str"},
}

_DUMMY_VALUES = {"str": "x", "int": 1, "float": 0.5, "bool": True,
                 "list": [], "dict": {}}


def test_role_registry_matches_brief_contract() -> None:
    assert set(llm.BENCH_ROLES) == set(EXPECTED_SCHEMAS)
    for name, role in llm.BENCH_ROLES.items():
        assert role.receipt_schema == EXPECTED_SCHEMAS[name], name
        assert (PROMPT_DIR / role.prompt_file).exists(), role.prompt_file
        assert name not in ROLES  # benchmark roles stay out of production
        assert {"Agent", "Task", "Skill"} <= set(role.disallowed), name
        assert role.corrective_attempts == 3, name
        assert role.postconditions == (), name  # documented choice: none
        for spec in role.receipt_schema.values():  # receipts-DSL sanity
            if isinstance(spec, tuple):
                assert spec[0] == "enum", name
            else:
                assert (spec[1:] if spec.startswith("?") else spec) in _TYPE_CHECKS, name
        dummy = {
            key: (
                spec[1]
                if isinstance(spec, tuple)
                else _DUMMY_VALUES[spec[1:] if spec.startswith("?") else spec]
            )
            for key, spec in role.receipt_schema.items()
        }
        assert validate_receipt(role.receipt_schema, dummy) == [], name


def test_pure_proposal_roles_carry_no_tools() -> None:
    for name in (
        "bench-rewarm-proposer",
        "bench-active-set",
        "bench-pool-proposer",
        "bench-pool-pairwise-judge",
    ):
        assert llm.BENCH_ROLES[name].tools == (), name
    assert llm.BENCH_ROLES["bench-hillclimb-editor"].tools == ("Read", "Edit")


# --- BoutSession ----------------------------------------------------------------


@needs_driver_session
def test_first_ask_fresh_second_ask_resumes(tmp_path) -> None:
    runner = FakeSessionRunner([
        {"receipt": {"configs": [{"depth": 2}]}},
        {"receipt": {"configs": [{"depth": 3}]}},
    ])
    session = make_bout(runner, tmp_path,
                        first_extras={"search_space": "SS", "budget": "10"})
    assert session.ask() == {"configs": [{"depth": 2}]}
    assert session.ask(extra={llm.OUTCOME_KEY: "O"}) == {"configs": [{"depth": 3}]}

    (_, ctx1), (_, ctx2) = runner.calls
    assert ctx1.resume_session_id is None  # fresh start
    assert ctx2.resume_session_id == "fake-sess-0001"  # chained via store
    assert (ctx1.invocation_id, ctx2.invocation_id) == (1, 2)
    assert ctx1.extra["search_space"] == "SS" and ctx1.extra["budget"] == "10"
    assert ctx2.extra == {llm.OUTCOME_KEY: "O"}  # first extras not re-sent


@needs_driver_session
def test_ask_extra_wins_first_extras_collision(tmp_path) -> None:
    runner = FakeSessionRunner([{"receipt": {}}])
    session = make_bout(runner, tmp_path, first_extras={"k": "first"})
    session.ask(extra={"k": "call"})
    assert runner.calls[0][1].extra["k"] == "call"


@needs_driver_session
def test_invocation_failed_propagates_and_chain_skips_failure(tmp_path) -> None:
    runner = FakeSessionRunner([
        {"receipt": {"ok": 1}},
        {"fail": ["no accepted receipt"]},
        {"receipt": {"ok": 3}},
    ])
    session = make_bout(runner, tmp_path)
    session.ask()
    with pytest.raises(InvocationFailed):
        session.ask()
    assert session.call_count == 2  # a failed ask still burned tokens: logged
    session.ask()
    ctx3 = runner.calls[2][1]
    assert ctx3.invocation_id == 3  # the failed call still consumed an id
    assert ctx3.resume_session_id == "fake-sess-0001"  # last GOOD invocation


@needs_driver_session
def test_bout_sessions_are_independent(tmp_path) -> None:
    runner = FakeSessionRunner([{"receipt": {}} for _ in range(4)])
    session_a = make_bout(runner, tmp_path)
    session_b = make_bout(runner, tmp_path)
    session_a.ask()
    session_b.ask()
    session_a.ask()
    session_b.ask()
    ctxs = [ctx for _, ctx in runner.calls]
    assert [c.invocation_id for c in ctxs] == [1, 2, 3, 4]  # shared store ids
    assert [c.resume_session_id for c in ctxs] == [
        None, None, "fake-sess-0001", "fake-sess-0002"]  # separate chains


@needs_driver_session
def test_usage_capture_without_events_file(tmp_path) -> None:
    runner = FakeSessionRunner([{"receipt": {}}, {"receipt": {}}])
    session = make_bout(runner, tmp_path)
    session.ask()
    session.ask()
    assert session.call_count == 2
    assert [e["invocation_id"] for e in session.usage_log] == [1, 2]
    assert all(e["input_tokens"] == 0 and e["output_tokens"] == 0
               for e in session.usage_log)
    assert session.totals() == {"llm_calls": 2, "llm_input_tokens": 0,
                                "llm_output_tokens": 0}


@needs_driver_session
def test_usage_capture_from_real_runner_events(tmp_path) -> None:
    """SDKSessionRunner + scripted client_factory (production FakeClient
    pattern): session_end usage rows, incl. corrective follow-ups, are
    summed per invocation; resume is wired into ClaudeAgentOptions."""
    pytest.importorskip("claude_agent_sdk")
    from driver.events import EventsLog
    from driver.receipts import ReceiptStore
    from driver.session import SDKSessionRunner

    role = llm.BENCH_ROLES["bench-rewarm-proposer"]
    store = ReceiptStore(tmp_path)
    behavior = [  # per query turn, across invocations
        {"accept": False, "usage": {"input_tokens": 10, "output_tokens": 4}},
        {"accept": True, "usage": {"input_tokens": 5, "output_tokens": 2,
                                   "cache_read_input_tokens": 7}},
        {"accept": True, "usage": {"input_tokens": 3, "output_tokens": 1}},
    ]
    captured_options = []

    class _Init:
        subtype = "init"

        def __init__(self, session_id):
            self.data = {"session_id": session_id}

    class _Result:
        subtype = "result"

        def __init__(self, usage):
            self.session_id = "sess-usage"
            self.is_error = False
            self.num_turns = 1
            self.total_cost_usd = 0.01
            self.usage = usage

    class _Client:
        def __init__(self, options):
            captured_options.append(options)
            self._inv = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, prompt):
            pass

        async def receive_response(self):
            if self._inv is None:
                # Nothing of this invocation is persisted yet, so the store's
                # next id IS this invocation's id.
                self._inv = store.next_invocation_id()
            yield _Init(f"sess-usage-{self._inv:04d}")
            turn = behavior.pop(0) if behavior else {"accept": False}
            if turn.get("accept"):
                store.persist_receipt(role.name, self._inv,
                                      {"configs": [{"depth": 2}],
                                       "rationale": "r"})
            yield _Result(turn.get("usage"))

    runner = SDKSessionRunner(model="test-model",
                              events=EventsLog(tmp_path),
                              client_factory=_Client)
    session = make_bout(runner, tmp_path)
    receipt1 = session.ask()
    receipt2 = session.ask()
    assert receipt1 == receipt2 == {"configs": [{"depth": 2}], "rationale": "r"}

    # Resume wiring through the REAL options builder.
    assert captured_options[0].resume is None
    assert captured_options[1].resume == "sess-usage-0001"

    first, second = session.usage_log
    assert first["invocation_id"] == 1 and first["session_ends"] == 2
    assert first["input_tokens"] == 15 and first["output_tokens"] == 6
    assert first["usage"]["cache_read_input_tokens"] == 7
    assert second["invocation_id"] == 2 and second["session_ends"] == 1
    assert second["input_tokens"] == 3 and second["output_tokens"] == 1
    assert session.totals() == {"llm_calls": 2, "llm_input_tokens": 18,
                                "llm_output_tokens": 7}


@needs_driver_session
def test_bout_session_factory(tmp_path) -> None:
    runner = FakeSessionRunner([{"receipt": {}}, {"receipt": {}}])
    factory = llm.make_bout_session_factory(runner=runner, run_dir=tmp_path,
                                            task="t", tag="tag")
    first = factory("bench-pool-proposer", first_extras={"a": "b"})
    second = factory("bench-pool-proposer")
    assert first.role.name == "bench-pool-proposer"
    assert first is not second  # a new bout is always a NEW session
    first.ask()
    second.ask()
    ctxs = [ctx for _, ctx in runner.calls]
    assert [c.resume_session_id for c in ctxs] == [None, None]
    with pytest.raises(KeyError):
        factory("no-such-role")


# --- format_history -------------------------------------------------------------


def _history_rows():
    H = checkpoint_mod.HistoryRow
    return [
        H(params={"x": 1}, score=5.0, status="ok", origin="phase_a"),
        H(params={"x": 2}, score=3.0, status="ok", origin="phase_a"),
        H(params={"x": 3}, score=4.0, status="ok", origin="bout_0"),
        H(params={"x": 4}, score=None, status="crash", origin="bout_0"),
        H(params={"x": 5}, score=2.5, status="ok", origin="bout_0"),
        H(params={"x": 6}, score=2.5, status="ok", origin="bout_0"),
    ]


def test_format_history_at_the_time_flags_default_incumbent() -> None:
    lines = llm.format_history(
        _history_rows(), incumbent_score_at_each_eval=math.inf).splitlines()
    assert len(lines) == 6  # one line per executed trial
    # Hand-computed against a running incumbent starting at +inf:
    # 5.0 improves (inf), 3.0 improves (5.0), 4.0 does not (3.0),
    # crash, 2.5 improves (3.0), 2.5 does not (equal is not strict).
    assert lines[0].startswith("#1 [phase_a]") and "score=5.0" in lines[0]
    assert "IMPROVED" in lines[0]
    assert "IMPROVED" in lines[1] and "score=3.0" in lines[1]
    assert "no improvement" in lines[2] and "score=4.0" in lines[2]
    assert "CRASH" in lines[3] and "score" not in lines[3].split("CRASH")[0]
    assert "IMPROVED" in lines[4] and "score=2.5" in lines[4]
    assert "no improvement" in lines[5]  # strict: equal score is not better


def test_format_history_with_starting_incumbent() -> None:
    lines = llm.format_history(
        _history_rows(), incumbent_score_at_each_eval=4.0).splitlines()
    # Hand-computed against a running incumbent starting at 4.0:
    # 5.0 no, 3.0 improves (4.0), 4.0 no (3.0), crash, 2.5 improves, 2.5 no.
    assert "no improvement" in lines[0]
    assert "IMPROVED" in lines[1]
    assert "no improvement" in lines[2]
    assert "CRASH" in lines[3]
    assert "IMPROVED" in lines[4]
    assert "no improvement" in lines[5]


def test_format_history_trial_rows_and_rejected_skip() -> None:
    T = state_mod.Trial
    trials = [
        T(config={"x": 1}, score=2.0, status="ok", source="tpe"),
        T(config={"x": 2}, score=None, status="preflight_rejected",
          source="tpe"),
        T(config={"x": 3}, score=None, status="crash", source="rewarm"),
    ]
    lines = llm.format_history(
        trials, incumbent_score_at_each_eval=math.inf).splitlines()
    assert len(lines) == 2  # preflight-rejected was never executed
    assert lines[0].startswith("#1 [tpe]") and "IMPROVED" in lines[0]
    assert lines[1].startswith("#2 [rewarm]") and "CRASH" in lines[1]


def test_format_history_empty() -> None:
    assert (
        llm.format_history([], incumbent_score_at_each_eval=1.0)
        == "(no executed trials yet)"
    )


# --- first_message_blocks --------------------------------------------------------


def test_first_message_blocks_cover_the_checklist(tmp_path) -> None:
    blocks = llm.first_message_blocks(
        make_checkpoint(tmp_path), make_contract(),
        protocol="POLL PROTOCOL TEXT", budget_remaining=7)
    assert set(blocks) == set(llm.BLOCK_KEYS)
    assert llm.EVIDENCE_KEY not in blocks

    search_space = blocks["search_space"]
    assert "lower-is-better" in search_space
    assert "PARAM_SCHEMA" in search_space  # parameter semantics
    assert "log-scale" in search_space
    assert "FIXED" in search_space  # degenerate seed dimension
    assert "tunable" in search_space

    candidate = blocks["candidate"]
    for marker in ("continuation", "cont_improved", "improve", "c3"):
        assert marker in candidate, marker

    assert "3.0" in blocks["incumbent"] and "depth" in blocks["incumbent"]

    history_lines = blocks["history"].removeprefix(llm.HISTORY_READING_NOTES).splitlines()
    assert len(history_lines) == 2
    assert "score=5.0" in history_lines[0]
    assert "CRASH" in history_lines[1]

    assert blocks["protocol"] == "POLL PROTOCOL TEXT"
    assert "7" in blocks["budget"]


def test_first_message_blocks_evidence(tmp_path) -> None:
    blocks = llm.first_message_blocks(
        make_checkpoint(tmp_path), make_contract(), protocol="p",
        budget_remaining=7, evidence=["block A", "block B"])
    assert "block A" in blocks[llm.EVIDENCE_KEY]
    assert "block B" in blocks[llm.EVIDENCE_KEY]


def test_first_message_blocks_trials_override(tmp_path) -> None:
    blocks = llm.first_message_blocks(
        make_checkpoint(tmp_path), make_contract(), protocol="p",
        budget_remaining=7, trials=[])
    assert blocks["history"] == llm.HISTORY_READING_NOTES + "(no executed trials yet)"


def test_first_message_blocks_live_incumbent_override(tmp_path) -> None:
    live = {"depth": 7, "lr": 0.02, "mode": "slow", "seed": 7}
    blocks = llm.first_message_blocks(
        make_checkpoint(tmp_path),
        make_contract(),
        protocol="p",
        budget_remaining=4,
        live_incumbent=(live, 2.5),
    )
    assert '"depth":7' in blocks["incumbent"]
    assert '"mode":"slow"' in blocks["incumbent"]
    assert "2.5" in blocks["incumbent"]
    assert "3.0" not in blocks["incumbent"]


def test_first_message_blocks_candidate_kind(tmp_path) -> None:
    contract = make_contract()
    explicit = llm.first_message_blocks(
        make_checkpoint(tmp_path, source={"candidate_id": "c9"}), contract,
        protocol="p", budget_remaining=1, candidate_kind="crossover")
    assert "crossover" in explicit["candidate"]
    derived = llm.first_message_blocks(
        make_checkpoint(tmp_path, source={"candidate_id": "provided_baseline"}),
        contract, protocol="p", budget_remaining=1)
    assert "provided-baseline" in derived["candidate"]
    unknown = llm.first_message_blocks(
        make_checkpoint(tmp_path, source={}), contract, protocol="p",
        budget_remaining=1)
    assert "unknown" in unknown["candidate"]


# --- outcome_message ---------------------------------------------------------------


def test_outcome_message_ok_improved() -> None:
    text = llm.outcome_message({"depth": 3}, status="ok", score=2.0,
                               incumbent_score=3.0, budget_remaining=4)
    assert "score = 2.0" in text and "IMPROVED" in text
    assert "3.0" in text and "4" in text


def test_outcome_message_ok_not_improved_strict() -> None:
    text = llm.outcome_message({"depth": 3}, status="ok", score=3.0,
                               incumbent_score=3.0, budget_remaining=4)
    assert "did NOT improve" in text  # equal is not strict improvement


def test_outcome_message_crash() -> None:
    text = llm.outcome_message({"depth": 3}, status="crash",
                               incumbent_score=3.0, budget_remaining=4)
    assert "CRASH" in text and "+inf" in text
    assert "did NOT improve" in text and "4" in text


def test_outcome_message_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        llm.outcome_message({}, status="ok", score=None,
                            incumbent_score=1.0, budget_remaining=1)
    with pytest.raises(ValueError):
        llm.outcome_message({}, status="weird", score=1.0,
                            incumbent_score=1.0, budget_remaining=1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_first_message_blocks_history_replays_from_trajectory_start(tmp_path) -> None:
    """PLAN §四: history flags are relative to the AT-THE-TIME incumbent —
    replayed from the trajectory's own start, never against the final
    checkpoint incumbent (which would flag every row 'no improvement')."""
    base = make_checkpoint(tmp_path)
    rows = (
        checkpoint_mod.HistoryRow(
            params=dict(base.incumbent.params, depth=1),
            score=5.0, status="ok", origin="phase_a"),
        checkpoint_mod.HistoryRow(
            params=dict(base.incumbent.params, depth=2),
            score=3.0, status="ok", origin="bout_0"),
        checkpoint_mod.HistoryRow(
            params=dict(base.incumbent.params, depth=3),
            score=4.0, status="ok", origin="bout_0"),
    )
    ckpt = dataclasses.replace(base, history=rows)

    blocks = llm.first_message_blocks(
        ckpt, make_contract(), protocol="p", budget_remaining=1)
    lines = blocks["history"].removeprefix(llm.HISTORY_READING_NOTES).splitlines()
    # 5.0 seeded the best-so-far, 3.0 tightened it, 4.0 did not beat it —
    # with the final incumbent (3.0) as reference all three would read
    # "no improvement".
    assert "IMPROVED" in lines[0] and "score=5.0" in lines[0]
    assert "IMPROVED" in lines[1] and "score=3.0" in lines[1]
    assert "no improvement" in lines[2] and "score=4.0" in lines[2]
