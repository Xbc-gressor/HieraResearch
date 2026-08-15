"""GPU-free same-pool selector experiment on frozen candidates.

For each frozen checkpoint this standalone runner:

1. asks the existing ``bench-pool-proposer`` for one POOL=5 proposal and its
   direct self-ranking;
2. keeps that exact filtered, production-cast pool fixed;
3. asks fresh, mutually isolated ``bench-pool-pairwise-judge`` sessions for
   single-order pair verdicts; and
4. compares the direct rank-1 config with the Copeland winner.

No candidate is executed: this script never imports or calls ``prepare.py``,
never invokes the benchmark objective, and consumes no admitted-objective or
GPU budget. It measures selector disagreement, not which selector is better.

The rank-1 candidate is compared with every other pool member first. If it
wins all of those edges, it is already the unique full-round-robin Copeland
winner and the remaining edges are skipped exactly. Otherwise the graph is
completed. With POOL=5 this costs 4--10 independent judge calls per pool.

Example (from the repository root)::

    uv run python tools/inner_benchmark/pairwise_shadow.py \
      --checkpoints-root ../checkpoints \
      --model grok/grok-4.6 \
      --out ../pairwise-shadow/grok46-all-checkpoints

The output is resumable: accepted pools and duel receipts are reused after a
restart, while every newly executed duel gets its own run directory and fresh
SDK session. Separate run directories also make bounded concurrency safe with
the current receipt invocation-id allocator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Callable, Iterable

_HERE = Path(__file__).resolve().parent
_TOOLS = _HERE.parent
_REPO = _TOOLS.parent
for _path in (_HERE, _TOOLS, _TOOLS / "tuners", _REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import checkpoint as checkpoint_mod  # noqa: E402
import llm  # noqa: E402
import space as space_mod  # noqa: E402
import state as state_mod  # noqa: E402
import tune_tools  # noqa: E402
from arms.pool import POOL_PROTOCOL, PoolDriver  # noqa: E402
from driver.events import EventsLog  # noqa: E402
from driver.session import SDKSessionRunner  # noqa: E402


ROLE = "bench-pool-pairwise-judge"
PROTOCOL_ID = "balanced-single-order-roundrobin-v1"
SCHEMA_VERSION = 1
PAIR_KEY = "pair"

PAIR_PROTOCOL = (
    "Same-pool shadow selector protocol. Compare only config A with config B "
    "as the next objective evaluation at this factual frozen-candidate state. "
    "Choose the config with lower expected score (lower is better; crash is "
    "worst). A/B positions are arbitrary. The proposer ranking and rationale, "
    "other pool members, other verdicts, and all unobserved outcomes are "
    "intentionally hidden. Return one forced A-or-B choice. No objective will "
    "be executed by this shadow experiment."
)


@dataclass(frozen=True)
class FrozenInput:
    checkpoint: checkpoint_mod.Checkpoint
    contract: space_mod.CandidateContract
    checkpoint_hash: str
    candidate_execution_revision: dict


@dataclass(frozen=True)
class Duel:
    ordinal: int
    a_index: int
    b_index: int


@dataclass(frozen=True)
class DuelVerdict:
    duel: Duel
    winner_index: int
    payload: dict


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_frozen(checkpoint_dir: Path) -> FrozenInput:
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint = checkpoint_mod.load_checkpoint(checkpoint_dir)
    contract = space_mod.read_contract(checkpoint.candidate_path)
    checkpoint_path = checkpoint_dir / checkpoint_mod.CHECKPOINT_FILENAME
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    revision = tune_tools._candidate_execution_revision(checkpoint.candidate_path)
    return FrozenInput(
        checkpoint=checkpoint,
        contract=contract,
        checkpoint_hash=digest,
        candidate_execution_revision=revision,
    )


def discover_checkpoint_dirs(
    explicit: Iterable[str] | None,
    roots: Iterable[str] | None,
) -> list[Path]:
    """Resolve explicit dirs or recursively enumerate checkpoint roots."""
    paths = [Path(path).resolve() for path in explicit or ()]
    for raw_root in roots or ():
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"checkpoint root not found: {root}")
        paths.extend(path.parent for path in root.rglob(checkpoint_mod.CHECKPOINT_FILENAME))
    unique = sorted(set(paths), key=str)
    if not unique:
        raise ValueError("no checkpoint.json found under --checkpoints-root")
    return unique


def _experiment_manifest(
    frozen: Iterable[FrozenInput],
    *,
    model: str,
    seed: int,
    repeats: int,
    budget: int,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen-candidate-same-pool-selector-shadow",
        "protocol_id": PROTOCOL_ID,
        "model": model,
        "llm": llm.LLMConfig(model=model).to_manifest(),
        "seed": seed,
        "repeats": repeats,
        "factual_budget_remaining": budget,
        "objective_calls": 0,
        "checkpoints": [
            {
                "checkpoint_id": item.checkpoint.checkpoint_id,
                "checkpoint_dir": str(item.checkpoint.checkpoint_dir.resolve()),
                "checkpoint_hash": item.checkpoint_hash,
                "candidate_execution_revision": item.candidate_execution_revision,
                # This experiment executes no objective, so an incomplete
                # target-machine remeasurement is allowed. Persist its state
                # to prevent selector disagreement from being misreported as
                # locally calibrated efficacy evidence.
                "remeasure": (item.checkpoint.extra or {}).get("remeasure"),
            }
            for item in frozen
        ],
    }


def _write_or_validate_manifest(out_dir: Path, expected: dict) -> None:
    path = out_dir / "manifest.json"
    if path.exists():
        observed = _read_json(path)
        if observed != expected:
            raise ValueError(
                f"{path} belongs to a different experiment; choose a new --out"
            )
        return
    _write_json(path, expected)


def _initial_state(item: FrozenInput, *, budget: int) -> state_mod.CellState:
    checkpoint = item.checkpoint
    contract = item.contract
    trials = [
        state_mod.Trial(
            config=dict(row.params),
            score=row.score,
            status=row.status,
            source=row.origin,
        )
        for row in checkpoint.history
    ]
    incumbent_identity = contract.params_identity(checkpoint.incumbent.params)
    if not any(
        contract.params_identity(trial.config) == incumbent_identity
        for trial in trials
    ):
        trials.insert(
            0,
            state_mod.Trial(
                config=dict(checkpoint.incumbent.params),
                score=checkpoint.incumbent.score,
                status="ok",
                source="incumbent",
            ),
        )
    return state_mod.CellState(
        incumbent_config=dict(checkpoint.incumbent.params),
        incumbent_score=checkpoint.incumbent.score,
        budget_remaining=budget,
        trials=trials,
        identity_fn=contract.params_identity,
    )


def _task_name(checkpoint: checkpoint_mod.Checkpoint) -> str:
    project = checkpoint.task.project or ""
    return project.rsplit("/", 1)[-1] if project else "unknown"


def _validate_saved_pool(data: dict, item: FrozenInput, *, model: str) -> list[dict]:
    expected = {
        "checkpoint_id": item.checkpoint.checkpoint_id,
        "checkpoint_hash": item.checkpoint_hash,
        "model": model,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"saved pool binding {key}={data.get(key)!r}, expected {value!r}")
    pool = data.get("executed_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("saved pool has no non-empty executed_pool")
    return _cast_pool(pool, item.contract)


def _cast_pool(configs: Iterable[dict], contract) -> list[dict]:
    pool: list[dict] = []
    identities: set[str] = set()
    for index, config in enumerate(configs):
        if not isinstance(config, dict):
            raise ValueError(f"pool[{index}] is not an object")
        cast = contract.cast(config)
        violations = tune_tools._bounds_violations(cast, contract.search_space)
        if violations:
            raise ValueError(f"pool[{index}] violates search space: {violations}")
        identity = contract.params_identity(cast)
        if identity in identities:
            raise ValueError(f"pool[{index}] duplicates another filtered member")
        identities.add(identity)
        pool.append(cast)
    return pool


def _obtain_pool(
    item: FrozenInput,
    *,
    replicate_dir: Path,
    model: str,
    budget: int,
    cli_path: str | None,
) -> tuple[list[dict], dict]:
    saved_path = replicate_dir / "pool.json"
    if saved_path.exists():
        saved = _read_json(saved_path)
        pool = _validate_saved_pool(saved, item, model=model)
        return pool, saved

    checkpoint = item.checkpoint
    proposer_dir = replicate_dir / "proposer"
    runner = SDKSessionRunner(
        model=model,
        events=EventsLog(proposer_dir),
        cli_path=cli_path,
    )
    factory = llm.make_bout_session_factory(
        runner=runner,
        run_dir=proposer_dir,
        task=_task_name(checkpoint),
        tag=f"{checkpoint.checkpoint_id}--selector-shadow",
    )
    ctx = SimpleNamespace(
        contract=item.contract,
        checkpoint=checkpoint,
        budget=budget,
        state=_initial_state(item, budget=budget),
        extras={"session_factory": factory},
    )
    driver = PoolDriver(ctx)
    proposal = driver.ask_pool()
    pool = _cast_pool(proposal["pool"], item.contract)
    filtered_ranks = [
        rank
        for rank, masked in enumerate(proposal["pool_duplicate_mask"], start=1)
        if not masked
    ]
    saved = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_hash": item.checkpoint_hash,
        "model": model,
        "pool_protocol": POOL_PROTOCOL,
        "pool_configs": proposal["pool_ranked"],
        "pool_order": proposal["order"],
        "pool_duplicate_mask": proposal["pool_duplicate_mask"],
        "executed_pool": pool,
        "executed_pool_proposer_ranks": filtered_ranks,
        "self_rank_index": 0,
        "self_rank_proposer_rank": filtered_ranks[0],
        "self_rank_config": pool[0],
        "proposer_rationale": proposal["rationale"],
        "proposer_attempts": proposal["attempts"],
        "proposer_usage": driver.totals(),
    }
    _write_json(saved_path, saved)
    return pool, saved


def balanced_duels(size: int, rng: random.Random) -> tuple[Duel, ...]:
    """One directed A/B presentation for every unordered pair.

    For odd pool sizes, the cyclic orientation gives every candidate exactly
    half its appearances as A and half as B. POOL=5 therefore gives 2/2.
    The rare even-size case (possible after duplicate filtering) is seeded and
    near-balanced, because exact per-candidate balance is impossible at odd
    degree.
    """
    if size < 2:
        return ()
    labels = list(range(size))
    rng.shuffle(labels)
    directed: list[tuple[int, int]] = []
    if size % 2 == 1:
        half = (size - 1) // 2
        for position, label in enumerate(labels):
            for offset in range(1, half + 1):
                directed.append((label, labels[(position + offset) % size]))
    else:
        for left in range(size):
            for right in range(left + 1, size):
                a, b = labels[left], labels[right]
                directed.append((a, b) if (left + right) % 2 else (b, a))

    expected = size * (size - 1) // 2
    unordered = {frozenset((a, b)) for a, b in directed}
    if len(directed) != expected or len(unordered) != expected:
        raise AssertionError("duel orientation did not cover each unordered pair once")

    first = sorted(
        ((a, b) for a, b in directed if 0 in (a, b)),
        key=lambda edge: edge[1] if edge[0] == 0 else edge[0],
    )
    rest = sorted(
        ((a, b) for a, b in directed if 0 not in (a, b)),
        key=lambda edge: tuple(sorted(edge)),
    )
    return tuple(
        Duel(ordinal=index + 1, a_index=a, b_index=b)
        for index, (a, b) in enumerate(first + rest)
    )


def run_exact_tournament(
    size: int,
    duels: tuple[Duel, ...],
    run_batch: Callable[[tuple[Duel, ...]], list[DuelVerdict]],
    tie_rng: random.Random,
) -> dict:
    """Run rank-1 proof edges first, then complete only when necessary."""
    if size == 1:
        return {
            "winner_index": 0,
            "verdicts": [],
            "observed_vote_counts": [0],
            "full_vote_counts": [0],
            "top_indexes": [0],
            "tie_break": "sole_survivor",
            "early_stopped": False,
            "full_round_robin_completed": True,
        }

    first = tuple(duel for duel in duels if 0 in (duel.a_index, duel.b_index))
    remaining = tuple(duel for duel in duels if 0 not in (duel.a_index, duel.b_index))
    verdicts = _validated_batch(first, run_batch(first))
    rank1_swept = all(verdict.winner_index == 0 for verdict in verdicts)
    early_stopped = bool(remaining) and rank1_swept
    if not early_stopped:
        verdicts.extend(_validated_batch(remaining, run_batch(remaining)))

    votes = [0] * size
    for verdict in verdicts:
        votes[verdict.winner_index] += 1

    if early_stopped:
        top = [0]
        selected = 0
        tie_break = "none"
        full_votes = None
    else:
        best = max(votes)
        top = [index for index, value in enumerate(votes) if value == best]
        selected = top[0] if len(top) == 1 else tie_rng.choice(top)
        tie_break = "none" if len(top) == 1 else "seeded_uniform"
        full_votes = votes

    return {
        "winner_index": selected,
        "verdicts": verdicts,
        "observed_vote_counts": votes,
        "full_vote_counts": full_votes,
        "top_indexes": top,
        "tie_break": tie_break,
        "early_stopped": early_stopped,
        "full_round_robin_completed": not early_stopped,
    }


def _validated_batch(
    requested: tuple[Duel, ...], observed: list[DuelVerdict]
) -> list[DuelVerdict]:
    by_ordinal = {verdict.duel.ordinal: verdict for verdict in observed}
    if set(by_ordinal) != {duel.ordinal for duel in requested}:
        raise ValueError("judge batch did not return exactly one verdict per duel")
    ordered = [by_ordinal[duel.ordinal] for duel in requested]
    for duel, verdict in zip(requested, ordered):
        if verdict.duel != duel or verdict.winner_index not in (
            duel.a_index,
            duel.b_index,
        ):
            raise ValueError(f"invalid verdict for duel {duel.ordinal}")
    return ordered


def _judge_one(
    duel: Duel,
    *,
    item: FrozenInput,
    pool: list[dict],
    common_blocks: dict,
    replicate_dir: Path,
    model: str,
    cli_path: str | None,
) -> DuelVerdict:
    duel_dir = replicate_dir / "duels" / f"duel-{duel.ordinal:02d}"
    result_path = duel_dir / "result.json"
    binding = {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_id": item.checkpoint.checkpoint_id,
        "checkpoint_hash": item.checkpoint_hash,
        "model": model,
        "a_index": duel.a_index,
        "b_index": duel.b_index,
        "config_a": pool[duel.a_index],
        "config_b": pool[duel.b_index],
    }
    if result_path.exists():
        payload = _read_json(result_path)
        for key, value in binding.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"{result_path}: saved duel binding {key} does not match"
                )
        return DuelVerdict(
            duel=duel,
            winner_index=int(payload["winner_index"]),
            payload={**payload, "reused": True},
        )

    blocks = dict(common_blocks)
    blocks[PAIR_KEY] = "\n".join(
        [
            "Config A:",
            json.dumps(pool[duel.a_index], ensure_ascii=False, separators=(",", ":")),
            "Config B:",
            json.dumps(pool[duel.b_index], ensure_ascii=False, separators=(",", ":")),
        ]
    )
    runner = SDKSessionRunner(
        model=model,
        events=EventsLog(duel_dir),
        cli_path=cli_path,
    )
    session = llm.BoutSession(
        role=llm.BENCH_ROLES[ROLE],
        runner=runner,
        run_dir=duel_dir,
        task=_task_name(item.checkpoint),
        tag=f"{item.checkpoint.checkpoint_id}--pairwise-shadow",
        first_extras=blocks,
    )
    receipt = session.ask()
    winner_index = duel.a_index if receipt["winner"] == "A" else duel.b_index
    payload = {
        **binding,
        "winner_label": receipt["winner"],
        "winner_index": winner_index,
        "reasoning": receipt.get("reasoning"),
        "usage": session.totals(),
        "usage_log": session.usage_log,
        "reused": False,
    }
    _write_json(result_path, payload)
    return DuelVerdict(duel=duel, winner_index=winner_index, payload=payload)


def _run_judge_batch(
    duels: tuple[Duel, ...],
    *,
    concurrency: int,
    judge: Callable[[Duel], DuelVerdict],
) -> list[DuelVerdict]:
    if not duels:
        return []
    workers = min(concurrency, len(duels))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge, duel): duel for duel in duels}
        results: list[DuelVerdict] = []
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _dimension_distance(contract, first: dict, second: dict) -> dict:
    deltas: dict[str, float] = {}
    for dim in contract.varying_dimensions:
        left, right = first[dim.name], second[dim.name]
        if dim.kind == "categorical":
            delta = 0.0 if left == right else 1.0
        elif dim.kind == "float" and dim.log:
            span = math.log(dim.hi) - math.log(dim.lo)
            delta = abs(math.log(float(left)) - math.log(float(right))) / span
        else:
            delta = abs(float(left) - float(right)) / (dim.hi - dim.lo)
        deltas[dim.name] = float(delta)
    values = list(deltas.values())
    return {
        "l0_dimensions": sum(value > 0.0 for value in values),
        "normalized_l1": sum(values),
        "normalized_linf": max(values, default=0.0),
        "per_dimension": deltas,
    }


def _has_cycle(size: int, verdicts: list[DuelVerdict]) -> bool:
    adjacency = [[] for _ in range(size)]
    for verdict in verdicts:
        loser = (
            verdict.duel.b_index
            if verdict.winner_index == verdict.duel.a_index
            else verdict.duel.a_index
        )
        adjacency[verdict.winner_index].append(loser)
    state = [0] * size

    def visit(node: int) -> bool:
        state[node] = 1
        for child in adjacency[node]:
            if state[child] == 1 or (state[child] == 0 and visit(child)):
                return True
        state[node] = 2
        return False

    return any(state[node] == 0 and visit(node) for node in range(size))


def _kendall_tau_b_against_proposer(votes: list[int]) -> float | None:
    concordant = discordant = tied = 0
    for better_rank in range(len(votes)):
        for worse_rank in range(better_rank + 1, len(votes)):
            if votes[better_rank] > votes[worse_rank]:
                concordant += 1
            elif votes[better_rank] < votes[worse_rank]:
                discordant += 1
            else:
                tied += 1
    if concordant + discordant == 0:
        return None
    denominator = math.sqrt((concordant + discordant) * (concordant + discordant + tied))
    return (concordant - discordant) / denominator


def _usage_totals(pool_record: dict, verdicts: list[DuelVerdict]) -> dict:
    proposer = pool_record.get("proposer_usage", {})
    totals = {
        "proposer_calls": int(proposer.get("llm_calls", 0)),
        "judge_calls": len(verdicts),
        "llm_input_tokens": int(proposer.get("llm_input_tokens", 0)),
        "llm_output_tokens": int(proposer.get("llm_output_tokens", 0)),
    }
    for verdict in verdicts:
        usage = verdict.payload.get("usage", {})
        totals["llm_input_tokens"] += int(usage.get("llm_input_tokens", 0))
        totals["llm_output_tokens"] += int(usage.get("llm_output_tokens", 0))
    totals["llm_calls"] = totals["proposer_calls"] + totals["judge_calls"]
    return totals


def _run_replicate(
    item: FrozenInput,
    *,
    out_dir: Path,
    replicate: int,
    model: str,
    seed: int,
    budget: int,
    concurrency: int,
    cli_path: str | None,
) -> dict:
    checkpoint = item.checkpoint
    replicate_dir = (
        out_dir / "checkpoints" / checkpoint.checkpoint_id / f"replicate-{replicate:03d}"
    )
    pool, pool_record = _obtain_pool(
        item,
        replicate_dir=replicate_dir,
        model=model,
        budget=budget,
        cli_path=cli_path,
    )
    common_blocks = llm.first_message_blocks(
        checkpoint,
        item.contract,
        protocol=PAIR_PROTOCOL,
        budget_remaining=budget,
        live_incumbent=(
            dict(checkpoint.incumbent.params),
            checkpoint.incumbent.score,
        ),
    )
    orientation_rng = random.Random(
        f"orientation:{seed}:{checkpoint.checkpoint_id}:{replicate}"
    )
    tie_rng = random.Random(f"tie:{seed}:{checkpoint.checkpoint_id}:{replicate}")
    duels = balanced_duels(len(pool), orientation_rng)

    def judge(duel: Duel) -> DuelVerdict:
        return _judge_one(
            duel,
            item=item,
            pool=pool,
            common_blocks=common_blocks,
            replicate_dir=replicate_dir,
            model=model,
            cli_path=cli_path,
        )

    def run_batch(batch: tuple[Duel, ...]) -> list[DuelVerdict]:
        return _run_judge_batch(batch, concurrency=concurrency, judge=judge)

    tournament = run_exact_tournament(len(pool), duels, run_batch, tie_rng)
    winner_index = tournament["winner_index"]
    verdicts: list[DuelVerdict] = tournament.pop("verdicts")
    proposer_ranks = pool_record.get("executed_pool_proposer_ranks")
    if not isinstance(proposer_ranks, list) or len(proposer_ranks) != len(pool):
        # Compatibility with a pool.json written by an interrupted early
        # version of this script. The mask still binds the original ranks.
        proposer_ranks = [
            rank
            for rank, masked in enumerate(
                pool_record["pool_duplicate_mask"], start=1
            )
            if not masked
        ]
    distance = _dimension_distance(item.contract, pool[0], pool[winner_index])
    full = tournament["full_round_robin_completed"]
    result = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_hash": item.checkpoint_hash,
        "replicate": replicate,
        "model": model,
        "protocol_id": PROTOCOL_ID,
        "objective_calls": 0,
        "pool_size": len(pool),
        "pool": pool,
        "self_rank_index": 0,
        "self_rank_proposer_rank": proposer_ranks[0],
        "self_rank_config": pool[0],
        "pairwise_winner_index": winner_index,
        "pairwise_winner_proposer_rank": proposer_ranks[winner_index],
        "proposer_rank_distance_from_self_rank": abs(
            proposer_ranks[winner_index] - proposer_ranks[0]
        ),
        "pairwise_winner_config": pool[winner_index],
        "top1_disagreement": winner_index != 0,
        "selected_config_distance": distance,
        **tournament,
        "cycle": _has_cycle(len(pool), verdicts) if full else None,
        "kendall_tau_b": (
            _kendall_tau_b_against_proposer(tournament["full_vote_counts"])
            if full
            else None
        ),
        "duels": [
            {
                "ordinal": verdict.duel.ordinal,
                "a_index": verdict.duel.a_index,
                "b_index": verdict.duel.b_index,
                "winner_index": verdict.winner_index,
                "winner_label": verdict.payload.get("winner_label"),
                "reasoning": verdict.payload.get("reasoning"),
                "reused": verdict.payload.get("reused", False),
            }
            for verdict in verdicts
        ],
        "usage": _usage_totals(pool_record, verdicts),
    }
    _write_json(replicate_dir / "result.json", result)
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(results: list[dict]) -> dict:
    disagreements = [result for result in results if result["top1_disagreement"]]
    ranks = Counter(str(result["pairwise_winner_proposer_rank"]) for result in results)
    rank_distances = [
        result["proposer_rank_distance_from_self_rank"] for result in results
    ]
    usage_keys = (
        "proposer_calls",
        "judge_calls",
        "llm_calls",
        "llm_input_tokens",
        "llm_output_tokens",
    )
    return {
        "status": "ok",
        "objective_calls": 0,
        "snapshots": len(results),
        "top1_disagreements": len(disagreements),
        "top1_disagreement_rate": len(disagreements) / len(results),
        "pairwise_winner_proposer_rank_distribution": dict(sorted(ranks.items())),
        "proposer_rank_distance_distribution": dict(
            sorted(Counter(str(value) for value in rank_distances).items())
        ),
        "sum_proposer_rank_distance_from_self_rank": sum(rank_distances),
        "mean_proposer_rank_distance_from_self_rank": _mean(rank_distances),
        "max_proposer_rank_distance_from_self_rank": max(rank_distances),
        "early_stop_count": sum(result["early_stopped"] for result in results),
        "full_round_robin_count": sum(
            result["full_round_robin_completed"] for result in results
        ),
        "mean_selected_config_l0_dimensions": _mean(
            [result["selected_config_distance"]["l0_dimensions"] for result in results]
        ),
        "mean_selected_config_normalized_l1": _mean(
            [result["selected_config_distance"]["normalized_l1"] for result in results]
        ),
        "mean_selected_config_normalized_linf": _mean(
            [result["selected_config_distance"]["normalized_linf"] for result in results]
        ),
        "mean_kendall_tau_b_on_completed_roundrobins": _mean(
            [
                result["kendall_tau_b"]
                for result in results
                if result["kendall_tau_b"] is not None
            ]
        ),
        "usage": {
            key: sum(result["usage"][key] for result in results)
            for key in usage_keys
        },
        "comparisons": [
            {
                "checkpoint_id": result["checkpoint_id"],
                "replicate": result["replicate"],
                "top1_same": not result["top1_disagreement"],
                "self_rank_proposer_rank": result["self_rank_proposer_rank"],
                "pairwise_winner_proposer_rank": result[
                    "pairwise_winner_proposer_rank"
                ],
                "proposer_rank_distance": result[
                    "proposer_rank_distance_from_self_rank"
                ],
                "selected_config_l0_dimensions": result[
                    "selected_config_distance"
                ]["l0_dimensions"],
                "selected_config_normalized_l1": result[
                    "selected_config_distance"
                ]["normalized_l1"],
                "selected_config_normalized_linf": result[
                    "selected_config_distance"
                ]["normalized_linf"],
                "early_stopped": result["early_stopped"],
                "judge_calls": result["usage"]["judge_calls"],
            }
            for result in results
        ],
        "disagreement_snapshots": [
            {
                "checkpoint_id": result["checkpoint_id"],
                "replicate": result["replicate"],
                "pairwise_winner_proposer_rank": result[
                    "pairwise_winner_proposer_rank"
                ],
                "selected_config_distance": result["selected_config_distance"],
            }
            for result in disagreements
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inner_benchmark.pairwise_shadow",
        description=(
            "Generate one frozen-candidate pool, then compare its direct "
            "self-rank winner with independent pairwise voting; never run "
            "the objective or GPU."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        action="append",
        help="frozen checkpoint directory; repeat for multiple candidates",
    )
    source.add_argument(
        "--checkpoints-root",
        action="append",
        help=(
            "recursively discover checkpoint.json files; repeat for multiple "
            "roots"
        ),
    )
    parser.add_argument("--model", required=True, help="pinned model for proposer and judges")
    parser.add_argument("--out", required=True, help="experiment output directory")
    parser.add_argument("--seed", type=int, default=1, help="orientation/tie seed")
    parser.add_argument("--repeats", type=int, default=1, help="fresh pools per checkpoint")
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="factual remaining budget shown to proposer/judges; no budget is consumed",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="maximum simultaneous isolated judge sessions",
    )
    parser.add_argument("--cli-path", default=None, help="optional Claude CLI path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the call envelope without invoking an LLM",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    checkpoint_dirs = discover_checkpoint_dirs(args.checkpoint, args.checkpoints_root)
    frozen = [_load_frozen(path) for path in checkpoint_dirs]
    ids = [item.checkpoint.checkpoint_id for item in frozen]
    if len(ids) != len(set(ids)):
        raise ValueError("--checkpoint contains duplicate checkpoint ids")

    pools = len(frozen) * args.repeats
    if args.dry_run:
        plan = {
            "checkpoints": ids,
            "model": args.model,
            "pools": pools,
            "objective_calls": 0,
            "proposer_calls": pools,
            "judge_calls_min_if_pool5": pools * 4,
            "judge_calls_max_if_pool5": pools * 10,
            "concurrency": args.concurrency,
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    out_dir = Path(args.out).resolve()
    manifest = _experiment_manifest(
        frozen,
        model=args.model,
        seed=args.seed,
        repeats=args.repeats,
        budget=args.budget,
    )
    _write_or_validate_manifest(out_dir, manifest)
    results = []
    for item in frozen:
        for replicate in range(1, args.repeats + 1):
            result = _run_replicate(
                item,
                out_dir=out_dir,
                replicate=replicate,
                model=args.model,
                seed=args.seed,
                budget=args.budget,
                concurrency=args.concurrency,
                cli_path=args.cli_path,
            )
            results.append(result)
            print(
                f"[shadow] {item.checkpoint.checkpoint_id} replicate={replicate} "
                f"self_rank=1 pairwise_rank={result['pairwise_winner_proposer_rank']} "
                f"judge_calls={result['usage']['judge_calls']}",
                flush=True,
            )
    summary = _summarize(results)
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
