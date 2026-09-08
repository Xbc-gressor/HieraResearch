from arms.pool_hebo_mace import PoolHeboMace, select_pool, _subprocess_rank_fn
from arms.pool import PoolDriver
import arm_api

class ExplicitE3U2(PoolHeboMace):
    name = "explicit_e3u2"
    pool_size = arm_api.POOL
    def run(self, ctx):
        explorer = PoolDriver(ctx, role="bench-pool-explorer", pool_size=3,
                              role_protocol="Explore diverse legal regions within the fixed search space.")
        exploiter = PoolDriver(ctx, role="bench-pool-exploiter", pool_size=2,
                               role_protocol="Exploit the incumbent locally with small legal steps.")
        try:
            while True:
                a, b = explorer.ask_pool(), exploiter.ask_pool()
                pool = (a["pool"] + b["pool"])
                if not pool: raise arm_api.ArmError("explicit_e3u2: empty merged pool")
                history = ctx.state.finite_unique_history()
                rank_fn = ctx.extras.get("hebo_rank_fn") or _subprocess_rank_fn
                idx, state = select_pool(rank_fn, ctx, history, pool) if len(history) >= arm_api.WARMUP else (0, {"ranker_fallback": True})
                chosen = pool[idx]
                before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(params=chosen, source=self.name, rationale=(a.get("rationale") or "") + " " + (b.get("rationale") or ""), arm_state={"pool_size": len(pool), **state})
                explorer.report_outcome(chosen, feedback, incumbent_before=before)
                exploiter.report_outcome(chosen, feedback, incumbent_before=before)
        finally:
            ctx.emit({**explorer.totals(), "exploit_llm_calls": exploiter.totals()["llm_calls"], "exploit_llm_input_tokens": exploiter.totals()["llm_input_tokens"], "exploit_llm_output_tokens": exploiter.totals()["llm_output_tokens"]})
ARM = ExplicitE3U2()
