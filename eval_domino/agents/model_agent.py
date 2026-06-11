"""model_agent — owns intake (parse + confirm) and model_prepare (resolve/fetch/adapt/smoke).
Runs on the Claude Agent SDK (local Claude Code CLI subscription auth — no ANTHROPIC_API_KEY)."""
from eval_domino.agents.sdk_base import SDKAgent
from eval_domino.skills import model_skills as m

REGISTRY = {
    "gpu_snapshot_summary": m.gpu_snapshot_summary,
    "save_eval_config": m.save_eval_config,
    "list_local_models": m.list_local_models,
    "resolve_model_ref": m.resolve_model_ref,
    "fetch_model": m.fetch_model,
    "check_benchmark_compat": m.check_benchmark_compat,
    "smoke_test_inference": m.smoke_test_inference,
    "write_launch_spec": m.write_launch_spec,
}

SYSTEM = ("You own model intake and preparation for benchmark evaluation. The contract you "
          "produce (launch_spec.json) is what the benchmark agent runs — never hand off a model "
          "that did not pass check_benchmark_compat AND smoke_test_inference on a real GPU. "
          "Adapt-check failures must reach the user as a diagnosis (what is wrong, how to fix), "
          "not a generic error. v1 pairing: pi05 family on the DOMINO benchmark.")


def make_agent(state: dict, skill_names: tuple, builtins: tuple = ()) -> SDKAgent:
    return SDKAgent("model_agent", state["run_id"],
                    {n: REGISTRY[n] for n in skill_names}, SYSTEM, builtins=builtins)
