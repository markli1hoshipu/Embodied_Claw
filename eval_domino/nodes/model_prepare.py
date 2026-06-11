"""model_prepare (model_agent): resolve -> fetch -> adapt-check -> smoke -> launch_spec."""
from eval_domino import tools
from eval_domino.nodes import agent_node, cfg_excerpt
from eval_domino.agents.model_agent import make_agent

SKILLS = ("list_local_models", "resolve_model_ref", "fetch_model", "check_benchmark_compat",
          "smoke_test_inference", "write_launch_spec")


def prompt(state: dict) -> str:
    rid = state["run_id"]
    gpus = (state["config"].get("resources") or {}).get("gpu_ids") or []
    return f"""TASK: prepare the model from CONFIG for evaluation and produce launch_spec.json. \
Sequence: resolve_model_ref (if kind=hf: fetch_model into the benchmark layout — transient \
download failures: retry twice, then escalate); check_benchmark_compat — on not-ok ESCALATE \
with its `diagnosis` verbatim plus what the user should do, do NOT proceed; \
smoke_test_inference on gpu_id={gpus[0] if gpus else 'MISSING'} (a failed smoke: read the \
log_tail, one bounded fix attempt if it is clearly environmental, else escalate with the \
tail); write_launch_spec(run_id="{rid}", spec_json=...) with keys model, benchmark, runtime \
(activation="source /work/markhsp/DOMINO/domino_env.sh", xla_mem_fraction=0.4), smoke \
(result_line from the smoke), example_cmd (one shard command).

SUCCESS: write_launch_spec ok -> complete_node(status="succeeded", \
artifact_paths=[launch_spec.json, launch_doc.md paths]).

CONFIG:
{cfg_excerpt(state, 'model', 'benchmark', 'resources')}"""


def _extra(state: dict, result: dict) -> dict:
    if not (tools.run_dir(state["run_id"]) / "launch_spec.json").exists():
        return {"__error__": "agent reported success but launch_spec.json was not written"}
    return {}


node = agent_node("model_prepare", lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file")),
                  prompt, requires=("resource_gate",), extra=_extra)
