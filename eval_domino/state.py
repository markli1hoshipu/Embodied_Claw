"""EvalState / EvalConfig — Model x Benchmark evaluation FSM state (see EVAL_PIPELINE_SPEC.md).
Reuses pipeline.state.StageStatus so the bridge/inbox/transitions contracts stay identical."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pipeline.state import StageStatus, pending


class EvalConfig(TypedDict, total=False):
    run_id: str
    description: str                  # raw user request, verbatim
    model: dict                       # family ("pi05") / ref (hf repo or local) / train_config_name
                                      # / model_name / checkpoint_step (None => latest)
    benchmark: dict                   # name ("domino") / tasks (list[str] or "all") / task_config
                                      # / episodes_per_task / seed
    resources: dict                   # gpus_requested (int, from confirm card) / gpu_ids (leased)


class EvalState(TypedDict):
    config: dict
    run_id: str
    intake: StageStatus
    resource_gate: StageStatus
    model_prepare: StageStatus
    bench_preflight: StageStatus
    run_matrix: StageStatus
    aggregate_report: StageStatus
    shared_notes: Annotated[dict, operator.or_]


STAGES: tuple[str, ...] = ("intake", "resource_gate", "model_prepare", "bench_preflight",
                           "run_matrix", "aggregate_report")


def init_state(run_id: str, config: dict | None = None) -> EvalState:
    return {"config": config or {}, "run_id": run_id, "shared_notes": {},
            **{s: pending() for s in STAGES}}  # type: ignore[typeddict-item]
