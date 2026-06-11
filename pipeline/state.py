"""PipelineState / StageStatus / RunConfig TypedDicts — spec PIPELINE_AGENTIC_SPEC.md section 6."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class SourceSpec(TypedDict, total=False):
    description: str                       # free-form; data_agent interprets
    hf_repo: str | None
    repo_type: Literal["dataset", "model"] | None
    kind: Literal["snapshot", "single_file_zip"] | None
    allow_patterns: list[str] | None
    filename: str | None
    local_dir: str | None


class RunConfig(TypedDict, total=False):
    run_id: str
    description: str                       # raw user request, verbatim
    data_request: dict                     # task_description / sources / filter_description (free-form)
    train_request: dict                    # base_model / num_train_steps / batch_size / peak_lr / fsdp_devices / save_interval / wandb_enabled
    outputs: dict                          # hf_dataset_repo / hf_model_repo (None => propose + confirm)


class StageStatus(TypedDict):
    status: Literal["pending", "running", "escalated", "succeeded", "failed", "skipped"]
    started_at: str | None
    finished_at: str | None
    artifact_paths: list[str]
    agent_thread_id: str | None            # which agent's transcript to read
    escalation: dict | None                # {"question","context","user_reply"}
    error: str | None


class PipelineState(TypedDict):
    config: dict
    run_id: str
    intake: StageStatus
    ingest: StageStatus
    filter_build: StageStatus
    upload_dataset: StageStatus            # Node 3 — parallel with norm_stats
    norm_stats: StageStatus                # Node 4
    train: StageStatus
    upload_model: StageStatus
    shared_notes: Annotated[dict, operator.or_]  # dict-merge reducer: safe under parallel writes


STAGES: tuple[str, ...] = ("intake", "ingest", "filter_build", "upload_dataset",
                           "norm_stats", "train", "upload_model")


def pending() -> StageStatus:
    return {"status": "pending", "started_at": None, "finished_at": None,
            "artifact_paths": [], "agent_thread_id": None, "escalation": None, "error": None}


def init_state(run_id: str, config: dict | None = None) -> PipelineState:
    # Every key explicit: re-invoking a thread_id merges input over the checkpoint.
    return {"config": config or {}, "run_id": run_id, "shared_notes": {},
            **{s: pending() for s in STAGES}}  # type: ignore[typeddict-item]
