"""TypedDict state for the pipeline, mirroring scripts/PIPELINE_LANGGRAPH_SPEC.md exactly."""
from __future__ import annotations

from typing import Literal, TypedDict

class SourceSpec(TypedDict):
    hf_repo: str               # "behavior-1k/2025-challenge-demos" | "n8wishh/failure_recovery" | ...
    repo_type: Literal["dataset", "model"]
    kind: Literal["snapshot", "single_file_zip"]
    allow_patterns: list[str] | None   # for "snapshot"
    filename: str | None               # for "single_file_zip"
    local_dir: str                     # absolute path

class RunConfig(TypedDict):
    run_id: str                # e.g. "perturb_recovery3"
    dataset_name: str          # final lerobot dir name
    train_config_name: str     # openpi TrainConfig name
    hf_model_repo: str
    hf_dataset_repo: str
    sources: list[SourceSpec]
    builder_script: str        # path under scripts/
    pca_thresh: float | None
    dup_factor: int
    train_script: str          # path to train_pi05_*.sh
    num_train_steps: int
    save_interval: int
    batch_size: int
    peak_lr: float

class StageStatus(TypedDict):
    status: Literal["pending", "running", "succeeded", "failed", "skipped"]
    started_at: str | None     # ISO-8601
    finished_at: str | None
    error: str | None          # truncated stderr if failed
    artifact_paths: list[str]

class PipelineState(TypedDict):
    config: RunConfig
    ingest: StageStatus
    filter_build: StageStatus
    norm_stats: StageStatus
    train: StageStatus
    upload: StageStatus

STAGES: tuple[str, ...] = ("ingest", "filter_build", "norm_stats", "train", "upload")

def pending() -> StageStatus:
    return {"status": "pending", "started_at": None, "finished_at": None, "error": None, "artifact_paths": []}

def init_state(config: RunConfig) -> PipelineState:
    # Every key explicit: re-invoking a thread_id merges input over the checkpoint, so omitted keys would leak.
    return {"config": config, "ingest": pending(), "filter_build": pending(),
            "norm_stats": pending(), "train": pending(), "upload": pending()}
