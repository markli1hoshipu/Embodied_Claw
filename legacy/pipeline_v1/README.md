# pi0.5 BEHAVIOR-1K LangGraph pipeline

Orchestrates the existing manual workflow — ingest → filter/build → norm-stats → train → HF upload —
as a LangGraph state machine. The underlying scripts are invoked as black-box CLIs and are never
modified. Spec: `scripts/PIPELINE_LANGGRAPH_SPEC.md`.

## Run

```bash
cd /work/markhsp/Embodied_Claw
export HF_TOKEN=hf_...          # required by ingest + upload; NEVER hardcoded anywhere
pipeline/.venv/bin/python -m pipeline.cli runs/perturb_recovery3.yaml   # one or more YAMLs, sequential
```

Preview without executing anything (side-effect free; also detects a live training process):

```bash
pipeline/.venv/bin/python -m pipeline.cli --dry-run runs/perturb_recovery3.yaml
```

Inspect the checkpointed state of a run:

```bash
pipeline/.venv/bin/python -m pipeline.cli --status perturb_recovery3
```

`--status` exit codes: `0` for a readable state **and** for a run that has never been executed
(it prints "never been executed"); `1` only for genuinely broken states (DB present but no
checkpoint for the thread).

## How it works

- State is the `PipelineState` TypedDict (`pipeline/state.py`): the YAML `RunConfig` + one
  `StageStatus` per stage.
- Every node checks its output artifacts at entry and returns `"skipped"` when they exist:
  - ingest: each source's `local_dir` has parquets AND matches the expected-counts manifest
    (`.pipeline_expected.json`, persisted on the first successful download from the repo file
    listing / completed unzip) — a partially-downloaded tree is re-ingested, not skipped.
    Pre-pipeline data without a manifest falls back to the weak parquet-presence check.
  - filter_build: `meta/info.json` exists and `total_episodes` == expected
    (source parquets matching THIS run's `allow_patterns` − PCA drops, perturb × dup_factor;
    reads BOTH `pca_filter/merged/` and `pca_filter/<run_id>/drop_list.json`)
  - norm_stats: `<lerobot>/norm_stats.json` exists
  - train: `<ckpt_dir>/<num_train_steps-1>/` contains `params/` + `_CHECKPOINT_METADATA`
  - upload: both HF repos exist AND hold the expected content (model: a `ckpt-*/params` file;
    dataset: `meta/info.json`) — existence alone is not enough, since an interrupted upload
    leaves a created-but-incomplete repo behind
- Conditional edges halt the graph as soon as any stage reports `"failed"`.
- Checkpointer: `SqliteSaver` → `/work/markhsp/Embodied_Claw/pipeline_runs/<run_id>.sqlite`
  (`thread_id` = `run_id`; invoked with `durability="sync"`). `pipeline_runs/` is git-ignored
  via the repo-root `.gitignore`.

## Resume after a crash

Just re-run the same command. Re-invoking the same `thread_id` restarts from START and the
skip-if-done checks fast-forward past completed stages; nodes are idempotent. A crashed node can
also be resumed in place by the checkpointer. State for debugging: `--status <run_id>`.

## Train node specifics (read before touching)

- **Attach mode**: before launching, the node pgreps `python.*scripts/train\.py.*<config_name>( |$)`
  (narrow pattern — bash watchers carry the substring too, and the trailing boundary avoids prefix
  collisions between config names, e.g. recovery vs recovery2/3). If a live process matches, it does
  NOT launch — the train .sh passes `--overwrite`, so a duplicate launch would wipe the in-flight
  run — and goes straight to the 60 s poll loop.
- A pgrep **error** (rc ≥ 2: fork failure, bad regex) is treated as "assume alive" — never as
  "no process" — so an unhandled error path can never double-launch. `_launch` re-checks liveness
  immediately before every launch (fresh and NCCL-relaunch) to close the preflight TOCTOU window,
  and `cli.run` holds an exclusive flock on `pipeline_runs/<run_id>.lock` so two CLI instances of
  the same run cannot overlap.
- Progress is read from the newest `logs/<config>_*.log` and `/tmp/train_<run_id>.log` (the latter
  only exists when the pipeline itself launched the script), with `\r → \n` translation because
  tqdm writes carriage returns. The heartbeat also reports the newest step dir in the ckpt dir as
  an independent progress signal.
- Pre-flight 8-GPU psum check runs only before a *fresh* launch (it can't run while GPUs are busy).
- Crash triage: NCCL `CUDA failure 401` / `nvls.cc` in log bytes written **after** attach/launch
  (stale logs from earlier runs are masked by size marks) → relaunch (max 2, after re-checking the
  final checkpoint); anything else fails the stage for a human.
- Overall poll timeout 24 h — patient enough for 4–6 h runs.

## Safety notes

- `filter_and_build` is gated hard: the builder `rmtree`s its output dir **including
  norm_stats.json**, so it only runs when info.json is missing or the episode count is stale —
  and it refuses to run at all while a live train process matches the config (the running job
  may be streaming from that very dataset dir).
- Upload staging uses hardlinks (`cp -al`) under `/work/markhsp/hf_staging/<run_id>/` — the
  per-run subdir is a deliberate deviation from the spec's flat `/work/markhsp/hf_staging/` for
  multi-run isolation (the resulting repo layout is identical); stale `ckpt-*` staging dirs from
  a prior attempt are removed and re-hardlinked. The final `59999` step is staged as `ckpt-60000`.
  `train_state/` and ckpt `assets/` are excluded; `norm_stats.json` is injected into each
  `ckpt-*/assets/` afterwards. `HF_HUB_ENABLE_HF_TRANSFER=0` and `HF_HUB_DISABLE_XET=1` are forced
  during the node (and restored afterwards). After uploading, the first staged file's hub SHA is
  compared against the local bytes for both repos.
- **Upload stall (spec failure mode 8)**: the automatic retry only covers `upload_large_folder`
  calls that *fail with an exception* — between attempts the resumable-upload cache
  `~/.cache/huggingface/upload` is cleared and HF dedups by SHA. A genuine silent **hang** on the
  final shard has no in-process watchdog: kill the pipeline process manually, run
  `rm -rf ~/.cache/huggingface/upload`, then rerun the same command — server-side SHAs resume and
  only missing shards re-transfer.

## Tests

```bash
cd /work/markhsp/Embodied_Claw
pipeline/.venv/bin/python -m pytest pipeline/tests/ -q
```

All subprocess / HuggingFace calls are mocked; tests never touch GPUs, the network, or real
dataset/checkpoint paths.
