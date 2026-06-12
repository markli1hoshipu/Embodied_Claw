# Embodied Claw — agentic pipeline

LangGraph state machine (7 nodes) where the work inside each node is done by one of three
long-lived Claude agents (`data_agent`, `training_agent`, `hf_agent`) with tool access to the
section-5 skills. Single source of truth: `/work/markhsp/openpi/scripts/PIPELINE_AGENTIC_SPEC.md`.

```
intake -> ingest -> filter_build -> { upload_dataset || norm_stats } -> train -> upload_model
 (data)    (data)      (data)          (hf)            (training)     (training)   (hf)
```

## Install

The repo-root venv is already set up (`/work/markhsp/Embodied_Claw/.venv`, python 3.11:
anthropic 0.109.0, langgraph 1.2.4 + sqlite saver, huggingface_hub, pytest). Heavy data-build
skills (`slice_and_renumber`, `build_curated_dataset`) lazily import pyarrow/numpy — install
them into the venv (`.venv/bin/pip install pyarrow numpy`) or rely on the skills that shell out
to the `xvla-stable` conda env (downloads, PCA, norm stats, training all do).

## Configure

- Agent auth — two interchangeable backends, picked by `pipeline.agents.base.make_backend`:
  - **claude-agent-sdk** (default when no key is set): subscription auth from the logged-in
    Claude Code CLI, same as eval_domino and the Slack operators. No API key needed.
    Escalations block in-tool on the mailbox (crash-safe: re-attach on node re-run).
  - **raw anthropic SDK**: used when `ANTHROPIC_API_KEY` is exported. Escalations use
    LangGraph interrupt/resume. Force either with `EMBODIED_CLAW_BACKEND=sdk|api`.
- `export HF_TOKEN=...` — required by ingest/upload skills. Never hardcoded.
- Model: defaults to `claude-sonnet-4-6`. Override globally `EMBODIED_CLAW_MODEL=...` or
  per-agent `EMBODIED_CLAW_MODEL_DATA_AGENT=claude-opus-4-8` etc. Adaptive thinking is on;
  `EMBODIED_CLAW_THINKING=off` disables. `EMBODIED_CLAW_MAX_ITER` caps the tool loop (40).
- Notifications: `config/notifications.toml` (`[notify] command`, `inbox_log`). Defaults:
  `notify-send` + append to `~/.cache/embodied_claw/inbox.log`.
- Path overrides (used by tests, useful for sandboxes): `EMBODIED_CLAW_RUNS`,
  `EMBODIED_CLAW_AGENTS`, `EMBODIED_CLAW_CONFIG`, `EMBODIED_CLAW_CACHE`.

## Run

```bash
cd /work/markhsp/Embodied_Claw
# request from a file you wrote:
mkdir -p runs/my_run && $EDITOR runs/my_run/request.txt
.venv/bin/python -m pipeline.cli run my_run
# or inline:
.venv/bin/python -m pipeline.cli run my_run --request "Train a pi0.5 on task-0 ... p98 ... 5x"
# detached (what the Slack bridge spawns); equivalent to: setsid .venv/bin/python -m pipeline.cli run my_run </dev/null >runs/my_run/driver.log 2>&1 &
.venv/bin/python -m pipeline.cli run my_run --detach
```

The driver is restart-safe: rerunning `run <run_id>` resumes from the sqlite checkpoint —
completed nodes never re-execute; a node killed mid-flight replays from its persisted agent
conversation; non-succeeded stages of a finished run are retried.

## Reply to escalations

Agents never guess on ESCALATE triggers — they write
`runs/<run_id>/escalations/<node>_<ts>_<uuid>.question.json`, notify, and the graph pauses
(checkpointed; survives kills). Reply via any of:

```bash
.venv/bin/python -m pipeline.cli inbox                                  # list pending
.venv/bin/python -m pipeline.cli reply --run-id my_run --node filter_build --option 1
.venv/bin/python -m pipeline.cli reply --run-id my_run --node filter_build --message "p98 but force-drop 280"
.venv/bin/python -m pipeline.cli reply --latest --message "go"
echo "1" > runs/my_run/escalations/<escalation_id>.reply.txt            # plain file drop
```

A reply file whose content is a single integer selects that option id; anything else is a
free-form message. The driver polls every 30s (`--poll`); after 24h it re-notifies and keeps
waiting — no auto-defaults, ever. The Slack bridge talks to the same files; the pipeline never
imports bridge code.

## Inspect state

- `runs/<id>/transitions.jsonl` — every stage status change (running/escalated/succeeded/...).
- `runs/<id>/agent_messages/<node>.jsonl` — full per-node Claude transcript (tool calls included).
- `runs/<id>/state.sqlite` — LangGraph checkpoint (thread_id == run_id).
- `runs/<id>/artifacts.json`, `summary.md` — written at run end; `config.json` from intake.
- `pipeline_runs/<id>` — symlink index of all runs.

## Memory

Each agent reads `agents/<name>/memory.md` at every node entry (inlined into the cached system
prompt) and may append lessons via its `append_memory` tool — append-only. To prune, edit the
file by hand (or ask Claude Code to); keep the `- [tag](file) — text` line format.

## Tests

```bash
.venv/bin/python -m pytest pipeline/tests/ -q
```

Fully offline: FakeAnthropic scripts tool_use sequences, subprocess/HfApi/pyarrow are mocked,
mailbox + runs live in tmp dirs. Covers the section-13 DoD: 7-node e2e from request.txt,
kill-mid-train resume (real child process + os._exit), escalated-node resume via file drop,
re-entry double-post guard, parallel fan-out/join, section-9 failure modes, mailbox format.

## Design notes / deviations

- `ask_user` pauses via langgraph `interrupt()` (resume value = parsed reply); the 30s poll
  lives in the CLI driver, so a waiting pipeline can be killed for free and resumed later.
  Stage status `escalated` is therefore recorded in `transitions.jsonl` + the question file,
  NOT in the checkpointed StageStatus while the node waits: `interrupt()` raises before the
  node returns, so `graph.get_state()` shows the stage as `pending` for the whole wait and the
  `escalation` dict (with `user_reply`) lands on the StageStatus only at node completion. This
  is the accepted reading of the sanctioned interrupt() deviation — the file mailbox is the
  contract; inbox/bridge/CLI all consume the files, never the checkpoint.
- Run-scoped agent context lives in `runs/<id>/agent_conv/<agent>.json` (saved every turn),
  not inside LangGraph state as spec section 3 words it. Deliberate: token-heavy transcripts
  replay from the exact pending tool call after a crash, which `state.sqlite` alone could not
  reconstruct. Caveat: manual checkpoint surgery (rollback/fork) does not roll back the
  conversation files — delete `agent_conv/` too if you ever rewind a checkpoint by hand.
- Agents call one tool at a time (`disable_parallel_tool_use`) so a crash mid-tool replays
  deterministically from the persisted conversation.
- `slice_and_renumber` (spec signature, returns a pyarrow Table) stays internal to the
  composite `build_curated_dataset`; the agent-facing `slice_and_renumber` tool is a thin
  wrapper requiring `dst_path` and returning `{"rows", "dst_path"}` (a Table cannot round-trip
  a JSON tool_result).
- Node 0's spec SKILLS list names `confirm_with_user`; intake confirmation flows through the
  generic `ask_user` builtin (functionally identical, also sanctioned by Node 0's TOOLS line).
  The authoritative section-5 skill tables never list `confirm_with_user`.
- Node 2's spec TOOLS line lists Write/Edit "for build scripts"; there is no separate
  write_file/edit_file builtin — bash heredocs (`cat > script.py <<'EOF' ...`) are the
  sanctioned write path for free-form build scripts.
- LOC budget (spec section 13, <=~1800 for "skills + agents + graph + mailbox + CLI"): the
  parenthetical sum (skills, agents/, graph.py, tools.py+inbox.py, cli.py) sits a few percent
  over the tilde'd cap (~1860) and ~2200 counting the templated `nodes/*.py` wrappers and
  `state.py`. The overage is review-mandated safety hardening (unconditional rebuild gates,
  launch/monitor liveness handling, structured train_request pass-through); accepted rather
  than shaved. If it must shrink, the 7 one-screen node files collapse into a single table.

## Entering the FSM at any point (reconcile)

`run` grounds stage status in **on-disk evidence** before invoking the graph (default on):
sources present → ingest seeded; `meta/info.json` with episodes → filter_build; `norm_stats.json`
→ norm_stats; complete final checkpoint (`<steps-1>/params + _CHECKPOINT_METADATA`) → train.
The graph then starts at the first stage the filesystem cannot vouch for — e.g. uploading an
already-trained checkpoint runs only the upload nodes. Uploads are never auto-seeded (proving
them needs the HF API; re-uploads dedup by SHA). The train node additionally carries its own
artifact guard: a complete final checkpoint can never be re-launched (`--overwrite` safety).

```bash
python -m pipeline.cli run <id>                  # auto-reconcile (prints the evidence table)
python -m pipeline.cli run <id> --no-reconcile   # v2 behavior: trust only graph state
python -m pipeline.cli run <id> --from train     # explicit entry point; conflicts need --force
python -m pipeline.cli node upload_model <id>    # ONE node in a one-node graph (own thread_id,
                                                 # same escalation/resume mechanics); --force
                                                 # bypasses unproven upstream requires-gates
```

Reconcile derives `dataset_name` from `outputs.hf_dataset_repo` and `train_config_name` via the
`pi05_<b1k_...>` convention — set both explicitly in config.json when they differ.
