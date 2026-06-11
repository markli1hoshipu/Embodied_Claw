# benchmark_agent memory (curated 2026-06-11 by operator after the cross-run incident;
# raw history preserved in memory.md.bak-incident-20260611)

## Operating rules (supersede anything you remember differently)
- There is NO auto-respawning harness. Drivers are restarted by the operator only. If you
  believe a driver restart is needed: ask_user, then stop.
- Stay inside YOUR run_id: its shards, its config.json, its GPUs/ports (resources.gpu_ids).
  Other runs' files, processes, GPUs, and ports are off-limits even when you can see the bug.
- If mcp__skills__ tools return "Stream closed": stop working. Do not drive the queue via Bash.

## DOMINO / pi05 server-client facts
- runs/<id>/config.json is re-read every run_pending_shards window and shard launch; harness
  code (eval_domino/) is imported once at driver start. Config = the live data-fix surface.
- Port 9100 is the system node_exporter (foreign, NEVER kill). Shard ports are 9210+gpu_id.
  Per-run GPU/port split must never overlap: two servers on one port means clients can eval
  the WRONG checkpoint and silently poison results recorded as 'done'.
- policy_model_server.py must not import sim deps; libGL exists only in
  /work/markhsp/miniforge3/envs/domino/lib (shard_cmd exports LD_LIBRARY_PATH for the server).
  The pi05 smoke test does not cover the server's import path.
- ModelServer.start() exits the process (os._exit(2)) on bind failure — a server that cannot
  bind must die, not linger holding ~57GB. Keep that patch.
- DOMINO script files are read fresh per shard launch (new processes) — DOMINO-repo fixes act
  without driver restarts. deploy_policy.py must keep jax imports lazy inside get_model()
  (the client env has no jax; module-level `from pi_model import *` breaks every shard).
- Client/server protocol (retrofit 2026-06-11): server dispatches {"cmd", "args"} as
  method(*args); ModelClient duck-types the PI0 surface eval() uses (reset/set_language/
  update_observation_window/get_action, observation_window mirror, pi0_step from yml) and
  raises RuntimeError with the server traceback on {"error"} responses. Cheap preflight with
  no model load: stub-PI0 ModelServer (pi05 venv) + real ModelClient (domino env), localhost.
- Ground truth that survives anything: DOMINO/eval_result/<task>/pi05/<task_config>/<model>/*/
  _metrics.json (parse_shard_result reads the newest per task). A task_plan.json reset costs
  re-run compute, not data. Do NOT hand-edit task_plan.json under a live driver (it rewrites
  the file every window).
- checkpoint_step=0 is falsy — never use `or` defaults on it (bit the zero-shot baseline once;
  harness fixed with `is not None`).
- The orchestrator auto-fails a shard at 3 launches. After a SYSTEMIC fix: preflight outside
  the queue first, then retry_shard a minimum probe set, revive the rest after one clean
  window. retry_shard also revives status=failed shards.
- run_pending_shards/shard_log_tail numeric params (max_minutes/max_bytes) work as integers
  now (tool-schema typing fixed 2026-06-11); calling with run_id only also remains fine.
