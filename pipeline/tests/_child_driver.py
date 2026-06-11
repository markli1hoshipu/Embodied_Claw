"""Subprocess driver for the kill-mid-train resume test (run with repo root on sys.path).

usage: python pipeline/tests/_child_driver.py <crash|resume> <run_id> <trace_file>
env:   EMBODIED_CLAW_RUNS / _AGENTS / _CONFIG / _CACHE point at the test tmp dir.

crash : happy-path through nodes 0-4, then os._exit(17) MID-train-node (simulates kill -9).
resume: invoke(None) on the same sqlite — only the dead node may re-execute.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.agents import base, training_agent  # noqa: E402
from pipeline.graph import build_graph, open_checkpointer  # noqa: E402
from pipeline.state import STAGES, init_state  # noqa: E402
from pipeline.tests import fakes  # noqa: E402

mode, run_id, trace = sys.argv[1], sys.argv[2], sys.argv[3]
LAUNCH = fakes.tool_use("launch_detached_train",
                        {"script_path": "s.sh", "log_path": "t.log", "config_name": "cfg"})

if mode == "crash":
    training_agent.REGISTRY["launch_detached_train"] = lambda **kw: os._exit(17)
    script = fakes.default_script(intake=fakes.intake_script(run_id), train=[[LAUNCH]])
else:
    training_agent.REGISTRY["launch_detached_train"] = \
        lambda **kw: {"attached": False, "pid": 4242}
    script = {"train": [fakes.done(paths=("/tmp/ckpt/59999",))],
              "upload_model": [fakes.done(paths=("hf://m",))]}

fake = fakes.FakeAnthropic(script, trace_file=trace)
base.get_client = lambda: fake
graph = build_graph(open_checkpointer(run_id))
conf = {"configurable": {"thread_id": run_id}}
graph.invoke(init_state(run_id) if mode == "crash" else None, conf, durability="sync")
vals = graph.get_state(conf).values
print("FINISHED " + json.dumps({s: vals[s]["status"] for s in STAGES}))
