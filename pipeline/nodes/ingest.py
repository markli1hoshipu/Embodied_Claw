"""Node 1 — ingest_data (data_agent, inherits Node 0 context)."""
from pipeline.agents.data_agent import make_agent
from pipeline.nodes import agent_node, cfg_excerpt

SKILLS = ("download_hf_snapshot", "download_hf_zip", "verify_dataset_integrity",
          "inventory_dataset")


def prompt(state: dict) -> str:
    return f"""TASK: download every source in data_request.sources to its deterministic local \
path and verify each download is complete and well-formed. Interpret free-form source \
descriptions; fill missing SourceSpec fields yourself or escalate. Finish with \
inventory_dataset per source, then `ls`-walk /work/markhsp/datasets/ (one level, via bash) and \
include EVERY dataset dir under that root — pre-existing ones too — in the summary table you \
pass to complete_node.

ESCALATE: HF 404 / repeated 429 after backoff (skills return an error dict), file counts \
inconsistent with the repo listing, or a data layout you have never seen (check memory first).

SUCCESS: every source verified ok -> complete_node(status="succeeded", artifact_paths=[the \
local source dirs]).

CONFIG:
{cfg_excerpt(state, 'data_request', 'description')}"""


node = agent_node("ingest", lambda s: make_agent(s, SKILLS, builtins=("bash", "read_file")),
                  prompt, requires=("intake",))
