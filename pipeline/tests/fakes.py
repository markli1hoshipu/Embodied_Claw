"""Test doubles: a scripted FakeAnthropic (no network, no key) + helpers + a fake HfApi.

FakeAnthropic dispatches on the '## NODE: <stage>' tag in the conversation, popping the next
scripted content-block list for that node. Responses validate against the real
anthropic.types.Message model, so the agent loop exercises the genuine pydantic surface.
"""
from __future__ import annotations

import threading
import uuid

import anthropic.types as at


def tool_use(name: str, input: dict | None = None) -> dict:
    return {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": name, "input": input or {}}


def done(status: str = "succeeded", paths: tuple = (), **kw) -> list:
    return [tool_use("complete_node", {"status": status, "artifact_paths": list(paths), **kw})]


class _Messages:
    def __init__(self, fake):
        self.fake = fake

    def create(self, **kw):
        return self.fake._create(**kw)


class FakeAnthropic:
    def __init__(self, script: dict[str, list], trace_file: str | None = None):
        self.script = {k: list(v) for k, v in script.items()}
        self.lock = threading.Lock()
        self.calls: list[dict] = []
        self.trace_file = trace_file
        self.messages = _Messages(self)

    @staticmethod
    def _node_tag(messages) -> str:
        tag = "?"
        for m in messages:
            content = m.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for b in blocks or []:
                if isinstance(b, dict) and b.get("type") == "text" and "## NODE: " in b.get("text", ""):
                    tag = b["text"].split("## NODE: ", 1)[1].split("\n", 1)[0].strip()
        return tag

    def _create(self, **kw):
        node = self._node_tag(kw.get("messages", []))
        with self.lock:
            self.calls.append({"node": node, "kwargs": kw})
            if self.trace_file:
                with open(self.trace_file, "a") as f:
                    f.write(node + "\n")
            queue = self.script.get(node) or self.script.get("*")
            if not queue:
                raise AssertionError(f"FakeAnthropic: no scripted response left for node {node!r}")
            content = queue.pop(0)
        content = content(kw) if callable(content) else content
        registered = {t["name"] for t in kw.get("tools", [])}
        for b in content:  # scripted tool calls must exist in the request's tools param —
            if b.get("type") == "tool_use":  # catches schema-wiring regressions immediately
                assert b["name"] in registered, (
                    f"FakeAnthropic: scripted tool {b['name']!r} is not registered for node "
                    f"{node!r} (request offered {sorted(registered)})")
        stop = "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn"
        return at.Message.model_validate({
            "id": f"msg_{uuid.uuid4().hex[:8]}", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-6", "content": content, "stop_reason": stop,
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})


def default_script(**overrides) -> dict:
    """A 7-node happy-path script: every node goes straight to complete_node."""
    script = {
        "intake": [
            [tool_use("write_run_config", {"run_id": "RID", "config": {}})],  # replaced in tests
            done(paths=("config.json",)),
        ],
        "ingest": [done(paths=("/tmp/src",))],
        "filter_build": [done(paths=("/tmp/lerobot/ds",))],
        "upload_dataset": [done(paths=("https://huggingface.co/datasets/x/y",))],
        "norm_stats": [done(paths=("/tmp/lerobot/ds/norm_stats.json",))],
        "train": [done(paths=("/tmp/ckpt/59999",))],
        "upload_model": [done(paths=("https://huggingface.co/x/z",))],
    }
    script.update(overrides)
    return script


MINIMAL_CONFIG = {
    "run_id": "RID", "description": "raw request",
    "data_request": {"task_description": "task-0 radio", "sources": [],
                     "filter_description": "PCA p98, dup 5x"},
    "train_request": {"base_model": "pi05_base", "num_train_steps": 60000, "batch_size": 32,
                      "peak_lr": 2.5e-5, "fsdp_devices": 8, "save_interval": 10000,
                      "wandb_enabled": False},
    "outputs": {"hf_dataset_repo": "Hoshipu/ds", "hf_model_repo": "Hoshipu/m"},
}


def intake_script(run_id: str) -> list:
    cfg = {**MINIMAL_CONFIG, "run_id": run_id}
    return [[tool_use("write_run_config", {"run_id": run_id, "config": cfg})],
            done(paths=(f"runs/{run_id}/config.json",))]


class FakeHfApi:
    """Stand-in for huggingface_hub.HfApi — records calls, serves canned listings."""

    def __init__(self, files: dict | None = None, raises: dict | None = None):
        self.files = files or {}
        self.raises = raises or {}
        self.calls: list = []

    def list_repo_files(self, repo_id, repo_type="model", **kw):
        self.calls.append(("list_repo_files", repo_id, repo_type))
        if repo_id in self.raises:
            raise self.raises[repo_id]
        if repo_id not in self.files:
            raise FileNotFoundError(f"404 Not Found: {repo_id}")
        return self.files[repo_id]

    def create_repo(self, repo_id, repo_type="model", exist_ok=False, **kw):
        self.calls.append(("create_repo", repo_id, repo_type))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw.get("path_in_repo")))

    def upload_large_folder(self, **kw):
        self.calls.append(("upload_large_folder", kw))

    def get_paths_info(self, repo_id, paths, repo_type="model"):
        self.calls.append(("get_paths_info", repo_id, tuple(paths)))
        return [type("PI", (), {"lfs": None, "blob_id": None})() for _ in paths]
