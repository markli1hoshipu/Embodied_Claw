import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root


@pytest.fixture(autouse=True)
def notify_calls(monkeypatch):
    """No test may exec a real notify-send (tests mock every subprocess): tools.notify is
    replaced everywhere with a recorder. Request this fixture to assert on notifications.
    Also pins the agent backend to the raw-anthropic path — the test harness (FakeAnthropic
    via patch_client) scripts that backend; the sdk backend would shell out to claude-code."""
    calls: list[str] = []
    monkeypatch.setattr("pipeline.tools.notify", lambda summary: calls.append(str(summary)))
    monkeypatch.setenv("EMBODIED_CLAW_BACKEND", "api")
    return calls


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated runs/ + agents/ + cache under tmp; no ANTHROPIC_API_KEY, no HF network."""
    runs = tmp_path / "runs"
    agents = tmp_path / "agents"
    runs.mkdir()
    for a in ("data_agent", "training_agent", "hf_agent"):
        (agents / a).mkdir(parents=True)
        (agents / a / "memory.md").write_text(f"# {a} memory\n- Hoshipu/example-{a}\n")
    monkeypatch.setenv("EMBODIED_CLAW_RUNS", str(runs))
    monkeypatch.setenv("EMBODIED_CLAW_AGENTS", str(agents))
    monkeypatch.setenv("EMBODIED_CLAW_CONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("EMBODIED_CLAW_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_FAKE_test_token")  # fake; nothing hits the network
    return tmp_path


@pytest.fixture
def run_dir(env):
    rd = env / "runs" / "tr1"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "request.txt").write_text("train pi0.5 on task-0 with perturb data, PCA p98, 5x dup")
    return rd


def patch_client(monkeypatch, fake):
    from pipeline.agents import base
    monkeypatch.setattr(base, "get_client", lambda: fake)


@pytest.fixture
def graph_env(run_dir, monkeypatch):
    """Build a graph wired to a FakeAnthropic; returns (graph, conf, fake, run_dir)."""
    from pipeline.graph import build_graph, open_checkpointer
    from pipeline.tests import fakes

    def make(script=None):
        script = script or fakes.default_script(intake=fakes.intake_script("tr1"))
        fake = fakes.FakeAnthropic(script)
        patch_client(monkeypatch, fake)
        graph = build_graph(open_checkpointer("tr1"))
        return graph, {"configurable": {"thread_id": "tr1"}}, fake
    return make


def write_min_config(rd: Path, run_id: str):
    from pipeline.tests.fakes import MINIMAL_CONFIG
    cfg = {**MINIMAL_CONFIG, "run_id": run_id}
    (rd / "config.json").write_text(json.dumps(cfg))
    return cfg
