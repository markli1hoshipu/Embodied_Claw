"""Spec section 9 failure-mode catalog — regression tests (rows not covered elsewhere noted)."""
import re
import subprocess
import types

import pytest

from pipeline import tools
from pipeline.skills import data_skills as d


def test_hf_429_backoff_then_classified(env, tmp_path, monkeypatch):
    """Row 1: HF Xet 429 -> exponential backoff with HF_HUB_DISABLE_XET=1, then a clean error."""
    calls = {"n": 0, "envs": []}

    def sh(cmd, env=None, timeout=None):
        calls["n"] += 1
        calls["envs"].append(env["HF_HUB_DISABLE_XET"])
        return types.SimpleNamespace(returncode=1, stdout="", stderr="HTTP 429 Too Many Requests")
    monkeypatch.setattr(tools, "sh", sh)
    monkeypatch.setattr(d, "BACKOFFS", (0, 0))
    out = d.download_hf_snapshot("r/x", "dataset", None, str(tmp_path / "dl"))
    assert "429" in out["error"] and "backoff exhausted" in out["error"]
    assert calls["n"] == 3 and set(calls["envs"]) == {"1"}  # 1 try + 2 backoff retries, Xet off


def test_pgrep_pattern_never_substring_matches(env, monkeypatch):
    """Row 2: bash watchers/tee carry 'scripts/train.py' in their cmdline; v1 regex contract."""
    seen = {}

    def run(args, capture_output=True, text=True):
        seen["pat"] = args[-1]
        return types.SimpleNamespace(returncode=0, stdout="200738\n200750\n", stderr="")
    monkeypatch.setattr(tools.subprocess, "run", run)
    assert tools.find_train_pids("cfg+v2 (x)") == [200738, 200750]
    pat = seen["pat"]
    assert pat == r"python.*scripts/train\.py.*" + re.escape("cfg+v2 (x)") + "( |$)"
    assert re.search(pat, "python scripts/train.py --config-name cfg+v2 (x) --overwrite")
    assert not re.search(pat, "bash -c tee scripts/train.py.log cfg+v2 (x)2")  # prefix collision
    assert not re.search(pat.replace(re.escape("cfg+v2 (x)"), re.escape("cfg")),
                         "python scripts/train.py --config-name cfg2 ")  # trailing boundary


def test_pgrep_rc_semantics(env, monkeypatch):
    def run_rc(rc, err=""):
        return lambda *a, **kw: types.SimpleNamespace(returncode=rc, stdout="", stderr=err)
    monkeypatch.setattr(tools.subprocess, "run", run_rc(1))
    assert tools.find_train_pids("cfg") == []                 # rc 1 == no match
    monkeypatch.setattr(tools.subprocess, "run", run_rc(2, "fork failed"))
    with pytest.raises(RuntimeError, match="pgrep failed"):
        tools.find_train_pids("cfg")                          # rc>=2 must NEVER read as 'not running'
    assert tools.live_pids("cfg") == [-1]                     # fail-safe sentinel: assume alive
    with pytest.raises(ValueError, match="config_name required"):
        tools.find_train_pids("")        # empty pattern would match EVERY train.py on the host


def test_video_shorter_than_parquet_caps(env, monkeypatch):
    """Row 3 — covered in depth in test_skills_data; assert the ffprobe binary contract here."""
    seen = {}

    def co(args, **kw):
        seen["args"] = args
        return b"42\n"
    monkeypatch.setattr(d.subprocess, "check_output", co)
    assert d.cap_at_video_length(100, "/v.mp4")["effective_rows"] == 42
    assert seen["args"][0] == tools.FFPROBE and "nb_frames" in " ".join(seen["args"])


def test_nccl_401_classify_and_safe_relaunch(env, tmp_path, monkeypatch):
    """Row 4: 401/nvls log -> nccl_nvls -> relaunch path keeps NVLS off and never
    double-launches (attach when a matching pid is alive)."""
    from pipeline.skills import training_skills as t
    assert tools.TRAIN_ENV["NCCL_NVLS_ENABLE"] == "0"
    log = tmp_path / "t.log"
    log.write_text("NCCL WARN cuda failure 401 at nvls.cc:123\n")
    assert t.classify_train_crash(str(log))["category"] == "nccl_nvls"
    monkeypatch.setattr(tools, "live_pids", lambda c: [31337])  # straggler still alive
    out = t.restart_with_workaround("nccl_nvls", "s.sh", str(log), "cfg")
    assert out["attached"] and out["pid"] == 31337  # attach, not duplicate launch


def test_wandb_image_guard_verified_in_preflight_row5(env, tmp_path):
    """Row 5: pre-flight greps scripts/train.py for the `if config.wandb_enabled:` guard
    around the first-batch wandb.Image gather (read-only; openpi is never edited)."""
    from pipeline.skills import training_skills as t
    patched = tmp_path / "train_patched.py"
    patched.write_text("    batch = next(data_iter)\n"
                       "    if config.wandb_enabled:\n"
                       "        images_to_log = [\n"
                       "            wandb.Image(np.concatenate([...], axis=1))\n"
                       "        ]\n")
    out = t.verify_train_script_patched(str(patched))
    assert out["ok"] and "guarded" in out["evidence"]
    unpatched = tmp_path / "train_unpatched.py"
    unpatched.write_text("    batch = next(data_iter)\n"
                         "    images_to_log = [wandb.Image(np.concatenate([...], axis=1))]\n")
    out = t.verify_train_script_patched(str(unpatched))
    assert not out["ok"] and "unguarded wandb.Image" in out["evidence"]
    assert not t.verify_train_script_patched(str(tmp_path / "missing.py"))["ok"]
    # the real openpi script is already patched — the pre-flight must agree (read-only check)
    assert t.verify_train_script_patched()["ok"]
    # and the skill is actually reachable from the train node's tool surface
    from pipeline.agents.training_agent import REGISTRY
    from pipeline.nodes.train import SKILLS
    assert "verify_train_script_patched" in REGISTRY and "verify_train_script_patched" in SKILLS


def test_upload_stall_row6_covered_in_hf_tests():
    """Row 6 (final-shard stall): real coverage lives in test_skills_hf — assert the covering
    tests still exist by name so a rename breaks this cross-reference."""
    from pipeline.tests import test_skills_hf
    assert callable(test_skills_hf.test_upload_resilient_stall_kill_clear_retry)
    assert callable(test_skills_hf.test_upload_resilient_exhausts_and_escalates)


def test_norm_stats_always_sampled_row8(env, tmp_path, monkeypatch):
    from pipeline.skills import training_skills as t
    seen = {}
    monkeypatch.setattr(tools, "sh", lambda cmd, env=None, timeout=None:
                        (seen.update(cmd=cmd, timeout=timeout),
                         types.SimpleNamespace(returncode=1, stdout="", stderr="x"))[1])
    t.compute_norm_stats_fast("cfg", dataset_dir=str(tmp_path))
    assert "--max-frames 50000" in seen["cmd"] and seen["timeout"] == 7200


def test_conda_activation_row7(env, tmp_path, monkeypatch):
    """Row 7: huggingface_hub import errors mean the conda env wasn't activated — every HF
    download command is prefixed with the xvla-stable activation."""
    seen = {}
    monkeypatch.setattr(tools, "sh", lambda cmd, env=None, timeout=None:
                        (seen.update(cmd=cmd),
                         types.SimpleNamespace(returncode=1, stdout="", stderr="boom"))[1])
    monkeypatch.setattr(d, "BACKOFFS", ())
    d.download_hf_zip("r/x", "f.zip", str(tmp_path))
    assert seen["cmd"].startswith(tools.CONDA)
