import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # /work/markhsp/Embodied_Claw
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import nodes  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return {
        "run_id": "testrun",
        "dataset_name": "b1k_test",
        "train_config_name": "pi05_b1k_test",
        "hf_model_repo": "Org/model",
        "hf_dataset_repo": "Org/dataset",
        "sources": [
            {"hf_repo": "org/official", "repo_type": "dataset", "kind": "snapshot",
             "allow_patterns": ["data/*"], "filename": None, "local_dir": str(tmp_path / "official")},
            {"hf_repo": "org/perturb", "repo_type": "dataset", "kind": "single_file_zip",
             "allow_patterns": None, "filename": "out.zip", "local_dir": str(tmp_path / "perturb")},
        ],
        "builder_script": "scripts/build_test.py",
        "pca_thresh": 4.83,
        "dup_factor": 5,
        "train_script": "scripts/train_test.sh",
        "num_train_steps": 60000,
        "save_interval": 10000,
        "batch_size": 32,
        "peak_lr": 2.5e-5,
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every real-world path constant in nodes.py into tmp_path."""
    for name in ("LEROBOT_ROOT", "CKPT_ROOT", "LOGS_DIR", "PCA_DIR", "STAGING_ROOT", "UPLOAD_CACHE",
                 "TMP_LOG_DIR"):
        p = tmp_path / name.lower()
        p.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(nodes, name, p)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return tmp_path
