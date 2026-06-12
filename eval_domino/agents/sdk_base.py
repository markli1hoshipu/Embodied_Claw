"""Shim: SDKAgent moved to pipeline.agents.sdk_base (the training pipeline now uses the same
subscription-auth backend). Import path kept for eval_domino agents/tests."""
from pipeline.agents.sdk_base import SDKAgent  # noqa: F401
