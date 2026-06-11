"""Benchmark adapters. Each module exposes: NAME, MIN_FREE_MB, list_tasks(), task_configs(),
shard_cmd(...), parse_shard_result(...), runtime_check_cmd(gpu_id). The FSM never hardcodes a
benchmark — adding one = adding a module here and naming it in the request."""
from __future__ import annotations

from importlib import import_module


def get(name: str):
    return import_module(f"eval_domino.benchmarks.{name}")
