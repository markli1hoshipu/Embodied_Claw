# hf_agent long-term memory

- [user-pref:naming](naming.md) — datasets: Hoshipu/b1k_<task>_<variant> (e.g. Hoshipu/b1k_perturb_recovery3_task0_curated); models: Hoshipu/pi05-b1kt0-<variant>-lr2.5e5 (e.g. Hoshipu/pi05-b1kt0-perturbrec3-lr2.5e5).
- [gotcha:upload-env](upload.md) — hf_transfer OFF (silent stalls) and Xet OFF (429s) for all uploads; train_state/ always excluded; ckpt assets/norm_stats.json injected per step AFTER the bulk upload.
- [gotcha:final-shard-stall](stall.md) — upload_large_folder can hang silently on the final shard; the resilient skill watches for no-progress, kills, clears ~/.cache/huggingface/upload, retries (server-side SHA dedup makes reruns cheap).
- [convention:staged-steps](steps.md) — upload steps {save_interval..num_train_steps}; disk dir N-1 stages as ckpt-N (59999 -> ckpt-60000).
