# training_agent long-term memory

- [gotcha:nccl-nvls](nccl_nvls.md) — this host needs NCCL_NVLS_ENABLE=0 (CUDA failure 401 at nvls.cc otherwise); already exported in TRAIN_ENV. If preflight still fails, escalate — out of known workarounds.
- [gotcha:num-workers](dataloader.md) — openpi num_workers=2 starves 8 GPUs; insert TrainConfig with num_workers=48 (the template does).
- [convention:wandb](wandb.md) — wandb_enabled=False unless the user asks; if true, WANDB_API_KEY must be set or escalate.
- [convention:final-ckpt](ckpt.md) — final save lands at step N-1 (59999 for a 60k run): checkpoints/<config>/<exp>/<step>/{params,_CHECKPOINT_METADATA}. Healthy end-of-run loss ~0.0035-0.008.
- [convention:norm-stats](norm.md) — always --max-frames 50000 (full pass takes 60+ min).
