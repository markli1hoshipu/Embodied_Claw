# data_agent long-term memory

- [convention:trim-task0](trim.md) — task-0 default skills_keep = {1, 2, 67} (move-to, pick-up, press); drop the place-on tail (skill_id 3).
- [gotcha:hf-xet](hf.md) — HF downloads run with HF_HUB_DISABLE_XET=1 and hf_transfer off; Xet 429s hard otherwise. Skills already retry 429 with backoff — escalate only after they give up.
- [layout:b1k-raw](layout.md) — official trees: data/task-XXXX/episode_NNNNNNNN.parquet, annotations/task-XXXX/episode_NNNNNNNN.json (skill_annotation segments with skill_id=[int], frame_duration=[s,e)), videos/task-XXXX/<video_key>/. Perturb zips extract to <zip-stem>/data/task-0000/.
- [gotcha:v3-truncated-videos](videos.md) — perturb v3 source had videos shorter than parquets for many episodes; always cap at the SHORTEST RGB stream (build_curated_dataset does this).
