# Alex GR00T fine-tuning Docker

Fully self-contained training pipeline: on **any** machine with Docker and the
NVIDIA container toolkit, two commands download everything (Isaac-GR00T code,
the Alex configs from the Arena fork, the LeRobot dataset, and the
`nvidia/GR00T-N1.6-3B` base model), fine-tune, and upload the resulting
checkpoint to HuggingFace. Nothing is needed from your own computer.

## On a new machine

```bash
# 1. Build (one-time per machine, ~30+ min — flash-attn compiles from source)
docker build -t alex-gr00t-train \
  https://github.com/EAOZONE/IsaacLab-Arena.git#main:isaaclab_arena_gr00t/training/docker

# 2. Train + upload
docker run --gpus all --shm-size 16g \
  -e HF_TOKEN=hf_xxx \
  -v alex_hf_cache:/cache/huggingface \
  -v "$PWD/checkpoints:/checkpoints" \
  alex-gr00t-train
```

- `HF_TOKEN` needs **write** access; it is verified (and the model repo
  created, private) *before* training starts, so a bad token fails in seconds,
  not hours.
- Mounting `/checkpoints` is recommended: the trainer auto-resumes from the
  last checkpoint there if the container is restarted.
- The `alex_hf_cache` volume keeps the ~6 GB base model across runs.
- To skip the 30-min build on each new machine, build once anywhere, then
  `docker tag` + `docker push` to Docker Hub and `docker pull` elsewhere.

The Alex embodiment configs are cloned from
`https://github.com/EAOZONE/IsaacLab-Arena` **at build time** — push your
config changes first, then rebuild with
`--build-arg ARENA_REF=$(git rev-parse HEAD)` (changing the build-arg also
busts the cached clone layer; plain rebuilds reuse the old cached checkout).

## Knobs (all via `-e`)

| Variable | Default | Meaning |
|---|---|---|
| `HF_DATASET_ID` | `H2Ozone/alex_microwave` | LeRobot dataset repo to download |
| `HF_MODEL_REPO` | `H2Ozone/alex_open_microwave_gr00t` | model repo to upload to |
| `SKIP_UPLOAD` | `0` | `1` = train only, no upload |
| `LOW_VRAM` | `0` | `1` = diffusion head only, batch 2 + grad-accum (≤16 GB GPUs) |
| `GLOBAL_BATCH_SIZE` | `8` (`2` if LOW_VRAM) | effective batch size |
| `MAX_STEPS` / `SAVE_STEPS` | `30000` / `5000` | training length / checkpoint cadence |
| `NUM_GPUS` | `1` | GPUs to train on |
| `UPLOAD_OPTIMIZER_STATE` | `0` | `1` = also upload optimizer/scheduler state |
| `WANDB_API_KEY` + `WANDB_MODE=online` | disabled | enable wandb logging |

Only the latest `checkpoint-N` is uploaded, under `checkpoint-N/` in the model
repo, with optimizer state stripped by default.

## Relation to the other training paths

- `alex_finetune_single_gpu.sh` — local host training against the
  `submodules/Isaac-GR00T` checkout (same pinned commit as this image).
- `alex_colab_finetune.ipynb` — Colab; this image replaces it for any machine
  where you can run Docker.
