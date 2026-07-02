# Alex lever GR00T fine-tuning (real robot data)

Train a GR00T N1.6 policy on the real-robot `alex_lever` LeRobot dataset
(joint-space, ZED stereo, ability-hand fingers). The playback script you already
use (`playback_lerobot_dataset.py`) reads the raw v3 layout; training needs the
GR00T episode-per-file layout produced by the converter.

## 1. Prepare the dataset (workstation)

From the repo root, with `ffmpeg` and `pandas`/`pyarrow` on the host:

```bash
bash isaaclab_arena_gr00t/training/prepare_alex_lever_dataset.sh
```

This writes `datasets/alex_lever_gr00t/` and fills the all-zero hand action
columns from measured state (`--action_from_state_dims 13:33`), matching what
the training Docker entrypoint does.

Optional — push to HuggingFace so the cluster can pull instead of rsync:

```bash
export HF_TOKEN=hf_xxx
export HF_DATASET_REPO=<your-org>/alex_lever_real
UPLOAD_HF=1 bash isaaclab_arena_gr00t/training/prepare_alex_lever_dataset.sh
```

## 2. Train on an H100 cluster (recommended)

Build the self-contained training image once (includes Isaac-GR00T, flash-attn,
Alex modality configs, and the v3 converter):

```bash
./isaaclab_arena_gr00t/training/docker/build.sh ghcr.io/<you>/alex-gr00t-train
docker push ghcr.io/<you>/alex-gr00t-train
```

On the cluster:

```bash
# One-time
apptainer pull $SCRATCH/alex-gr00t-train.sif docker://ghcr.io/<you>/alex-gr00t-train:latest
rsync -av datasets/alex_lever_gr00t/ $SCRATCH/alex_lever_gr00t/

# Submit (defaults: 8× H100, batch 64, 15k steps)
export HF_TOKEN=hf_xxx
export HF_MODEL_REPO=<your-org>/alex_lever_gr00t_real
sbatch isaaclab_arena_gr00t/training/cluster/alex_lever_finetune.slurm
```

The job bind-mounts your converted dataset (`SKIP_DOWNLOAD=1`), fine-tunes with
full backbone tuning (`LOW_VRAM=0`), and uploads the latest checkpoint to
`HF_MODEL_REPO`. Resubmit after preemption — training resumes from
`$SCRATCH/alex_lever_checkpoints/`.

**H100 knobs** (export before `sbatch`):

| Variable | Default | Notes |
|---|---|---|
| `NUM_GPUS` | 8 | Match `--gres=gpu:N` in the SLURM header |
| `GLOBAL_BATCH_SIZE` | 64 | 32–128 works well on H100 |
| `MAX_STEPS` | 15000 | 29 real episodes; raise if you add data |
| `SAVE_STEPS` | 2500 | |
| `HF_MODEL_REPO` | — | Where the checkpoint is uploaded |

To pull from HuggingFace instead of rsync, omit `SKIP_DOWNLOAD` and set
`HF_DATASET_ID=<your-org>/alex_lever_real` in the Apptainer env (see
`docker/entrypoint.sh`).

See also `docker/README.md` for Singularity/enroot variants.

## 3. Train locally (single GPU)

For a laptop GPU (≤16 GB), use diffusion-head-only mode:

```bash
bash isaaclab_arena_gr00t/training/prepare_alex_lever_dataset.sh
export DATASET_PATH=datasets/alex_lever_gr00t
export OUTPUT_DIR=~/models/alex_lever_gr00t_finetune
LOW_VRAM=1 bash isaaclab_arena_gr00t/training/alex_finetune_single_gpu.sh
```

## 4. Deploy in Arena

After training, serve the checkpoint from the Isaac-GR00T env:

```bash
cd submodules/Isaac-GR00T
MODEL="$(hf download <your-org>/alex_lever_gr00t_real)/checkpoint-15000"
uv run python gr00t/eval/run_gr00t_server.py \
  --model_path="$MODEL" \
  --embodiment_tag=NEW_EMBODIMENT \
  --host=127.0.0.1 --port=5555
```

Evaluate in the Arena container:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner_cli.py \
  --policy_config isaaclab_arena_gr00t/policy/config/alex_lever_fingers_gr00t_closedloop_config.yaml \
  --policy_host 127.0.0.1 --policy_port 5555 \
  alex_lever_teleop --embodiment alex_v2_lever_fingers_joint_pos
```

Pair with `embodiment_tag: NEW_EMBODIMENT` and
`modality_config_path: isaaclab_arena_gr00t/embodiments/alex/alex_lever_data_config.py`
in the closed-loop config (already set).
