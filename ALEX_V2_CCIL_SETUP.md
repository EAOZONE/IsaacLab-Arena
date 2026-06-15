# Isaac Lab-Arena — Alex V2 + CCIL Setup Guide

End-to-end setup for running **Alex V2 (ability-hands) simulations** in Isaac Lab-Arena and
training/evaluating a **CCIL** behavioral-cloning policy on the `alex_open_microwave` task.

This is the "get my machine working from scratch" guide. It assumes the layout on this host
(`bpratt@Legion`) but calls out the generic defaults so it transfers to another box.

---

## 0. What you're setting up

Three pieces that work together:

| Piece | Where it runs | Purpose |
|---|---|---|
| **Arena Docker container** | local clone, `--net=host` | Isaac Sim + Arena package: sim, eval, data convert |
| **Alex V2 ability-hands embodiment** | inside the container | the robot model, mounted from the IHMC Alex SDK |
| **CCIL training** | a *separate* Python 3.8 conda env | trains the BC policy offline; only a TorchScript artifact crosses into Arena |

CCIL trains **outside** the container (it needs a `d3rlpy` fork pinned to Python 3.8, incompatible
with Isaac Sim's interpreter). Only a dependency-free `policy.pt` is handed back to Arena for
closed-loop evaluation.

---

## 1. Host prerequisites

- Docker + the NVIDIA Container Toolkit (GPU passthrough).
- An NVIDIA GPU. (This box: 16 GB — enough for one sim *or* one GR00T server, not both at once.)
- `pre-commit` on the host (lint/format hooks run on the host, not in the container):
  ```bash
  pre-commit install     # run once per fresh clone, from the repo root
  ```
- The **IHMC Alex SDK** cloned on the host (provides the Alex V2 description + Ability Hand URDFs):
  - This box: `~/ws_alex/repository-group/ihmc-alex-sdk`
  - Must contain `alex-models/alex_V2_description/` and the Ability Hand models.

---

## 2. Host directory layout & mounts

`run_docker.sh` mounts three host dirs into the container. Defaults are under `$HOME`; this box
overrides `-m` to point at the Alex SDK:

| Host path (this box) | Container path | `run_docker.sh` flag | Holds |
|---|---|---|---|
| `~/datasets` | `/datasets` | `-d` (default `$HOME/datasets`) | demos, CCIL pickles, policies |
| `~/ws_alex/repository-group/ihmc-alex-sdk` | `/models` | `-m` (default `$HOME/models`) | Alex V2 description + Ability Hands |
| `~/eval` | `/eval` | `-e` (default `$HOME/eval`) | eval videos/outputs (optional) |
| `~/IsaacLab-Arena` | `/workspaces/isaaclab_arena` | (always) | the repo itself |

> A mount is only added **if the host dir exists** — create `~/eval` first if you want it.

Relevant `/datasets` contents for the microwave task:
```
/datasets/alex_microwave/
├── demo.hdf5                      # raw Arena teleop demos (50 trajectories)
├── demo/lerobot/                  # LeRobot-format copy of the demos
└── ccil/
    ├── alex_microwave.pkl         # demos converted to CCIL format
    ├── ccil_bc_meta.json          # verification metadata for Arena
    ├── policy_naive.pt            # plain BC baseline (no augmentation)
    └── policy_ccil.pt             # BC on CCIL-augmented data
```

---

## 3. Build & start the Arena container

Don't hardcode the container name — each clone gets its own (`isaaclab_arena-<suffix>`, suffix =
repo dir name after `IsaacLab-Arena`). On this box it is `isaaclab_arena-latest`.

```bash
cd ~/IsaacLab-Arena

# First time / rebuild: -m points at the Alex SDK so /models has the robot description.
./docker/run_docker.sh -m ~/ws_alex/repository-group/ihmc-alex-sdk

# Useful flags:
#   -r   rebuild the image
#   -R   rebuild without cache
#   -g   also install GR00T N1.6 deps (only if you'll run GR00T)
#   -d / -e   override the datasets / eval mounts
```

`run_docker.sh` runs the container with `--net=host` and `--ipc=host`. **`--net=host` matters**:
`localhost` inside the container is the host, so a policy server bound to `127.0.0.1` on the host
is reachable from inside without rebinding.

### Running commands in the container

Inside the container, `python` is aliased to `/isaac-sim/python.sh`. From the host, exec as your
user (not root) and use the explicit interpreter path:

```bash
ARENA=isaaclab_arena-latest    # or discover via the dev-container skill
docker exec "$ARENA" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh <script.py> ..."
```

---

## 4. Verify the Alex V2 sim

The Alex model is auto-discovered from `/models`. Resolution order (`alex.py:_alex_models_dir`):
1. `$ALEX_MODELS_DIR` if set inside the container, else
2. `/models` or `/models/alex-models` (so either the SDK root or `alex-models` alone works as `-m`).

Ability Hand URDFs are found under the SDK automatically, or set `$ABILITY_HAND_MODELS_DIR`.

**Embodiment names (Alex V2):**
- `alex_v2_ability_hands` — Pink-IK embodiment (first 14 action dims = wrist-pose targets).
- `alex_v2_ability_hands_joint_pos` — **34-DOF absolute joint-position** embodiment. **Use this for
  CCIL** (the policy outputs joint positions, not IK targets).

Quick visual smoke test (Kit viewer):
```bash
docker exec "$ARENA" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
   isaaclab_arena_environments/alex_playground_environment.py \
   --viz kit --enable_cameras --embodiment alex_v2_ability_hands_joint_pos"
```

If the model is missing you'll get an assertion naming the expected
`alex_V2_description` path under `/models` — fix the `-m` mount and re-run.

---

## 5. The microwave task

`alex_open_microwave` (`isaaclab_arena_environments/alex_open_microwave_environment.py`):
- Task: `OpenDoorTask` on a microwave (Lightwheel asset, cached under
  `~/.cache/lightwheel_sdk/`).
- Reset: microwave starts **20% open**; **success = ≥80% open** (`openness_threshold=0.8`).
- Episode length: **`100/30 ≈ 3.33 s`** (hardcoded; override at eval with `--episode_length_s`).
- Success metric: `SuccessRateMetric` via the `success` termination.

---

## 6. CCIL training & evaluation

Full detail lives in **`isaaclab_arena_ccil/training/README.md`** — this is the condensed flow.

### 6a. Convert demos → CCIL pickle (in the container; needs only h5py + numpy)
```bash
/isaac-sim/python.sh isaaclab_arena_ccil/data/convert_hdf5_to_ccil.py \
  --hdf5_file /datasets/alex_microwave/demo.hdf5 \
  --out_file  /datasets/alex_microwave/ccil/alex_microwave.pkl
```
Or via the driver: `HDF5=/datasets/alex_microwave/demo.hdf5 bash isaaclab_arena_ccil/training/run_ccil_pipeline.sh convert`

Produces a list of `{"observations": (T,49), "actions": (T,34)}` dicts — 50 trajectories,
**2,385 `(state, action)` datapoints** total (~48 timesteps each at 30 fps).

### 6b. Set up the CCIL Python 3.8 env (offline, once)
```bash
git clone https://github.com/personalrobotics/CCIL.git && cd CCIL
conda create -n CCIL python=3.8.10 -y && conda activate CCIL
pip install "pip<24.1"      # gym==0.19.0 has legacy metadata newer pip rejects
pip install -e .            # after trimming install_requires — see CCIL README §2
```
Copy `alex_microwave.pkl` to this machine; the config uses an absolute path to it.

### 6c. Train (run from the CCIL repo root; config is positional with `key value` overrides)

Config: `isaaclab_arena_ccil/training/alex_microwave.yml` (copy into CCIL's `config/`).

**Baseline — plain BC (no augmentation):**
```bash
python correct_il/train_bc_policy.py config/alex_microwave.yml policy.naive true
```

**CCIL — augmented BC (the contribution):**
```bash
python correct_il/train_dynamics_model.py config/alex_microwave.yml   # Lipschitz dynamics model
python correct_il/gen_aug_label.py        config/alex_microwave.yml   # corrective synthetic labels
python correct_il/train_bc_policy.py       config/alex_microwave.yml   # BC on real + synthetic
```

Outputs land under `output/alex_microwave/seed42/alex_open_microwave/`. The TorchScript `policy.pt`
**bakes its scalers onto `cuda:0`** — it must be loaded and run on CUDA.

### 6d. Export verification metadata
```bash
cp output/alex_microwave/seed42/alex_open_microwave/policy/.../policy.pt \
   /datasets/alex_microwave/ccil/policy.pt
python isaaclab_arena_ccil/training/export_bc_to_torch.py \
  --policy_pt /datasets/alex_microwave/ccil/policy.pt \
  --pickle    /datasets/alex_microwave/ccil/alex_microwave.pkl \
  --out_meta  /datasets/alex_microwave/ccil/ccil_bc_meta.json
```

### 6e. Evaluate in Arena (in the container)

**Argument order matters**: the env name is an argparse subparser — global/policy args go
**before** `alex_open_microwave`, env args (`--embodiment`) after.

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --device cuda --enable_cameras --viz kit \
  --num_episodes 20 \
  --policy_type isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy \
  --model_path /datasets/alex_microwave/ccil/policy.pt \
  --meta_path  /datasets/alex_microwave/ccil/ccil_bc_meta.json \
  --policy_device cuda \
  alex_open_microwave \
  --embodiment alex_v2_ability_hands_joint_pos
```

The CCIL policy is **state-based** (49-D joint state in → 34-D joint targets out), TorchScript with
scalers baked in — no joint remapping, no inference server.

---

## 7. CCIL configuration reference (`alex_microwave.yml`)

| Group | Key | Value | Meaning |
|---|---|---|---|
| dynamics | `lipschitz_type` | `soft_sampling` | Lipschitz penalty style |
| dynamics | `lipschitz_constraint` | **2.0** | bound on the dynamics slope → how far off-data you can trust it |
| dynamics | `soft_lipschitz_penalty_weight` | 5e-4 | penalty strength |
| dynamics | `layers` / `activation` | `[512,512]` / relu | dynamics MLP |
| aug | `type` | `backward_euler` | corrective-label integration scheme |
| aug | `epsilon` | **10.0** | augmentation step size (how far `ŝ` deviates from data) |
| aug | `num_labels` | 20 | candidate labels per anchor (filtered; see note) |
| aug | `max_iter` / `delta` | 50 / 1e-5 | corrective solve budget / tolerance |
| policy | `lr` / `batch_size` / `train_epochs` | 1e-3 / 256 / 200 | BC training |

**Datapoint accounting:** a *demonstration* is one teleop trajectory (you have **50**); a
*datapoint* is one `(state, action)` timestep (**2,385** total). CCIL augmentation produces
corrective `(perturbed_state, recovery_action)` datapoints anchored to the real ones — **not** a
fixed multiple. Observed counts:

| Lipschitz | Aug method | Synthetic datapoints |
|---|---|---|
| L = 2.0 | backward_euler | **2,385** (≈1:1 with real) |
| L = 3.0 | forward_euler | **1,999** |

(The `num_labels: 20` config vs. ~1 retained label per anchor is unresolved — `gen_aug_label.py`
filters candidates; read it if you need the exact definition.)

---

## 8. Evaluation & robustness tooling

These eval flags (in `policy_runner.py`) are useful for stress-testing any policy on the task:

- **`--episode_length_s <s>`** — shorten episodes so they reset sooner (overrides the env's 3.33 s).
- **Perturbation ("poke")** — apply an external wrench to a robot link mid-rollout to test recovery:
  ```bash
  --poke --poke_body RIGHT_GRIPPER_Z_LINK \
  --poke_force -5 -5 -5 --poke_torque -10 -10 -10 \
  --poke_start_step 40 --poke_duration 10
  ```
  - `RIGHT_GRIPPER_Z_LINK` = the **hand** (right EEF). `RIGHT_WRIST_Z_LINK`/`RIGHT_ELBOW_Y_LINK` are
    further up the arm.
  - A **red arrow** marker shows where/when the poke fires (disable with `--no-poke_marker`).
  - The poke **ramps per episode**: episode *k* applies *k×* the base wrench (and a bigger arrow),
    so one 20-episode run sweeps disturbance magnitude — handy for a Lipschitz-vs-recovery study.
- **Video**: `--video --video_dir <dir>` records the viewport (works headless with `--headless`).

---

## 9. Tests

Three required phases, run **inside** the container (see the `run-tests` skill):
```bash
# no-cameras, with-cameras, with-subprocess — run all three before pushing
docker exec "$ARENA" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest <...>"
```

---

## 10. Gotchas (learned the hard way)

- **`--policy_device cuda` is mandatory for CCIL** — d3rlpy bakes the scaler constants onto
  `cuda:0`; loading on CPU breaks the policy.
- **Eval is deterministic.** The microwave resets to the *same* pose/openness every episode and
  `seed=None`, so a deterministic policy gives identical episodes — "25 episodes" is really 1 trial
  ×25 (success rate is binary). Add initial-state randomization for a real `k/N` success rate.
- **Check which `.pt` you're evaluating.** `policy.pt`, `policy_naive.pt`, `policy_ccil.pt` can be
  distinct exports (different md5s) — verify by comparing outputs on a fixed input before trusting
  results. A mislabel will silently invert your conclusion.
- **VRAM**: this box's 16 GB fits *one* of {Isaac Sim eval, GR00T server}. Free one before the other.
- **Argument ordering**: global/policy args before the env name, env args after.

---

## 11. GR00T on the same task

An alternative to CCIL: fine-tune **NVIDIA GR00T N1.6** (a 3B vision-language-action model) on the
same Alex demos and evaluate it closed-loop. Unlike CCIL (state-based, local TorchScript), GR00T is
**vision-based** (reads the stereo ZED cameras) and runs as a **two-process** setup — an inference
**server** that holds the model and an Arena **client** that drives the sim.

Pipeline: HDF5 demos → LeRobot dataset → fine-tune → checkpoint → serve → evaluate.

### 11a. Convert demos → LeRobot (in the container)

GR00T trains on the LeRobot format, not the CCIL pickle. The conversion reads stereo ZED RGB plus
`robot_joint_pos`/`processed_actions`:
```bash
/isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/alex_open_microwave_config.yaml
```
Edit the YAML for your paths: `data_root`, `hdf5_name`, `language_instruction`
(default `"Open the microwave."`). Output: `<data_root>/<hdf5_stem>/lerobot/`. Push that dataset to
a HuggingFace repo (e.g. `H2Ozone/alex_microwave`) for the training image to pull, or point local
training at it directly.

### 11b. Fine-tune

Base model: `nvidia/GR00T-N1.6-3B`. Modality config:
`isaaclab_arena_gr00t/embodiments/alex/alex_data_config.py`. Two routes:

**Route A — self-contained Docker image (any machine, no local setup).** Downloads code, dataset,
and base model, fine-tunes, and uploads the checkpoint to HuggingFace:
```bash
docker build -t alex-gr00t-train \
  https://github.com/EAOZONE/IsaacLab-Arena.git#main:isaaclab_arena_gr00t/training/docker

docker run --gpus all --shm-size 16g \
  -e HF_TOKEN=hf_xxx \                          # needs WRITE access (verified before training)
  -v alex_hf_cache:/cache/huggingface \         # keeps the ~6 GB base model across runs
  -v "$PWD/checkpoints:/checkpoints" \           # auto-resumes from last checkpoint here
  alex-gr00t-train
```
Knobs (all via `-e`): `HF_DATASET_ID` (`H2Ozone/alex_microwave`), `HF_MODEL_REPO`
(`H2Ozone/alex_open_microwave_gr00t`), `MAX_STEPS`/`SAVE_STEPS` (`30000`/`5000`), `GLOBAL_BATCH_SIZE`,
`NUM_GPUS`, `LOW_VRAM=1` (≤16 GB GPUs: diffusion-head-only + grad-accum), `SKIP_UPLOAD=1`. Only the
latest `checkpoint-N` is uploaded, under `checkpoint-N/` in the model repo. See
`isaaclab_arena_gr00t/training/docker/README.md` (incl. Apptainer/SLURM for clusters).

**Route B — local single GPU** against the `submodules/Isaac-GR00T` checkout (GR00T's own env):
```bash
export DATASET_PATH=<data_root>/<hdf5_stem>/lerobot
export OUTPUT_DIR=~/models/alex_open_microwave_finetune
bash isaaclab_arena_gr00t/training/alex_finetune_single_gpu.sh
```
Honors `MAX_STEPS`, `SAVE_STEPS`, `BASE_MODEL_PATH`, `LOW_VRAM=1` (uses
`launch_finetune_low_vram.py` with gradient checkpointing). Writes `checkpoint-N/` under `OUTPUT_DIR`.

### 11c. Get the trained checkpoint

If the model was pushed to HF, pull it (downloads into the HF hub cache — **not** a `git clone`, so
no `git lfs pull` needed):
```bash
MODEL="$(hf download H2Ozone/alex_open_microwave_gr00t)/checkpoint-30000"
```
`hf download` prints/returns the cached snapshot path; the real checkpoint is the **`checkpoint-N/`
subfolder** (it holds `config.json`, `experiment_cfg/`, the safetensors shards). Don't point at the
snapshot root, and don't `cp` the snapshot elsewhere — that breaks the cache's relative symlinks.

### 11d. Start the inference server (GR00T env, on the host)

```bash
cd submodules/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
  --model_path="$MODEL" \
  --embodiment_tag=NEW_EMBODIMENT \
  --host=127.0.0.1 --port=5555
```
Loads ~7 GB. Leave it running. `127.0.0.1` is fine because the Arena container is `--net=host`.

### 11e. Evaluate in Arena (in the container)

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --device cuda --enable_cameras --viz kit \
  --num_episodes 20 \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/alex_manip_gr00t_closedloop_config.yaml \
  --policy_device cuda \
  --remote_host localhost --remote_port 5555 \
  --num_envs 1 \
  alex_open_microwave \
  --embodiment alex_v2_ability_hands_joint_pos
```
The config carries the language instruction, joint-space maps, and `pov_cam_name_sim`
(`zed_left_cam_rgb`/`zed_right_cam_rgb`) the policy reads — so `--enable_cameras` is required and the
embodiment must expose the ZED cameras (`alex_v2_ability_hands_joint_pos` does). Override the prompt
with `--language_instruction "…"`. The same `--poke` / `--episode_length_s` robustness flags apply.

> **VRAM**: the server (~7 GB) and Isaac Sim eval share the GPU here. On 16 GB, stop other GPU users
> (e.g. a CCIL eval) first, or run the server on a second GPU/host and set `--remote_host` to its IP.

### GR00T vs CCIL at a glance

| | CCIL | GR00T |
|---|---|---|
| Input | 49-D joint state | stereo ZED RGB + state |
| Model | small MLP, TorchScript | 3B VLA |
| Runtime | in-container, pure PyTorch | server + client (2 processes) |
| Training env | py3.8 CCIL conda env | GR00T env / Docker image |
| Artifact into Arena | `policy.pt` + meta | nothing — talks to server |
