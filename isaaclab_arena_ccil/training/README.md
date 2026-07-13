# CCIL-BC on Isaac Lab Arena (Alex_V2 open-microwave)

Train the [CCIL](https://github.com/personalrobotics/CCIL) behavioral-cloning policy on Arena
teleop demos and evaluate it closed-loop in Arena.

**Why two environments?** CCIL uses a custom `d3rlpy` fork pinned to **Python 3.8**, which is
incompatible with Arena's Isaac Sim interpreter (py3.12/3.13). So training/export happen in a
separate offline env (e.g. on `gpu2`), and only a dependency-free TorchScript artifact crosses
into Arena. Inference in Arena is pure PyTorch — no d3rlpy, no server.

State/action spaces: `robot_joint_pos` (49) in, raw Pink IK `actions` (34) out — left/right
EE target poses (pos 3 + quat 4 each = 14) followed by 20 ability-hand finger joints. The
policy output is fed to an **IK-in-the-loop embodiment** (`alex_v2_ability_hands`), which
resolves the EE targets to a whole-body solution — so the same policy transfers to the real
robot's IK streamer. (To train the legacy direct-joint policy instead, pass
`--action_key processed_actions` to the converter and evaluate with `alex_v2_ability_hands_joint_pos`.)

---

## 1. Convert demos to the CCIL pickle (Arena container)

Only needs `h5py` + `numpy`:

```bash
/isaac-sim/python.sh isaaclab_arena_ccil/data/convert_hdf5_to_ccil.py \
  --hdf5_file /datasets/alex_microwave/demo.hdf5 \
  --out_file  /datasets/alex_microwave/ccil/alex_microwave.pkl
```

Produces a list of `{"observations": (T,49), "actions": (T,34)}` trajectory dicts.

For stereo ZED visuomotor BC, include the recorded camera streams and resize them to
128×128:

```bash
/isaac-sim/python.sh isaaclab_arena_ccil/data/convert_hdf5_to_ccil.py \
  --hdf5_file /datasets/alex_microwave/demo.hdf5 \
  --out_file  /datasets/alex_microwave/ccil/alex_microwave_visual.pkl \
  --image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --image_size 128 128
```

The visual pickle keeps the state/action fields and adds
`images/{zed_left_cam_rgb,zed_right_cam_rgb}` as `(T,3,128,128)` uint8 arrays.

## Convert a LeRobot dataset to CCIL

For LeRobot datasets, convert directly from either a local LeRobot directory or a Hugging
Face dataset repo. For example, `H2Ozone/test_obs_new` stores 48-D `observation.state`,
46-D `action`, and ZED videos named `observation.images.cam_zed_left/right`:

```bash
/isaac-sim/python.sh isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py \
  --repo_id H2Ozone/test_obs_new \
  --out_file /datasets/test_obs_new/ccil/test_obs_new.pkl
```

To include the ZED videos for visual BC:

```bash
/isaac-sim/python.sh isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py \
  --repo_id H2Ozone/test_obs_new \
  --out_file /datasets/test_obs_new/ccil/test_obs_new_visual.pkl \
  --image_keys observation.images.cam_zed_left observation.images.cam_zed_right \
  --output_image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --image_size 128 128
```

## 2. Set up the CCIL Python 3.8 env (offline)

```bash
git clone https://github.com/personalrobotics/CCIL.git && cd CCIL
conda create -n CCIL python=3.8.10 -y && conda activate CCIL
```

CCIL's `setup.py` lists benchmark-env packages we don't use, and one of them
(`gym-pybullet-drones`) has a build backend (`poetry-core@main`) that **fails on Python 3.8**.
For the state-based Alex task, trim `install_requires` to the minimal set actually imported by
the BC / dynamics / augmentation path (verified by grep): keep `d3rlpy` (the fork) +
`torch torch gym scipy PyYAML tqdm tabulate tensorboard matplotlib pygame`; **remove**
`gym-pybullet-drones`, `metaworld`, `f110_gym`, `d4rl`, `mjrl`. Also make the dead `import d4rl`
in `correct_il/utils.py` optional (`try/except ImportError`).

```bash
pip install "pip<24.1"   # gym==0.19.0 has legacy metadata pip>=24.1 rejects
pip install -e .
```

Copy `alex_microwave.pkl` to the CCIL machine (the config below uses an absolute path).

## 3. Train

CCIL scripts take the config as a **positional** arg with dotted `key value` overrides, and are
run from the CCIL repo root. A ready config lives at `config/alex_microwave.yml` (a copy is tracked
in this package as `alex_microwave.yml`). `state_dim=49`/`action_dim=34` are inferred from the
pickle; `env:` is never instantiated for training.

### Phase 1 — plain BC baseline (de-risk the integration first)
```bash
python correct_il/train_bc_policy.py config/alex_microwave.yml policy.naive true
```

`train_bc_policy.py` saves the greedy policy as a TorchScript `policy.pt` under
`output/alex_microwave/seed42/alex_open_microwave/policy/naive/` (scalers baked in: observation
`standard`, action `min_max`). **Note:** d3rlpy bakes the scaler constants to the training device
(`cuda:0`), so `policy.pt` must be loaded/run on CUDA.

### Phase 2 — CCIL augmentation (the actual contribution)
Set `policy.naive: false` and run the full pipeline:
```bash
python correct_il/train_dynamics_model.py config/alex_microwave.yml   # Lipschitz dynamics
python correct_il/gen_aug_label.py        config/alex_microwave.yml   # corrective labels -> augmented pickle
`python correct_il/train_bc_policy.py      config/alex_microwave.yml `  # BC on augmented data
```

Only the training data changes; export + Arena serving are identical.

## 4. Export verification metadata

Stage `policy.pt` next to the pickle (it gets a mount into the Arena container) and run the export
on CUDA (the script auto-selects it):

```bash
cp output/alex_microwave/seed42/alex_open_microwave/policy/naive/policy.pt \
   /datasets/alex_microwave/ccil/policy.pt
python isaaclab_arena_ccil/training/export_bc_to_torch.py \
  --policy_pt /datasets/alex_microwave/ccil/policy.pt \
  --pickle    /datasets/alex_microwave/ccil/alex_microwave.pkl \
  --out_meta  /datasets/alex_microwave/ccil/ccil_bc_meta.json
```

## 5. Evaluate in Arena

`policy_device` defaults to `cuda` (required — see the baked-device note above).

**Argument ordering matters:** the environment name is an argparse *subparser*, so policy and global
args must come **before** `alex_open_microwave`, and only env-specific args (`--embodiment`) after it.

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --device cuda --enable_cameras \
  --num_episodes 20 \
  --policy_type isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy \
  --model_path /datasets/alex_microwave/ccil/policy.pt \
  --meta_path /datasets/alex_microwave/ccil/ccil_bc_meta.json \
  --policy_device cuda \
  alex_open_microwave \
  --embodiment alex_v2_ability_hands
```

Use ``alex_v2_ability_hands`` (the Pink IK embodiment): the CCIL policy now outputs the raw
34-dim Pink IK action — the first 14 dims are left/right EE target poses (pos + quat) and the
last 20 are ability-hand finger joints. The IK embodiment resolves those EE targets to a
whole-body joint solution each step. (A policy trained on ``processed_actions`` instead would
require ``alex_v2_ability_hands_joint_pos``, which replays the 14 wrist/arm dims as direct
joint targets.)

The `OpenDoorTask` success term reports the success rate.

> Verified end-to-end: a `policy.pt` trained under torch 2.1.0 (CCIL env) loads through Arena's
> torch 2.10.0 and reproduces the CCIL reference actions to ~2e-7, so the TorchScript path is the
> primary route. The `state_dict` fallback below is only needed if a future torch breaks it.

## Cross-version fallback

If `torch.jit.load(policy.pt)` fails in Arena (TorchScript produced by an older torch), regenerate
a plain `state_dict` + full normalization meta from the d3rlpy model in the CCIL env (extract the
policy MLP weights and the `standard` obs scaler `mean`/`std` and `min_max` action scaler
`min`/`max` into `ccil_bc_meta.json` with `hidden_units`/`activation`). `CCILBCPolicy` then
reconstructs the MLP from that meta. See `ccil_bc_policy.py:_load_model`.

## Visual BC with ZED stereo

The ZED image path is BC-only: it trains on recorded samples with images plus robot state.
It does not run CCIL corrective dynamics/augmentation, because those generated labels do
not have corresponding camera frames.

```bash
/isaac-sim/python.sh isaaclab_arena_ccil/training/train_visual_bc.py \
  --pickle /datasets/alex_microwave/ccil/alex_microwave_visual.pkl \
  --out_policy /datasets/alex_microwave/ccil/visual_policy.pt \
  --out_meta /datasets/alex_microwave/ccil/visual_bc_meta.json \
  --image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --epochs 200
```

Evaluate the visual policy with cameras enabled:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --device cuda --enable_cameras \
  --num_episodes 20 \
  --policy_type isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy \
  --model_path /datasets/alex_microwave/ccil/visual_policy.pt \
  --meta_path /datasets/alex_microwave/ccil/visual_bc_meta.json \
  --policy_device cuda \
  --use_images \
  --image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --image_size 128 128 \
  alex_open_microwave \
  --embodiment alex_v2_ability_hands
```

To replay a recorded LeRobot episode through the CCIL visual policy, add the
replay-video override. For the ``test_obs_new`` adapter, recorded state and ZED frames
advance together so the multimodal policy input remains synchronized; actions still
drive the live simulator. In Kit, ``--show_replay_video`` opens dockable windows with
the recorded ZED frames:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --device cuda --enable_cameras \
  --num_episodes 20 \
  --policy_type isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy \
  --model_path /datasets/test_obs_new/ccil/visual_policy.pt \
  --meta_path /datasets/test_obs_new/ccil/visual_bc_meta.json \
  --policy_device cuda \
  --state_adapter test_obs_new \
  --use_images \
  --image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --image_size 128 128 \
  --replay_video_dataset_path /datasets/test_obs_new/lerobot \
  --replay_video_keys cam_zed_left cam_zed_right \
  --replay_video_episode 0 \
  --replay_video_stride 1 \
  --show_replay_video \
  alex_empty \
  --embodiment alex_v2_ability_hands
```
