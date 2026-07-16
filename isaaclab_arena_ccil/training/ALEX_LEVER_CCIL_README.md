# Alex Lever: Mimic to LeRobot v3 to CCIL

This is the workflow for the Alex lever policy trained from Mimic-generated data using the
`H2Ozone/test_obs_new`-style schema:

- `observation.state`: 48 dims
- `action`: 46 dims
- videos: `observation.images.cam_zed_left`, `observation.images.cam_zed_right`

The environment-side eval path should use `alex_lever_turn --test_obs_new_io`, which exposes a
native 48-D policy observation and 46-D action term. Do not evaluate this dataset through the old
34-D action interface.

## 0. Container

From the host repo root:

```bash
cd /home/bpratt/IsaacLab-Arena
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
```

## 1. After Mimic Generation Finishes

Expected generated HDF5:

```bash
/datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new.hdf5
```

To generate a domain-randomized replacement dataset for the current `another_try_lever.usd`
fixture, run the lever Mimic pipeline with conservative pose jitter and handle-color variation:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena/scripts/imitation_learning/run_lever_mimic_pipeline.py \
  --work_dir /datasets/alex_lever_sim/mimic_dr_test_obs_new \
  --record_count 20 \
  --generated_count 400 \
  --generation_num_envs 10 \
  --device cuda \
  --enable_cameras \
  --overwrite \
  --usd isaaclab_arena/assets/lever_sim/another_try_lever.usd \
  --table none \
  --success_angle_threshold 0.523599 \
  --lever_dr \
  --lever_pose_dr_xy_jitter 0.01 \
  --lever_pose_dr_yaw_jitter_deg 5.0
'
```

Use `/datasets/alex_lever_sim/mimic_dr_test_obs_new/generated.hdf5` as the source HDF5 for the
conversion steps below. For `another_try_lever.usd`, pose is sampled conservatively at process/env
setup time; per-reset root motion for this base-object USD remains disabled to avoid the known GPU
PhysX instability.

Sanity-check the corrected wrist quaternion packing. For each wrist, action `quat_x` should be
closer to state `quat_x` than state `quat_y`, and action `quat_y` should be closer to state
`quat_y` than state `quat_x`.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh - <<PY
import h5py, numpy as np
p="/datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new.hdf5"
with h5py.File(p, "r") as f:
    d=f["data/demo_0"]
    s=d["observation.state"][:]
    a=d["action"][:]
for name, offset in [("left", 0), ("right", 7)]:
    qx = offset + 3
    qy = offset + 4
    print(name, "action qx vs state qx", np.mean(np.abs(a[:, qx] - s[:, qx])))
    print(name, "action qx vs state qy", np.mean(np.abs(a[:, qx] - s[:, qy])))
    print(name, "action qy vs state qy", np.mean(np.abs(a[:, qy] - s[:, qy])))
    print(name, "action qy vs state qx", np.mean(np.abs(a[:, qy] - s[:, qx])))
PY
'
```

## 2. HDF5 to GR00T-Layout LeRobot

This first converter writes the repo's older GR00T/LeRobot layout. It is an intermediate format,
not the final v3 dataset.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/alex_lever_mimic_80_test_obs_new_config.yaml
'
```

Output:

```bash
/datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new/lerobot
```

## 3. GR00T-Layout LeRobot to LeRobot v3

This is the required step for standard LeRobot v3.0. It creates `meta/tasks.parquet`,
`meta/episodes/chunk-000/file-000.parquet`, chunked `data/`, `videos/`, and `meta/stats.json`.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && rm -rf /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3 && /isaac-sim/python.sh \
  isaaclab_arena_gr00t/lerobot/convert_gr00t_to_lerobot_v3.py \
  --input_dir /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new/lerobot \
  --output_dir /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3
'
```

Validate the v3 files:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
find /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3/meta -maxdepth 4 -type f | sort
find /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3/data -maxdepth 3 -type f | sort
find /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3/videos -maxdepth 5 -type f | sort | head
'
```

## 4. LeRobot v3 to CCIL Pickles

State-only:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && mkdir -p /datasets/alex_lever_sim/ccil && /isaac-sim/python.sh \
  isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py \
  --lerobot_dir /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3 \
  --out_file /datasets/alex_lever_sim/ccil/quest_alex_empty_mimic_80_test_obs_new.pkl
'
```

Visual:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py \
  --lerobot_dir /datasets/alex_lever_sim/quest_alex_empty_mimic_80_test_obs_new_v3 \
  --out_file /datasets/alex_lever_sim/ccil/quest_alex_empty_mimic_80_test_obs_new_visual.pkl \
  --image_keys observation.images.cam_zed_left observation.images.cam_zed_right \
  --output_image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --image_size 128 128
'
```

## 5. Train State-Only CCIL

State-only CCIL runs in the separate Python 3.8 CCIL env, not the Isaac container.

```bash
source /home/bpratt/miniconda3/etc/profile.d/conda.sh
conda activate CCIL
cd /home/bpratt/CCIL
rm -rf /home/bpratt/datasets/alex_lever_sim/ccil/ccil_output/seed42/alex_lever_turn
```

Naive BC baseline:

```bash
python correct_il/train_bc_policy.py \
  /home/bpratt/datasets/alex_lever_sim/ccil/configs/alex_lever_test_obs_new_naive.yml
```

Full CCIL:

```bash
python correct_il/train_dynamics_model.py \
  /home/bpratt/datasets/alex_lever_sim/ccil/configs/alex_lever_test_obs_new_ccil.yml
python correct_il/gen_aug_label.py \
  /home/bpratt/datasets/alex_lever_sim/ccil/configs/alex_lever_test_obs_new_ccil.yml
python correct_il/train_bc_policy.py \
  /home/bpratt/datasets/alex_lever_sim/ccil/configs/alex_lever_test_obs_new_ccil.yml
```

Stage the full CCIL policy:

```bash
cp /home/bpratt/datasets/alex_lever_sim/ccil/ccil_output/seed42/alex_lever_turn/policy/backward_euler_soft_samplingL2.0/policy.pt \
   /home/bpratt/datasets/alex_lever_sim/ccil/policy_test_obs_new_ccil.pt
```

Export Arena metadata:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena_ccil/training/export_bc_to_torch.py \
  --policy_pt /datasets/alex_lever_sim/ccil/policy_test_obs_new_ccil.pt \
  --pickle /datasets/alex_lever_sim/ccil/quest_alex_empty_mimic_80_test_obs_new.pkl \
  --out_meta /datasets/alex_lever_sim/ccil/ccil_bc_meta_test_obs_new_ccil.json
'
```

## 6. Train Visual BC

Visual BC uses recorded image frames and does not use CCIL dynamics augmentation.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena_ccil/training/train_visual_bc.py \
  --pickle /datasets/alex_lever_sim/ccil/quest_alex_empty_mimic_80_test_obs_new_visual.pkl \
  --out_policy /datasets/alex_lever_sim/ccil/visual_policy_test_obs_new.pt \
  --out_meta /datasets/alex_lever_sim/ccil/visual_bc_meta_test_obs_new.json \
  --image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --epochs 200
'
```

## 7. Evaluate in the Lever Env

Use `--test_obs_new_io`. The action table should show `shape: 46`, and the policy observation
table should show `shape: (48,)`.

State-only CCIL:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena/evaluation/policy_runner.py \
  --device cuda \
  --viz kit \
  --num_episodes 10 \
  --policy_type isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy \
  --model_path /datasets/alex_lever_sim/ccil/policy_test_obs_new_ccil.pt \
  --meta_path /datasets/alex_lever_sim/ccil/ccil_bc_meta_test_obs_new_ccil.json \
  --policy_device cuda \
  --state_adapter test_obs_new \
  alex_lever_turn \
  --embodiment alex_v2_ability_hands \
  --test_obs_new_io \
  --usd isaaclab_arena/assets/lever_sim/another_try_lever.usd \
  --table none \
  --success_angle_threshold 0.523599 \
  --lever_dr \
  --lever_pose_dr_xy_jitter 0.01 \
  --lever_pose_dr_yaw_jitter_deg 5.0
'
```

Visual BC:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c '
cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
  isaaclab_arena/evaluation/policy_runner.py \
  --device cuda \
  --viz kit \
  --enable_cameras \
  --num_episodes 10 \
  --policy_type isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy \
  --model_path /datasets/alex_lever_sim/ccil/visual_policy_test_obs_new.pt \
  --meta_path /datasets/alex_lever_sim/ccil/visual_bc_meta_test_obs_new.json \
  --policy_device cuda \
  --state_adapter test_obs_new \
  --use_images \
  --image_keys zed_left_cam_rgb zed_right_cam_rgb \
  --image_size 128 128 \
  alex_lever_turn \
  --embodiment alex_v2_ability_hands \
  --test_obs_new_io \
  --usd isaaclab_arena/assets/lever_sim/another_try_lever.usd \
  --table none \
  --success_angle_threshold 0.523599 \
  --lever_dr \
  --lever_pose_dr_xy_jitter 0.01 \
  --lever_pose_dr_yaw_jitter_deg 5.0
'
```

## Common Failure Modes

- `Action Terms (shape: 34)`: the command is missing `--test_obs_new_io`.
- Wrist `quat_x` tracks `quat_y`: the dataset was generated before the quat-packing fix. Regenerate Mimic.
- `gen_dataset`-style HF dataset with only generic video rows: you skipped the v3 conversion. Run
  `convert_gr00t_to_lerobot_v3.py` and use the `_v3` directory.
- Policy loads but moves strangely: make sure the policy was retrained after regenerating the corrected HDF5.
