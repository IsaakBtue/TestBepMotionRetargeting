# Booster T1 dataset (from Unitree G1 goalkeeper `.pt`)

This folder holds **converted** motion files: same tensor schema as the source G1 goalkeeper trajectories, but `joint_position` / `joint_velocity` are laid out for **Booster T1 serial (23 revolute joints)**.

- `joint_id.txt` — column index → T1 joint name (matches `manual_g1_to_t1_23_mapping.json` `target_joint_order`).
- `*.pt` — produced by `batch_convert_g1_pt_to_t1.py` (or `convert_g1_pt_to_t1_23_manual.py` per file).

**Not yet retargeted in Cartesian space:** this is **manual joint-axis mapping** only. Root pose (`base_position`, `base_pose`) and link tensors are still from the G1 recording until you add IK or a second-stage retarget.

**Source columns:** the goalkeeper `joint_id.txt` on the G1 side lists **21** actuated joints. There is no separate wrist-roll channel in that file, so T1 `Left_Elbow_Yaw` / `Right_Elbow_Yaw` stay **zero** until you map from another signal or run IK. Head joints (`AAHead_yaw`, `Head_pitch`) are also **unmapped** in the starter JSON.

## Regenerate

From the repo root (with `HumanoidDataGeneration` activated):

```bash
python /home/isaak/BEP/ConvertData/batch_convert_g1_pt_to_t1.py \
  --input-dir /home/isaak/BEP/ConvertData/export/motion_dataset \
  --output-dir /home/isaak/BEP/ConvertData/datasets/t1_from_g1_goalkeeper/motions \
  --joint-id /home/isaak/BEP/ConvertData/export/motion_dataset/joint_id.txt
```

Copy `joint_id.txt` from this folder next to `motions/` if your loader expects it there.

## Visualize (Booster T1 in MuJoCo)

Uses the same `RobotMotionViewer` as `./scripts/visualize.sh`, but reads the **converted `.pt`** (not the GMR `.pkl`).

```bash
conda activate HumanoidDataGeneration
python /home/isaak/BEP/ConvertData/visualize_t1_goalkeeper_pt.py \
  --motion-pt /home/isaak/BEP/ConvertData/datasets/t1_from_g1_goalkeeper/motions/leftjump.pt
```

Requires a GUI (`DISPLAY` set). Test without a window: `--validate-only`.
