# ConvertData tooling

This folder now includes scripts for a manual G1 (dataset `.pt`) -> Booster T1 (23-DoF) workflow.

## 1) Visualize a raw G1 `.pt` in MuJoCo

```bash
python /home/isaak/BEP/ConvertData/visualize_g1_pt_mujoco.py \
  --motion-pt /path/to/input_motion.pt \
  --joint-id /home/isaak/BEP/ConvertData/export/motion_dataset/joint_id.txt \
  --g1-urdf /home/isaak/BEP/ConvertData/export/unitreeg1/urdf/g1_23.urdf \
  --fps 30 \
  --quat-order xyzw \
  --allow-missing-meshes
```

Notes:

- If `ConvertData/export/unitreeg1/urdf/g1_23.urdf` exists, it is the default `--g1-urdf` (otherwise it falls back to `export/unitree_g1/...`).
- The stock G1 URDF references `../meshes/*.STL`. If you did not copy `meshes/`, pass `--allow-missing-meshes` to strip mesh visuals/collisions so MuJoCo can still load for a kinematic preview.
- The viewer writes a temporary runtime URDF **next to the source URDF** (a hidden `.g1_runtime_*` file) so `../meshes/...` paths still resolve correctly. It is deleted on exit unless you pass `--keep-runtime-urdf`.
- The script normalizes MuJoCo `meshdir` for the G1 URDF so meshes resolve from `../meshes`.
- The viewer also uncomments the `floating_base_joint` block by default so `base_position` / `base_pose` from the `.pt` actually apply. Use `--no-enable-floating-base` if you want joint-only playback.
- In headless terminals (`DISPLAY` not set), the script skips opening the interactive viewer and exits after successful model+motion load.

If root orientation looks wrong, retry with `--quat-order wxyz`.

## 2) Edit the manual mapping file

Start from:

`/home/isaak/BEP/ConvertData/manual_g1_to_t1_23_mapping.json`

- `target_joint_order`: target Booster T1 23-DoF order
- `mappings`: one mapping per source->target joint with optional:
  - `scale` (default `1.0`) for sign/axis fixes
  - `offset` (default `0.0`) for zero-position offsets

Two target joints are intentionally left unmapped initially:
- `AAHead_yaw`
- `Head_pitch`

## 3) Convert the `.pt`

```bash
python /home/isaak/BEP/ConvertData/convert_g1_pt_to_t1_23_manual.py \
  --input-pt /path/to/input_motion.pt \
  --output-pt /path/to/output_t1_23.pt \
  --mapping-json /home/isaak/BEP/ConvertData/manual_g1_to_t1_23_mapping.json
```

Add `--strict` to fail if any target joint remains all-zero.

### Batch: whole folder → T1 dataset folder

Default output: `ConvertData/datasets/t1_from_g1_goalkeeper/motions/` (see `datasets/t1_from_g1_goalkeeper/README.md`).

```bash
python /home/isaak/BEP/ConvertData/batch_convert_g1_pt_to_t1.py \
  --input-dir /home/isaak/BEP/ConvertData/export/motion_dataset
```
