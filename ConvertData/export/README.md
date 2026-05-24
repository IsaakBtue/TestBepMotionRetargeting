# Export bundle — G1 motion → Booster T1 retargeting

This folder is a **read-only snapshot** (copies) of assets under `Humanoid-Goalkeeper/`. It is **not** used by training code unless you point your own scripts here. Regenerate with the shell commands at the bottom if sources change.

## Layout

| Path | Contents |
|------|-----------|
| `motion_dataset/` | Goalkeeper imitation trajectories (`*.pt`) and `joint_id.txt` (dataset joint index → name). Same as `legged_gym/resources/datasets/goalkeeper/`. |
| `unitree_g1/` | Unitree G1 URDFs (`urdf/g1_29.urdf`, `urdf/g1_23.urdf`) and all `meshes/*.STL` referenced by the URDF. |
| `booster_t1/` | Booster Robotics T1 URDF/XML from `Humanoid-Goalkeeper/Boosterversion/booster_t1/` for **target** kinematics when retargeting. |

## Motion file format (`.pt`)

Files are `torch.load` dictionaries (one trajectory per file). `MotionLib` in the upstream code expects each dict to expose at least the keys below (shapes: `T` = time steps).

| Key | Role |
|-----|------|
| `base_position` | Root position, shape `(T, 3)`. |
| `base_pose` | Root orientation as quaternion `(T, 4)` (loaded as tensor; code converts to roll–pitch–yaw). |
| `joint_position` | Joint angles `(T, J_dataset)`; columns indexed by `joint_id.txt`. |
| `joint_velocity` | Joint velocities, same layout as positions. |
| `link_position` | Keyframe / link positions `(T, K, 3)` — `K` matches env `keyframe_names` order in code. |
| `link_oritentation` | Link quaternions `(T, K, 4)` (typo preserved from upstream). |
| `lin_velocity` | Per-keyframe linear velocity `(T, K, 3)`. |
| `link_angular_velocity` | Per-keyframe angular velocity `(T, K, 3)`. |

Joint names for the **21 columns** in the dataset mapping are listed in `motion_dataset/joint_id.txt` (index → name). The simulation stacks **29 DoF** G1 control; upstream maps dataset joints into `dof_names` via that file inside `MotionLib`.

Default dataset settings in upstream config: folder `resources/datasets/goalkeeper`, motion nominal rate **30 Hz** (`frame_rate` in `g1_29_config.py` class `dataset`).

## Retargeting checklist (Booster T1)

1. Parse **source** kinematic tree from `unitree_g1/urdf/g1_29.urdf` (or `g1_23.urdf` if you match that asset).
2. Parse **target** from `booster_t1/T1_locomotion.urdf` (or `T1_serial.urdf` as needed).
3. Build a **joint / link map** from G1 dataset indices and URDF joint names to T1 joint names and DoF order.
4. For each `.pt`, map `joint_position` / `joint_velocity` (and optionally root + keyframes) into T1’s tensor layout; recompute or approximate root/keyframe terms if skeletons differ.
5. Save new tensors in the same logical schema your Booster env expects (may differ from G1).

## Source paths in repo

- Motion: `Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/`
- G1: `Humanoid-Goalkeeper/legged_gym/resources/robots/g1/`
- T1: `Humanoid-Goalkeeper/Boosterversion/booster_t1/`

## Refresh copies

```bash
EXPORT=/home/isaak/BEPImitationlearning/export
GK=/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper
mkdir -p "$EXPORT/motion_dataset" "$EXPORT/unitree_g1" "$EXPORT/booster_t1"
cp -a "$GK/legged_gym/resources/datasets/goalkeeper/." "$EXPORT/motion_dataset/"
cp -a "$GK/legged_gym/resources/robots/g1/." "$EXPORT/unitree_g1/"
cp -a "$GK/Boosterversion/booster_t1/." "$EXPORT/booster_t1/"
```
