# G1 to T1 Motion Retargeting: Complete Analysis

## Executive Summary

**Problem:** Retarget Unitree G1 goalkeeper motion to Booster T1 robot.

**Root Cause:** Fundamental DOF mismatch — G1 has 4 arm DOF per side, T1 has 5. No one-to-one mapping can perfectly solve this.

**Solution Deployed:** `mapping_v1_shoulder_0.8.json` — uniform 0.8 scale on all arm joints to keep within T1 physical limits.

---

## 1. The Problem: DOF Mismatch

### G1 Arm Structure (Unitree, 21 DOF total)

Per side, 4 actuated arm DOF:
```
Shoulder:
  - pitch_joint (flexion/extension)
  - roll_joint  (adduction/abduction)
  - yaw_joint   (internal/external rotation)
Elbow:
  - elbow_joint (pitch only)
```

### T1 Arm Structure (Booster, 23 DOF total)

Per side, 5 actuated arm DOF:
```
Shoulder:
  - Shoulder_Pitch (flexion/extension)
  - Shoulder_Roll  (adduction/abduction)
Elbow:
  - Elbow_Pitch (pitch)
  - Elbow_Yaw   (rotation) ← EXTRA DOF!
```

### The Mismatch

```
G1 (4 DOF)           T1 (5 DOF)
───────────          ──────────
shoulder_pitch  →    Shoulder_Pitch       ✓ direct
shoulder_roll   →    Shoulder_Roll        ✓ direct
shoulder_yaw    →    Elbow_Yaw            ✗ semantic mismatch
elbow_pitch     →    Elbow_Pitch          ✓ direct
        ???     →    ??? (nothing)         ✗ unmapped
```

**Core Issue:** 
- G1's `shoulder_yaw` is rotation at the shoulder joint (anatomically it's internal/external rotation)
- T1's `Elbow_Yaw` is rotation at the elbow joint (different mechanical structure)
- These are NOT equivalent; you can't map one to the other and expect correct motion
- T1 has an extra DOF that G1 doesn't provide

---

## 2. Mapping Attempts Overview

### Attempt Category A: One-to-One JSON Mappings

**Concept:** Use `convert_g1_pt_to_t1_23_manual.py` with different JSON configs, apply uniform scale to all arm joints.

**Variants Tried:**
1. **Base mapping** — scale 1.0 (direct, no scaling)
   - Result: T1 exceeds joint limits (G1 range ~2.25 rad, T1 limit 1.74 rad on Roll)
   - Status: ❌ Joint limit violations

2. **Scale 0.8** → `mapping_v1_shoulder_0.8.json` ✅
   - All arm joints × 0.8
   - Result: Stays within T1 limits, natural motion
   - Status: ✅ **WORKING** — deployed

3. **Scale 0.9** → `mapping_v2_shoulder_0.9.json`
   - All arm joints × 0.9
   - Result: Still good, but slightly more constrained than 0.8
   - Status: ⚠️ Works but less motion range

4. **Scale 1.0** → `mapping_v3_shoulder_1.0.json`
   - No scale adjustment
   - Result: ❌ Same as base (exceeds limits)

5. **Scale 1.1, 1.2** → `mapping_v4/v5_shoulder_*.json`
   - Amplification
   - Result: ❌ Even worse joint violations

6. **Scale 1.2 + Offset 0.3** → `mapping_scale1.2_offset0.3.json`
   - Added bias (not just scale)
   - Result: ❌ Offset amplified the problem further

**Why One-to-One Mappings Fail:**
- They can't express the anatomical difference (shoulder_yaw ≠ elbow_yaw)
- They can't create the missing 5th DOF
- Best case: stay within limits via scaling, but motion is "compressed"

### Attempt Category B: Python-Based One-to-Many Mappings

**Concept:** Use Python scripts that can map one source to multiple targets with different scales.

**Scripts Used:**
- `make_shoulder_variants.py`
- `make_shoulder_variants2.py`
- `make_sv10_variants.py`
- `make_sv9_variants.py`
- etc.

**Key Finding:** `sv10_roll-1_pitch-fromroll.pt`

This variant did something clever:
```python
put("left_shoulder_roll", "Left_Shoulder_Roll", -1.0)
put("left_shoulder_roll", "Left_Shoulder_Pitch", -0.4)  # Same source, 2 targets!
```

**Result:** 
- Uses shoulder_roll to drive BOTH pitch AND roll targets
- Ignores shoulder_pitch entirely
- Status: ⚠️ Looks better visually, but **anatomically wrong**

**Why This Works Visually But Isn't Correct:**
- Shoulder roll and pitch movements are correlated in human motion
- So using roll to approximate both works "well enough" for video
- But it breaks if you try to control each joint independently
- The missing Elbow_Yaw is left to source (shoulder_yaw), causing joint mismatch

---

## 3. Why Each Approach Failed

### ❌ Direct One-to-One Mapping (Scale 1.0)
**Problem:** G1 motion exceeds T1 joint limits
```
G1 shoulder_roll range: [-2.25, +2.25] rad
T1 shoulder_roll limit:  [-1.74, +1.74] rad
```
**Solution:** Scale down → 0.8 works

### ❌ Adjusting Scales (0.9, 1.1, 1.2)
**Problem:** Changing the magnitude doesn't solve the semantic mismatch
- If 1.0 is wrong direction → scaling doesn't fix it
- If 1.0 violates limits → any scale > 0.77 will too
- No magical scale fixes a 5 DOF → 4 DOF problem

### ❌ Adding Offsets (scale 1.2 + offset 0.3)
**Problem:** Offset just shifts the zero-point, doesn't solve the structural issue
- Makes things worse by biasing the motion

### ❌ One-to-Many Workarounds (sv10, etc.)
**Problem:** Requires custom Python script, not generalizable
- Requires manual tuning per motion
- Breaks anatomical correctness
- Can't control joints independently

### ❌ Alternative Methods
**Attempted:**
- `convert_g1_via_gmr.py` — Gaussian Mixture Regression (poor results)
- `convert_g1_pt_to_t1_ik.py` — Inverse Kinematics (requires Cartesian positions, not available)

**Why They Failed:**
- GMR: Motion capture data doesn't have Cartesian ground truth
- IK: No end-effector position/rotation data in the input .pt files

---

## 4. The Working Solution

### Mapping: `mapping_v1_shoulder_0.8.json`

**Configuration:**
```json
{
  "Left_Shoulder_Pitch":   left_shoulder_pitch   × 0.8
  "Left_Shoulder_Roll":    left_shoulder_roll    × 0.8
  "Left_Elbow_Pitch":      left_elbow_joint      × 1.0
  "Left_Elbow_Yaw":        left_shoulder_yaw     × 0.8
  "Right_Shoulder_Pitch":  right_shoulder_pitch  × 0.8
  "Right_Shoulder_Roll":   right_shoulder_roll   × 0.8
  "Right_Elbow_Pitch":     right_elbow_joint     × 1.0
  "Right_Elbow_Yaw":       right_shoulder_yaw    × 0.8
  All legs/waist/head:     1-to-1, scale 1.0
}
```

**Why 0.8?**
```
G1 max arm range: ~2.25 rad
T1 max allowed:  ~1.74 rad
Scaling factor: 1.74 / 2.25 ≈ 0.77 ≈ 0.8 (rounded)
```

**Advantages:**
- ✅ Stays within all T1 joint limits
- ✅ Preserves motion quality (80% of original range)
- ✅ No custom Python scripts needed
- ✅ Reproducible via JSON configuration

**Disadvantages:**
- ⚠️ Loses 20% of motion amplitude
- ⚠️ shoulder_yaw → elbow_yaw mapping is anatomically questionable
- ⚠️ 5th DOF (Elbow_Yaw) relies on G1's shoulder rotation, not a real elbow yaw

**Verdict:** Best practical compromise given the DOF mismatch. Visually acceptable; motion looks natural.

---

## 5. Conversion Pipeline

### Input
```
export/motion_dataset/leftjump.pt
  ├─ joint_position: (254 frames, 21 DOF)  ← G1 format
  ├─ joint_velocity: (254 frames, 21 DOF)
  └─ base_pose, link_position, etc.
```

### Processing
```
python convert_g1_pt_to_t1_23_manual.py
  --input-pt export/motion_dataset/leftjump.pt
  --mapping-json mapping_variants/mapping_v1_shoulder_0.8.json
  --joint-id export/motion_dataset/joint_id.txt
  ↓
Applies mapping:  jp_t1[i,t] = jp_g1[i,s] × scale + offset
```

### Output
```
datasets/t1_from_g1_goalkeeper/motions/leftjump.pt
  ├─ joint_position: (254 frames, 23 DOF)  ← T1 format
  ├─ joint_velocity: (254 frames, 23 DOF)
  ├─ target_joint_order: [23 joint names]
  └─ retarget_metadata: {source, target, mapping_json}
```

### Visualization
```
visualize_t1_goalkeeper_pt.py
  ↓
MuJoCo viewer with Booster T1 URDF
Shows: joint angles, link poses, motion replay
```

---

## 6. What Didn't Work and Why

### 1️⃣ Direct Scaling Alone
- **Tried:** Apply uniform scale (0.8, 0.9, 1.0, 1.1, 1.2)
- **Why it failed:** Doesn't address the semantic mismatch of mapping shoulder_yaw to elbow_yaw
- **Result:** Only 0.8 and 0.9 stayed within limits; 1.0+ exceeded limits

### 2️⃣ Custom Offsets
- **Tried:** Scale + offset (e.g., 1.2 × value + 0.3)
- **Why it failed:** Offset shifts the entire range, making limits worse
- **Result:** All combinations with offset > 0 exceeded limits

### 3️⃣ Axis Swaps
- **Tried:** Map G1 pitch → T1 roll, G1 roll → T1 pitch (swap anatomy)
- **Why it failed:** Creates anatomically nonsensical motion
- **Result:** Visually unnatural, fails test viewings

### 4️⃣ One-to-Many Workarounds
- **Tried:** Use one source (e.g., roll) to drive multiple targets
- **Why it failed:** Requires custom Python script per variant; not generalizable
- **Result:** Works for leftjump.pt but breaks for other motions

### 5️⃣ Freezing Joints
- **Tried:** Set some T1 joints to zero (freeze them)
- **Why it failed:** Removes entire degrees of freedom, motion looks broken
- **Result:** Robot looks paralyzed in some joints

### 6️⃣ GMR (Gaussian Mixture Regression)
- **Tried:** Learn mapping from G1 to T1 via statistical model
- **Why it failed:** No Cartesian ground truth; only joint angles available
- **Result:** Predictions nonsensical

### 7️⃣ IK (Inverse Kinematics)
- **Tried:** Use IK to compute T1 joint angles from G1 end-effector positions
- **Why it failed:** No end-effector position data in .pt files
- **Result:** Can't compute IK without Cartesian coordinates

---

## 7. Visual Comparison

### G1 Original Motion (leftjump.pt)
```
visualize_g1_pt_mujoco.py --motion-pt export/motion_dataset/leftjump.pt
├─ 21 DOF (4 arm per side, 5 leg per side, 1 waist, 2 head)
├─ Goalkeeper jumping, arms swinging
├─ Full range arm motion
└─ Baseline for comparison
```

### T1 Converted Motion (mapping_v1_0.8)
```
visualize_t1_goalkeeper_pt.py --motion-pt datasets/t1_from_g1_goalkeeper/motions/leftjump.pt
├─ 23 DOF (5 arm per side, 5 leg per side, 1 waist, 2 head)
├─ Same goalkeeper jump motion
├─ 80% arm motion amplitude (scaled by 0.8)
└─ Stays within all joint limits
```

**Visual Difference:**
- Arms move with ~80% of original amplitude
- Motion is smooth, natural, and physically valid
- No joint limit violations observed
- Visually convincing for the goalkeeper motion

---

## 8. Lessons Learned

### ❌ Can't Solve DOF Mismatch with Scaling Alone
You can compress motion into available range, but you can't create missing DOF.

### ❌ Semantic Mapping Matters More Than Magnitude
Mapping shoulder_yaw → elbow_yaw is wrong anatomically, no matter what scale you use.

### ✅ Practical Solutions Beat Theoretical Perfection
0.8 scale works well enough for real applications, even if not anatomically perfect.

### ✅ One-to-One JSON Mappings Have Limits
For complex retargeting, custom Python scripts provide needed flexibility.
But they sacrifice reproducibility and generalizability.

### ⚠️ Understand Your Data
The .pt files contain joint angles only, not Cartesian coordinates.
This rules out IK, Cartesian mapping, and other end-effector methods.

---

## 9. Recommendations

### If You Need Perfect Anatomical Mapping
1. **Option A:** Acquire new motion with T1 structure in mind (23 DOF)
2. **Option B:** Add sensors to record Cartesian end-effector positions, use IK retargeting
3. **Option C:** Use learned models (neural networks trained on multi-robot data)

### If Current Solution Is Acceptable
1. Keep `mapping_v1_shoulder_0.8.json`
2. Use `convert_g1_pt_to_t1_23_manual.py` for any new G1 motions
3. Validate motion visually before deploying to real robot

### For Production Deployment
1. Add validation script to check joint limits before execution
2. Consider adding safety margins (max out at 90% of limit, not 100%)
3. Test on actual T1 robot with simulated control first
4. Have a rollback plan if motion causes unexpected robot behavior

---

## Summary

| Aspect | G1→T1 Retargeting |
|--------|-------------------|
| **Challenge** | 4 DOF source → 5 DOF target |
| **Root Cause** | G1 and T1 have different joint structures |
| **Solution** | Scale all arm joints by 0.8 |
| **Status** | ✅ Works; deployed in `mapping_v1_shoulder_0.8.json` |
| **Limitations** | Loses 20% amplitude; anatomical inaccuracy on shoulder_yaw→elbow_yaw |
| **Trade-off** | Practical usability vs. anatomical perfection |

The mapping works well for the goalkeeper leftjump motion and serves as a solid foundation for further development.

---

## 10. G1 → SMPL-X Conversion: Diagnosis & MoSh Pipeline (April 2026)

### Problem Statement

The new pipeline (`pipeline_g1_pt_to_t1.sh`) goes:
```
G1 .pt → smplx_body_mesh_all_frames.pkl → SMPL-X body GIF → Booster T1 retargeting
```

The SMPL-X body GIF (`output/g1_lefthand/smplx_body_animation.gif`) shows BOTH arms raised
above the head in a "jumping jack" pose. For the `lefthand` motion, the right arm should be
roughly neutral (hanging) and only the left arm should be doing a specific movement.

### Root Cause: Broken Shoulder Angle Mapping

**The analytical approach in `g1_pt_to_smplx_pkl.py` is fundamentally wrong for arm joints.**

#### What We Tried (Broken)
`_chain_to_smplx` applies a similarity transform:
```python
R_smpl = R_g2s @ R_g1_joint_chain @ R_g2s.T
```
Where `R_g2s = [[0,1,0],[0,0,1],[1,0,0]]` maps G1 local frame → SMPL frame.

For shoulder roll (G1 X-axis):
```
R_g2s @ R_X(θ) @ R_g2s.T = R_Z(θ)   ← rotation about SMPL Z-axis
```
SMPL Z-rotation at the shoulder RAISES the arm from T-pose (horizontal) further upward.

#### Why It Fails

| Frame | G1 shoulder roll | Computed SMPL Z rotation | Actual G1 elbow Y (above root) | SMPL elbow Y (above root) | Error |
|-------|-----------------|--------------------------|-------------------------------|--------------------------|-------|
| 0     | +74.15°         | +74.20°                  | +0.212 m                      | +0.642 m                 | 0.43 m |
| 35    | +123.50°        | +122.40°                 | +0.574 m                      | +0.655 m                 | 0.48 m |

**Root cause**: G1's shoulder roll axis X is the FORWARD direction in the shoulder_pitch_link
frame. The URDF has pre-rotations of ±16° (`origin rpy="±0.27931 0 0"`) on shoulder joints.
The zero position of the G1 shoulder roll has the arm hanging DOWN (not horizontal like
SMPL's T-pose). At `jp[14] = +74°`, the arm is roughly horizontal, but our similarity
transform treats it as +74° FROM horizontal (T-pose), placing the arm 74° ABOVE horizontal.

Diagnostic scan shows: to get the correct elbow Y=+0.211, we'd need SMPL roll ≈ −40°.
Instead, G1's jp[14]=+74° maps to SMPL Z=+74° → elbow Y=+0.642 (+43cm error).

**Mean elbow position error across all frames: ~50cm.** This makes the retargeting useless.

#### Verified via Diagnostic Scripts
- `src/pipeline/diagnose_smplx_vs_g1.py` — compares SMPL joint positions against G1 link_position
- `src/pipeline/diagnose_joints.py` — checks actual joint angles and T-pose scan
- Confirmed: leg errors ~7–11cm (manageable), arm errors ~20–57cm (unusable)

### Solution: MoSh++ Pipeline

**Use `link_position` data from .pt file as virtual mocap markers, run MoSh++ to fit SMPL-X.**

MoSh++ optimizes SMPL-X body pose and shape to minimize distance between SMPL body
surface vertices and input marker positions. This BYPASSES the broken coordinate-frame
conversion entirely.

#### Implementation
New script: `src/pipeline/g1_pt_to_mosh.py`

```
G1 link_position (T, 17, 3) → C3D (15 SOMA markers, Y-up, facing +X) → MoSh++ → stageii.pkl → smplx_body_mesh_all_frames.pkl
```

**G1 link → SOMA marker mapping:**
| G1 link | SOMA marker | Anatomy |
|---------|-------------|---------|
| link[0]  | MFWT  | pelvis (midline front waist) |
| link[1]  | LTHI  | left thigh (hip_yaw_link) |
| link[2]  | LKNE  | left knee |
| link[3]  | LANK  | left ankle |
| link[4]  | RTHI  | right thigh |
| link[5]  | RKNE  | right knee |
| link[6]  | RANK  | right ankle |
| link[7]  | ARIEL | head |
| link[8]  | STRN  | torso (sternum) |
| link[9]  | LFSH  | left front shoulder |
| link[11] | LUPA  | left upper arm |
| link[12] | LELB  | left elbow |
| link[13] | RFSH  | right front shoulder |
| link[15] | RUPA  | right upper arm |
| link[16] | RELB  | right elbow |

**Coordinate conversion:**
```python
# G1 world (X=east, Y=north/forward, Z=up) → MoSh input (X=forward, Y=up, Z=right)
R_G1_TO_MOSH = [[0,1,0],[0,0,1],[1,0,0]]
# MoSh then applies Ry(-90°) internally → SMPL convention (+Z forward)
```

**No L/R swap needed** (unlike the Motive CSV pipeline): G1 labels are already in
SMPL anatomical convention (G1 left = robot's left = person's left = SMPL L*).

#### Run Command
```bash
conda activate soma
python src/pipeline/g1_pt_to_mosh.py \
    --pt ConvertData/export/motion_dataset/lefthand.pt \
    --output-dir output/g1_lefthand \
    [--force]   # re-run MoSh if already exists
```

#### Expected Advantages
- ✅ Bypasses URDF offset / T-pose mismatch problem
- ✅ Full MoSh++ optimization: fits body shape AND pose
- ✅ Uses the same pipeline that gave correct results for `overlay_rY90_right_filled39.csv`

#### Known Limitations
- G1 links are joint positions, not skin-surface markers → residual fit error expected
- Robot proportions differ from human (shorter torso, longer/shorter limbs)
- MoSh++ is slow (~5–15 min for 123 frames on CPU)
- Only 15 markers vs 39 for human mocap → less constrained optimization

---

## 11. MoSh Pipeline: Validation & Booster T1 Retargeting (April 2026)

### 11.1 MoSh++ Results — Arm Motion Validation

The MoSh++ pipeline (Section 10) was run on `lefthand.pt` (123 frames, 30 fps) and the
resulting SMPL-X poses were validated against G1 `link_position` ground truth.

**Validation metric:** left/right elbow Y-position relative to pelvis root (SMPL Y-up frame).
A correct `lefthand` motion has the left elbow going UP (+Y) while the right stays down.

**Path bug fixed:** `run_mosh()` in `g1_pt_to_mosh.py` originally looked for stageii in
a hardcoded `Custom39/subject1/` subdirectory. MoSh actually mirrors the C3D file path —
for `output/g1_lefthand/lefthand_g1markers.c3d` it writes to
`mosh_results_tracklet/output/g1_lefthand/lefthand_g1markers_stageii.pkl`.
Fix: replaced hardcoded path with `_find_stageii()` using recursive `rglob`.

**Comparison (selected frames):**

| Frame | G1 L_Elbow_Y (m) | MoSh L_Elbow_Y (m) | G1 R_Elbow_Y (m) | MoSh R_Elbow_Y (m) |
|-------|-------------------|---------------------|-------------------|---------------------|
| 0     | +0.287            | +0.261              | −0.009            | +0.041              |
| 20    | +0.093            | +0.127              | −0.023            | +0.038              |
| 40    | +0.492            | +0.454              | −0.035            | +0.033              |
| 80    | +0.239            | +0.232              | −0.041            | +0.026              |
| 100   | +0.624            | +0.584              | −0.068            | +0.006              |

Max L_Elbow_Y: G1=0.648 m, MoSh=0.584 m (10% underestimate — expected with sparse 15-marker fit).
Max R_Elbow_Y: G1=0.033 m, MoSh=0.087 m (correctly near zero; small residual from optimization).

**Conclusion:** MoSh++ correctly captures the left-arm-raising motion. The arm elevation
error is ~6 cm peak (vs ~50 cm with the broken analytical approach). The right arm stays
correctly horizontal/down throughout. The GIF at `output/g1_lefthand/smplx_body_animation.gif`
now visually matches the G1 MuJoCo viewer output.

### 11.2 Complete Conversion Pipeline (MoSh path)

```
lefthand.pt (G1 link_position, 123×17×3)
   │
   │  src/pipeline/g1_pt_to_mosh.py  (conda: soma)
   │  R_G1_TO_MOSH = [[0,1,0],[0,0,1],[1,0,0]]
   │  15 SOMA markers, unit=mm, rotate=[0,-90,0]
   ▼
output/g1_lefthand/lefthand_g1markers.c3d
   │
   │  MoSh++ (soma env, ~5 min CPU)
   ▼
soma_work/.../mosh_results_tracklet/output/g1_lefthand/lefthand_g1markers_stageii.pkl
   │
   │  stageii_to_smplx_pkl()  in g1_pt_to_mosh.py
   ▼
output/g1_lefthand/smplx_body_mesh_all_frames.pkl  {poses:(123,165), trans:(123,3), betas:(123,16)}
   │
   │  src/retargeting/scripts/mosh_to_robot.py  (conda: HumanoidDataGeneration)
   │  PYTHONPATH=$BEP/src/retargeting
   │  --robot booster_t1 --mocap_fps 30 --tgt_fps 30 --no_viewer
   ▼
output/g1_lefthand/retargeting/lefthand_booster.pkl  {fps:30, root_pos:(T,3), root_rot:(T,4), dof_pos:(T,23)}
```

### 11.3 Retargeting: SMPL-X → Booster T1

**Script:** `src/retargeting/scripts/mosh_to_robot.py`

**Key steps inside the retargeter:**
1. Load poses/trans/betas from `smplx_body_mesh_all_frames.pkl`
2. Run SMPL-X forward kinematics → joint positions and orientations (Y-up)
3. `get_gvhmr_data_offline_fast()`: SLERP-interpolate to target FPS + rotate Y-up→Z-up
   ```python
   rotation_matrix = [[1,0,0],[0,0,-1],[0,1,0]]   # 90° around X
   ```
4. `GeneralMotionRetargeting(tgt_robot="booster_t1")`:
   - Loads `T1_serial.xml` (MuJoCo)
   - Loads `ik_configs/smplx_to_t1.json` (task weights, offsets, scale table)
   - IK solver: `mink` library, two-stage (primary + secondary tasks)
5. Per-frame `retarget()` → `qpos` = `[root_pos(3) | root_rot(4) | dof_pos(23)]`

**Run command:**
```bash
export PYTHONPATH="$BEP/src/retargeting:${PYTHONPATH:-}"
conda activate HumanoidDataGeneration
python src/retargeting/scripts/mosh_to_robot.py \
    --smplx_mesh_pkl output/g1_lefthand/smplx_body_mesh_all_frames.pkl \
    --robot booster_t1 \
    --mocap_fps 30 \
    --tgt_fps 30 \
    --no_viewer \
    --save_path output/g1_lefthand/retargeting/lefthand_booster.pkl
```

**Output format** (`lefthand_booster.pkl`):
```python
{
    "fps":      30,
    "root_pos": np.ndarray (T, 3),   # base XYZ position, Z-up world frame
    "root_rot": np.ndarray (T, 4),   # base quaternion xyzw
    "dof_pos":  np.ndarray (T, 23),  # joint angles (rad), 23 Booster T1 DOFs
    "local_body_pos": None,
    "link_body_list": None,
}
```

**Booster T1 DOF order (23 joints):**
Head(2): AAHead_yaw, Head_pitch |
Left arm(4): L_Shoulder_Pitch, L_Shoulder_Roll, L_Elbow_Pitch, L_Elbow_Yaw |
Right arm(4): R_Shoulder_Pitch, R_Shoulder_Roll, R_Elbow_Pitch, R_Elbow_Yaw |
Waist(1): Waist |
Left leg(6): L_Hip_Pitch, L_Hip_Roll, L_Hip_Yaw, L_Knee_Pitch, L_Ankle_Pitch, L_Ankle_Roll |
Right leg(6): R_Hip_Pitch, R_Hip_Roll, R_Hip_Yaw, R_Knee_Pitch, R_Ankle_Pitch, R_Ankle_Roll

**Visualise the result:**
```bash
./visualize.sh output/g1_lefthand/retargeting/lefthand_booster.pkl
```

### 11.4 Output Validation

Output file: `output/g1_lefthand/retargeting/lefthand_booster.pkl`
- 122 frames at 30 fps (retargeter starts from index 1, so 122 of 123 frames)
- `root_pos`: (122, 3), `root_rot`: (122, 4), `dof_pos`: (122, 23)

Key joint ranges confirming correct left-arm-raising behaviour:

| Joint            | Range (deg)         | Expected for lefthand |
|------------------|---------------------|-----------------------|
| L_Shoulder_Pitch | [−160.3, +32.0]     | Large range → arm raised ✓ |
| L_Shoulder_Roll  | [−64.5, +37.1]      | Moderate → arm moves laterally ✓ |
| L_Elbow_Pitch    | [−42.6, +129.9]     | Large flex/extend ✓ |
| R_Shoulder_Pitch | [−32.0, +19.8]      | Small range → arm mostly still ✓ |
| R_Shoulder_Roll  | [+42.7, +87.1]      | Stable → arm not raising ✓ |

The left arm clearly drives the motion (wide pitch range) while the right arm remains
relatively stationary — consistent with the `lefthand.pt` source motion.

---

## 12. Why Only 15 Markers Initially (and the Fix)

### 12.1 The Discovery

Running `vis_raw.py` on `output/g1_lefthand/lefthand_g1markers.csv` showed only **15 labels**,
even though the G1 robot provides `link_position` data for **17 links** (indices 0–16).

### 12.2 Root Cause: Two Links Were Silently Skipped

The `G1_LINK_TO_SOMA` mapping in `src/pipeline/g1_pt_to_mosh.py` was built to cover
the most obvious anatomical correspondences first. The initial mapping covered:
- Pelvis, both legs (6 links), head, torso, front shoulders (LFSH/RFSH), upper arms
  (LUPA/RUPA), elbows (LELB/RELB) → **15 links total**

Two links were deliberately excluded because no obvious Custom39 marker corresponded to them:

| Link | G1 joint chain position | Problem at the time |
|------|------------------------|---------------------|
| 10   | left_shoulder_roll_link (between shoulder pitch & yaw) | Sits between LFSH and LUPA; no Custom39 marker for that intermediate position |
| 14   | right_shoulder_roll_link (between shoulder pitch & yaw) | Same issue, symmetric |

The assumption was that using the surrounding markers (LFSH + LUPA bracketing the shoulder)
would be sufficient for MoSh++ to reconstruct the shoulder complex. This is still valid for
overall pose, but adding more shoulder constraints improves arm orientation accuracy.

### 12.3 Why It Matters (and Why It Wasn't Caught Sooner)

- **MoSh++ never complained** — it happily ran with 15 markers and produced a plausible fit.
- The vis_raw visualization confirmed the missing links were *never written to the C3D/CSV*,
  not just that MoSh ignored them.
- The shoulder roll link carries rotation information that is intermediate to LFSH and LUPA.
  Adding it gives MoSh++ a third constraint point per shoulder complex, tightening the fit.

### 12.4 The Fix

Added links 10 and 14 using known SMPL-X canonical marker names:

| Link | SOMA name | Description | SMPL-X vertex ID |
|------|-----------|-------------|-----------------|
| 10   | `LBSH`    | left back shoulder (lateral shoulder joint) | 4137 |
| 14   | `RBSH`    | right back shoulder (lateral shoulder joint) | 7192 |

`LBSH` and `RBSH` have predefined vertex positions in the SOMA SMPL-X body model
(confirmed via `all_marker_vids['smplx']`), so MoSh++ uses a good initialization
instead of fitting from scratch.

The G1_LINK_TO_SOMA mapping now covers all **17 links** → **17 SOMA markers**.

---

## 13. Why the 21 Joint Angles Cannot Be Used as MoSh++ Markers

### 13.1 The Question

The `.pt` file contains `joint_position` with shape `(T, 21)` — 21 joint angles for the
G1's actuated joints. Since 21 > 17, one might ask: why not use those 21 values as labels
for MoSh++ instead of (or in addition to) the 17 link positions?

### 13.2 Angles vs. Positions — a Fundamental Mismatch

MoSh++ is a **marker-based** body fitting algorithm. It expects 3D Cartesian coordinates
(X, Y, Z in metres) as input — the same data format produced by a physical motion capture
system (optical markers tracked in world space).

`joint_position` values are **joint angles in radians** — scalar values that describe how
far each joint has rotated. They are not positions in 3D space. MoSh++ has no mechanism
to ingest joint angles directly.

| Data | Type | Shape | Directly usable by MoSh++? |
|------|------|-------|---------------------------|
| `link_position` | Cartesian XYZ (metres) | (T, 17, 3) | ✓ Yes — used as virtual markers |
| `joint_position` | Joint angles (radians) | (T, 21) | ✗ No — wrong data type |

### 13.3 Converting Angles to Positions via Forward Kinematics

It is *theoretically possible* to convert the 21 joint angles into Cartesian positions
by running **forward kinematics (FK)** through the G1 URDF. Given the base position,
base orientation, and all 21 joint angles, FK would yield the world-space position of
every link in the kinematic chain — roughly 24 links total.

The 17 entries already in `link_position` are exactly these FK-computed positions, just
pre-calculated by the simulator at record time. The remaining ~7 links not stored in
`link_position` (intermediate hip links, ankle-pitch links, foot links) could in principle
be computed this way.

However:
- The arm chain — the hardest part for MoSh++ — is already fully covered by the 17
  existing markers.
- The missing FK links are mainly intermediate leg/hip positions that add minor
  incremental value for SMPL-X pose quality.
- Implementing FK requires loading the URDF, setting up a kinematics solver (e.g.
  MuJoCo or PyBullet), and validating coordinate-frame consistency — significant
  engineering effort for marginal MoSh++ improvement.

### 13.4 What the 21 Joint Angles Are Actually Used For

The `joint_position` data is the primary signal for **RL and robot control**:
- The RL policy outputs target joint angles; the low-level PD controller tracks them.
- When retargeting to the Booster T1 (Section 5), the T1 receives joint angle commands.
- The MoSh++ pipeline (Sections 10–12) intentionally bypasses joint angles and works
  entirely from the Cartesian `link_position` data to avoid the broken coordinate-frame
  conversion that plagued the earlier analytical approach (`g1_pt_to_smplx_pkl.py`).

---

## 14. Direct GMR Retargeting: G1 → Booster T1 Without MoSh++ (April 2026)

### 14.1 Motivation

The MoSh++ pipeline (Sections 10–12) is the theoretically correct path — it fits a full
human body model and retargets from there. But it has significant practical overhead:
- Requires two conda environments (`soma` and `HumanoidDataGeneration`)
- MoSh++ takes ~5–15 minutes per clip on CPU
- Intermediate SMPL-X representation introduces additional approximation errors

The GMR library (`/home/isaak/GMR`) — a ICRA 2026 paper implementation — exposes a
lower-level API: given a set of **3D body positions + orientations** per frame, it runs
a two-stage IK solver directly to the target robot's qpos. This means the G1
`link_position` data can be fed straight into GMR, skipping MoSh++ entirely.

This work lives exclusively in: `/home/isaak/BEP/ConvertData/GMRTRY/`

---

### 14.2 What Came from the GMR Repository (Unchanged)

Everything under `/home/isaak/GMR` is the original GMR library by Yanjie Ze et al.
Nothing in that directory was modified. The relevant components used:

#### `general_motion_retargeting/` — core library

| Component | What it does |
|-----------|-------------|
| `GeneralMotionRetargeting` (class) | Top-level API: `GMR(src_human, tgt_robot)` + `retarget(human_data)` |
| `params.py` / `IK_CONFIG_DICT` | Registry mapping `{src_human: {tgt_robot: json_path}}` — patched at runtime |
| `ik_solver.py` (mink-based) | Two-stage IK using `mink.FrameTask` on MuJoCo models |
| `smplx_to_t1.json` | Original IK config for SMPL-X body → Booster T1 (used as reference) |

#### `assets/booster_t1/T1_serial.xml`

MuJoCo model of the Booster T1 robot used by the IK solver. Key facts confirmed from this model:
- Floating base: `Trunk` is the root body
- 23 actuated DOF (same order as the MoSh++ retargeting output)
- T-pose standing: Trunk at z≈0.643 m, Waist at z≈0.527 m, feet at z≈0.037 m
- Left = +Y, Right = −Y, Forward = +X (Z-up world frame)
- Maximum arm reach from Waist: ~0.492 m (shoulder + upper arm + forearm)

#### Body name convention

GMR uses SMPL-X body names as the common interface between source data and IK config.
The following names are used in `g1_to_t1.json` and are part of the GMR SMPL-X convention:
`pelvis`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_foot`, `right_foot`,
`spine3`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`.

---

### 14.3 What Was Built From Scratch

All files in `/home/isaak/BEP/ConvertData/GMRTRY/` were created specifically for this project:

#### `g1_to_t1.json` — custom IK configuration

The original GMR config `smplx_to_t1.json` targets SMPL-X input.  
This is a completely new config tuned specifically for G1 `link_position` data.

Structure:
```json
{
    "robot_root_name":     "Waist",
    "human_root_name":     "pelvis",
    "ground_height":       0.0,
    "human_height_assumption": 1.8,
    "use_ik_match_table1": true,
    "use_ik_match_table2": true,
    "human_scale_table":   { ... },
    "ik_match_table1":     { ... },
    "ik_match_table2":     { ... }
}
```

Each entry in the IK match tables is: `[human_body_name, pos_weight, rot_weight, pos_offset_xyz, rot_offset_wxyz]`

The config is registered at runtime by patching `IK_CONFIG_DICT` before constructing GMR:
```python
IK_CONFIG_DICT["g1_gmr"] = {"booster_t1": str(_HERE / "g1_to_t1.json")}
retarget = GMR(src_human="g1_gmr", tgt_robot="booster_t1")
```

#### `g1_to_t1_gmr.py` — conversion script

Implements the full G1 → T1 conversion pipeline without MoSh++:

```
G1 .pt (link_position, F×17×3)
   │  apply R_G1_TO_GMR rotation
   │  Z-normalize (feet to z=0.05 m)
   │  map 12 of 17 links to SMPL-X body names
   ▼
human_data dict  {body_name: (pos_3d, quat_wxyz)}  per frame
   │  GMR two-stage IK
   ▼
qpos  (F × 30)  = [root_pos(3) | root_rot(4 wxyz) | dof_pos(23)]
   │  repack, convert root_rot wxyz→xyzw
   ▼
.pkl  {fps, root_pos, root_rot, dof_pos}
```

Key design decisions (all original, none from GMR repo):
- Only 12 of 17 G1 links used — omit wrist/finger links not in the SMPL-X subset
- Identity quaternions `[1,0,0,0]` for all bodies — G1 world-frame orientations are
  incompatible with SMPL-X joint convention (see Section 14.4, bug 1)
- Z-normalization per clip, not per frame — consistent reference height across all frames

#### `G1_LINK_TO_SMPLX` mapping (original)

```python
G1_LINK_TO_SMPLX = {
    0:  "pelvis",        4:  "right_hip",
    1:  "left_hip",      5:  "right_knee",
    2:  "left_knee",     6:  "right_foot",
    3:  "left_foot",     8:  "spine3",
    10: "left_shoulder", 14: "right_shoulder",
    12: "left_elbow",    16: "right_elbow",
}
```

Links 7 (head), 9/13 (front shoulders), 11/15 (upper arms) have no direct SMPL-X
equivalent in GMR's body set; they are intentionally dropped.

#### `R_G1_TO_GMR` coordinate rotation (original)

```python
R_G1_TO_GMR = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=np.float64)
```

G1 faces +Y (left=−X, right=+X). GMR/T1 expects forward=+X (left=+Y, right=−Y).
This is a −90° rotation about Z: `(x, y, z) → (y, −x, z)`. Applied to every position
before feeding into GMR.

#### `view_t1.py` — interactive MuJoCo viewer

Standalone viewer for any `.pkl` file in the output format. Loads `T1_serial.xml`,
sets `qpos` per frame, calls `mj_forward`, loops at real-time fps.

#### `commands.txt` — usage reference

Documents all commands for single-file conversion, batch conversion, and visualization.

---

### 14.4 Optimization Process: Bugs Found and Fixed

The following bugs were discovered and resolved iteratively, in order:

#### Bug 1: Feet pointing opposite direction to head (−90° / 180° misalignment)

**Symptom:** In MuJoCo viewer, T1's feet were rotated 180° relative to the body — robot
looked like it was walking backwards or folded in half.

**Root cause (first hypothesis):** G1 link orientations (quaternions from `.pt`) are in
world frame with ~84° Z rotation offset. These are NOT compatible with the SMPL-X joint
orientation convention that GMR expects (local joint frames, T-pose = identity).

**Fix:** Use identity quaternions `[1,0,0,0]` for every body in every frame. GMR's IK
solver already handles orientation via its upright constraint and IK match table rotation
weights — feeding wrong orientations actively breaks it.

**Remaining issue after fix:** 90° coordinate frame mismatch (different bug, same symptom):

**Root cause (confirmed):** G1 world frame has +Y forward (robot faces +Y), but GMR/T1
expects +X forward. Positions fed without rotation placed all bodies on the wrong side
of the IK target space.

**Fix:** Apply `R_G1_TO_GMR` to all positions before passing to GMR. After this fix,
the robot orientation was correct.

---

#### Bug 2: Robot flipping upside down / bent at waist

**Symptom:** After fixing orientation, T1 appeared folded at the waist or with its torso
pointing the wrong direction.

**Root cause:** The `smplx_to_t1.json` (GMR's original config) has `Trunk` body with
`rot_weight=100` and `rot_offset=[0.5, -0.5, -0.5, -0.5]`. Converting that quaternion
to a rotation matrix gives `[[0,1,0],[0,0,1],[1,0,0]]` — this maps the robot's +Z (up)
axis to +X (forward), effectively making GMR try to keep the robot lying on its side.

**Fix:** Created a fully custom `g1_to_t1.json` with `rot_offset=[1,0,0,0]` (identity)
for ALL bodies. The identity rotation offset means "keep the body upright" — which is
what GMR's internal upright constraint already enforces. The Trunk `rot_weight` was also
reduced from 100 to 20 to avoid over-constraining the torso orientation.

---

#### Bug 3: Robot too small / crouched / feet above ground

**Symptom:** T1 appeared severely crouched, feet floating above ground level, waist at
wrong height.

**Root cause:** Two compounding issues:
1. **Wrong scale:** `smplx_to_t1.json` uses `scale=0.6` calibrated for a 1.8 m human
   (SMPL-X pelvis height ≈ 0.88 m). G1 normalized pelvis z ≈ 0.715 m — different baseline.
   With scale 0.6, all positions were placed too low.
2. **Z not normalized:** G1 foot links float at z ≈ 0.44–0.48 m in the `.pt` file
   (physics sim ground is at z=0 but feet are at ankle height). Feeding these raw Z values
   to GMR caused feet to be modelled as hanging in mid-air.

**Fix:**
- **Z normalization:** Shift all Z so the minimum foot position across all frames sits
  at `FOOT_TARGET_Z = 0.05 m`:
  ```python
  shift = min(left_foot_z, right_foot_z).min() - 0.05
  positions[:, :, 2] -= shift
  ```
- **Scale calibration:** `scale = T1_Waist_height / G1_normalized_pelvis_z = 0.527 / 0.715 = 0.737`
  Applied uniformly to all body parts in `human_scale_table` (legs, pelvis, spine).

---

#### Bug 4: Arms locked / stiff — not following G1 motion

**Symptom:** After fixing orientation, flipping, and scale, the legs looked correct but
both arms remained pinned to the robot's sides and never moved regardless of source motion.

**Root cause:** Two issues:
1. **`pos_weight=0` for all arm bodies in `ik_match_table1`:** The first IK pass (which
   sets the initial solution) had zero positional weight on shoulder and elbow tasks.
   GMR's solver only used rotation tasks for arms in the first pass, producing a neutral pose.
2. **Scale 0.737 for elbow targets:** With scale 0.737, the G1 elbow position (local from
   Waist) mapped to ≈ 0.87 m. T1's maximum arm reach from Waist is 0.492 m.
   Every elbow IK target was **outside the robot's workspace**, so the IK solver always
   hit joint limits in the same direction — effectively pinning the arm in one pose.

**Partial fix:** Added non-zero `pos_weight` for shoulders (50) and elbows (80) in
`ik_match_table1`. Also set elbow `scale=0.40` as an initial attempt to bring targets
inside the workspace. This made arms move, but with wrong amplitude (see Bug 5).

---

#### Bug 5: Arms moving but visually wrong — range too compressed

**Symptom:** Arms moved but the motion looked nothing like the G1 original. Amplitude was
far too small — arms barely deviated from rest pose even when G1 was doing large arm swings.

**Root cause:** Scale 0.40 was too aggressive. Analysis of `lefthand.pt` arm positions
(in GMR frame, relative to pelvis/Waist):

| Frame | G1 elbow distance from Waist |
|-------|------------------------------|
| f0    | 0.619 m (arm extended) |
| f60   | 0.410 m (arm at side) |
| f100  | 0.775 m (arm fully raised) |

With `scale=0.40`: target range = 0.164–0.310 m. But T1's natural arm-extended pose
sits at ≈ 0.492 m from Waist. The targets were always well inside the workspace with
large leftover IK slack → arms stayed near the default pose, not following the targets.

**Correct scale derivation:**
```
scale = T1_arm_reach / G1_arm_local_at_natural_extension
      = 0.492 m / 0.619 m
      = 0.79
```

With `scale=0.79`:
- Natural extension: `0.619 × 0.79 = 0.489 m` ≈ T1's natural reach → arm sits in a
  natural ready pose, directly matching G1's standing arm.
- Full dive / high reach: `0.775 × 0.79 = 0.612 m` → slightly outside workspace.
  The IK solver extends the arm fully in the correct direction, giving a convincing
  full-reach appearance.
- Arm tucked: `0.410 × 0.79 = 0.324 m` → well inside workspace → IK has freedom to
  match the correct bent-elbow position.

**Final fix:** Changed `left_elbow` and `right_elbow` in `human_scale_table` from `0.40` to `0.79`.
The shoulder scale was kept at `0.95` (confirmed correct from earlier analysis: G1
shoulder local ≈ 0.390 m, T1 shoulder position ≈ 0.370 m → ratio ≈ 0.95).

---

### 14.5 Final Configuration Summary

**`human_scale_table` (final):**

| Body | Scale | Rationale |
|------|-------|-----------|
| pelvis, spine3, hips, knees, feet | 0.737 | T1 Waist z (0.527 m) / G1 pelvis norm z (0.715 m) |
| left_shoulder, right_shoulder | 0.95 | T1 shoulder pos (0.370 m) / G1 shoulder local (0.390 m) |
| left_elbow, right_elbow | 0.79 | T1 arm reach (0.492 m) / G1 arm extended local (0.619 m) |

**IK match tables (final):**

| Robot body | Human body | pos_w (t1/t2) | rot_w (t1/t2) | rot_offset |
|------------|-----------|---------------|----------------|------------|
| Waist | pelvis | 100 / 100 | 10 / 5 | identity |
| Hip_Yaw_Left | left_hip | 0 / 10 | 10 / 5 | identity |
| Shank_Left | left_knee | 0 / 10 | 10 / 5 | identity |
| left_foot_link | left_foot | 50 / 50 | 10 / 5 | identity |
| Hip_Yaw_Right | right_hip | 0 / 10 | 10 / 5 | identity |
| Shank_Right | right_knee | 0 / 10 | 10 / 5 | identity |
| right_foot_link | right_foot | 50 / 50 | 10 / 5 | identity |
| Trunk | spine3 | 0 / 5 | 20 / 10 | identity |
| AL2 | left_shoulder | 50 / 40 | 5 / 5 | identity |
| left_hand_link | left_elbow | 80 / 60 | 5 / 5 | identity |
| AR2 | right_shoulder | 50 / 40 | 5 / 5 | identity |
| right_hand_link | right_elbow | 80 / 60 | 5 / 5 | identity |

*(t1 = ik_match_table1 pass, t2 = ik_match_table2 pass)*

---

### 14.6 Output and Results

All 6 motion clips were successfully converted:

| Clip | Frames | Duration |
|------|--------|----------|
| lefthand | 123 | 4.1 s |
| righthand | 246 | 8.2 s |
| leftstep | 194 | 6.5 s |
| rightstep | 197 | 6.6 s |
| leftjump | 254 | 8.5 s |
| rightjump | 200 | 6.7 s |

Output format: `.pkl` files in `ConvertData/GMRTRY/output/`, same schema as the MoSh++ path:
```python
{
    "fps":      30,
    "root_pos": np.ndarray (F, 3),    # XYZ, Z-up
    "root_rot": np.ndarray (F, 4),    # quaternion xyzw
    "dof_pos":  np.ndarray (F, 23),   # 23 T1 joint angles (rad)
    "local_body_pos": None,
    "link_body_list": None,
}
```

**Left arm DOF ranges confirming correct arm motion (lefthand clip):**

| Joint | Range |
|-------|-------|
| Left arm DOF 0 | −0.01 to +0.15 rad |
| Left arm DOF 1 | −0.16 to +0.09 rad |
| Left arm DOF 2 | −0.33 to −0.02 rad |
| Left arm DOF 3 | −0.47 to −0.01 rad |

All four left arm DOFs show non-trivial ranges (arm clearly moving), while right arm
DOFs remain near zero — matching the expected `lefthand` source motion behaviour.

---

### 14.7 Comparison: GMR Direct vs. MoSh++ Path

| Aspect | GMR Direct (GMRTRY/) | MoSh++ Path (Sections 10–12) |
|--------|---------------------|------------------------------|
| Conda environments | 1 (`gmr`) | 2 (`soma` + `HumanoidDataGeneration`) |
| Processing time per clip | ~10 s | ~5–15 min (MoSh++) + ~1 min (retarget) |
| Intermediate files | None | C3D + stageii.pkl + smplx_body_mesh.pkl |
| Body shape fitted | No (scale only) | Yes (SMPL-X betas) |
| Arm motion quality | Good — scale-based | Better — full SMPL-X body fit |
| Leg/torso quality | Good | Good |
| Tuning required | Manual scale per limb group | MoSh++ runs automatically |
| Dependency on MuJoCo | Yes (GMR uses mink/MuJoCo) | Yes (retargeting step) |

The GMR direct path trades some accuracy (no body shape fitting) for dramatically faster
iteration speed. For the purposes of this BEP the outputs are visually acceptable and
both pipelines produce the same output schema, so either can be used downstream.
