# OptiTrack Motive, operator handbook

Step-by-step workflow from powering the system to exporting a CSV you can use in this repository’s pipeline. For processing that CSV (filter → MoSh++ → retarget → viewer), see the [README](README.md).

---

## 1. Startup

1. **Start the PC** and log in using the password on the **white paper** next to the computer.
2. **Power the OptiTrack system**: switch on the **extension cord** (power strip) so the cameras and hub receive power.
3. On the **desktop**, double-click **OptiTrack Motive** (shortcut on the **middle left** of the screen).
4. In Motive, click **Open** (top left). Navigate to the **TechUnited** folder and open the **three items that are not folders**, these are your **calibration and asset** files.
5. When everything is loaded correctly, you should see **blue rings** on the cameras in the 3D view.

---

## 2. Recording

1. **Put on the mocap suit** (markers placed as required for your layout).
2. Press the **red Record** button at the **bottom** of the Motive window to start.
3. When finished, press **Record** again to **stop**.

---

## 3. Open your take (recording pane)

1. Click the **third icon** in the small group at the **top right** of the Motive window, this opens the **Recording** pane (see screenshot below).

![Recording pane and main layout, use the third top-right icon to open this pane](SupportingPhotos/Screenshot%202026-04-17%20154609.png)

2. In the **left-hand list** of recordings, **double-click** the take you just captured. New takes are often **far down** the list.
3. After the take loads, check the **Assets** area (top left): if everything went well, you should see **High39V2Humanoid** and **Lower39V2Humanoid** listed for this recording.

If you **do not** see those two assets, use the section **Creating assets** below before continuing.

---

## 4. Manual labeling (marker pane)

With the recording open, switch to the **Labeling** workflow (see the same screenshot: **Labeling** tab, marker list, 3D view).

1. Note which **tools** are active in the **middle left** (highlighted buttons), use the **selection** tool as needed.
2. In the marker list, **click a name** (e.g. `Marker1`, `Marker2`, …).
3. In the **3D view**, **click the orange dot** that should correspond to that marker to assign the label.
4. Press **Space** to **play** the take, then **pause** again. Some markers may **lose** labels during playback, that is normal. **Re-assign** any that drop out so the trajectory stays consistent and the recording quality is good enough for the pipeline.

Repeat until labeling is stable enough for your segment.

---

## 5. Export tracking data

1. In the **Recording** pane, **right-click** the **name of your recording** in the list.
2. Choose **Export Tracking Data** (see screenshot).

![Right-click recording → Export Tracking Data](SupportingPhotos/Screenshot%202026-04-17%20161515.png)

3. In the export dialog, use these settings (see screenshot):

![CSV export settings, verify Markers, Header Information, device/world options, units, and rotation type](SupportingPhotos/Screenshot%202026-04-17%20164157.png)

**Recommended export settings:**

| Setting | Value |
|--------|--------|
| **Export type** | CSV (`*.csv`) |
| **Scale** | `1` |
| **Markers** | On |
| **Unlabeled Markers** | Off |
| **Quality Statistics** | Off |
| **Rigid Bodies / Rigid Body Markers / Bones / Bone Markers** | Off |
| **Header Information** | On |
| **Export Device Data** | On |
| **Use World Coordinates** | On |
| **Rotation type** | Quaternion |
| **Units** | Meters |
| **Frame range** | Working range (start/end as needed) |

4. Before exporting, confirm in the **Assets** list that only **High39V2Humanoid** and **Lower39V2Humanoid** are **solved**: you should see **gray checkmarks** to the **left and right** of each name.
5. Click **Export**, choose a save location, then copy the **CSV** to your machine (e.g. **USB stick**) for use with `data/` and the pipeline in the README.

---

<details>
<summary><strong>Creating assets</strong> (only if High39 / Lower39 are missing)</summary>

Use this when **High39V2Humanoid** and **Lower39V2Humanoid** do not appear after loading the take, for example if the humanoid assets were removed from the project.

![Builder / asset setup, select marker counts and follow the template mapping](SupportingPhotos/Screenshot%202026-04-17%20164157.png)

1. Open the **Builder** pane (first pane in the layout shown above, left side of the Motive window).
2. For **Lower39V2Humanoid**, set the marker count to **20**.
3. For **High39V2Humanoid**, set the marker count to **19**.
4. Apply the **marker index → name mapping** from **`SupportingPhotos/Template.png`** (placeholder, add the diagram when it is ready). Match each rigid-body marker index to the correct anatomical label as in that template.
5. Return to **Section 4** (manual labeling) and label markers in the 3D view as described, then export as in **Section 5**.

</details>

---

## Quick checklist

- [ ] TechUnited calibration + three non-folder files opened; cameras show blue rings  
- [ ] Take recorded and opened from the recording pane  
- [ ] High39 + Lower39 present (or assets recreated via **Creating assets**)  
- [ ] Markers labeled; playback checked; dropped labels fixed  
- [ ] Export Tracking Data with CSV settings above; only High/Lower solved  
- [ ] CSV copied off the capture PC  

Then continue with [Running the Full Pipeline](README.md#running-the-full-pipeline) in the README.
