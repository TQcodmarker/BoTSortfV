# DeepSORVF++ Implementation Context

This repository contains the original DeepSORVF demo and a BoT-SORT subtree for
the DeepSORVF++ refactor. Future coding work should follow the data facts below
instead of guessing input formats.

## Dataset Layout

Example clip:

```text
clip-01/
  2022_06_04_12_05_12_12_07_02_b.mp4
  camera_para.txt
  ais/
    2022_06_04_12_05_12.csv
    2022_06_04_12_05_13.csv
    ...
  gt/
    clip-01_gt_detection.txt
    clip-01_gt_tracking.txt
    clip-01_gt_fusion.txt
    clip-01_gt.mp4
```

The AIS files are split by wall-clock second. File names encode time as:

```text
YYYY_MM_DD_HH_MM_SS.csv
```

The video file name encodes the initial and final times. The current demo reads
the initial time from the video file name in `utils/file_read.py`.

## AIS CSV Format

AIS records are raw geographic observations, not pre-projected image points.

Sample header from `clip-01/ais/2022_06_04_12_05_12.csv`:

```text
,mmsi,lon,lat,speed,course,heading,type,timestamp
```

Sample row:

```text
0,110000000,114.32583,30.60115833,0.9,142.5,511,18,1654315502004
```

Field meanings used by existing code:

```text
mmsi       anonymized vessel id
lon        longitude in degrees
lat        latitude in degrees
speed      AIS speed over ground, treated as knots by existing projection code
course     course over ground in degrees
heading    heading in degrees; 511 appears as unavailable/invalid AIS heading
type       AIS vessel/type code
timestamp  Unix epoch timestamp in milliseconds
```

Important: do not treat AIS as high frequency. The demo updates AIS processing
only near each whole-second boundary, while video frames advance at frame time
steps.

## Video Time Model

The current demo derives:

```python
fps = int(cap.get(5))
t = int(1000 / fps)
```

Each frame advances time by `t` milliseconds through `utils/file_read.update_time`.
The returned `timestamp` is a Unix epoch timestamp in milliseconds. The returned
`Time_name` is used to find the per-second AIS CSV file.

In DeepSORVF++ code, timestamp deltas for AIS reliability should be computed in
seconds:

```text
delta_t_sec = abs(frame_timestamp_ms - ais_timestamp_ms) / 1000.0
```

Use exponential decay for delayed AIS:

```text
lambda(delta_t) = exp(-kappa * delta_t_sec)
```

Discard or ignore AIS when `delta_t_sec > ais_max_age`.

## Camera Parameters and Projection

`clip-01/camera_para.txt` contains one Python-list-like row:

```text
[114.32722222222222, 30.60027777777778, 352, -4, 20, 55, 30.94, 2391.26, 2446.89, 1305.04, 855.214]
```

Existing parameter order in `utils/AIS_utils.py`:

```text
0 lon_cam
1 lat_cam
2 shoot_hdir
3 shoot_vdir
4 height_cam
5 FOV_hor
6 FOV_ver
7 f_x
8 f_y
9 u0
10 v0
```

Existing projection utilities:

```text
utils/AIS_utils.py::visual_transform(lon_v, lat_v, camera_para, shape)
utils/AIS_utils.py::transform(AIS_current, AIS_vis, camera_para, shape)
utils/AIS_utils.py::AISPRO.process(camera_para, timestamp, Time_name)
```

`visual_transform` returns image coordinates `(x, y)` using the top-left image
origin convention. Existing code passes `im_shape = [cap.get(3), cap.get(4)]`,
so shape is `[width, height]`.

For DeepSORVF++, prefer reusing the existing projection math initially. The AIS
virtual observation should use projected `x, y`, while width and height must
come from the Kalman prediction, not AIS.

## Existing AIS Low-Frequency Logic

Existing `utils/AIS_utils.py` behavior:

```text
AISPRO.process(...) updates AIS only when timestamp % 1000 < t.
data_pred(...) propagates vessels missing from the current second using speed
and course.
data_pre(...) uses pyproj.Geod.fwd and converts speed in knots to meters through
distance = speed * dt_hours * 1852.
AISPRO.time_lim = 2 means projected AIS history is retained for 2 minutes in
the original visualization/fusion pipeline.
```

For DeepSORVF++, do not blindly reuse 2 minutes as a KF fusion window. Use a much
shorter `ais_max_age` for state-level virtual observation, e.g. 2 seconds by
default, with reliability decay.

## Ground Truth / Result Text Format

Examples from `clip-01/gt/clip-01_gt_tracking.txt`:

```text
2,0,308,683,439,84,1,1,1,1
2,1,1355,647,125,55,1,1,1,1
```

Examples from `clip-01/gt/clip-01_gt_fusion.txt`:

```text
2,380000000,308,683,439,84,1,1,1,1
2,290000000,1355,647,125,55,1,1,1,1
```

Interpretation:

```text
tracking: frame_id, track_id, x, y, w, h, ...
fusion:   frame_id, mmsi,     x, y, w, h, ...
```

Bounding boxes are stored as top-left `x, y, w, h`.

## DeepSORVF++ Coding Constraints

1. AIS state fusion must be optional and backward-compatible with existing
   BoT-SORT calls.
2. `BoTSORT.update(output_results, img)` should keep working. Extended calls may
   pass `ais_frame` and `timestamp`.
3. AIS virtual observation should be:

```text
z_virtual = [x_AIS, y_AIS, w_KF, h_KF]
R_virtual = diag([small_xy_var, small_xy_var, large_scale_var, large_scale_var])
```

4. AIS delay should increase position variance or lower AIS association weight.
5. If no valid AIS exists for a frame, tracking must degrade to vanilla BoT-SORT.
6. Avoid global optimization. Per-frame NumPy/PyTorch matrix operations and
   Hungarian matching are acceptable for the real-time target.
7. CMC compensation must keep AIS projected points in the same image coordinate
   frame as BoT-SORT tracks before virtual observation or AIS association cost.

