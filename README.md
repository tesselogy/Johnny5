# Johnny5
Amalias robot


## L2 yoga pose classification (Roboflow-ready)

L2 now includes `PoseClassifier` (`l2_pose_classifier.py`) that classifies poses by **joint angles** (not raw coordinates).

1. Export/prepare your yoga dataset labels (e.g. from Roboflow).
2. Build per-class **angle prototypes** and save them to `pose_prototypes.json` in repository root:

```json
{
  "tree_pose": {
    "left_elbow": 168.5,
    "right_elbow": 172.0,
    "left_hip": 161.2,
    "right_hip": 172.3,
    "left_knee": 48.2,
    "right_knee": 177.1
  },
  "warrior_2": {
    "left_elbow": 175.0,
    "right_elbow": 174.1,
    "left_knee": 91.3,
    "right_knee": 171.8
  }
}
```

- Angles are in degrees `[0..180]`.
- Supported angle names: `left/right_elbow`, `left/right_shoulder`, `left/right_hip`, `left/right_knee`.
- Config options: `POSE_PROTOTYPES_PATH`, `POSE_MIN_VISIBLE_KEYPOINTS` in `config.py`.
- If prototypes are absent/insufficient, classifier falls back to heuristic `asana` labels from L1.
