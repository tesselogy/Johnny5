# Johnny5
Amalias robot


## L2 yoga pose classification (Roboflow-ready)

L2 now includes `PoseClassifier` (`l2_pose_classifier.py`) that can classify poses using normalized keypoint prototypes.

1. Export/prepare your yoga dataset labels (e.g. from Roboflow) and build per-class prototype keypoints.
2. Save them to `pose_prototypes.json` in repository root with format:

```json
{
  "tree_pose": {"nose": [0.51, 0.12], "left_shoulder": [0.43, 0.30]},
  "warrior_2": {"nose": [0.50, 0.10], "left_shoulder": [0.35, 0.29]}
}
```

- Coordinates are normalized to `[0, 1]` inside person crop.
- Config options: `POSE_PROTOTYPES_PATH`, `POSE_MIN_VISIBLE_KEYPOINTS` in `config.py`.
- If prototypes are absent, classifier falls back to heuristic `asana` labels from L1.
