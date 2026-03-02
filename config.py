
TICK_INTERVAL = 1.0
FRAME_INTERVAL = 0.03

LOW_THRESHOLD = 0.6
RECOGNITION_THRESHOLD = 0.75

GRACE_PERIOD_SEC = 300

# Postgres config
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "johnny5"
PG_USER = "dmitriysarychev"
PG_PASSWORD = "postgres"

# L2 pose classification
POSE_PROTOTYPES_PATH = "pose_prototypes.json"
POSE_MIN_VISIBLE_KEYPOINTS = 6

# L2 pose smoothing
POSE_DISTANCE_EMA_ALPHA = 0.35
POSE_SWITCH_MARGIN = 0.06

# L2 MLP pose classifier
POSE_MLP_MODEL_PATH = "pose_mlp.pt"
POSE_MLP_LABELS_PATH = "pose_labels.json"
