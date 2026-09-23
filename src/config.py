# ─────────────────────────────────────────────────────────────
# Central configuration AND shared ML utilities.
# Imported by: pipeline.py, collect_data.py, app.py
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
from collections import deque

import mediapipe as mp
_mp_hands   = mp.solutions.hands          # type: ignore[attr-defined]
_mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
DOCS_DIR     = os.path.join(BASE_DIR, "docs")

DATA_CSV     = os.path.join(DATA_DIR,   "gesture_data.csv")
MODEL_FILE   = os.path.join(MODEL_DIR,  "gesture_model.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR,  "label_encoder.pkl")
CM_IMAGE     = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# ── Gestures ───────────────────────────────────────────────────
GESTURES            = ["palm", "dislike", "rock", "ok", "fist"]
SAMPLES_PER_GESTURE = 5

GESTURE_TO_CMD = {
    "palm":        "FORWARD",
    "dislike":     "REVERSE",
    "rock":  "LEFT",
    "ok": "RIGHT",
    "fist":        "STOP",
}

# ── MediaPipe ──────────────────────────────────────────────────
NUM_LANDMARKS           = 21
FEATURES_PER_LANDMARK   = 3
FEATURE_DIM             = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE  = 0.6

# ── Camera ─────────────────────────────────────────────────────
CAM_INDEX  = 0
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 30

# ── Training ───────────────────────────────────────────────────
TEST_SIZE           = 0.2
RANDOM_STATE        = 42
CV_FOLDS            = 5
RF_N_ESTIMATORS     = 100
RF_MIN_SAMPLES_LEAF = 2
MLP_HIDDEN_LAYERS   = (128, 64)
MLP_MAX_ITER        = 500

# ── Inference ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.85
VOTE_WINDOW          = 5

# ── MQTT — HiveMQ Cloud ────────────────────────────────────────
MQTT_BROKER      = "29455b01c27447b488b1ec93488ce95d.s1.eu.hivemq.cloud"
MQTT_PORT_TLS    = 8883   # ESP8266 TCP + TLS
MQTT_PORT_WSS    = 8884   # Browser WebSocket + TLS
MQTT_USERNAME    = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD    = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC_CMD   = "gesturecar/command"
MQTT_TOPIC_STATUS= "gesturecar/status"

# ── Flask ──────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000


# ══════════════════════════════════════════════════════════════
# SHARED ML UTILITIES
# ══════════════════════════════════════════════════════════════

def build_hand_detector(detection_conf: float, tracking_conf: float):
    """
    Create MediaPipe Hands detector.
    Call once — reuse across all frames.
    """
    return _mp_hands.Hands(           # type: ignore[attr-defined]
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = detection_conf,
        min_tracking_confidence  = tracking_conf,
    )


def extract_features(landmarks) -> np.ndarray:
    """
    21 MediaPipe landmarks → 63-dim normalised float32 vector.
    Step 1: subtract wrist → position invariant.
    Step 2: divide by max abs → scale invariant.
    Returns values in [-1, +1].
    """
    wx, wy, wz = landmarks[0].x, landmarks[0].y, landmarks[0].z
    coords = []
    for lm in landmarks:
        coords.extend([lm.x - wx, lm.y - wy, lm.z - wz])

    coords  = np.array(coords, dtype=np.float32)
    max_val = np.max(np.abs(coords))

    if max_val < 1e-6:
        return np.zeros(63, dtype=np.float32)

    return coords / max_val


def process_frame(frame, detector):
    """
    Run MediaPipe on one BGR frame.
    Returns (result, rgb_frame).
    """
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = detector.process(rgb)
    rgb.flags.writeable = True
    return result, rgb


def get_landmark_list(result):
    """
    Extract first hand landmarks from MediaPipe result.
    Returns landmark list or None.
    """
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0].landmark
    return None


# Expose drawing utils for files that need it
# Import via: from config import mp_drawing_utils
mp_drawing_utils = _mp_drawing  # type: ignore[attr-defined]
mp_hands_module  = _mp_hands    # type: ignore[attr-defined]


class FastVoteBuffer:
    """
    Asymmetric vote buffer.
    STOP  → 1 frame  (~33ms)  — safety first.
    MOTION → 2 frames (~66ms) — reduced from 3 for faster response.
    """
    MOTION_VOTES = 2   # reduced from 3 for faster command response
    MOTION_CMDS  = {"FORWARD", "REVERSE", "LEFT", "RIGHT"}

    def __init__(self, confidence_threshold: float):
        self.buf         = deque(maxlen=self.MOTION_VOTES)
        self.conf_thresh = confidence_threshold
        self.last_stable = "STOP"

    def update(self, label: str, confidence: float,
               hand_present: bool) -> str:
        if not hand_present or confidence < self.conf_thresh:
            self.buf.clear()
            self.last_stable = "STOP"
            return "STOP"

        cmd = label.upper()
        if cmd == "STOP":
            self.buf.clear()
            self.last_stable = "STOP"
            return "STOP"

        if cmd in self.MOTION_CMDS:
            self.buf.append(cmd)
            if (len(self.buf) >= self.MOTION_VOTES and
                    all(v == self.buf[0] for v in self.buf)):
                self.last_stable = self.buf[0]
                return self.last_stable

        return self.last_stable

    def force_stop(self):
        self.buf.clear()
        self.last_stable = "STOP"