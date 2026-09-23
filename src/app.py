# ─────────────────────────────────────────────────────────────
# Flask backend server — LAPTOP MODE for development/testing.
# Runs MediaPipe + ML inference locally on laptop.
# Sends commands to ESP8266 via HiveMQ MQTT broker.
# ─────────────────────────────────────────────────────────────

import os, sys, cv2, pickle, time, threading, numpy as np
import mediapipe as mp
import paho.mqtt.client as paho_mqtt
from flask import Flask, Response, render_template_string, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_FILE, ENCODER_FILE,
    CAM_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    FLASK_HOST, FLASK_PORT,
    GESTURES, GESTURE_TO_CMD,
    MQTT_BROKER, MQTT_PORT_TLS,
    MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_TOPIC_CMD, MQTT_TOPIC_STATUS,
    # Utilities now live in config.py
    build_hand_detector, process_frame,
    get_landmark_list, extract_features, FastVoteBuffer,
)

mp_draw_utils = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_hands_mod  = mp.solutions.hands          # type: ignore[attr-defined]

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Gesture RC Car — Dev Dashboard</title>
<!-- Use cdnjs instead of unpkg — not blocked by Edge tracking prevention -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
:root{
  --bg:#0f1117;--surface:#1a1d27;--border:#2a2d3a;
  --text:#e8eaf0;--muted:#6b7080;
  --green:#00e676;--blue:#4dabf7;--amber:#ffb300;
  --red:#ff5252;--purple:#b39ddb;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
  font-family:'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  min-height:100vh;
}

/* ── Header ── */
header{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:14px 24px;display:flex;justify-content:space-between;
  align-items:center;
}
.logo{display:flex;align-items:center;gap:12px;}
.logo h1{font-size:18px;font-weight:700;}
.logo p{font-size:12px;color:var(--muted);margin-top:2px;}
.pills{display:flex;gap:8px;flex-wrap:wrap;}
.pill{
  font-size:11px;padding:5px 12px;border-radius:20px;
  border:1px solid var(--border);color:var(--muted);
  background:var(--bg);transition:all .3s;white-space:nowrap;
}
.pill.on {color:var(--green);border-color:var(--green);}
.pill.err{color:var(--red);border-color:var(--red);}
.pill.warn{color:var(--amber);border-color:var(--amber);}

/* ── Main grid ── */
main{
  display:grid;
  grid-template-columns:1fr 380px;
  grid-template-rows:auto auto;
  gap:16px;padding:16px 24px;
  max-width:1400px;margin:0 auto;
}

/* ── Cards ── */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:18px;
}
.card h2{
  font-size:11px;text-transform:uppercase;letter-spacing:.8px;
  color:var(--muted);margin-bottom:14px;font-weight:500;
}

/* ── Camera ── */
.feed-card{grid-row:1 2;}
.video-wrap{
  position:relative;background:#000;border-radius:8px;
  overflow:hidden;aspect-ratio:4/3;
}
#video-feed{width:100%;height:100%;object-fit:cover;display:block;}
.fps-badge{
  position:absolute;bottom:10px;right:10px;
  background:rgba(0,0,0,.7);color:var(--muted);
  font-size:11px;padding:3px 8px;border-radius:4px;
  font-family:monospace;
}

/* ── Right column ── */
.right-col{
  grid-column:2;grid-row:1/3;
  display:flex;flex-direction:column;gap:16px;
}

/* ── Command ── */
.cmd-display{
  font-size:48px;font-weight:900;letter-spacing:2px;
  text-align:center;padding:16px 0;
  color:var(--red);transition:color .2s;
}
.cmd-display.FORWARD{color:var(--green);}
.cmd-display.REVERSE{color:var(--blue);}
.cmd-display.LEFT   {color:var(--purple);}
.cmd-display.RIGHT  {color:var(--amber);}
.gesture-row,.conf-row{
  display:flex;align-items:center;gap:10px;
  margin-bottom:8px;font-size:13px;
}
.row-label{color:var(--muted);min-width:80px;font-size:11px;
  text-transform:uppercase;letter-spacing:.5px;}
.conf-bar-wrap{flex:1;display:flex;align-items:center;gap:8px;}
.conf-track{
  flex:1;height:8px;background:var(--bg);
  border-radius:4px;overflow:hidden;
}
.conf-fill{
  height:100%;background:var(--green);
  border-radius:4px;width:0;transition:width .2s;
}
.conf-val{font-size:12px;color:var(--muted);min-width:36px;}

/* ── D-pad ── */
.dpad{display:flex;flex-direction:column;align-items:center;gap:6px;}
.dpad-row{display:flex;gap:6px;}
.dpad-btn{
  width:58px;height:58px;border-radius:10px;
  background:var(--bg);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;color:var(--muted);transition:all .15s;
}
.dpad-btn.active{
  background:#0d2b1a;border-color:var(--green);
  color:var(--green);box-shadow:0 0 14px rgba(0,230,118,.3);
}
.dpad-btn.active.stop-btn{
  background:#2a0d0d;border-color:var(--red);
  color:var(--red);box-shadow:0 0 14px rgba(255,82,82,.3);
}

/* ── MQTT status ── */
.mqtt-info{
  background:var(--bg);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;
  font-size:12px;color:var(--muted);line-height:1.8;
}
.mqtt-info strong{color:var(--text);}

/* ── Stats ── */
.stat-row{
  display:flex;justify-content:space-between;
  padding:7px 0;border-bottom:1px solid var(--border);
  font-size:13px;
}
.stat-row:last-child{border-bottom:none;}
.stat-row span:last-child{
  font-family:monospace;color:var(--blue);font-weight:600;
}

/* ── Log ── */
.log-card{grid-column:1;grid-row:2;}
.log-table{width:100%;border-collapse:collapse;font-size:13px;}
.log-table th{
  text-align:left;padding:6px 12px;color:var(--muted);
  font-weight:500;border-bottom:1px solid var(--border);
  font-size:11px;text-transform:uppercase;
}
.log-table td{padding:8px 12px;border-bottom:1px solid var(--border);}
.log-table tr:last-child td{border-bottom:none;}
.log-empty{color:var(--muted);text-align:center;padding:20px;}
.cmd-tag{font-weight:700;}
.cmd-tag.FORWARD{color:var(--green);}
.cmd-tag.REVERSE{color:var(--blue);}
.cmd-tag.LEFT   {color:var(--purple);}
.cmd-tag.RIGHT  {color:var(--amber);}
.cmd-tag.STOP   {color:var(--red);}

@media(max-width:900px){
  main{grid-template-columns:1fr;grid-template-rows:auto;}
  .right-col{grid-column:1;grid-row:auto;}
  .log-card{grid-column:1;grid-row:auto;}
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <span style="font-size:24px">⬡</span>
    <div>
      <h1>Gesture RC Car</h1>
      <p>Real-time hand gesture control — Laptop Dev Mode</p>
    </div>
  </div>
  <div class="pills">
    <span class="pill" id="pill-cam">● Camera</span>
    <span class="pill" id="pill-ml">● ML Model</span>
    <span class="pill" id="pill-mqtt">● MQTT/Car</span>
    <span class="pill" id="pill-ws">● WebSocket</span>
  </div>
</header>

<main>

  <!-- Camera feed -->
  <section class="card feed-card">
    <h2>Live Camera Feed</h2>
    <div class="video-wrap">
      <img id="video-feed" src="/video" alt="Camera stream"/>
      <div class="fps-badge" id="fps-badge">0 fps</div>
    </div>
  </section>

  <!-- Right column -->
  <section class="right-col">

    <!-- Active command -->
    <div class="card">
      <h2>Active Command</h2>
      <div class="cmd-display STOP" id="cmd-display">STOP</div>
      <div class="gesture-row">
        <span class="row-label">Gesture</span>
        <span id="gesture-label">—</span>
      </div>
      <div class="conf-row">
        <span class="row-label">Confidence</span>
        <div class="conf-bar-wrap">
          <div class="conf-track">
            <div class="conf-fill" id="conf-fill"></div>
          </div>
          <span class="conf-val" id="conf-val">0%</span>
        </div>
      </div>
    </div>

    <!-- D-pad -->
    <div class="card">
      <h2>Direction</h2>
      <div class="dpad">
        <div class="dpad-btn" id="btn-FORWARD">▲</div>
        <div class="dpad-row">
          <div class="dpad-btn" id="btn-LEFT">◀</div>
          <div class="dpad-btn stop-btn" id="btn-STOP">■</div>
          <div class="dpad-btn" id="btn-RIGHT">▶</div>
        </div>
        <div class="dpad-btn" id="btn-REVERSE">▼</div>
      </div>
    </div>

    <!-- MQTT connection info -->
    <div class="card">
      <h2>MQTT / Car Connection</h2>
      <div class="mqtt-info" id="mqtt-info">
        <strong>Broker:</strong> HiveMQ Cloud<br>
        <strong>Topic:</strong> gesturecar/command<br>
        <strong>Status:</strong> <span id="mqtt-status-text">Connecting…</span>
      </div>
    </div>

    <!-- Session stats -->
    <div class="card">
      <h2>Session Stats</h2>
      <div id="stats-container">
        <div class="stat-row"><span>Forward</span><span id="stat-forward">0</span></div>
        <div class="stat-row"><span>Reverse</span><span id="stat-reverse">0</span></div>
        <div class="stat-row"><span>Left</span><span id="stat-left">0</span></div>
        <div class="stat-row"><span>Right</span><span id="stat-right">0</span></div>
        <div class="stat-row"><span>Stop</span><span id="stat-stop">0</span></div>
      </div>
    </div>

  </section>

  <!-- Command log -->
  <section class="card log-card">
    <h2>Command Log</h2>
    <table class="log-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Command</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody id="log-body">
        <tr><td colspan="3" class="log-empty">Waiting for gestures…</td></tr>
      </tbody>
    </table>
  </section>

</main>

<script>
// ── SocketIO connection ─────────────────────────────────────────
// Connect to Flask-SocketIO on same origin
const socket = io({
  transports: ['websocket', 'polling'],
  upgrade: true,
});

// ── DOM refs ────────────────────────────────────────────────────
const cmdDisplay   = document.getElementById('cmd-display');
const gestureLabel = document.getElementById('gesture-label');
const confFill     = document.getElementById('conf-fill');
const confVal      = document.getElementById('conf-val');
const fpsBadge     = document.getElementById('fps-badge');
const logBody      = document.getElementById('log-body');
const mqttStatus   = document.getElementById('mqtt-status-text');
const pillCam      = document.getElementById('pill-cam');
const pillMl       = document.getElementById('pill-ml');
const pillMqtt     = document.getElementById('pill-mqtt');
const pillWs       = document.getElementById('pill-ws');

const DPAD_BTNS = {
  FORWARD: document.getElementById('btn-FORWARD'),
  REVERSE: document.getElementById('btn-REVERSE'),
  LEFT:    document.getElementById('btn-LEFT'),
  RIGHT:   document.getElementById('btn-RIGHT'),
  STOP:    document.getElementById('btn-STOP'),
};

const STAT_ELS = {
  forward: document.getElementById('stat-forward'),
  reverse: document.getElementById('stat-reverse'),
  left:    document.getElementById('stat-left'),
  right:   document.getElementById('stat-right'),
  stop:    document.getElementById('stat-stop'),
};

const CMD_COLORS = {
  FORWARD:'#00e676', REVERSE:'#4dabf7',
  LEFT:'#b39ddb', RIGHT:'#ffb300', STOP:'#ff5252',
};

// ── WebSocket connection status ─────────────────────────────────
socket.on('connect', () => {
  pillWs.classList.add('on');
  pillWs.classList.remove('err');
  pillWs.textContent = '● WebSocket';
  console.log('[WS] Connected to Flask server');
});

socket.on('disconnect', () => {
  pillWs.classList.remove('on');
  pillWs.classList.add('err');
  pillWs.textContent = '● WebSocket';
  console.log('[WS] Disconnected');
});

socket.on('connect_error', (err) => {
  pillWs.classList.add('err');
  console.error('[WS] Error:', err.message);
});

// ── State update from inference thread ──────────────────────────
let lastCmd = null;

socket.on('state_update', (data) => {
  const cmd        = data.command || 'STOP';
  const gesture    = data.gesture || '—';
  const confidence = data.confidence || 0;
  const fps        = data.fps || 0;
  const carConn    = data.car;

  // Gesture label
  gestureLabel.textContent = gesture;

  // Confidence bar
  const pct = Math.round(confidence * 100);
  confFill.style.width      = pct + '%';
  confFill.style.background = pct >= 85 ? '#00e676' : '#4dabf7';
  confVal.textContent       = pct + '%';

  // Command display
  cmdDisplay.textContent      = cmd;
  cmdDisplay.className        = 'cmd-display ' + cmd;

  // D-pad highlight
  Object.entries(DPAD_BTNS).forEach(([key, el]) => {
    el.classList.toggle('active', key === cmd);
  });

  // FPS
  fpsBadge.textContent = fps + ' fps';

  // Camera pill — always green if receiving data
  pillCam.classList.add('on');

  // ML pill
  pillMl.classList.add('on');

  // MQTT/Car pill
  if (carConn) {
    pillMqtt.classList.add('on');
    pillMqtt.classList.remove('err', 'warn');
    pillMqtt.textContent   = '● MQTT/Car';
    mqttStatus.textContent = 'Connected to HiveMQ ✓ — Car receiving commands';
    mqttStatus.style.color = '#00e676';
  } else {
    pillMqtt.classList.remove('on');
    pillMqtt.classList.add('warn');
    pillMqtt.textContent   = '● MQTT/Car';
    mqttStatus.textContent = 'MQTT connecting… check HiveMQ credentials in app.py';
    mqttStatus.style.color = '#ffb300';
  }

  // Session stats
  if (data.stats) {
    Object.entries(data.stats).forEach(([gesture, count]) => {
      const el = STAT_ELS[gesture];
      if (el) el.textContent = count;
    });
  }

  // Command log
  if (data.log && data.log.length > 0 && cmd !== lastCmd) {
    lastCmd = cmd;
    const rows = [...data.log].reverse().slice(0, 10).map(entry => {
      const color = CMD_COLORS[entry.cmd] || '#e8eaf0';
      return `<tr>
        <td>${entry.time}</td>
        <td><span class="cmd-tag ${entry.cmd}" style="color:${color}">${entry.cmd}</span></td>
        <td>${Math.round(entry.conf * 100)}%</td>
      </tr>`;
    }).join('');
    logBody.innerHTML = rows || '<tr><td colspan="3" class="log-empty">No commands yet</td></tr>';
  }
});

// ── Periodic status poll fallback ───────────────────────────────
// If WebSocket fails, poll REST endpoint every 2s
setInterval(() => {
  if (!socket.connected) {
    fetch('/status')
      .then(r => r.json())
      .then(data => {
        gestureLabel.textContent = data.gesture || '—';
        cmdDisplay.textContent   = data.command || 'STOP';
        cmdDisplay.className     = 'cmd-display ' + (data.command || 'STOP');
        fpsBadge.textContent     = (data.fps || 0) + ' fps';
      })
      .catch(() => {});
  }
}, 2000);
</script>
</body>
</html>
"""

# ── Flask + SocketIO ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(
    __name__,
    template_folder = os.path.join(BASE_DIR, "templates"),
    static_folder   = os.path.join(BASE_DIR, "static"),
)
app.config['SECRET_KEY'] = 'gesture_car_dev_2024'
socketio = SocketIO(
    app,
    cors_allowed_origins = "*",
    async_mode          = 'threading',
    logger              = False,
    engineio_logger     = False,
)

# ── Shared state ──────────────────────────────────────────────
frame_lock = threading.Lock()
state = {
    "frame":         None,
    "gesture":       "—",
    "confidence":    0.0,
    "command":       "STOP",
    "car_connected": False,
    "cam_connected": False,
    "fps":           0.0,
    "session_stats": {"forward": 0, "reverse": 0, "left": 0, "right": 0, "stop": 0},
    "last_commands": [],
}


# ── MQTT sender ───────────────────────────────────────────────
class MQTTSender:
    """
    Sends commands to ESP8266 via HiveMQ MQTT broker.
    Connects over TLS port 8883.
    """

    def __init__(self):
        self.connected  = False
        self.last_cmd   = None
        self.last_sent  = 0.0

        # Validate config before connecting
        if "xxxxxxxx" in MQTT_BROKER:
            print("\n" + "!"*55)
            print("  WARNING: MQTT_BROKER not configured in config.py")
            print("  Update MQTT_BROKER with your HiveMQ cluster URL")
            print("  Car will not receive commands until this is set")
            print("!"*55 + "\n")
            return

        self.client = paho_mqtt.Client(
            client_id = "gesture_laptop_dev",
            protocol  = paho_mqtt.MQTTv311,
        )
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        # TLS for port 8883
        try:
            self.client.tls_set()
        except Exception as e:
            print(f"[MQTT] TLS setup error: {e}")
            return

        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish    = self._on_publish

        try:
            print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT_TLS}")
            self.client.connect(MQTT_BROKER, MQTT_PORT_TLS, keepalive=30)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")
            print("[MQTT] Check: internet connection, broker URL, credentials")

    def _on_connect(self, client, userdata, flags, rc):
        rc_meanings = {
            0: "Connected successfully",
            1: "Wrong protocol version",
            2: "Invalid client ID",
            3: "Broker unavailable",
            4: "Wrong username or password",
            5: "Not authorised",
        }
        if rc == 0:
            self.connected = True
            print(f"[MQTT] ✓ Connected to HiveMQ")
            # Subscribe to car status to confirm car is online
            self.client.subscribe(MQTT_TOPIC_STATUS)
        else:
            self.connected = False
            meaning = rc_meanings.get(rc, f"Unknown error rc={rc}")
            print(f"[MQTT] ✗ Connection refused: {meaning}")
            if rc == 4:
                print("[MQTT] Check MQTT_USERNAME and MQTT_PASSWORD in config.py")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect rc={rc} — will auto-reconnect")
        else:
            print("[MQTT] Disconnected cleanly")

    def _on_publish(self, client, userdata, mid):
        pass  # called when message delivered

    def send(self, command: str):
        if not hasattr(self, 'client') or not self.connected:
            return

        now       = time.time()
        changed   = command != self.last_cmd
        heartbeat = now - self.last_sent > 0.4

        if changed or heartbeat:
            try:
                result = self.client.publish(
                    MQTT_TOPIC_CMD, command, qos=0, retain=False)
                if result.rc == paho_mqtt.MQTT_ERR_SUCCESS:
                    self.last_cmd  = command
                    self.last_sent = now
            except Exception as e:
                print(f"[MQTT] Publish error: {e}")

    def close(self):
        if hasattr(self, 'client'):
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


# ── Model loader ──────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run: python src/train_model.py"
        )
    with open(MODEL_FILE,   'rb') as f: model   = pickle.load(f)
    with open(ENCODER_FILE, 'rb') as f: encoder = pickle.load(f)
    print(f"[ML] Model loaded — classes: {list(encoder.classes_)}")
    return model, encoder


# ── Frame annotation ────────────────────────────────────────── 

mp_draw      = mp_draw_utils

COMMAND_COLORS = {
    "FORWARD": (0,  220, 100),
    "REVERSE": (0,  120, 255),
    "LEFT":    (200,100, 255),
    "RIGHT":   (255,180, 0),
    "STOP":    (60, 60,  220),
}

def annotate_frame(frame, gesture, confidence, command, mqtt_connected):
    h, w      = frame.shape[:2]
    cmd_color = COMMAND_COLORS.get(command, (200, 200, 200))

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 90), (18, 18, 18), -1)

    # Gesture
    conf_color = (0,220,100) if confidence >= CONFIDENCE_THRESHOLD \
                             else (80,80,220)
    cv2.putText(frame, f"Gesture: {gesture}",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, conf_color, 2)

    # Confidence bar
    bar_w = int(confidence * 200)
    cv2.rectangle(frame, (16, 50), (216, 66), (50,50,50), -1)
    cv2.rectangle(frame, (16, 50), (16+bar_w, 66), conf_color, -1)
    cv2.putText(frame, f"{confidence:.0%}",
                (224, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1)

    # Command
    cv2.putText(frame, command,
                (w-180, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cmd_color, 2)

    # Bottom bar — show MQTT status
    cv2.rectangle(frame, (0, h-28), (w, h), (18,18,18), -1)
    mqtt_text  = "MQTT: HiveMQ ✓" if mqtt_connected else "MQTT: connecting..."
    mqtt_color = (0,200,100) if mqtt_connected else (100,100,255)
    cv2.putText(frame, f"LAPTOP MODE  |  {mqtt_text}",
                (16, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mqtt_color, 1)

    return frame


# ── Inference thread ──────────────────────────────────────────
def inference_thread(model, encoder):
    detector    = build_hand_detector(
        MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE)
    sender      = MQTTSender()
    vote_buf    = FastVoteBuffer(CONFIDENCE_THRESHOLD)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)

    state["cam_connected"] = cap.isOpened()
    if not cap.isOpened():
        print(f"[CAM] ERROR: Cannot open camera index {CAM_INDEX}")
        print("Change CAM_INDEX in config.py")
        return

    print(f"[CAM] Opened {CAM_WIDTH}×{CAM_HEIGHT}")

    fps          = 0.0
    t_prev       = time.time()
    gesture      = "—"
    confidence   = 0.0
    command      = "STOP"
    emit_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        if lms is not None:
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,180,255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=1),
            )
            features   = extract_features(lms).reshape(1, -1)
            proba      = model.predict_proba(features)[0]
            idx        = int(np.argmax(proba))
            confidence = float(proba[idx])
            gesture_label = encoder.classes_[idx]
            gesture       = gesture_label
            confidence    = float(proba[idx])
            command_str   = GESTURE_TO_CMD.get(gesture_label, "STOP")
        else:
            gesture     = "No hand"
            confidence  = 0.0
            command_str = "STOP"    # ← always defined before vote_buf call

        command = vote_buf.update(command_str, confidence, lms is not None)
        sender.send(command)

        # Update session stats
        cmd_key = command.lower()
        if cmd_key in state["session_stats"]:
            state["session_stats"][cmd_key] += 1

        # Rolling command log
        log = state["last_commands"]
        log.append({
            "cmd":  command,
            "time": time.strftime("%H:%M:%S"),
            "conf": round(confidence, 3),
        })
        if len(log) > 20:
            log.pop(0)

        # FPS
        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        # Annotate
        display = annotate_frame(
            frame.copy(), gesture, confidence,
            command, sender.connected
        )

        with frame_lock:
            state["frame"]         = display
            state["gesture"]       = gesture
            state["confidence"]    = round(confidence, 3)
            state["command"]       = command
            state["fps"]           = round(fps, 1)
            state["car_connected"] = sender.connected

        # Emit to browser ~10 Hz
        emit_counter += 1
        if emit_counter % 3 == 0:
            socketio.emit('state_update', {
                "gesture":    gesture,
                "confidence": round(confidence, 3),
                "command":    command,
                "fps":        round(fps, 1),
                "car":        sender.connected,
                "stats":      state["session_stats"],
                "log":        state["last_commands"][-5:],
            })

    cap.release()
    detector.close()
    sender.close()


# ── MJPEG stream ──────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = state["frame"]

        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera initializing…",
                        (140, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (100,100,100), 2)
            frame = placeholder

        ret, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buf.tobytes()
            + b'\r\n'
        )
        time.sleep(0.033)


# ── Flask routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )

@app.route('/status')
def status():
    return jsonify({
        "gesture":    state["gesture"],
        "confidence": state["confidence"],
        "command":    state["command"],
        "fps":        state["fps"],
        "car":        state["car_connected"],
        "cam":        state["cam_connected"],
        "stats":      state["session_stats"],
        "log":        state["last_commands"],
    })

@app.route('/ping')
def ping():
    return jsonify({"ok": True, "time": time.strftime("%H:%M:%S")})

@app.route('/favicon.ico')
def favicon():
    return '', 204


# ── SocketIO events ───────────────────────────────────────────
@socketio.on('connect')
def on_ws_connect():
    print(f"[WS] Browser connected")

@socketio.on('disconnect')
def on_ws_disconnect():
    print(f"[WS] Browser disconnected")


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    print("═" * 55)
    print("  Gesture Car — Laptop Development Mode")
    print("═" * 55)
    print(f"  MQTT Broker : {MQTT_BROKER}")
    print(f"  MQTT Topic  : {MQTT_TOPIC_CMD}")
    print(f"  Camera      : index {CAM_INDEX}")
    print("─" * 55)

    model, encoder = load_model()

    t = threading.Thread(
        target   = inference_thread,
        args     = (model, encoder),
        daemon   = True,
        name     = "InferenceThread",
    )
    t.start()
    print(f"[THREAD] Inference thread started")
    print(f"[SERVER] Open in browser: http://localhost:{FLASK_PORT}")
    print("─" * 55)

    socketio.run(
        app,
        host         = FLASK_HOST,
        port         = FLASK_PORT,
        debug        = False,
        use_reloader = False,
        allow_unsafe_werkzeug = True,
    )