# 👋 Gesture Car

### **Control a real car with your hand. No app. No remote. Just your phone.**

[🚗 Live Demo — Open on any phone](https://yash-patil-26.github.io/Gesture_car/)

> **Show your hand → browser understands the gesture → car moves.**

[🎥 **Watch the Demo**](https://github.com/user-attachments/assets/304a75fb-850f-4141-9bdb-6fb4176caa54)

---

## ⚡ What is it?

**Gesture Car** is a browser-based gesture-controlled robotic car.

Your phone camera detects your hand using **MediaPipe**, a lightweight **Random Forest model** classifies the gesture directly in the browser, and the resulting command is sent through **MQTT** to an **ESP8266** that drives the car.

**No camera frames leave your phone.**

```text
📱 Phone Camera
      ↓
🖐️ MediaPipe
      ↓
🧠 ONNX Random Forest
      ↓
☁️ HiveMQ MQTT
      ↓
📡 ESP8266
      ↓
⚙️ L298N
      ↓
🚗 4× TT Motors
```

---

## ✋ Control With Your Hand

|     Gesture    |   Command   |
| :------------: | :---------: |
|   ✋ Open Palm  | **FORWARD** |
|     ✊ Fist     |   **STOP**  |
|     🤘 Rock    |   **LEFT**  |
|      👌 OK     |  **RIGHT**  |
| 👎 Thumbs Down | **REVERSE** |

### 🛑 Built for reliable control

Movement commands require **2 consecutive matching frames** to avoid accidental triggers.

**STOP requires only 1 frame.**

If the ESP8266 receives no command for **>2 seconds**, its watchdog automatically stops the motors.

---

## 🧠 The Interesting Part

The ML does **not** run on a server.

The phone performs:

```text
Camera
  ↓
21 Hand Landmarks
  ↓
63 Features
  ↓
Random Forest
  ↓
Gesture
```

The trained model is exported to **ONNX** and executed using **ONNX Runtime Web**.

### Model

|                     |                                  |
| ------------------- | -------------------------------- |
| Algorithm           | Random Forest · 100 trees        |
| Input               | 21 landmarks × XYZ = 63 features |
| Classes             | 5 gestures                       |
| Training data       | ~25,000 balanced samples         |
| Validation accuracy | ~97–99%                          |
| Browser inference   | ~5 ms                            |

---

## 🌐 Why This Architecture?

A browser served over HTTPS cannot simply connect to an insecure `ws://` endpoint.

So the system uses:

**Browser → WSS → HiveMQ → MQTT/TCP → ESP8266**

This lets an ordinary HTTPS webpage control an ESP8266 without changing the hardware architecture.

---

## 🔧 Hardware

| Component           | Role                |
| ------------------- | ------------------- |
| **ESP8266 NodeMCU** | Wireless controller |
| **L298N**           | Motor driver        |
| **4× TT Motors**    | Differential drive  |
| **2× 18650**        | Power               |
| **Acrylic Chassis** | Vehicle body        |
| **SPST Switch**     | Master power        |

---

## 🛠️ Tech Stack

**Python · scikit-learn · MediaPipe.js · ONNX · ONNX Runtime Web · JavaScript · MQTT · HiveMQ · ESP8266 · Arduino · L298N · GitHub Pages**

---

## 📁 Project Structure

```text
Gesture_Car/
├── docs/                  # Browser controller + ONNX model
├── esp8266/               # Car firmware
├── src/                   # Data collection + ML pipeline
├── outputs/               # Model evaluation
└── requirements.txt
```

---

## 🚀 Run It

```bash
git clone https://github.com/yash-patil-26/Gesture_Car.git
cd Gesture_Car

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Train:

```bash
python src/collect_data.py
python src/pipeline.py --stage train,export
```

Then flash the ESP8266 firmware and open:

[https://yash-patil-26.github.io/Gesture_car/](https://yash-patil-26.github.io/Gesture_car/?utm_source=chatgpt.com)

---

## 🎯 The Whole Project in One Line

> **A phone becomes the eyes, ML becomes the brain, MQTT becomes the nervous system, and the ESP8266 turns a hand gesture into motion.**

---

### 📜 License

MIT License
