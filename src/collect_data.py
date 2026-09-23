# ─────────────────────────────────────────────────────────────
# Interactive webcam data collection.
# appends to data/gesture_data.csv.
# ─────────────────────────────────────────────────────────────

import cv2
import csv
import os
import sys
import time

import mediapipe as mp
mp_draw      = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_hands_mod = mp.solutions.hands          # type: ignore[attr-defined]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GESTURES, FEATURE_DIM,
    DATA_CSV, DATA_DIR,
    CAM_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    build_hand_detector, process_frame,
    get_landmark_list, extract_features,
)


PRE_RECORD_HOLD_SEC = 3.0   # stable hold required before recording
MIN_RECORD_SEC      = 2.0   # minimum recording duration

STYLE_READY = mp_draw.DrawingSpec(
    color=(0, 200, 100), thickness=2, circle_radius=3)
STYLE_REC   = mp_draw.DrawingSpec(
    color=(0, 140, 255), thickness=2, circle_radius=3)
STYLE_CONN  = mp_draw.DrawingSpec(
    color=(255, 255, 255), thickness=1)


def ask_sample_count() -> int:
    print("\n" + "─" * 50)
    print("  How many people will record gestures?")
    print("  Each person records 5 samples per gesture.")
    print("  Total = people × 5")
    print("─" * 50)
    while True:
        raw = input("  Enter number of people [default 1]: ").strip()
        if raw == "":
            people = 1
        else:
            try:
                people = int(raw)
                if people < 1 or people > 100:
                    print("  Enter a number between 1 and 100")
                    continue
            except ValueError:
                print("  Enter a whole number")
                continue

        samples = people * 5
        print(f"\n  {people} person(s) × 5 samples = {samples} samples per gesture")
        print(f"  Total dataset addition: {samples * len(GESTURES)} rows")
        confirm = input("  Continue? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            return samples
        print("  Enter number of people again.")


def ask_gesture_selection() -> list:
    print("\n" + "─" * 50)
    print("  Which gestures to collect?")
    for i, g in enumerate(GESTURES):
        print(f"    {i+1}. {g}")
    print("    A. All (default)")
    print("─" * 50)
    raw = input("  Enter numbers separated by comma, or A: ").strip()
    if not raw or raw.upper() == "A":
        return GESTURES
    selected = []
    for part in raw.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(GESTURES):
                selected.append(GESTURES[idx])
        except ValueError:
            pass
    return selected if selected else GESTURES


def draw_progress_bar(frame, elapsed, total, label, color):
    h, w  = frame.shape[:2]
    pct   = min(elapsed / max(total, 0.001), 1.0)
    fill  = int(pct * (w - 40))
    cv2.rectangle(frame, (20, 8), (w - 20, 26), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, 8), (20 + fill, 26), color, -1)
    cv2.putText(frame, label,
                (24, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (240, 240, 240), 1)


def draw_sample_bar(frame, count, target):
    h, w = frame.shape[:2]
    pct  = count / max(target, 1)
    fill = int(pct * (w - 40))
    cv2.rectangle(frame, (20, h - 28), (w - 20, h - 12), (40,40,40), -1)
    cv2.rectangle(frame, (20, h - 28), (20+fill, h - 12), (0,200,100), -1)
    cv2.putText(frame, f"Samples: {count} / {target}",
                (24, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1)


def phase_ready(gesture_name, cap, detector) -> bool:
    """Show READY screen. User positions hand. SPACE starts countdown."""
    print(f"\n  Gesture: '{gesture_name}'")
    print("  Position your hand. Press SPACE when ready. Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        if result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS, STYLE_READY, STYLE_CONN)

        detected     = lms is not None
        detect_color = (0, 200, 100) if detected else (60, 60, 220)
        detect_text  = "Hand detected" if detected else "No hand — move into frame"

        cv2.putText(frame, f"GESTURE: {gesture_name.upper()}",
                    (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, detect_text,
                    (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, detect_color, 1)
        cv2.putText(frame, "SPACE = start  |  Q = quit",
                    (20, frame.shape[0] - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140,140,140), 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not detected:
                continue
            return True
        if key == ord('q'):
            return False


def phase_stability(gesture_name, cap, detector) -> bool:
    """
    Enforce PRE_RECORD_HOLD_SEC seconds of stable hold.
    Timer resets if hand disappears.
    """
    print(f"  Hold gesture steady for {PRE_RECORD_HOLD_SEC:.0f} seconds…")
    hold_start   = None
    hold_elapsed = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        if lms is not None:
            if hold_start is None:
                hold_start = time.time()
            hold_elapsed = time.time() - hold_start
            mp_draw.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
                STYLE_CONN)
        else:
            hold_start   = None
            hold_elapsed = 0.0

        remaining = PRE_RECORD_HOLD_SEC - hold_elapsed
        draw_progress_bar(
            frame, hold_elapsed, PRE_RECORD_HOLD_SEC,
            f"Hold steady… {max(remaining, 0):.1f}s",
            (0, 200, 255))

        cv2.putText(frame, f"STABILISING: {gesture_name.upper()}",
                    (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
        status_text  = "Keep gesture still" if lms else "Hand lost — reposition!"
        status_color = (180,180,180) if lms else (60,60,220)
        cv2.putText(frame, status_text,
                    (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if hold_elapsed >= PRE_RECORD_HOLD_SEC:
            print("  Stability confirmed — recording…")
            return True


def phase_record(gesture_name, target_count, cap, detector, csv_writer) -> int:
    """
    Record target_count samples.
    Enforces MIN_RECORD_SEC minimum duration.
    Both count AND time must complete before finishing.
    """
    count     = 0
    rec_start = time.time()

    while count < target_count:
        ret, frame = cap.read()
        if not ret:
            continue
        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        elapsed    = time.time() - rec_start
        time_done  = elapsed >= MIN_RECORD_SEC
        count_done = count >= target_count

        if lms is not None:
            features = extract_features(lms)
            csv_writer.writerow([gesture_name] + features.tolist())
            count += 1
            mp_draw.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS, STYLE_REC, STYLE_CONN)

        draw_progress_bar(
            frame, elapsed, max(MIN_RECORD_SEC, 1),
            f"Recording: {elapsed:.1f}s (min {MIN_RECORD_SEC:.0f}s)",
            (0, 200, 100) if time_done else (0, 160, 255))
        draw_sample_bar(frame, count, target_count)

        status_color = (0, 200, 100) if lms else (60, 60, 220)
        status_text  = "RECORDING" if lms else "RECORDING — no hand!"
        cv2.putText(frame, f"{status_text}: {gesture_name.upper()}",
                    (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,160,255), 2)
        cv2.putText(frame, f"Collected {count}/{target_count}  |  {elapsed:.1f}s",
                    (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"  Stopped early at {count} samples")
            break
        if count_done and time_done:
            break

    elapsed = time.time() - rec_start
    print(f"  Saved {count} samples in {elapsed:.1f}s")
    return count


def show_summary(gesture_counts: dict):
    import pandas as pd
    print("\n" + "─" * 50)
    print("  Session summary")
    print("─" * 50)
    for g, c in gesture_counts.items():
        print(f"  {g:12s}: {c:4d} new samples")
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        print("\n  Dataset totals:")
        for label, count in df['label'].value_counts().sort_index().items():
            bar = "█" * min(int(count / 100), 40)
            print(f"  {label:12s}: {count:6,}  {bar}")
        print(f"\n  Total rows : {len(df):,}")
    print("─" * 50)


def init_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        with open(DATA_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label'] + [f'f{i}' for i in range(FEATURE_DIM)])
        print(f"Created new dataset: {DATA_CSV}")
    else:
        print(f"Appending to existing dataset: {DATA_CSV}")


def main():
    print("═" * 50)
    print("  Gesture Car — Data Collection")
    print("═" * 50)

    init_csv()
    target_count      = ask_sample_count()
    selected_gestures = ask_gesture_selection()

    print(f"\n  Collecting {target_count} samples × {len(selected_gestures)} gestures")
    print(f"  Stability hold before recording : {PRE_RECORD_HOLD_SEC:.0f}s")
    print(f"  Minimum recording time          : {MIN_RECORD_SEC:.0f}s")
    input("\n  Press ENTER to open camera…")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAM_INDEX}")
        print("Change CAM_INDEX in config.py")
        return

    detector      = build_hand_detector(
        MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE)
    gesture_counts = {}

    try:
        with open(DATA_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, gesture in enumerate(selected_gestures):
                print(f"\n{'─'*50}")
                from config import GESTURE_TO_CMD
                cmd = GESTURE_TO_CMD.get(gesture, gesture.upper())
                print(f"  Gesture {i+1}/{len(selected_gestures)}: {gesture.upper()} → command: {cmd}")
                print(f"{'─'*50}")

                if not phase_ready(gesture, cap, detector):
                    print("  Quit by user")
                    break
                if not phase_stability(gesture, cap, detector):
                    print("  Quit by user")
                    break

                saved = phase_record(gesture, target_count, cap, detector, writer)
                gesture_counts[gesture] = saved
                f.flush()

                if i < len(selected_gestures) - 1:
                    print("  Next gesture in 2 seconds…")
                    time.sleep(2)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    show_summary(gesture_counts)
    print("\n  Next step: python src/pipeline.py --stage train,export")


if __name__ == "__main__":
    main()