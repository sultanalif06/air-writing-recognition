import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "knn_model_v2.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder_v2.pkl"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

target_points = 50
min_points_threshold = 20

points = []
is_recording = False
last_prediction = ""
sentence = ""


def draw_text_with_bg(frame, text, pos, font_scale, color, bg_color, thickness=1, padding=6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def run_prediction():
    global points, last_prediction, sentence
    points_np = np.array(points)
    indices = np.linspace(0, len(points_np) - 1, target_points)
    x_interp = interp1d(np.arange(len(points_np)), points_np[:, 0], kind='linear')(indices)
    y_interp = interp1d(np.arange(len(points_np)), points_np[:, 1], kind='linear')(indices)

    x_centered = x_interp - np.mean(x_interp)
    y_centered = y_interp - np.mean(y_interp)
    max_range = max(np.ptp(x_centered), np.ptp(y_centered)) + 1e-8
    x_scaled = x_centered / max_range
    y_scaled = y_centered / max_range

    feature_vector = np.concatenate([x_scaled, y_scaled]).reshape(1, -1)
    pred_class = model.predict(feature_vector)[0]
    last_prediction = label_encoder.inverse_transform([pred_class])[0]
    sentence += last_prediction
    print(f"Predicted: {last_prediction}  |  Sentence: {sentence}")
    points = []


cap = cv2.VideoCapture(0)
cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Writing", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Top status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    if is_recording:
        draw_text_with_bg(frame, "  REC", (10, 36), 0.8, (255, 255, 255), (0, 0, 200), thickness=2, padding=8)
    else:
        draw_text_with_bg(frame, "PAUSED", (10, 36), 0.8, (220, 220, 220), (70, 70, 70), thickness=2, padding=8)

    # Points progress bar
    bar_x, bar_y, bar_w, bar_h = 140, 16, 200, 20
    progress = min(len(points) / min_points_threshold, 1.0)
    bar_fill_color = (0, 200, 100) if progress >= 1.0 else (0, 140, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), bar_fill_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (160, 160, 160), 1)
    cv2.putText(frame, f"Points: {len(points)}", (bar_x + bar_w + 8, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

    # Last prediction badge (top-right)
    if last_prediction:
        pred_text = f"Last: {last_prediction}"
        (tw, _), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        draw_text_with_bg(frame, pred_text, (w - tw - 28, 36), 0.8,
                          (255, 255, 255), (20, 120, 20), thickness=2, padding=8)

    # Hand tracking
    hand_detected = bool(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            index_tip = hand.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            if is_recording:
                cv2.circle(frame, (x, y), 12, (0, 0, 220), -1)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                points.append((x, y))
            else:
                cv2.circle(frame, (x, y), 10, (0, 200, 255), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

    if not hand_detected:
        msg = "No hand detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        draw_text_with_bg(frame, msg, (w // 2 - tw // 2, h // 2 + th // 2),
                          0.7, (255, 255, 255), (0, 0, 180), padding=8)

    # Gradient trail (orange at start, blue at fingertip)
    if len(points) > 1:
        for i in range(1, len(points)):
            alpha = i / len(points)
            color = (int(255 * (1 - alpha)), 80, int(255 * alpha))
            cv2.line(frame, points[i - 1], points[i], color, 2)

    # Sentence buffer panel
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h - 82), (w, h - 42), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay3, 0.8, frame, 0.2, 0)

    display_sentence = (sentence[-38:] if len(sentence) > 38 else sentence) + "|"
    if not sentence and not is_recording:
        cv2.putText(frame, "Press SPACE to start writing...", (10, h - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    else:
        cv2.putText(frame, display_sentence, (10, h - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Bottom controls bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 40), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0)
    cv2.putText(frame, "SPACE: Start/Save    BKSP: Delete    ENTER: Space    C: Clear stroke    R: Reset    Q: Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.imshow("Air Writing", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if not is_recording:
            is_recording = True
        else:
            is_recording = False
            if len(points) >= min_points_threshold:
                run_prediction()
            else:
                print(f"Not enough points to predict ({len(points)}/{min_points_threshold})")
                points = []

    elif key == 8:  # Backspace
        if sentence:
            sentence = sentence[:-1]
            last_prediction = sentence[-1] if sentence else ""

    elif key == 13:  # Enter — add space between words
        sentence += " "

    elif key == ord('c'):
        points = []
        print("Stroke cleared")

    elif key == ord('r'):
        sentence = ""
        last_prediction = ""
        points = []
        print("Sentence reset")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
