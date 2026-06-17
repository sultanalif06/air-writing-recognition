import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import joblib

model = joblib.load("knn_model_v2.pkl")
label_encoder = joblib.load("label_encoder_v2.pkl")

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

target_points = 50 
min_points_threshold = 20 

points = []
is_recording = False

# Open webcam
cap = cv2.VideoCapture(0)
print("Controls:")
print("- SPACE: Start/Pause Recording")
print("- 'p': Predict current gesture")
print("- 'c': Clear points")
print("- 'q': Quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Recording status
    status_text = "RECORDING" if is_recording else "PAUSED"
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Points: {len(points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            index_tip = hand.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            if is_recording:
                points.append((x, y))



    cv2.imshow("Air Writing Inference V2", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '): 
        is_recording = not is_recording
        print("Recording:", "ON" if is_recording else "OFF")

    elif key == ord('p') and len(points) >= min_points_threshold:
        # Resample
        points_np = np.array(points)
        indices = np.linspace(0, len(points_np) - 1, target_points)
        x_interp = interp1d(np.arange(len(points_np)), points_np[:, 0], kind='linear')(indices)
        y_interp = interp1d(np.arange(len(points_np)), points_np[:, 1], kind='linear')(indices)

        # Centering
        x_centered = x_interp - np.mean(x_interp)
        y_centered = y_interp - np.mean(y_interp)

        # Scaling
        max_range = max(np.ptp(x_centered), np.ptp(y_centered)) + 1e-8
        x_scaled = x_centered / max_range
        y_scaled = y_centered / max_range

        feature_vector = np.concatenate([x_scaled, y_scaled]).reshape(1, -1)

        # Predict
        pred_class = model.predict(feature_vector)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        print(f"Predicted Letter: {pred_label}")

        # Show result
        cv2.putText(frame, f"Prediction: {pred_label}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Air Writing Inference V2", frame)
        cv2.waitKey(2000)

        points = []

    elif key == ord('c'):
        points = []
        print("Points cleared")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
