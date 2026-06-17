import cv2
import mediapipe as mp
import pandas as pd
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


points = []
label = input("Enter the letter you are writing (A-Z): ").upper()
cap = cv2.VideoCapture(0)
start_time = time.time()


is_paused = True  #
pause_message = "PAUSED - Press SPACE to start recording"

print("Controls:")
print("- SPACE: Start/Pause recording")
print("- 's': Save current data to CSV")
print("- 'q': Quit")
print("- 'c': Clear current points without saving")
print("\nStarting in PAUSED mode. Press SPACE when ready to record.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    status_text = "RECORDING" if not is_paused else "PAUSED"
    status_color = (0, 255, 0) if not is_paused else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"Points: {len(points)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    instructions = [
        "SPACE: Start/Pause",
        "S: Save data",
        "C: Clear points",
        "Q: Quit"
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (10, frame.shape[0] - 120 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            # Get index fingertip 
            h, w, _ = frame.shape
            index_tip = hand.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            
            #  Show the finger position
            circle_color = (0, 255, 0) if not is_paused else (0, 255, 255) 
            cv2.circle(frame, (x, y), 8, circle_color, -1)
            
            if not is_paused:
                points.append((x, y))
                
                # Trails
                if len(points) > 1:
                    for i in range(max(0, len(points)-20), len(points)-1):
                        pt1 = points[i]
                        pt2 = points[i+1]
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    cv2.imshow("Air Writing - Letter: " + label, frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        is_paused = not is_paused
        if is_paused:
            print("Recording PAUSED. Position your finger and press SPACE to continue.")
        else:
            print("Recording STARTED. Draw your letter!")
    
    # Save data
    elif key == ord('s') and points:  
        is_paused = True 
        df = pd.DataFrame(points, columns=["x", "y"])
        df["label"] = label
        filename = f"airwriting_{label}_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(points)} points.")
        print("Recording PAUSED. Position finger for next recording and press SPACE.")
        points = [] 
    
    elif key == ord('c'): 
        points = []
        print("Points cleared. Ready for new recording.")
    
    elif key == ord('q'): 
        break
    
    elif key == ord('s') and not points: 
        print("No data to save. Record some points first!")

cap.release()
cv2.destroyAllWindows()
print("Air writing data collection ended.")