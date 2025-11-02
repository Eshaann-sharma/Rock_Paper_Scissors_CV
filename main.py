import cv2
import mediapipe as mp
import time
import random
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

options = ['Rock', 'Paper', 'Scissors']

# --- IMPROVED GESTURE DETECTION FUNCTION ---
def get_gesture(landmarks):
    # Get y positions for fingertips and MCP joints
    tips = [landmarks[i].y for i in [4, 8, 12, 16, 20]]
    mcps = [landmarks[i].y for i in [2, 5, 9, 13, 17]]

    fingers = []
    for tip, mcp in zip(tips[1:], mcps[1:]):  # skip thumb
        fingers.append(1 if tip < mcp else 0)
    thumb_open = 1 if tips[0] < mcps[0] - 0.05 else 0

    total = sum(fingers) + thumb_open

    # Rock = all fingers down
    if total == 1:
        return "Rock"
    # Paper = all fingers up
    elif total == 5:
        return "Paper"
    # Scissors = index and middle up
    elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers) == 2:
        return "Scissors"
    else:
        return "Unknown"

# --- WINNER LOGIC ---
def get_result(user, computer):
    if user == computer:
        return "Draw!"
    elif (user == "Rock" and computer == "Scissors") or \
         (user == "Paper" and computer == "Rock") or \
         (user == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "You Lose!"

# --- MAIN PROGRAM ---
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
state = "waiting"  # waiting, countdown, show_result
gesture = "Unknown"
computer_choice = ""
result = ""
countdown_start = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ui_width = int(w * 0.4)
    camera_width = w - ui_width
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- STATE: COUNTDOWN ---
    if state == "countdown":
        elapsed = time.time() - countdown_start
        remaining = max(0, 2 - elapsed)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = get_gesture(hand_landmarks.landmark)

        ui_panel = 255 * np.ones((h, ui_width, 3), dtype=np.uint8)
        cv2.putText(ui_panel, "Show Your Gesture!", (30, 180), font, 1, (0, 0, 0), 2)
        cv2.putText(ui_panel, f"Time Left: {remaining:.1f}s", (30, 230), font, 0.9, (0, 0, 255), 3)
        cv2.putText(ui_panel, f"Detected: {gesture}", (30, 280), font, 0.8, (0, 100, 0), 2)

        if elapsed >= 2:
            computer_choice = random.choice(options)
            result = get_result(gesture, computer_choice)
            state = "show_result"
            result_start = time.time()

    # --- STATE: SHOW RESULT ---
    elif state == "show_result":
        ui_panel = 255 * np.ones((h, ui_width, 3), dtype=np.uint8)
        cv2.putText(ui_panel, f"Computer: {computer_choice}", (30, 180), font, 0.9, (0, 0, 0), 2)
        cv2.putText(ui_panel, f"You: {gesture}", (30, 230), font, 0.9, (0, 100, 0), 2)
        cv2.putText(ui_panel, f"Result: {result}", (30, 280), font, 1, (0, 150, 0), 3)
        cv2.putText(ui_panel, "Press 'P' to Play Again", (30, 340), font, 0.6, (100, 100, 100), 2)
        cv2.putText(ui_panel, "Press 'Q' to Quit", (30, 370), font, 0.6, (100, 100, 100), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- STATE: WAITING ---
    else:
        ui_panel = 255 * np.ones((h, ui_width, 3), dtype=np.uint8)
        cv2.putText(ui_panel, "Press 'S' to Start!", (40, 180), font, 1, (0, 0, 0), 3)
        cv2.putText(ui_panel, "Press 'Q' to Quit", (40, 230), font, 0.8, (100, 100, 100), 2)

    # --- Combine and Display ---
    combined = np.hstack((ui_panel, frame[:, :camera_width]))
    cv2.imshow("Rock Paper Scissors", combined)

    # --- Key Handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and state == "waiting":
        countdown_start = time.time()
        state = "countdown"
    elif key == ord('p') and state == "show_result":
        countdown_start = time.time()  # start next round directly
        gesture = "Unknown"
        state = "countdown"

cap.release()
cv2.destroyAllWindows()
