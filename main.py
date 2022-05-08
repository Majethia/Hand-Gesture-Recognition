import pickle
import cv2
import mediapipe as mp
import pandas as pd

with open("best1.pickle", "rb") as f:
        model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

gestures = {'callme': 'Call Me', 'd': 'One Minute', 'l': 'Lost', 'v':'Victory', 'fist': 'Fist', 'palm':'Wait', 'rock':'Rock', 'r':'Promise'}

print(len(gestures))

cols = ["wrist_x", 'wrist_y', 'thumb_cmc_x', 'thumb_cmc_y', 'thumb_mcp_x', 'thumb_mcp_y', 'thumb_ip_x', 'thumb_ip_y', 'thumb_tip_x', 'thumb_tip_y', 'index_mcp_x', 'index_mcp_y', 'index_pip_x', 'index_pip_y', 'index_dip_x', 'index_dip_y', 'index_tip_x', 'index_tip_y', 'middle_mcp_x', 'middle_mcp_y', 'middle_pip_x', 'middle_pip_y', 'middle_dip_x', 'middle_dip_y', 'middle_tip_x', 'middle_tip_y', 'ring_mcp_x', 'ring_mcp_y', 'ring_pip_x', 'ring_pip_y', 'ring_dip_x', 'ring_dip_y', 'ring_tip_x', 'ring_tip_y', 'pinky_mcp_x', 'pinky_mcp_y', 'pinky_pip_x', 'pinky_pip_y', 'pinky_dip_x', 'pinky_dip_y', 'pinky_tip_x', 'pinky_tip_y']


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x, y, c = image.shape
        prediction = "None"
        if results.multi_hand_landmarks:
            landmarks = [[]]
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks[0].append(lm.x)
                    landmarks[0].append(lm.y)
               

            d = pd.DataFrame(landmarks, columns=cols) 
            prediction = gestures[model.predict(d)[0]]
        image = cv2.flip(image, 1)
        image = cv2.putText(image, prediction, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False) 
        
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
