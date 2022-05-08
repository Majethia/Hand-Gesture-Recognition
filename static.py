import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = []

LANDMARKS = []

for i in range(1600):
    # Images from the dataset, files too large to include in the repo.
    IMAGE_FILES.append(f"original_images\original_images\\r\\r{i}.jpg")


# For static images:
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        landmarks = []
        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y))
            LANDMARKS.append(landmarks)
        print(idx)


with open("data2.csv", "a") as f:
    final = ''
    for i in LANDMARKS:
        for j in i:
            final += f"{j[0]},{j[1]},"
        final += "r\n"
    f.write(final)

