import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']
model.verbose = 0

cap = cv2.VideoCapture(2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.3, max_num_hands=2)

labelsDict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I or J', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'Z'}
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dataOut = []

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    handsResults = hands.process(imgRGB)

    if handsResults.multi_hand_landmarks:
        for handLandmarks in handsResults.multi_hand_landmarks:
            for i in range(len(handLandmarks.landmark)):
                x = handLandmarks.landmark[i].x
                y = handLandmarks.landmark[i].y
                dataOut.append(x)
                dataOut.append(y)
            mp_drawing.draw_landmarks(frame, handLandmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        if len(dataOut) < 84:
            dataOut = np.pad(dataOut, (0, 84 - len(dataOut)), 'constant')
        # Truncate or pad the item to exactly 84 elements
        dataOut = dataOut[:84]

        print(dataOut)

        prediction = model.predict(np.array([dataOut]))

        cv2.putText(frame, prediction[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)