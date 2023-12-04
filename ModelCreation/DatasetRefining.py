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

hands = mp_hands.Hands(min_detection_confidence=0.3, max_num_hands=1)

labelsDict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I" ,9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}
while True:
    ret, frame = cap.read()
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

        prediction = model.predict(np.array([dataOut]))

        outText = "prediction: {prediction} if this is wrong, press T and input the correct letter".format(prediction=prediction[0])

        cv2.putText(frame, outText, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)