import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model file direct from pickle as it keeps the model as an object
model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']
model.verbose = 0

# Open the webcam
cap = cv2.VideoCapture(2)

# Set up mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.3, max_num_hands=2)

labelsDict = {}

print(model_dict['orderedLabels'])

# Dictionary to convert the label number to a letter
for label in model_dict['orderedLabels']:
    print(label)
    labelsDict[model_dict['orderedLabels'].index(label)] = label

print(labelsDict)

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

        # print(dataOut)

        prediction = model.predict(np.array([dataOut]))

        cv2.putText(frame, prediction[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('save.jpg', frame)
    if cv2.waitKey(1) == ord('q'):
        break
