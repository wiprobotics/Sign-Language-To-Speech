import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

datasetFolder = "ModelCreation/" + "dataset"

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpStyle = mp.solutions.drawing_styles

hands = mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data = []
labels = []

for folder in tqdm(os.listdir(datasetFolder)):
    fullPath = "ModelCreation/" + datasetFolder + "/" + folder
    # print("Processing folder " + fullPath)
    for file in tqdm(os.listdir(fullPath)):
        imagePath = "ModelCreation/" + fullPath + "/" + file
        img = cv2.imread(imagePath)
        dataOut = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        handsResults = hands.process(imgRGB)

        if handsResults.multi_hand_landmarks:
            for handLandmarks in handsResults.multi_hand_landmarks:
                for i in range(len(handLandmarks.landmark)):
                    x = handLandmarks.landmark[i].x
                    y = handLandmarks.landmark[i].y
                    dataOut.append(x)
                    dataOut.append(y)

            data.append(dataOut)
            labels.append(folder)

        # else:
        #     print("No hand found in " + imagePath)

with open("ModelCreation/" + "data.pickle", "wb") as f:
    pickle.dump({'data': data, 'labels': labels}, f)
f.close()
        

            