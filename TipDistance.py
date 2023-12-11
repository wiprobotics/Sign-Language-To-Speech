import os
import pickle
import numpy as np
import math

datasetDir = os.listdir("ModelCreation/dataset")
datasetDir.sort()

for file in datasetDir:
    outData = []
    if file.endswith(".pickle"):
        print(f"Loading dataset {file}...")
        dataPath = "ModelCreation/dataset/" + file
        dataDictionary = pickle.load(open(dataPath, "rb"))
        data = dataDictionary['data']
        label = str(file.split("-")[0])
        for landmarks in data:

            # Pass the item if it is less than 43 elements to account for single hand images
            if len(landmarks) < 43:
                outData.append(landmarks)
                continue

            # Fill the item with 0s if it is less than 84
            if len(landmarks) < 84:
                padding = (0, 84 - len(landmarks))
                item = np.pad(landmarks, padding, 'constant')
            # Truncate or pad the item to exactly 84 elements
            landmarks = landmarks[:84]
            hand1tips = [[], []]
            hand2tips = [[], []]
            
            # Get the coordinates of the tips of the fingers for hand 1
            for h1 in range(0, 41, 8):
                hand1tips[0].append(landmarks[h1])      # x for tip of finger
                hand1tips[1].append(landmarks[h1 + 1])  # y for tip of finger
                if (h1 + 2) < 41:
                    hand1tips[0].append(landmarks[h1 + 2])  # x for base of next finger
                    hand1tips[1].append(landmarks[h1 + 3])  # y for base of next finger

            # Get the coordinates of the tips of the fingers for hand 2
            for h2 in range(42, 84, 8):
                hand2tips[0].append(landmarks[h2])      # x for tip of finger
                hand2tips[1].append(landmarks[h2 + 1])  # y for tip of finger
                if (h2 + 2) < 84:
                    hand2tips[0].append(landmarks[h2 + 2])  # x for base of next finger
                    hand2tips[1].append(landmarks[h2 + 3])  # y for base of next finger

            # Calculate the distance between each tip of hand 1 and each tip of hand 2
            distances = []
            for point1 in range(len(hand1tips[0])):
                for point2 in range(len(hand2tips[0])):
                    # Calculate the distance between the two points
                    distance = math.sqrt((hand1tips[0][point1] - hand2tips[0][point2])**2 + (hand1tips[1][point1] - hand2tips[1][point2])**2)
                    distances.append(distance)

            landmarks = np.append(landmarks, distances)
            print(len(landmarks))

            outData.append(landmarks)

        # Save the data
        with open(dataPath, 'wb') as f:
            pickle.dump({'data': outData}, f)

