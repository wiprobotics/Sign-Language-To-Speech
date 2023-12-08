import pickle

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

allData = []
labels = []
orderedLabels = []

datasetDir = os.listdir("ModelCreation/dataset")
datasetDir.sort()

for file in datasetDir:
    if file.endswith(".pickle"):
        print(f"Loading dataset {file}...")
        data_path = "ModelCreation/dataset/" + file
        dataDictionary = pickle.load(open(data_path, "rb"))
        data = np.asarray(dataDictionary['data'], dtype=object)
        label = str(file.split("-")[0])
        for landmarks in data:
            # Fill the item with 0s if it is less than 120
            if len(landmarks) < 120:
                padding = (0, 120 - len(landmarks))
                landmarks = np.pad(landmarks, padding, 'constant')
            # Truncate or pad the item to exactly 84 elements
            landmarks = landmarks[:120]
            allData.append(landmarks)
            labels.append(label)
        if label not in orderedLabels:
            orderedLabels.append(label)

num_classes = len(orderedLabels)
input_shape = (120,)

model = keras.Sequential([
    layers.Flatten(input_shape=(input_shape)),  # Input layer
    layers.Dense(128, activation='relu'),       # Hidden layer with ReLU activation
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Number of samples: " + str(len(allData)) + "\nNumber of labels: " + str(len(labels)))

xTrain = np.array(allData, dtype=np.float32)
yTrain = np.array(labels)

labelEncoder = LabelEncoder()
yTrain = labelEncoder.fit_transform(yTrain)

xTrain, xTest, yTrain, yTest = train_test_split(xTrain, yTrain, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

print("Training model...")
model.fit(xTrain, yTrain, epochs=10, verbose=1, validation_data=(xTest, yTest), batch_size=32)

yPredict = model.predict(xTest)

test_loss, test_acc = model.evaluate(xTest, yTest)
print(f'Test accuracy: {test_acc}')

model.save("model.h5")

with open("modeldata.pickle", "wb") as f:
    pickle.dump({'orderedLabels': orderedLabels, 'labelEncoder': labelEncoder}, f)


print("Model saved to model.h5")
