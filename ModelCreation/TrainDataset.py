import pickle
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

allData = []
labels = []
orderedLabels = []

for item in os.listdir("ModelCreation/dataset"):
    if item.endswith(".pickle"):
        print("Loading dataset " + item + "...")
        dataDictionary = pickle.load(open("ModelCreation/dataset/" + item, "rb"))
        data = np.asarray(dataDictionary['data'], dtype=object)
        label = str(item.split("-")[0])
        for item in data:
            # Fill the item with 0s if it is less than 90
            if len(item) < 84:
                item = np.pad(item, (0, 84 - len(item)), 'constant')
            # Truncate or pad the item to exactly 84 elements
            item = item[:84]
            # print(item)
            allData.append(item)
            labels.append(label)
        if label not in orderedLabels:
            orderedLabels.append(label)

print("Number of samples: " + str(len(allData)) + "\nNumber of labels: " + str(len(labels)))

xTrain, xTest, yTrain, yTest = train_test_split(allData, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

model = RandomForestClassifier(verbose=1)

print("Training model...")
model.fit(xTrain, yTrain)

yPredict = model.predict(xTest)

score = accuracy_score(yPredict, yTest)

print("Accuracy: {:.1f}%".format(round((score * 100), 2)))

f = open("model.pickle", "wb")
pickle.dump({"model": model, "labels": labels, "orderedLabels": orderedLabels}, f)
f.close()

print("Model saved to model.pickle")
