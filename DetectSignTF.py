import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

from tkinter import ttk
from PIL import Image, ImageTk


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Sign Language Detector")
        # self.master.geometry("800x600")
        # self.master.resizable(False, False)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Sign Language Detector", font=("Arial", 24))
        self.label.grid(row=0, column=0, columnspan=5, pady=10)

        self.canvas = tk.Canvas(self, width=640, height=480)
        self.canvas.grid(row=1, column=0, columnspan=3, rowspan=2)

        self.columns = ("out", "probability")

        self.treeview = ttk.Treeview(self, columns=self.columns, show="headings")
        self.treeview.grid(row=1, column=3, columnspan=2, padx=10, pady=10)

        self.treeview.heading("out", text="Output")
        self.treeview.heading("probability", text="Probability")
        
        self.buttonFrame = tk.Frame(self)
        self.buttonFrame.grid(row=2, column=3, columnspan=2, padx=10, pady=10)

        self.expandDatasetButton = tk.Button(self.buttonFrame, text="Start Word Detect", command=ToggleWordDetect)
        self.expandDatasetButton.grid(row=0, column=0, padx=10, pady=10)
   
        self.recordButton = tk.Button(self.buttonFrame, text="Start Recording", command=ToggleRecording)
        self.recordButton.grid(row=0, column=1, padx=10, pady=10)
       
        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.grid(row=3, column=0, columnspan=5, pady=10)


def ToggleWordDetect():
    global wordDetect
    if wordDetect:
        wordDetect = False
        app.expandDatasetButton.config(text="Start Word Detect")
    else:
        wordDetect = True
        app.expandDatasetButton.config(text="Stop Word Detect")


def ToggleRecording():
    global recording
    global outRecorder
    if recording:
        recording = False
        outRecorder.release()
        app.recordButton.config(text="Start Recording")
    else:
        recording = True
        app.recordButton.config(text="Stop Recording")


def LoadModel():
    global model, modelDict, labelsDict
    # Load the model
    model = keras.models.load_model("Requirements/model.h5")
    model.verbose = False

    # Load the model dictionary
    modelDict = pickle.load(open("Requirements/modeldata.pickle", "rb"))

    # Dictionary to convert the label number to a letter
    labelsDict = {}

    # Dictionary to convert the label number to a letter
    for label in modelDict['orderedLabels']:
        labelsDict[modelDict['orderedLabels'].index(label)] = label


def LoadMediapipe():
    global mp_hands, mp_drawing, mp_drawing_styles, hands
    # Set up mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(min_detection_confidence=0.3, max_num_hands=2)


# Create the tkinter window
root = tk.Tk()
app = Application(master=root)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outRecorder = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
recording = False
wordDetect = False

LoadModel()
LoadMediapipe()

# Open the webcam
cap = cv2.VideoCapture(2)

# Main loop
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

        hand1tips = [[], []]
        hand2tips = [[], []]

        if len(handsResults.multi_hand_landmarks) > 1:
            # Get the coordinates of the tips of the fingers for hand 1
            for h1 in range(0, 41, 8):
                hand1tips[0].append(dataOut[h1])
                hand1tips[1].append(dataOut[h1 + 1])

            # Get the coordinates of the tips of the fingers for hand 2
            for h2 in range(42, 84, 8):
                hand2tips[0].append(dataOut[h2])
                hand2tips[1].append(dataOut[h2 + 1])

            # Calculate the distance between each tip of hand 1 and each tip of hand 2
            distances = []
            for point1 in range(len(hand1tips[0])):
                for point2 in range(len(hand2tips[0])):
                    # Calculate the distance between the two points
                    distance = math.sqrt((hand1tips[0][point1] - hand2tips[0][point2])**2 + (hand1tips[1][point1] - hand2tips[1][point2])**2)
                    distances.append(distance)

            dataOut = np.append(dataOut, distances)
        else:
            dataOut = np.pad(dataOut, (0, 120 - len(dataOut)), 'constant')

        dataOut = np.array(dataOut).reshape(1, 120)

        # print(dataOut)

        predictions = model.predict(dataOut)

        # Get the prediction in letter form
        letterPrediction = np.argmax(predictions[0])

        text = "Prediction: " + str(labelsDict[letterPrediction] + ":" + str(round((predictions[0][letterPrediction] * 100), 1)) + "%")

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        app.treeview.delete(*app.treeview.get_children())
        for i, out in enumerate(predictions[0]):
            probability = round((out * 100), 1)  # Use 'out' instead of 'predictions[0][i]'
            app.treeview.insert("", tk.END, values=(labelsDict[i], probability))
          
    outFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    app.photo = ImageTk.PhotoImage(image=Image.fromarray(outFrame))
    app.canvas.create_image(0, 0, image=app.photo, anchor="nw")
    if recording:
        print("Recording")
        outRecorder.write(frame)
    root.update()

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) == ord('s'):
    #     cv2.imwrite('save.jpg', frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break
