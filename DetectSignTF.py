import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import math
import threading
import time

from tensorflow import keras

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
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        self.canvas = tk.Canvas(self, width=640, height=480)
        self.canvas.grid(row=1, column=0, columnspan=1, rowspan=1)

        self.holdingFrame = tk.Frame(self)
        self.holdingFrame.grid(row=1, column=1, columnspan=1, padx=10, pady=10)

        self.wordDetectFrame = tk.Frame(self.holdingFrame)
        self.wordDetectFrame.grid(row=0, column=0, padx=10, pady=10)

        self.wordDetectLabel = tk.Label(self.wordDetectFrame, text="Word Detect", font=("Arial", 18))
        self.wordDetectLabel.grid(row=0, column=0, columnspan=1, pady=10)

        self.wordDetectTimerText = tk.Label(self.wordDetectFrame, text="Hold sign for: ")
        self.wordDetectTimerText.grid(row=1, column=0, columnspan=1, pady=10)

        self.wordDetectTimer = tk.Label(self.wordDetectFrame, text="2")
        self.wordDetectTimer.grid(row=1, column=1, columnspan=1, pady=10)

        self.wordDetectText = tk.Text(self.wordDetectFrame, height=5, width=20)
        self.wordDetectText.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.infoPanelFrame = tk.Frame(self.holdingFrame)
        self.infoPanelFrame.grid(row=1, column=0, padx=10, pady=10)

        self.infoPanelLabel = tk.Label(self.infoPanelFrame, text="Info Panel", font=("Arial", 18))
        self.infoPanelLabel.grid(row=0, column=0, columnspan=1, pady=10)

        self.columns = ("out", "probability")

        self.treeview = ttk.Treeview(self.infoPanelFrame, columns=self.columns, show="headings", height=5)
        self.treeview.grid(row=1, column=0, columnspan=1, padx=10, pady=10)

        self.style = ttk.Style()
        self.style.configure("Treeview.column", font=(None, 100))

        self.treeview.heading("out", text="Output")
        self.treeview.heading("probability", text="Probability")

        self.buttonFrame = tk.Frame(self.infoPanelFrame)
        self.buttonFrame.grid(row=2, column=0, columnspan=1, padx=10, pady=10)

        self.wordDetectButton = tk.Button(self.buttonFrame, text="Start Word Detect", command=ToggleWordDetect)
        self.wordDetectButton.grid(row=0, column=0, padx=10, pady=10)

        self.recordButton = tk.Button(self.buttonFrame, text="Start Recording", command=ToggleRecording)
        self.recordButton.grid(row=0, column=1, padx=10, pady=10)
       
        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.grid(row=3, column=0, columnspan=2, pady=10)


class CustomTimer(threading.Timer):
    started_at = None

    def start(self):
        self.started_at = time.time()
        threading.Timer.start(self)
 
    def elapsed(self):
        return time.time() - self.started_at

    def remaining(self):
        return self.interval - self.elapsed()


def ToggleWordDetect():
    global wordDetect
    if wordDetect:
        wordDetect = False
        app.wordDetectButton.config(text="Start Word Detect")
    else:
        wordDetect = True
        app.wordDetectButton.config(text="Stop Word Detect")


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


def LoadWordDictionary():
    global wordDict
    wordDict = []
    with open("Requirements/dict", "r") as f:
        for line in f:
            wordDict.append(line.strip())


def DetectWord(alreadyDetected):
    potentialWords = []
    for word in wordDict:
        listWord = list(word)
        failed = False

        if len(alreadyDetected) < len(word):
            maxCount = len(alreadyDetected)
        else:
            maxCount = len(word)

        for count in range(maxCount):
            if alreadyDetected[count] != listWord[count]:
                failed = True

        if failed is False:
            potentialWords.append(word)

    return potentialWords


def TimerEnd():
    global timerOn
    global wordDetect
    global lastLetter
    global alreadyDetected
    global timer
    timer = CustomTimer(4, TimerEnd)
    timerOn = False
    lastLetter = ""
    alreadyDetected.append(labelsDict[letterPrediction])
    print(DetectWord(alreadyDetected))


# Create the tkinter window
root = tk.Tk()
app = Application(master=root)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outRecorder = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
timer = CustomTimer(4, TimerEnd)

recording = False
wordDetect = False
timerOn = False

lastLetter = ""
alreadyDetected = []

LoadModel()
LoadMediapipe()
LoadWordDictionary()

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

        predictions = model.predict(dataOut)

        # Get the prediction in letter form
        letterPrediction = np.argmax(predictions[0])

        text = "Prediction: " + str(labelsDict[letterPrediction])

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        predictions_list = [(labelsDict[i], round((out * 100), 1)) for i, out in enumerate(predictions[0]) if round((out * 100), 1) > 0]
        predictions_list.sort(key=lambda x: x[1], reverse=True)

        if wordDetect:
            if lastLetter == "":
                lastLetter = labelsDict[letterPrediction]
                print("First letter detected")

            elif lastLetter == labelsDict[letterPrediction] and timerOn is False:
                timerOn = True
                timer.start()
                print("Timer started")

            elif lastLetter != labelsDict[letterPrediction]:
                lastLetter = labelsDict[letterPrediction]
                timer.cancel()
                timer = CustomTimer(4, TimerEnd)
                timerOn = False
                print("Timer cancelled")

        app.treeview.delete(*app.treeview.get_children())
        for label, probability in predictions_list:
            app.treeview.insert("", tk.END, values=(label, probability))

    else:
        app.treeview.delete(*app.treeview.get_children())
        cv2.putText(frame, "No hands detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        timer.cancel()
        timer = CustomTimer(4, TimerEnd)
        timerOn = False

    if timerOn:
        app.wordDetectTimer.config(text=str(round(timer.remaining(), 2)))
    else:
        app.wordDetectTimer.config(text="4")
    outFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    app.photo = ImageTk.PhotoImage(image=Image.fromarray(outFrame))
    app.canvas.create_image(0, 0, image=app.photo, anchor="nw")
    app.wordDetectText.delete("1.0", tk.END)
    app.wordDetectText.insert(tk.END, " ".join(alreadyDetected))
    if recording:
        print("Recording")
        outRecorder.write(frame)
    root.update()
