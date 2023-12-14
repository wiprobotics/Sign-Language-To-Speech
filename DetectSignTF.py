import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import math
import threading
import time
import gtts
import os
import sys

from tensorflow import keras
from tkinter import ttk
from PIL import Image, ImageTk


# The main tkinter application window
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
       
        self.quit = tk.Button(self, text="QUIT", fg="red", command=QuitApp)
        self.quit.grid(row=3, column=0, columnspan=2, pady=10)


# A custom timer class that allows for the elapsed time to be calculated
class CustomTimer(threading.Timer):
    started_at = None

    def start(self):
        self.started_at = time.time()
        threading.Timer.start(self)
 
    def elapsed(self):
        return time.time() - self.started_at

    def remaining(self):
        return self.interval - self.elapsed()


# Function to toggle the word detect button and feature
def ToggleWordDetect():
    global wordDetect
    if wordDetect:
        wordDetect = False
        app.wordDetectButton.config(text="Start Word Detect")
    else:
        wordDetect = True
        app.wordDetectButton.config(text="Stop Word Detect")


# Function to toggle the recording button and feature
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


# Function to load the model and required dictionaries
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


# Function to load the mediapipe library
def LoadMediapipe():
    global mp_hands, mp_drawing, mp_drawing_styles, hands
    # Set up mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(min_detection_confidence=0.3, max_num_hands=2)


# Function which is run when the word detect timer ends
def TimerEnd():
    global timerOn
    global wordDetect
    global lastLetter
    global alreadyDetected
    global timer
    timer = CustomTimer(4, TimerEnd)
    timerOn = False
    lastLetter = ""
    if labelsDict[letterPrediction] not in commandWords:
        alreadyDetected.append(labelsDict[letterPrediction])
    else:
        if labelsDict[letterPrediction] == "Backspace":
            if len(alreadyDetected) > 0:
                alreadyDetected.pop()
        elif labelsDict[letterPrediction] == "Enter":
            print("Enter detected")
            outWord = ""
            for item in alreadyDetected:
                outWord += item
            alreadyDetected = []
            SpeakWord(outWord)
        elif labelsDict[letterPrediction] == "Space":
            alreadyDetected.append(" ")


# Speaks a word using the gtts library
def SpeakWord(word):
    tts = gtts.gTTS(word)
    tts.save("word.mp3")
    os.system("mpg123 word.mp3")


# Function to quit the application
def QuitApp():
    cap.release()
    outRecorder.release()
    sys.exit()


# Create the tkinter window
root = tk.Tk()
app = Application(master=root)

# Object variables and initialisation
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outRecorder = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
timer = CustomTimer(4, TimerEnd)

# Boolean variables
recording = False
wordDetect = False
timerOn = False

# String and list variables
lastLetter = ""
alreadyDetected = []
commandWords = ["Backspace", "Enter", "Space"]
bannedWords = [""]

# Run the initialisation functions
LoadModel()
LoadMediapipe()

# Open the webcam feed
cap = cv2.VideoCapture(2)

# Main loop
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dataOut = []

    # Convert the frame to RGB for mediapipe
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the hand landmarks from the frame
    handsResults = hands.process(imgRGB)

    # If there are hands in the frame run through them and extract x and y coordinates
    if handsResults.multi_hand_landmarks:
        for handLandmarks in handsResults.multi_hand_landmarks:
            for i in range(len(handLandmarks.landmark)):
                x = handLandmarks.landmark[i].x
                y = handLandmarks.landmark[i].y
                dataOut.append(x)
                dataOut.append(y)
            mp_drawing.draw_landmarks(frame, handLandmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        # Pad the data to 84 elements incase their is only one hand
        if len(dataOut) < 84:
            dataOut = np.pad(dataOut, (0, 84 - len(dataOut)), 'constant')
        # Truncate or pad the item to exactly 84 elements
        dataOut = dataOut[:84]

        # Arrays to hold the x and y coordinates of the tips and base of the fingers and palms
        hand1tips = [[], []]
        hand2tips = [[], []]

        # If there are two hands in the frame calculate the distance between each tip of hand 1 and each tip of hand 2
        if len(handsResults.multi_hand_landmarks) > 1:
            # Get the coordinates of the tips of the fingers for hand 1
            for h1 in range(0, 41, 8):
                hand1tips[0].append(dataOut[h1])      # x for tip of finger
                hand1tips[1].append(dataOut[h1 + 1])  # y for tip of finger
                if (h1 + 2) < 41:
                    hand1tips[0].append(dataOut[h1 + 2])  # x for base of next finger
                    hand1tips[1].append(dataOut[h1 + 3])  # y for base of next finger

            # Get the coordinates of the tips of the fingers for hand 2
            for h2 in range(42, 84, 8):
                hand2tips[0].append(dataOut[h2])      # x for tip of finger
                hand2tips[1].append(dataOut[h2 + 1])  # y for tip of finger
                if (h2 + 2) < 84:
                    hand2tips[0].append(dataOut[h2 + 2])  # x for base of next finger
                    hand2tips[1].append(dataOut[h2 + 3])  # y for base of next finger
            
            # Calculate the distance between each tip of hand 1 and each tip of hand 2
            distances = []
            for point1 in range(len(hand1tips[0])):
                for point2 in range(len(hand2tips[0])):
                    # Calculate the distance between the two points
                    distance = math.sqrt((hand1tips[0][point1] - hand2tips[0][point2])**2 + (hand1tips[1][point1] - hand2tips[1][point2])**2)
                    distances.append(distance)

            dataOut = np.append(dataOut, distances)
        else:
            dataOut = np.pad(dataOut, (0, 205 - len(dataOut)), 'constant')

        # Reshape the data to be a 1x205 array for the tensorflow model
        dataOut = np.array(dataOut).reshape(1, 205)

        # Get the prediction from the model
        predictions = model.predict(dataOut)

        # Get the prediction in letter form
        letterPrediction = np.argmax(predictions[0])

        # Display the prediction on the screen
        text = "Prediction: " + str(labelsDict[letterPrediction])
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Get only the predictions with a probability of over 0 and put them 
        # in a sorted list based on probability
        predictions_list = [(labelsDict[i], round((out * 100), 1)) for i, out in enumerate(predictions[0]) if round((out * 100), 1) > 0]
        predictions_list.sort(key=lambda x: x[1], reverse=True)

        # If word detect is on then run through the letter detection logic
        if wordDetect:
            # If the last letter is empty then set it to the current letter
            if lastLetter == "":
                lastLetter = labelsDict[letterPrediction]
                print("First letter detected")

            # If the last letter is the same as the current letter then start the timer
            elif lastLetter == labelsDict[letterPrediction] and timerOn is False:
                timerOn = True
                timer.start()
                print("Timer started")

            # If the last letter is not the same as the current letter
            # (if you're signing something else) then reset the timer
            elif lastLetter != labelsDict[letterPrediction]:
                lastLetter = labelsDict[letterPrediction]
                timer.cancel()
                timer = CustomTimer(4, TimerEnd)
                timerOn = False
                print("Timer cancelled")

        # Display the predictions in the info panel table
        app.treeview.delete(*app.treeview.get_children())
        for label, probability in predictions_list:
            app.treeview.insert("", tk.END, values=(label, probability))

    # If there are no hands in the frame then display a message on the screen
    else:
        app.treeview.delete(*app.treeview.get_children())
        cv2.putText(frame, "No hands detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        timer.cancel()
        timer = CustomTimer(4, TimerEnd)
        timerOn = False

    # Display the word detect timer and the already detected words
    if timerOn:
        app.wordDetectTimer.config(text=str(round(timer.remaining(), 2)))
    else:
        app.wordDetectTimer.config(text="4")
    outFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    app.photo = ImageTk.PhotoImage(image=Image.fromarray(outFrame))
    app.canvas.create_image(0, 0, image=app.photo, anchor="nw")
    app.wordDetectText.delete("1.0", tk.END)
    app.wordDetectText.insert(tk.END, " ".join(alreadyDetected))

    # If recording is on then write the frame to the output video
    if recording:
        print("Recording")
        outRecorder.write(frame)

    # Update the tkinter window
    root.update()
