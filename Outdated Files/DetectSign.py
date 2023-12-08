import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

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

        self.expandDatasetButton = tk.Button(self, text="Expand Dataset", command=ExpandDataset)
        self.expandDatasetButton.grid(row=2, column=3, padx=10, pady=10)
        
        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.grid(row=3, column=0, columnspan=5, pady=10)

def ExpandDataset():
    pass


# Create the tkinter window
root = tk.Tk()
app = Application(master=root)

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

# Dictionary to convert the label number to a letter
labelsDict = {}

# Dictionary to convert the label number to a letter
for label in model_dict['orderedLabels']:
    labelsDict[model_dict['orderedLabels'].index(label)] = label

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

        # print(dataOut)

        prediction = model.predict(np.array([dataOut]))

        probability = model.predict_proba(np.array([dataOut]))

        probability = round(np.amax(probability[0]) * 100, 2)

        cv2.putText(frame, str(prediction[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        app.treeview.delete(*app.treeview.get_children())
        for out in prediction:
            app.treeview.insert("", tk.END, values=(out, probability))
            
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    app.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    app.canvas.create_image(0, 0, image=app.photo, anchor="nw")
    root.update()

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) == ord('s'):
    #     cv2.imwrite('save.jpg', frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break
