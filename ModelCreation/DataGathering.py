import os
import cv2
import mediapipe as mp
import pickle


def IndividualData(classObj):   
    outArray = []
    classDir = datasetDir + "/" + str(classObj)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        showFrame = frame.copy()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        handsResults = hands.process(rgbFrame)

        if handsResults.multi_hand_landmarks:
            # print(len(handsResults.multi_hand_landmarks))
            # print the hands on the frame
            
            for handLandmarks in handsResults.multi_hand_landmarks:
                print(handLandmarks.landmark[mpHands.HandLandmark.PINKY_TIP].x)
                mpDrawing.draw_landmarks(showFrame, handLandmarks, mpHands.HAND_CONNECTIONS, mpDrawingStyles.get_default_hand_landmarks_style(), mpDrawingStyles.get_default_hand_connections_style())
        
        frameText = ('Now collecting data for class ' + str(classObj) + ' press "Q" to start')
        cv2.putText(showFrame, frameText, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', showFrame)
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite("ahhh.jpg", showFrame)
    
    for i in range(100):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        showFrame = frame.copy()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        handsResults = hands.process(rgbFrame)

        if handsResults.multi_hand_landmarks:
            # print the hands on the frame
            for handLandmarks in handsResults.multi_hand_landmarks:
                mpDrawing.draw_landmarks(showFrame, handLandmarks, mpHands.HAND_CONNECTIONS, mpDrawingStyles.get_default_hand_landmarks_style(), mpDrawingStyles.get_default_hand_connections_style())

        frameText = ('Starting Collection in ' + str(100 - i))
        cv2.putText(showFrame, frameText, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", showFrame)
        cv2.waitKey(1)

    pic = 0
    while pic < datasetSize:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        showFrame = frame.copy()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgbFlip = cv2.flip(rgbFrame, 1)

        dataOut = []
        dataOutFlipped = []

        handsResults = hands.process(rgbFrame)
        handsResultsFlipped = handsFlipped.process(rgbFlip)

        if handsResults.multi_hand_landmarks:
            if len(handsResultsFlipped.multi_hand_landmarks) == 2:
                print("Two hands detected, for double hand sign...")
                # print("Only one hand detected, skipping...")
                frameText = ('Collecting' + str(pic) + "/" + str(datasetSize))
                cv2.putText(showFrame, frameText, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("frame", showFrame)
                cv2.waitKey(1)
                pic += 1
            elif classObj in listOfSingleHandSigns:
                print("Only one hand detected, for single hand sign...")
                frameText = ('Collecting' + str(pic) + "/" + str(datasetSize))
                cv2.putText(showFrame, frameText, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("frame", showFrame)
                cv2.waitKey(1)
                pic += 1
            else:
                continue
    

        if handsResults.multi_hand_landmarks:
            # print(len(handsResults.multi_hand_landmarks))
            # print the hands on the frame
            for handLandmarks in handsResults.multi_hand_landmarks:
                mpDrawing.draw_landmarks(showFrame, handLandmarks, mpHands.HAND_CONNECTIONS, mpDrawingStyles.get_default_hand_landmarks_style(), mpDrawingStyles.get_default_hand_connections_style())
                for point in range(len(handLandmarks.landmark)):
                    x = handLandmarks.landmark[point].x
                    y = handLandmarks.landmark[point].y
                    dataOut.append(x)
                    dataOut.append(y)

            outArray.append(dataOut)
        
        if handsResultsFlipped.multi_hand_landmarks:
            if len(handsResultsFlipped.multi_hand_landmarks) == 2:
                print("Two hands detected, for double hand sign...")
                for handLandmarks in handsResultsFlipped.multi_hand_landmarks:
                    for point in range(len(handLandmarks.landmark)):
                        x = handLandmarks.landmark[point].x
                        y = handLandmarks.landmark[point].y
                        dataOutFlipped.append(x)
                        dataOutFlipped.append(y)

            elif classObj in listOfSingleHandSigns:
                print("Only one hand detected, for single hand sign...")
                for handLandmarks in handsResultsFlipped.multi_hand_landmarks:
                    for point in range(len(handLandmarks.landmark)):
                        x = handLandmarks.landmark[point].x
                        y = handLandmarks.landmark[point].y
                        dataOutFlipped.append(x)
                        dataOutFlipped.append(y)
            
            else:
                print("Only one hand detected, skipping...")

                outArray.append(dataOutFlipped)

        frameText = ('Collecting' + str(pic) + "/" + str(datasetSize))
        cv2.putText(showFrame, frameText, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", showFrame)
        cv2.waitKey(1)
        print(len(outArray))
        
    with open("ModelCreation/dataset/" + str(classObj) + "-3-data.pickle", "wb") as f:
        pickle.dump({'data': outArray}, f)
    

    print("Class " + str(classObj) + " done")

width, height = 1280, 720

listOfClasses = ["Backspace", "Enter", "Space"]
listOfSingleHandSigns = ["Backspace", "Enter", "Space"]
datasetSize = 500

datasetDir = "ModelCreation/dataset"
if not os.path.exists(datasetDir):
    os.makedirs(datasetDir)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

hands = mpHands.Hands(min_detection_confidence=0.3, max_num_hands=2)
handsFlipped = mpHands.Hands(min_detection_confidence=0.3, max_num_hands=2)

for classObj in listOfClasses:
    IndividualData(classObj)
    

