import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from drawLandmarks import *
from imgHandling import *
from processResults import *

def main():

    baseOptions = python.BaseOptions('gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=baseOptions)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    global drawUtils
    drawUtils = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        frameForDetection, frameForOutput = collectFrame(capIn= cap)
        mpImg = convertToMpImage(frameForDetection)

        recognitionResults, label = processResults(mpImg, recognizer)

        outputFrame(frameForOutput, label, recognitionResults)

        if cv2.waitKey(1) == ord('q'):
            break
    

    cap.release()

main()
