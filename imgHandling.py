import numpy as np
import mediapipe as mp
import cv2
from drawLandmarks import *

def collectFrame(capIn: cv2.VideoCapture) -> tuple[np.ndarray, np.ndarray]:

    _, frameTemp = capIn.read()
    frameColored = cv2.flip(frameTemp, 1)
    frameOut = cv2.cvtColor(frameColored, cv2.COLOR_BGR2RGB)

    return frameOut, frameColored

def convertToMpImage(cv2Img):

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2Img)

def outputFrame (frameIn: np.ndarray, text:str, results):

    frameIn = draw_landmarks_on_image(frameIn, results)
    
    frameOut = cv2.putText(frameIn, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frameOut)
