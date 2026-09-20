import cv2
import numpy as np
import os

haar_path = os.path.join(
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

haar_cascade = cv2.CascadeClassifier(haar_path)

if haar_cascade.empty():
    print("Failed to load Haar Cascade")
else:
    print("Haar Cascade ready")

recognizer = cv2.face.LBPHFaceRecognizer_create()

def test() :
    print("Hello World")
