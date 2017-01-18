import cv2, imutils
import numpy as np

def recognize():
    camera = cv2.VideoCapture(0)

    while(camera.isOpened()):
        _, frame = camera.read()
        cv2.imshow('Hand gesture recognition', frame)

        k = cv2.waitKey(10)
        if k == 27:  # esc
            break
