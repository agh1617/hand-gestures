import cv2, imutils
import numpy as np
import math
import code

def recognize():
    camera = cv2.VideoCapture(0)

    while(camera.isOpened()):
        _, frame = camera.read()

        cv2.rectangle(frame,(450,450),(50,50),(0,255,0),0)

        crop_img = frame[50:450, 50:450]

        gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray,(35,35),0)

        _, tresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        image, contours, hierarchy = cv2.findContours(tresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contour = max(contours, key = lambda x: cv2.contourArea(x))

        hull = cv2.convexHull(contour)

        drawing = np.zeros(crop_img.shape,np.uint8)

        cv2.drawContours(drawing,[contour],0,(0,255,0),0)
        cv2.drawContours(drawing,[hull],0,(0,0,255),0)

        hull = cv2.convexHull(contour,returnPoints = False)
        defects = cv2.convexityDefects(contour,hull)

        if defects is not None:
            count_defects = 0
            cv2.drawContours(tresh, contours, -1, (0,255,0), 3)

            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]

                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                if angle <= 90:
                    count_defects += 1
                    cv2.circle(drawing,far,5,[0,0,255],-1)

                cv2.line(drawing,start,end,[0,255,0],2)

            ellipse = cv2.fitEllipse(contour)
            _, size, _ = ellipse
            w,h = size

            text_position = (frame.shape[1] - 100,100)

            if count_defects > 0:
                cv2.putText(frame,"N", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 3, 5)
            elif w/h > 0.5:
                cv2.putText(frame,"K", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 3, 5)
            else:
                cv2.putText(frame,"P", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 3, 5)

            frame[50:450, 50:450] = drawing

        cv2.imshow('Hand gesture recognition', frame)

        k = cv2.waitKey(10)
        if k == 27:  # esc
            break

    camera.release()
    cv2.destroyAllWindows()
