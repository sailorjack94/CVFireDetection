import cv2
import numpy as numpy
import matplotlib.pyplot as plt

 

cam = cv2.VideoCapture(0)

lower_bound = numpy.array([11,33,111])
upper_bound = numpy.array([90,255,255])

while(cam.isOpened()):

    ret, frame = cam.read()
    frame = cv2.resize(frame,(1280,720))
    frame = cv2.flip(frame,1)
    frame_smooth = cv2.GaussianBlur(frame,(7,7),0)

    mask = numpy.zeros_like(frame)
    mask[0:720, 0:1280] = [255,255,255]

    amplify = cv2.bitwise_and(frame_smooth, mask)

    hsv = cv2.cvtColor(amplify,cv2.COLOR_BGR2HSV)

    binary = cv2.inRange(hsv, lower_bound, upper_bound)

    check_for_fire = cv2.countNonZero(binary)


    if int(check_for_fire) >= 20000 :

        cv2.putText(frame,"Fire!",(300,60),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),2)


    cv2.imshow("Fire Detection",frame)


    if cv2.waitKey(10) == 27 :
        break

 

cam.release()

cv2.destroyAllWindows()

