import cv2
import numpy as np
import playsound
import matplotlib.pyplot as plt

fire_alarm = 0
alarm = False

# video = cv2.VideoCapture("test_fire.mp4")
# video = cv2.VideoCapture("test_no_fire.mp4")
video = cv2.VideoCapture("cat.mp4")
# video = cv2.VideoCapture(0)

def sound_alarm():
    playsound.playsound("alarm.wav")


while True:

    ret, frame = video.read()
    frame = cv2.resize(frame, (1000,600)) 
    blur = cv2.GaussianBlur(frame, (15,15), 0)
    # Thermal View
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [20, 35, 35]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
 
    output = cv2.bitwise_and(frame, hsv, mask=mask)

    # pixel detection
    total = cv2.countNonZero(mask)

    if int(total) > 30000:
        print ("fire detected")
        fire_alarm += 1

        if fire_alarm > 10:
            if alarm == False:
                # sound_alarm()
                alarm = True


 
    if ret == False:
        break

    if cv2.waitKey(10) == 27 :

        break

    label = "Fire!"
    label2 = "No Fire Detected..."
    # Change rendered video her
    if fire_alarm > 10:
        frame = cv2.putText(frame, label, (30, 100),
            cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
    else:
        frame = cv2.putText(frame, label2, (10, 25),
            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Basic CV Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video.release()