import sys

sys.path.append("/Users/niall/codeclan_work/final_project/")

from fireDetectCNN import inceptionMap
import cv2
import math as m
from fireDetectCNN.inceptionMap import construct_inceptionv1onfire
import os

# InceptionCNN

if __name__ == '__main__':

    model = construct_inceptionv1onfire (224, 224, training=False)
    # model.load(os.path.join("models/InceptionV4-OnFire", "inceptionv4onfire"),weights_only=True)
    model.load(os.path.join("modelsExperimental/InceptionV1-OnFire", "inceptiononv1onfire"),weights_only=True)  
    print("[INFO] Loaded CNN network weights ...")

    # network input sizes - model layout must match weights pattern
    rows = 224
    cols = 224

    # display and loop settings
    windowName = "Inception V1"
    keepProcessing = True

    # initialise webcam input
    video = cv2.VideoCapture(0)
    print("[INFO] Loaded video ...")

    # open window
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    # grab video info
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000/fps)

    while (keepProcessing):
        start_t = cv2.getTickCount()
        ret, frame = video.read()
        if not ret:
            print("[INFO] ... end of video file reached")
            break

        # re-size image to network input size and perform prediction
        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

        # perform prediction on the image frame which is:
        # - an image (tensor) of dimension 224 x 224 x 3
        # - a 3 channel colour image with channel ordering BGR (not RGB)
        output = model.predict([small_frame])

        # label image based on prediction
        if round(output[0][0]) == 1: # equiv. to 0.5 threshold in [Dunnings / Breckon, 2018],  [Samarth/Bhowmik/Breckon, 2019] test code
            cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),10,cv2.LINE_AA)
        else:

            cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)

        # stop the timer and convert to ms
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # video stream display
        cv2.imshow(windowName, frame)

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, frame_time - int(m.ceil(stop_t)))) & 0xFF

        # exit key
        if (key == ord('x')):
            keepProcessing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)










# Refactored from [Dunnings / Breckon, 2018],  [Samarth/Bhowmik/Breckon, 2019]