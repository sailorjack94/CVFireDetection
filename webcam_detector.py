from __future__ import print_function

class VideoStream:
    def __init__(self, src=0, usePiCamera=False, resolution=(720, 480),
        framerate=32):
            self.stream = cv2.VideoCapture(src=src)

    def start(self):
        # start the threaded video stream
        return self.stream.start()
 
    def update(self):
        # grab the next frame from the stream
        self.stream.update()
 
    def read(self):
        # return the current frame
        return self.stream.read()
 
    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()
            
from tensorflow import *
import cv2
import imutils
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
# from threading import Thread
import numpy as np
import time
import os
import datetime
from pygame import mixer
 

# initialize the total number of frames that *consecutively* contain fire
# along with threshold required to trigger the fire alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20
# initialize the fire alarm
FIRE = False


# make sure pointing to correct model - using raks-bubblebeam
print("[INFO] loading model...")
MODEL_PATH = "./models/inferno_model.h5"

# Experimental CNN - Colab best run
# MODEL_PATH = "./models/experimentalCNN.h5"
model = keras.models.load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)
start = time.time()
f = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    #A variable f to keep track of total number of frames read
    f += 1
    frame = imutils.resize(frame, width=400)
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
 
    # classify the input image and initialize the label and
    # probability of the prediction
    begin = time.time()
    (fire, notFire) = model.predict(image)[0]
    terminate = time.time()

    label = "---"
    proba = notFire
    # check to see if fire was detected using our convolutional
    # neural network
    if fire > notFire:
        # update the label and prediction probability
        label = "Fire!"
        proba = fire
 
        # increment the total number of consecutive frames that
        # contain fire
        TOTAL_CONSEC += 1
        if not FIRE and TOTAL_CONSEC >= TOTAL_THRESH:
            # indicate that fire has been found
            FIRE = True
            #CODE FOR NOTIFICATION SYSTEM HERE
	    #A siren will be played indefinitely on the speaker
            mixer.init()
            mixer.music.load('./alarm.wav')
            mixer.music.play(-1)
            # otherwise, reset the total number of consecutive frames and the
    # fire alarm
    else:
        TOTAL_CONSEC = 0
        FIRE = False
        
        # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    #fps.update()
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        end = time.time()
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = f/ seconds
print("Estimated frames per second : {0}".format(fps))
cv2.destroyAllWindows()
vs.stop()
