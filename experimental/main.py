# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import time
from tensorflow import *
import cv2
import imutils
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import os
import datetime
from pygame import mixer
import matplotlib.pyplot as mpl
import matplotlib as mp
from imutils.video.webcamvideostream import WebcamVideoStream


class PhotoBoothApp:
	def __init__(self, vs, outputPath):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.outputPath = outputPath
		self.frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None

		# create a button, that when pressed, will take the current
		# frame and save it to file
		btn = tki.Button(self.root, text="Snapshot!",
										 command=self.takeSnapshot)
		btn.pack(side="bottom", fill="both", expand="yes", padx=10,
				 pady=10)

		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("PyImageSearch PhotoBooth")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def videoLoop(self):
		TOTAL_CONSEC = 0
		TOTAL_THRESH = 20
		# initialize the fire alarm
		FIRE = False
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				start = time.time()
				f = 0
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				f += 1
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=720)
				# prepare the image to be classified by our deep learning network
				image = cv2.resize(self.frame, (224, 224))
				image2 = image
				image = image.astype("float") / 255.0
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)

				MODEL_PATH = "./models/njtest1.h5"
				# Experimental CNN - Colab best run
				# MODEL_PATH = "./models/experimentalCNN.h5"
				model = keras.models.load_model(MODEL_PATH)

				begin = time.time()
				(fire, notFire) = model.predict(image)[0]
				terminate = time.time()

				label = "No Fire Detected."
				proba = notFire
				# check to see if fire was detected using our convolutional
				# neural network
				if fire > notFire:
					FIRE = False
					# update the label and prediction probability
					label = "Fire!"
					proba = fire
					# increment the total number of consecutive frames that
					# contain fire
					TOTAL_CONSEC += 1

				if not FIRE and TOTAL_CONSEC >= TOTAL_THRESH:
					FIRE = True
					# CODE FOR NOTIFICATION SYSTEM HERE
					# A siren will be played indefinitely on the speaker
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
			self.frame = cv2.putText(self.frame, label, (10, 25),
								cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

			#  represents images in BGR order; however PIL
			# represents images in RGB order, so we need to swap
			# the channels, then convert to PIL and ImageTk format
			image2 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
			image2 = Image.fromarray(image2)
			image2 = ImageTk.PhotoImage(image2)

			# if the panel is not None, we need to initialize it
			if self.panel is None:
				self.panel = tki.Label(image=image2)
				self.panel.image = image2
				self.panel.pack(side="left", padx=10, pady=10)

				# otherwise, simply update the panel
			else:
				self.panel.configure(image=image2)
				self.panel.image = image2

		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")

	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))

		# save the file
		cv2.imwrite(p, self.frame.copy())
		print("[INFO] saved {}".format(filename))

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()
