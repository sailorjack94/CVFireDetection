import sys
import os
import tkinter as tk
from tkinter import ttk
from tkinter import *
from ttkthemes import ThemedTk


gui = ThemedTk(theme="yaru")
gui.title('CV Fire Detector Launcher')
gui.geometry("400x400")
gui.configure(bg='white')

def launchCVDetector():
    os.system('python3 cv_fire_by_colour.py')

def launchInferno():
    os.system('python3 webcam_detector.py')

def launchInception():
    os.system('python3 inceptionRunner.py')



heading = Label(gui, text="Select your Application...", bg="white", fg="black", pady=10)
heading.pack(pady=10)

button0 = ttk.Button(gui, text="CV Colour Detector", command=launchCVDetector)
button0.pack(pady=10)
# button0.grid(columnspan=3, column=2, row=2, sticky='n', pady=10)

button1=ttk.Button(gui,text="Inferno Fire Detector (TensorFlow/Keras - YOLO5)",command= launchInferno)
button1.pack(pady=10)
# button1.grid(columnspan=3, column=2, row=3, sticky='n', pady=10)

button2 = ttk.Button(gui, text='InceptionV01 Detector(GoogLeNet - Inception)', command=launchInception)
button2.pack(pady=10)
# button2.grid(columnspan=3, column=2, row=4, sticky='n', pady=10)

ttk.Button(gui, text="Quit" ,command=gui.destroy).pack(pady=10)



gui.mainloop()