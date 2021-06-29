from flask import Flask
from flask.templating import render_template
from flask.wrappers import Response
from webcam_detector import VideoStream
import threading

app = Flask(__name__)

if __name__ == "__main__":
    threading.Thread(target=app.run).start()

@app.route('/')
def index():
    return render_template('index.html')

