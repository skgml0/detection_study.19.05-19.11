from imutils.video import VideoStream
from flask import Response,request
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import pymysql
# from camera import Camera
from flask import render_template

# initialize a flask object
app = Flask(__name__,static_url_path='/static')


@app.route("/")
def login1():
    return render_template('login.html')


@app.route("/app")
def app1():
    # return the rendered template
    return render_template("app.html")


@app.route("/room")
def room1():
    # return the rendered template
    return render_template("room.html")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)


