from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
#from camera import Camera

from flask import render_template
db=pymysql.connect(host="",port=,user="",password="",db="",charset="utf8")
cursor=db.cursor()

# initialize a flask object
app = Flask(__name__)

@app.route("/",methods=['POST'])
def login1():
    if request.method =='POST':
        login_info=request.form

        USER =login_info['userID']
        sql ="SELECT * FROM USER WHERE userID=%s"
        rows_count =cursor.execute(sql,USER)

        if rows_count>0:
            user_info= cursor.fetchone()
            print("user info: ", user_info)

        else:
            print('User does not exist')

        return redirect(request.url)
    return render_template('login.html')

@app.route("/app")
def app1():
    # return the rendered template
    return render_template("app.html")

@app.route("/room")
def room1():
    # return the rendered template
    return render_template("room.html")

@app.route('/call')
def video_test():
    print("call!!")
    test_tt()
    
import os
from subprocess import call
def test_tt():
    os.system('python3 pi_face_recognitiontest.py --cascade haarcascade_frontalface_default.xml        --encodings encodings.pickle')
    #call('python3 pi_face_recognition.py --cascade haarcascade_frontalface_default.xml        --encodings encodings.pickle')
# check to see if this is the main thread of execution
if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug=True)


