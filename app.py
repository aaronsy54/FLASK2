from cv2 import VideoCapture
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import io
import base64,cv2
import numpy as np
from flask_cors import CORS,cross_origin
from engineio.payload import Payload
import torch
import numpy as np
from PIL import Image

Payload.max_decode_packets = 4000
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def get_yolov5():
    # local best.pt
    model = torch.hub.load('./yolov5', 'custom', path='yolov5/models/best.pt', source = 'local')  # local repo
    model.conf = 0.85
    return model

model = get_yolov5()

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

@socketio.on('catch-frame')
def catch_frame(data):

    emit('response_back', data)  

def moving_average(x):
    return np.mean(x)

global stringData
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]
@socketio.on('image')
def two(data_image):
    global stringData
    frame = (readb64(data_image))
    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]    
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData




def videocap():
    
       while True:
        cam=cv2.VideoCapture(stringData)
            ## read the camera frame
        success,frame1=cam.read()
        if not success:
            break
        else:
            results=model(frame1)
            results.render()
            ret,buffer=cv2.imencode('.jpg',frame1)
            frame1=buffer.tobytes()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
  

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model.html',methods=['POST', 'GET'])
def test():
    return render_template('model.html')

@app.route('/webcam.html',methods=['POST', 'GET'])
def webcam2():
    return render_template('webcam.html')

@app.route('/video')
def video():
    return Response(videocap(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app,port=9990 ,debug=True)
   
