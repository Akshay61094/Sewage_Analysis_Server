from flask import Flask, render_template, Response
import cv2
from flask_cors import CORS
from flask import jsonify

URL='http://192.168.31.180:8080/video'

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


def gen():
    global URL
    print(URL)
    vidcap = cv2.VideoCapture('http://192.168.43.1:8080/video')

    while (True):
        success, frame = vidcap.read()
        # cv2.imshow(frame)
        # print(frame.shape)

        #	process_video()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hello')
def hello():
    return jsonify("hello")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # cap = cv2.VideoCapture('http://192.168.31.180:8080/video')
    # while (True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
