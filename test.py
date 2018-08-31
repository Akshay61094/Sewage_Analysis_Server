from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
	print("anupam************************************")
	vidcap = cv2.VideoCapture('http://192.168.43.1:8080/video')

	while(True):
		success,frame = vidcap.read()
		#cv2.imshow(frame)
		print(frame.shape)
	
#	process_video()

		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
