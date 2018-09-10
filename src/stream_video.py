from flask import Flask, render_template, Response
import cv2
from flask_cors import CORS
from flask import request
from flask import jsonify
import tensorflow as tf

URL='http://192.168.31.180:8080/video'
#
app = Flask(__name__)
# CORS(app)
# app.config["DEBUG"] = True
# from tensorflow.python.saved_model import tag_constants
# data_dir = './data'
# runs_dir = './runs'
# image_shape = (160, 576)
# import scipy
# import numpy as np
# sess = tf.Session(graph=tf.Graph())
# # with tf.Session(graph=tf.Graph()) as sess:
# tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],'model')
#
# input_img = sess.graph.get_tensor_by_name('image_input:0')
# logits = sess.graph.get_tensor_by_name('logits:0')
# keep_prob =sess.graph.get_tensor_by_name('keep_prob:0')
# def pro_image(image):
#     image = scipy.misc.imresize(image, image_shape)
#     im_softmax = sess.run(
#     [tf.nn.softmax(logits)],
#     {keep_prob: 1.0, input_img: [image]})
#     im_softmax_c = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
#     im_softmax_t = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
#     segmentation_c = (im_softmax_c > 0.5).reshape(image_shape[0], image_shape[1], 1)
#     segmentation_t = (im_softmax_t > 0.5).reshape(image_shape[0], image_shape[1], 1)
#     mask_c = np.dot(segmentation_c, np.array([[0, 255, 0, 127]]))
#     mask_c = scipy.misc.toimage(mask_c, mode="RGBA")
#     mask_t = np.dot(segmentation_t, np.array([[255, 0, 0, 127]]))
#     mask_t = scipy.misc.toimage(mask_t, mode="RGBA")
#     street_im = scipy.misc.toimage(image)
#     street_im.paste(mask_c, box=None, mask=mask_c)
#     street_im.paste(mask_t, box=None, mask=mask_t)
#     return np.array(street_im)
def gen():
    global URL
    print(URL)
    vidcap = cv2.VideoCapture('http://10.41.25.253:8080/video')
    
    while (True):
        success, frame = vidcap.read()
        print(frame.shape)
        test = pro_image(frame)
        # cv2.imshow(frame)
        # print(frame.shape)

        #	process_video()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', test)[1].tostring() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hello')
def hello():
    region = request.args.get('region',None)
    pipe = request.args.get('pipe',None)
    return jsonify({'region':region,'pipe':pipe})

@app.route('/process_video', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    content = request.get_json()
    print (content)
    return 'JSON posted'

if __name__ == '__main__':
    print("**********************hello")
    app.run(host='0.0.0.0', debug=True, use_reloader=False)

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
