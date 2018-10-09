from flask import Flask, render_template, Response
import cv2
import os.path
from flask_cors import CORS
from flask import request
from flask import jsonify
import tensorflow as tf
import matplotlib.pyplot as plt
import LiveProcessing
#from src import LiveProcessing

URL='http://10.156.172.197:8080/video'

app = Flask(__name__)
CORS(app)
region_selected = "Region1"
pipe_selected = "Pipe1"
def processVideo(inputUrl):
    from tensorflow.python.saved_model import tag_constants
    data_dir = './data'
    runs_dir = './runs'
    image_shape = (160, 576)
    import scipy
    import numpy as np
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], '../model')

        input_img = sess.graph.get_tensor_by_name('image_input:0')
        logits = sess.graph.get_tensor_by_name('logits:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')

        def process_image(image):
            # print("input shape",image.shape)
            input_shape=image.shape
            # print("shape of input",input_shape)
            image = scipy.misc.imresize(image, image_shape)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, input_img: [image]})
            im_softmax_c = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
            im_softmax_t = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation_c = (im_softmax_c > 0.5).reshape(image_shape[0], image_shape[1], 1)
            segmentation_t = (im_softmax_t > 0.4).reshape(image_shape[0], image_shape[1], 1)
            mask_c = np.dot(segmentation_c, np.array([[0, 255, 0, 127]]))
            mask_c = scipy.misc.toimage(mask_c, mode="RGBA")
            mask_t = np.dot(segmentation_t, np.array([[255, 0, 0, 127]]))
            mask_t = scipy.misc.toimage(mask_t, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask_c, box=None, mask=mask_c)
            street_im.paste(mask_t, box=None, mask=mask_t)
            str_im=np.array(street_im)
            str_im=scipy.misc.imresize(str_im, input_shape)
            # print("shape of output", str_im.shape)
            return str_im

        from moviepy.editor import VideoFileClip

        output_location = '../videos/output/'+inputUrl+'_output.mp4'
        video_input = VideoFileClip('../videos/input/'+inputUrl)

        video_output = video_input.fl_image(process_image)  # NOTE: this function expects color images!!

        video_output.write_videofile(output_location, audio=False)

# def gen(flag):
#     global region_selected
#     global pipe_selected
#     if(flag==False):
#         vidcap = cv2.VideoCapture('../videos/input/'+region_selected + pipe_selected+'.mp4')

#         while (True):
#             success, frame = vidcap.read()
#             plt.imshow(frame)
#             if frame is not None:
#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' +cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n')
#     else:
#         vidcap = cv2.VideoCapture('../videos/output/' + region_selected + pipe_selected + '_output.mp4')

#         while (True):
#             success, frame = vidcap.read()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    global region_selected
    global pipe_selected
    return Response(open('../videos/input/Region1Pipe1.mp4', "rb"), mimetype="video/mp4")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hello')
def hello():
    region = request.args.get('region',None)
    pipe = request.args.get('pipe',None)
    return jsonify({'region':region,'pipe':pipe})

@app.route('/input_video_feed')
def input_video_feed():
    global region_selected
    global pipe_selected
    return Response(open('../videos/input/' + region_selected + pipe_selected , "rb"), mimetype="video/mp4")

@app.route('/output_video_feed')
def output_video_feed():
    global region_selected
    global pipe_selected
    return Response(open('../videos/output/' + region_selected + pipe_selected + '_output.mp4', "rb"), mimetype="video/mp4")


@app.route('/process_video', methods = ['POST'])
def postJsonHandler():
    # print (request.is_json)
    global region_selected
    global  pipe_selected

    content = request.get_json()
    # print (content)
    region_selected = content['region']
    pipe_selected = content['pipe']
    print(region_selected,pipe_selected)
    inputUrl = region_selected+pipe_selected
    if not os.path.exists('../videos/output/' + region_selected + pipe_selected + '_output.mp4'):
        processVideo(inputUrl)
    return jsonify('Ok')


@app.route('/live_output_video_feed')
def live_output_video_feed():
    return Response(LiveProcessing.live_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_input_video_feed')
def live_input_video_feed():
    print("in 1")
    return Response(LiveProcessing.gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
