import cv2
import tensorflow as tf
video_capture = cv2.VideoCapture(-1)

# URL='http://10.156.172.197:8080/video'
#from src.videowrite import VideoWriter
from videowrite import VideoWriter

URL='http://192.168.43.1:8080/video'

# vwriter = VideoWriter("./inputRecordVideo.mp4", 480, 720, 30)

def live_feed():
    global video_capture

    # from tensorflow.python.saved_model import tag_constants
    data_dir = '../data'
    runs_dir = '../runs'
    image_shape = (160, 576)
    import scipy
    import numpy as np
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],'../live_model_backup')
    input_img = sess.graph.get_tensor_by_name('image_input:0')
    logits = sess.graph.get_tensor_by_name('logits:0')
    keep_prob =sess.graph.get_tensor_by_name('keep_prob:0')
    def pro_image(image):
        input_shape = image.shape
        image = scipy.misc.imresize(image, image_shape)
        im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_img: [image]})
        im_softmax_c = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        im_softmax_t = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation_c = (im_softmax_c > 0.3).reshape(image_shape[0], image_shape[1], 1)
        segmentation_t = (im_softmax_t > 0.3).reshape(image_shape[0], image_shape[1], 1)
        mask_c = np.dot(segmentation_c, np.array([[0, 255, 0, 127]]))
        mask_c = scipy.misc.toimage(mask_c, mode="RGBA")
        mask_t = np.dot(segmentation_t, np.array([[0, 0, 255, 127]]))
        mask_t = scipy.misc.toimage(mask_t, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask_c, box=None, mask=mask_c)
        street_im.paste(mask_t, box=None, mask=mask_t)
        str_im=np.array(street_im)
        str_im=scipy.misc.imresize(str_im, input_shape)
        return str_im
    i = 0
    while(True):
        ret, frame = video_capture.read()
        if not ret:
            continue
        # vwriter.write(frame)
        processed_frame = pro_image(frame)
        cv2.imwrite("imgs/" + str(i).zfill(5)+".jpg", processed_frame)
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', processed_frame)[1].tostring() + b'\r\n')
        # cv2.imshow(processed_frame)

def gen():
    # vidcap = cv2.VideoCapture(-1)
    while (True):
        success, frame = video_capture.read()
        if frame is not None:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n')



   
