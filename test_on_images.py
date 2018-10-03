import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def test():
    from tensorflow.python.saved_model import tag_constants
    data_dir = '../data'
    runs_dir = '../runs'
    image_shape = (160, 576)

    import scipy
    import numpy as np
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],'../model')

      input_img = sess.graph.get_tensor_by_name('image_input:0')
      logits = sess.graph.get_tensor_by_name('logits:0')
      keep_prob =sess.graph.get_tensor_by_name('keep_prob:0')

      helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_img)
    
    
if __name__ == '__main__':

    test()

