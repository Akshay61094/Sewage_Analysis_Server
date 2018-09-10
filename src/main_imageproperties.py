import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy
import cv2
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph=tf.get_default_graph()
    input_image=graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3=  graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer_3, layer_4, layer_7
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    #layer 7 convolved 1x1
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,1,padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),)

    #layer 7 upsampled twice
    layer_7_upsampled =  tf.layers.conv2d_transpose(conv_1x1, filters=num_classes, kernel_size=(3, 3),
                                                          strides=(2, 2), padding='same',
                                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # layer 4 convolved 1x1 to match the size to 2, so that it can be added to 7th layer
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                            padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # resized layer 4 and 7 combined
    layer_4_7_combined = tf.add(layer_7_upsampled, layer4_1x1)

    layer47_upsampled = tf.layers.conv2d_transpose(layer_4_7_combined, filters=num_classes, kernel_size=(3, 3),
                                                   strides=(2, 2), name="layer47_upsampled", padding='same',
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # layer 4 convolved 1x1 to match the size to 2, so that it can be added to 8th layer
    layer3_1x1_out = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                      name="new_layer3_1x1_out",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer_3_8_combined = tf.add(layer3_1x1_out, layer47_upsampled)

    layer_3_8_combined_upsampled = tf.layers.conv2d_transpose(layer_3_8_combined, filters=num_classes, kernel_size=(16, 16),
                                                             strides=(8, 8), padding='same',name="final_layer_upsampled_8x",
                                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return layer_3_8_combined_upsampled

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    cross_entropy_loss += tf.losses.get_regularization_loss()
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    print("training....")

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        print("EPOCH {} ...".format(i + 1))
        tot_loss=0.0
        batch_count=0
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.7,
                                          learning_rate: 0.0005})
            batch_count+=1
            tot_loss+=loss
            print("epoch no",i,"batch_number",batch_count,"loss = {:.3f}".format(loss))
        print()
        print("Avg loss: = {:.3f}".format(tot_loss/batch_count))
        

tests.test_train_nn(train_nn)


def run():
    num_classes = 3
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road_1/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 25
        batch_size = 2

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_img, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_img,
                 correct_label, keep_prob, learning_rate)

        # Saving
        inputs = {
            "input_img": input_img,
             "logits": logits,
             "keep_prob":keep_prob
        }
        outputs = {
            "logits": logits
        }
#         outputs = {"prediction": model_output}
        tf.saved_model.simple_save(
            sess, 'model/', inputs,outputs
        )
       

        # OPTIONAL: Apply the trained model to a video

def process_frame(image_file):
  image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
  image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  im_softmax = sess.run(
      [tf.nn.softmax(logits)],
      {keep_prob: 1.0, image_pl: [image]})
  im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
  segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
  mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
  mask = scipy.misc.toimage(mask, mode="RGBA")
  street_im = scipy.misc.toimage(image)
  street_im.paste(mask, box=None, mask=mask)
  
  return street_im
        
        
      
def restore_and_predict():
    from tensorflow.python.saved_model import tag_constants
    data_dir = './data'
    runs_dir = './runs'
    image_shape = (160, 576)

    import scipy
    import numpy as np
    sess = tf.Session()
    # with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],'model')

    input_img = sess.graph.get_tensor_by_name('image_input:0')
    logits = sess.graph.get_tensor_by_name('logits:0')
    keep_prob =sess.graph.get_tensor_by_name('keep_prob:0')
    def process_image(image):
        image = scipy.misc.imresize(image, image_shape)
        im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_img: [image]})
        im_softmax_c = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        im_softmax_t = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation_c = (im_softmax_c > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_t = (im_softmax_t > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask_c = np.dot(segmentation_c, np.array([[0, 255, 0, 127]]))
        mask_c = scipy.misc.toimage(mask_c, mode="RGBA")
        mask_t = np.dot(segmentation_t, np.array([[255, 0, 0, 127]]))
        mask_t = scipy.misc.toimage(mask_t, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask_c, box=None, mask=mask_c)
        street_im.paste(mask_t, box=None, mask=mask_t)
        image_properties(segmentation_c,segmentation_t)
        cv2.imshow('gfg',np.array(street_im))
        cv2.waitKey()
        return np.array(street_im) 
    from moviepy.editor import VideoFileClip

    output_location = 'merged_video_out.mp4'
    video_input = VideoFileClip("merged_video.mp4")

    video_output = video_input.fl_image(process_image).subclip(2,4) #NOTE: this function expects color images!!
    video_output.write_videofile(output_location, audio=False)
    video_input.reader.close()
    video_input.audio.reader.close_proc()
    video_output.reader.close()
    video_output.audio.reader.close_proc()
    # img = scipy.misc.imread("/home/allahbaksh/semantic_segmentation_cracks/data/data_road_1/testing/image_2/c4.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    # process_image(img)
    

def image_properties(segmentation_c, segmentation_t):

    frame_count=0
    detected = False
    crack_detected = False
    root_detected = False
    fault_not_found_duration = 0

    crack_set=[]
    crack_start=[]
    crack_end=[]
    crack_found_duration = 0
    crack_not_found_duration = 0

    root_set=[]
    root_start=[]
    root_end=[]
    root_not_found_duration = 0
    root_found_duration = 0

    crack_pixel_count = np.size(np.where(segmentation_c == True))
    root_pixel_count = np.size(np.where(segmentation_t == True))

    if(crack_pixel_count > 450):
        crack_detected = True
    if(root_pixel_count > 450):
        root_detected = True
    if(crack_detected or root_detected):
        detected = True

    if detected:
        if (crack_detected)
            crack_found_duration += 1 
            if crack_found_duration == 10:
                crack_start.append(frame_count)
            if crack_found_duration >= 10:
                crack_frames.append(frames)
                crack_not_found_duration=0
            
        if (root_detected)
            root_found_duration += 1 
            if root_found_duration ==10:
                root_start.append(frame_count)
            if root_found_duration >= 10:
                root_frames.append(frames)
                root_not_found_duration=0
            
    else :
        fault_not_found_duration +=1
        if(fault_not_found_duration == 10):
            end.append(frame_count)

        if(fault_not_found_duration >10):
        if crack_frames != null:
            crack_set.append[crack_frames]
        if root_frames != null:
            root_set.append(root_frames)
        
        crack_frames=[]
        root_frames=[]
        crack_found_duration = 0
        root_found_duration = 0  
    frame_count+=1 



def liveProcess():
    from tensorflow.python.saved_model import tag_constants
    data_dir = './data'
    runs_dir = './runs'
    image_shape = (160, 576)
    import scipy
    import numpy as np
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],'model')

      input_img = sess.graph.get_tensor_by_name('image_input:0')
      logits = sess.graph.get_tensor_by_name('logits:0')
      keep_prob =sess.graph.get_tensor_by_name('keep_prob:0')
      def pro_image(image):
        image = scipy.misc.imresize(image, image_shape)
        im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_img: [image]})
        im_softmax_c = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        im_softmax_t = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation_c = (im_softmax_c > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_t = (im_softmax_t > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask_c = np.dot(segmentation_c, np.array([[0, 255, 0, 127]]))
        mask_c = scipy.misc.toimage(mask_c, mode="RGBA")
        mask_t = np.dot(segmentation_t, np.array([[255, 0, 0, 127]]))
        mask_t = scipy.misc.toimage(mask_t, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask_c, box=None, mask=mask_c)
        street_im.paste(mask_t, box=None, mask=mask_t)
        return np.array(street_im)
    return pro_image
      
    
if __name__ == '__main__':
    #run()
    restore_and_predict()
#   process_video()
