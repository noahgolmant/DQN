import tensorflow as tf
import numpy as np

NUM_CHANNELS = 4 # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = None # random initialization seed
NUM_ACTIONS = 4  # number of actions for this game
BATCH_SIZE = 100
def weight_variable(shape, sdev=0.1):
    initial = tf.truncated_normal(shape, stddev=sdev, seed=SEED)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
   initial = tf.constant(constant, shape=shape)
   return tf.Variable(initial)


train_data_node = tf.placeholder(
    tf.float32,
    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

# construct convolution layers
conv1_weights = weight_variable([8, 8, NUM_CHANNELS, 32])
conv1_biases  =  bias_variable([32])

conv2_weights = weight_variable([4, 4, 32, 64])
conv2_biases = bias_variable([64])

conv3_weights = weight_variable([3, 3, 64, 64])
conv3_biases = bias_variable([64])

# construct fully connected and output layers
fc_weights = weight_variable([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512])
fc_biases = bias_variable([512])

output_weights = weight_variable([512, NUM_ACTIONS])
output_biases = bias_variable([NUM_ACTIONS])

def model(data):
    # convolution with 4x4 stride
    conv = tf.nn.conv2d(data, conv1_weights, strides = [1, 4, 4, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # convolution with 2x2 stride
    conv = tf.nn.conv2d(relu, conv2_weights, strides = [1, 2, 2, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

    # 1x1 stride
    conv = tf.nn.conv2d(relu, conv3_weights, strides = [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

    # reshape to fully connected layer
    relu_shape = relu.get_shape().as_list()
    reshape = tf.reshape(
        relu,
        [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, fc_weights) + fc_biases)

    # dropout ?
    return tf.matmul(hidden, output_weights) + output_biases

