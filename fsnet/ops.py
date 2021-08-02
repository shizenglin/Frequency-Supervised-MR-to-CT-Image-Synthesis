import tensorflow.compat.v1 as tf
import numpy as np

#######################
# 3d functions
#######################

def batch_normalization(input, name="instance_norm"):
    with tf.variable_scope(name):
        
        input_shape = input.get_shape()
        # operates on all dims except the last dim
        params_shape = input_shape[-1:]
        axes = [1,2,3]
        
        # create trainable variables and moving average variables
        beta = tf.get_variable(
            'beta',
            shape=params_shape,
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32, trainable=True)

        gamma = tf.get_variable(
            'gamma',
            shape=params_shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32, trainable=True)
        
        mean, variance = tf.nn.moments(input, axes=axes, keep_dims = True)
        eps = 1e-5
        inv = tf.rsqrt(variance + eps)
        normalized = (input-mean)*inv
        
        return  gamma*normalized + beta

# convolution
def conv3d(input, output_chn, kernel_size=3, stride=1, dilation=(1,1,1), use_bias=True, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            dilation_rate=dilation,padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=None, use_bias=use_bias, name=name)


def conv_bn_relu(input, output_chn, kernel_size=3, stride=1, dilation=(1,1,1), use_bias=True, name='conv_bn_relu'):
    with tf.variable_scope(name):
        conv = conv3d(input, output_chn, kernel_size, stride, dilation, use_bias, name='conv')
        bn = batch_normalization(conv, name="batch_norm")
        relu = tf.nn.leaky_relu(bn, name='relu')

    return relu

# deconvolution
def Deconv3d(input, output_chn, kernel_size, stride, name):
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    filter = tf.get_variable(name+"/filter", shape=[kernel_size, kernel_size, kernel_size, output_chn, in_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=None)

    conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * stride, in_height * stride, in_width * stride, output_chn],
                                  strides=[1, stride, stride, stride, 1], padding="SAME", name=name)
    return conv


def deconv_bn_relu(input, output_chn, kernel_size=4, stride=2, name='deconv'):
    with tf.variable_scope(name):
        conv = Deconv3d(input, output_chn, kernel_size, stride, name='deconv')
        bn = batch_normalization(conv, name="batch_norm")
        relu = tf.nn.leaky_relu(bn, name='relu')
    return relu

def decom_map(input_score, input_map):
    input_score_1 = input_score[:,:,:,:,0]
    input_score_2 = input_score[:,:,:,:,1]
    dyn_input_shape = tf.shape(input_map)
    input_score_1=input_score_1[:,:,:,:,tf.newaxis]
    input_score_2=input_score_2[:,:,:,:,tf.newaxis]
       
    input_score_1 = tf.tile(input_score_1,[1,1,1,1,dyn_input_shape[4]])
    input_score_2 = tf.tile(input_score_2,[1,1,1,1,dyn_input_shape[4]])
   
    return tf.multiply(input_score_1, input_map), tf.multiply(input_score_2, input_map)

def global_conv3d(input, output_chn, kernel_size=3, stride=1, dilation=(1,1,1), use_bias=True, name='gconv'):
    with tf.variable_scope(name):
        conv_x1 = conv_bn_relu(input, output_chn, (kernel_size,1,1), name='conv_x1')
        conv_y1 = conv_bn_relu(conv_x1, output_chn, (1,kernel_size,1), name='conv_y1')
        conv_z1 = conv_bn_relu(conv_y1, output_chn, (1,1,kernel_size), name='conv_z1')
        
        conv_x2 = conv_bn_relu(input, output_chn, (1,kernel_size,1), name='conv_x2')
        conv_y2 = conv_bn_relu(conv_x2, output_chn, (1,1,kernel_size), name='conv_y2')
        conv_z2 = conv_bn_relu(conv_y2, output_chn, (kernel_size,1,1), name='conv_z2')

        conv_x3 = conv_bn_relu(input, output_chn, (1,1,kernel_size), name='conv_x3')
        conv_y3 = conv_bn_relu(conv_x3, output_chn, (kernel_size,1,1), name='conv_y3')
        conv_z3 = conv_bn_relu(conv_y3, output_chn, (1,kernel_size,1), name='conv_z3')
        
        return conv_z1+conv_z2+conv_z3

def Inception_layer(inputI, name='incep'):
    with tf.variable_scope(name):
        conv1_1 = conv_bn_relu(input=inputI, output_chn=64, kernel_size=1, name='conv1_1')

        conv2_1 = conv_bn_relu(input=inputI, output_chn=96, kernel_size=1, name='conv2_1')
        conv2_2 = conv_bn_relu(input=conv2_1, output_chn=128, kernel_size=3, name='conv2_2')

        conv3_1 = conv_bn_relu(input=inputI, output_chn=16, kernel_size=1, name='conv3_1')
        conv3_2 = conv_bn_relu(input=conv3_1, output_chn=32, kernel_size=5, name='conv3_2')

        pool1 = tf.layers.max_pooling3d(inputs=inputI, pool_size=3, strides=1, padding='same', name='pool1')
        conv4_1 = conv_bn_relu(input=pool1, output_chn=32, kernel_size=1, name='conv4_1')                 

        return tf.concat([conv1_1, conv2_2, conv3_2, conv4_1], axis=4)
