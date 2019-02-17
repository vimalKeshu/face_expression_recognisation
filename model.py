# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def extract_features(features, bottleneck_layer_size, image_size, mode):
    # Input layer
    input = tf.reshape(features, [-1, image_size, image_size, 1])

    net = tf.layers.conv2d(inputs=input, 
                    filters=64, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.conv2d(inputs=net, 
                    filters=64, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.dropout(inputs=net, 
                    rate=0.25, 
                    training= (mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.conv2d(inputs=net, 
                    filters=128, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.conv2d(inputs=net, 
                    filters=128, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.dropout(inputs=net, 
                    rate=0.25, 
                    training= (mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.conv2d(inputs=net, 
                    filters=256, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.conv2d(inputs=net, 
                    filters=256, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.conv2d(inputs=net, 
                    filters=256, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.dropout(inputs=net, 
                    rate=0.25, 
                    training= (mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.conv2d(inputs=net, 
                    filters=256, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.conv2d(inputs=net, 
                    filters=256, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.conv2d(inputs=net, 
                    filters=256, 
                    kernel_size=[3, 3], 
                    padding="same", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.nn.elu(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    dropout = tf.layers.dropout(inputs=net, 
                    rate=0.25, 
                    training= (mode == tf.estimator.ModeKeys.TRAIN))

    # Dense Layer
    shape = dropout.get_shape()[1]
    flat  = tf.reshape(dropout, [-1, shape * shape * 256])
    dense = tf.layers.dense(inputs=flat, 
                    units=1024,  
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.nn.elu)
    dropout = tf.layers.dropout(inputs=dense, 
                    rate=0.25, 
                    training= (mode == tf.estimator.ModeKeys.TRAIN))

    dense = tf.layers.dense(inputs=dropout, 
                    units=1024,  
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.nn.elu)
    dropout = tf.layers.dropout(inputs=dense, 
                    rate=0.25, 
                    training= (mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, 
                    units=bottleneck_layer_size, 
                    name="logits", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    activation=None) 
    return logits