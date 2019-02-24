"""
The main CNN
Based heavily on model from a_project_basic.pdf (cite better)
"""

import tensorflow as tf

import numpy as np

def inference():
    # Placeholder should be hidden from the user of this class
    # Emulate scikit-learn api

    # self.model = inference() in __init__
    """
    Build the CNN model.
    Args:
        images0: {tf.placeholder} -- Images placeholder, from inputs()
    Returns:
        sigmoid_linear: Output tensor with the computed probabilities.
    """

    # conv1
    with tf.variable_scope("conv1") as scope:
        weights = tf.Variable(tf.truncated_normal([16,2,1,10],stddev=0.1)) # explain hyperparams, magic numbers
        conv = tf.nn.conv2d(images, weights, [1,1,1,1],padding="VALID")
        biases = tf.Variable(tf.constant(0.1,shape=[10])) # hyperp
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.Variable(tf.truncated_normal([11,3,10,20],stddev=0.1))
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding="VALID")
        biases = tf.Variable(tf.constant(0.1,shape=[20]))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

    # fully-connected1
    with tf.variable_scope("fully-connected1") as scope:
        reshape = tf.reshape(pool2, [-1,58*1*20])
        weights = tf.Variable(tf.truncated_normal([58*1*20,256],stddev=0.1))
        biases = tf.Variable(tf.constant(0.1,shape=[256]))
        local = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # fully-connected2
    with tf.variable_scope("fully-connected2") as scope:
        #dropout
        local3_drop =tf.nn.dropout(local, 0.5)
        weights = tf.Variable(tf.truncated_normal([256, num_freqs],stddev=0.1))
        biases = tf.Variable(tf.constant(0.1,shape=[num_freqs]))
        sigmoid_linear = tf.nn.sigmoid(tf.matmul(local3_drop, weights) + biases, name=scope.name)

    return sigmoid_linear

class MusesheetsCNN:
    # TODO:
    # Initializer, model field,
    # maybe hold generator
    # wrapper functions for feeding

    def __init__(self, num_freqs):
        device_name = tf.test.gpu_device_name()
        if not device_name == '/device:GPU:0':
            #raise SystemError('GPU device not found')
            print("GPU device not found, using {}".format(device_name))
        print('Found GPU at: {}'.format(device_name))


    # This network class is the model

    def loss(logits, labels):
        """
        Calculates the loss from the logits and the labels
        Args:
            logits: Logits from inference(), float - [batch_size, num_classes]
            labels: Labels tensor, int32 - [batch_size, num_classes]
        Returns:
            cross_entropy: Loss tensor of type float
        """
        cross_entropy = -tf.reduce_sum(labels*tf.log(logits+1e-10)+(1-labels)*tf.log(1-logits+1e-10))
        # factor in musicality
        return cross_entropy

    def evaluation(logits, labels, threshold):
        """
        Evaluate the quality of the logits at predicting the label.
        Args:
            logits: Logits from inference(), float - [batch_size, num_classes]
            labels: Labels tensor, int32 - [batch_size, num_classes]
            threshold: Threshold applied to the logits.
        Returns:
            accuracy: Compute precision of predicting.
        """
        pred = tf.cast(tf.greater(logits,threshold),"float")
        correct_prediction = tf.cast(tf.equal(pred, labels), "float")
        accuracy = tf.reduce_mean(correct_prediction)
        return accuracy

    def training(loss, learning_rate):
        """
        Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        Args:
            loss: Loss tensor, from loss().
            learning_rate: The learning rate to use for gradient descent.
        Returns:
            train_op: The Op for training.
        """
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Use the optimizer to apply the gradients that minimize the loss
        train_op = optimizer.minimize(loss)
        return train_op
