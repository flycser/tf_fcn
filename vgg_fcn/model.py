#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : model
# @Date : 08/21/2018 12:37:46
# @Poject : tf_fcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc : Implementations of FCN-32s, FCN-16s, FCN-8s models

import os
import sys

proj_root = ''
sys.path.append(os.path.abspath(proj_root)) # add project root into path variable when run this code in termw

import logging
from logging.config import fileConfig

fileConfig('../logging.conf')
# logger = logging.getLogger('fei')

import math
import time

import numpy as np
import tensorflow as tf
import scipy.misc as misc

VGG_MEAN = [103.939, 116.779, 123.68] # TODO: need to be verified

class FCN(object):
    def __init__(self, channel, num_classes, learning_rate, weight_decay, data_dir):

        self.logger = logging.getLogger('fei')
        self.logger.debug('Initialize model parameters and load trained vgg net parameters.')

        self.channel = channel
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_dict = np.load(data_dir, encoding='latin1').item()
        self.weights = []
        self.loss = 0.
        self.current_step = 0

        self.build()

        # initilize tensorflow session
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        self.logger.debug('Fully Convolutional Network was constructed.')

    def build(self):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('input'):
            self.images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channel], name='images')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='labels') # the value of each label denotes the index of corresponding class
            self.dropout = tf.placeholder_with_default(0., shape=(), name='dropout')

        with tf.name_scope('arch'):
            # convert rgb to bgr
            red, green, blue = tf.split(value=self.images, num_or_size_splits=self.channel, axis=3)
            bgr = tf.concat(values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], axis=3)

            # layer 1
            self.conv1_1 = self.conv_layer(bgr, name='conv1_1')
            self.conv1_2 = self.conv_layer(self.conv1_2, name='conv1_2')
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            # layer 2
            self.conv2_1 = self.conv_layer(self.pool1, name='conv2_1')
            self.conv2_2 = self.conv_layer(self.conv2_1, name='conv2_2')
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            # layer 3
            self.conv3_1 = self.conv_layer(self.pool2, name='conv3_1')
            self.conv3_2 = self.conv_layer(self.conv3_1, name='conv3_2')
            self.conv3_3 = self.conv_layer(self.conv3_2, name='conv3_3')
            self.pool3 = self.max_pool(self.conv3_3, name='pool3')

            # layer 4
            self.conv4_1 = self.conv_layer(self.pool3, name='conv4_1')
            self.conv4_2 = self.conv_layer(self.conv4_1, name='conv4_2')
            self.conv4_3 = self.conv_layer(self.conv4_2, name='conv4_3')
            self.pool4 = self.max_pool(self.conv4_3, name='pool4')

            # layer 5
            self.conv5_1 = self.conv_layer(self.pool4, name='conv5_1')
            self.conv5_2 = self.conv_layer(self.conv5_1, name='conv5_2')
            self.conv5_3 = self.conv_layer(self.conv5_2, name='conv5_3')
            self.pool5 = self.max_pool(self.conv5_3, name='pool5')

            # build fully convolutional layers
            weight_6 = self.get_variable_norm(shape=[7, 7, 512, 4096], name='W6')
            bias_6 = self.get_bias_const(shape=[4096], name='b6')
            self.conv6 = tf.nn.bias_add(tf.nn.conv2d(self.pool5, weight_6, strides=[1, 1, 1, 1], padding='SAME'), bias_6)
            self.relu6 = tf.nn.relu(self.conv6)
            self.dropout_relu6 = tf.nn.dropout(self.relu6, keep_prob=1. - self.dropout)

            weight_7 = self.get_variable_norm(shape=[1, 1, 4096, 4096], name='W7')
            bias_7 = self.get_bias_const(shape=[4096], name='b7')
            self.conv7 = tf.nn.bias_add(tf.nn.conv2d(self.dropout_relu6, weight_7, strides=[1, 1, 1, 1], padding='SAME'), bias_7)
            self.relu7 = tf.nn.relu6(self.conv7)
            self.dropout_relu7 = tf.nn.dropout(self. relu7, keep_prob=1. - self.dropout)

            # output layer
            weight_8 = self.get_variable_norm(shape=[1, 1, 4096, self.num_classes], name='W8')
            bias_8 = self.get_bias_const(shape=[self.num_classes], name='b8')
            self.conv8 = tf.nn.bias_add(tf.nn.conv2d(self.dropout_relu7, weight_8, strides=[1, 1, 1, 1], padding='SAME'), bias_8)

            # upsample
            conv_tr1_shape = self.pool4.get_shape()
            weight_tr1 = self.get_variable_norm(shape=[4, 4, conv_tr1_shape[3], self.num_classes], name='W_t1') # ???
            bias_tr1 = self.get_bias_const([conv_tr1_shape[3]], name='b_t1')
            self.conv_tr1 = self.conv_transpose_layer(self.conv8, weight_tr1, bias_tr1, output_shape=tf.shape(self.pool4))
            self.fuse1 = tf.add(self.conv_tr1, self.pool4, name='fuse_1')

            conv_tr2_shape = self.pool3.get_shape()
            weight_tr2 = self.get_variable_norm(shape=[4, 4, conv_tr2_shape[3].value, conv_tr1_shape[3].value], name='W_t2')
            bias_tr2 = self.get_bias_const(shape=[conv_tr2_shape[3].value], name='b_t2')
            self.conv_tr2 = self.conv_transpose_layer(self.fuse1, weight_tr2, bias_tr2, output_shape=tf.shape(self.pool3))
            self.fuse2 = tf.add(self.conv_tr2, self.pool3, name='fuse_2')

            shape = tf.shape(self.images)
            weight_tr3 = self.get_variable_norm(shape=[16, 16, self.num_classes, conv_tr2_shape[3].value], name='W_t3')
            bias_tr3 = self.get_bias_const(shape=[self.num_classes], name='b_t3')
            self.logits = self.conv_transpose_layer(self.fuse2, weight_tr3, bias_tr3, output_shape=[shape[0], shape[1], shape[2], self.num_classes], stride=8)
            self.pred_cidx = tf.argmax(self.logits, dimension=3, name='Pred') # predicted index of class for each specific pixel

        with tf.name_scope('loss'):
            # regularization
            for weight in self.weights:
                self.loss += tf.nn.l2_loss(weight)
            self.loss *= self.weight_decay # weight decay

            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.labels, squeeze_dims=[3]), logits=self.logits, name='cross_entropy'))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step,)


    def conv_layer(self, value, name):
        """
        Construct convolutional layer, initialized by specific parameters
        :param value:
        :param name:
        :return:
        """

        with tf.name_scope(name):
            filter = self.get_conv_filter(name)

            conv = tf.nn.conv2d(value, filter, [1, 1, 1, 1], padding='SAME')

            biases = self.get_biases(name)
            conv_biases = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv_biases)

            return relu

    def get_conv_filter(self, name):
        filter = tf.Variable(self.data_dict[name][0], name='filter_' + name)

        self.weights.append(filter)

        return filter

    def get_biases(self, name):
        return tf.Variable(self.data_dict[name][1], name='biases_' + name)

    def max_pool(self, value, name):
        """
        Construct max pool layer
        :param value:
        :param name:
        :return:
        """

        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_variable_norm(self, shape, stddev=0.02, name=None):
        """
        Get a variable initialized by random normal distribution, mean=0, stddev=0.02, or load the variable from specific parameters dictionary
        :param shape:
        :param stddev:
        :param name:
        :return:
        """

        init = tf.truncated_normal(shape, stddev=stddev)

        if name is None:
            return tf.Variable(init)
        else:
            return tf.get_variable(name, initializer=init)

    def get_bias_const(self, shape, name=None):
        init = tf.constant(0., shape=shape)

        if name is None:
            return tf.Variable(init)
        else:

            return tf.get_variable(name, initializer=init)

    # a filter initialization function is implemented in another code, not used for the moment in current file
    def get_conv_tr_variable(self, shape, name):
        width, height = shape[0], shape[1]

        f = math.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([shape[0], shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(shape)
        for i in range(shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name=name, initializer=init, shape=weights.shape)

    def conv_transpose_layer(self, value, filter, bias, output_shape, stride=2):

        # if output_shape is None, double stretching input image shape
        if output_shape is None:
            output_shape = value.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = filter.get_shape().as_list()[2] # filter, 4-D tensor, [height, width, output_channels, in_channels]

        conv = tf.nn.conv2d_transpose(value, filter=filter, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

        return tf.nn.bias_add(conv, bias=bias)

    def train(self, train_images, train_labels, val_images=None, val_labels=None, test_images=None, test_labels=None, dropout=0.3, batch_size=5, max_epochs=0, shuffle=True):



        train_images = np.array(train_images, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.int)

        if val_images is not None and val_labels is not None:
            val_images = np.array(val_images, dtype=np.float32)
            val_labels = np.array(val_labels, dtype=np.int)

        if test_images is not None and test_labels is not None:
            test_images = np.array(test_images, dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.int)

        self.logger.debug('Start training process ...')
        for epoch in range(max_epochs):
            if shuffle:
                self.logger.debug('Shuffle input training dataset.')
                order = np.random.permutation(np.arange(len(train_images)))
                train_images = train_images[order]
                train_labels = train_labels[order]

            epoch_loss = [] # used for calculating mean loss of each epoch
            for idx in np.arange(0, len(train_images), batch_size):
                start = time.time()
                # get one batch of training data
                batch_images = train_images[idx:min(len(train_images), idx+batch_size)]
                batch_labels = train_labels[idx:min(len(train_labels), idx+batch_size)]

                self.current_step, _, loss = self.sess.run([self.global_step, self.train_op, self.loss], feed_dict={
                    self.images: batch_images,
                    self.labels: batch_labels,
                    self.dropout: dropout
                })
                epoch_loss.append(loss)

                self.logger.debug('Epoch={:d}, step={:d}, batch mean loss={:.5f}, running time={:.5f}.'.format(epoch, self.current_step, loss, time.time() - start))

            self.logger.debug('Epoch={:d}, mean loss.'.format(epoch, np.mean(epoch_loss)))

            # validation
            if val_images is not None and val_labels is not None:
                pred_cidx, loss = self.sess.run([self.pred_cidx, self.loss], feed_dict={
                    self.images: val_images,
                    self.labels: val_labels
                })

                # display IOU results, per class IOU and mean IOU
                class_IOU, class_weight = self.get_IOU(pred_cidx, val_labels, num_classes=self.num_classes)
                for cidx in range(self.num_classes):
                    self.logger.debug('Class={:d}, class mean IOU={.5f}.'.format(cidx, class_IOU[cidx]))
                self.logger.debug('Validation mean loss={:.5f}, mean IOU={:.5f}.'.format(loss, np.mean(class_IOU)))

            # testing
            if test_images is not None and test_labels is not None:
                pred_cidx, loss = self.sess.run([self.pred_cidx, self.loss],        feed_dict={
                    self.images: test_images,
                    self.labels: test_labels
                })

                # display IOU results
                class_IOU, class_weight = self.get_IOU(pred_cidx, gt_labels=test_labels, num_classes=self.num_classes)
                for cidx in range(self.num_classes):
                    self.logger.debug('Class={:d}, class mean IOU={.5f}.'.format(cidx, class_IOU[cidx]))
                self.logger.debug('Test mean loss={:.5f}, mean IOU={:.5f}.'.format(loss, np.mean(class_IOU)))

    def predict(self):
        pass

    def get_IOU(self, pred_labels, gt_labels, num_classes):
        # verify dimension of labels firstly
        assert (32, 32) == pred_labels.shape[1:] # the last dimension = 1, so after squeeze, the dimension channel is removed and the shape of labels should be (?, height, width)
        assert (32, 32) == gt_labels.shape[1:]

        class_IOU = np.zeros(num_classes)
        class_weight = np.zeros(num_classes) # number of pixel per class, not used for the moment

        for cidx in range(num_classes):
            intersection = np.float32(np.sum((pred_labels == gt_labels) * (gt_labels == cidx)))
            union = np.sum(pred_labels == cidx) + np.sum(gt_labels == cidx) - intersection
            if union > 0:
                class_IOU[cidx] = intersection / union
                class_weight[cidx] = union

        return class_IOU, class_weight

def data_reader(image_dir, label_dir):
    file_names = [fn for fn in os.listdir(image_dir) if fn.endswith('.PNG') or fn.endswith('.JPG') or fn.endswith('.TIF') or fn.endswith('.GIF') or fn.endswith('.png') or fn.endswith('.jpg') or fn.endswith('.tif') or fn.endswith('.gif')]

    imgs = []
    lbls = []
    for fn in file_names:
        img = misc.imread(os.path.join(image_dir, fn), mode='RGB')
        imgs.append(img)
        lbl_fn = fn[0:-4] + '.png'
        lbl = misc.imread(os.path.join(label_dir, lbl_fn), mode='L')
        lbls.append(lbl)

    return imgs, lbls

if __name__ == '__main__':


    train_image_dir = '/network/rit/lab/ceashpc/fjie/tmp/ICML2017DVN/FCN/glass_vessels/Materials_In_Vessels/Train_Images'
    test_image_dir = '/network/rit/lab/ceashpc/fjie/tmp/ICML2017DVN/FCN/glass_vessels/Materials_In_Vessels/Test_Images_All'
    label_dir = '/network/rit/lab/ceashpc/fjie/tmp/ICML2017DVN/FCN/glass_vessels/Materials_In_Vessels/LiquidSolidLabels'
    train_imgs, train_lbls = data_reader(train_image_dir, label_dir)
    test_imgs, test_lbls = data_reader(test_image_dir, label_dir)


    channel = 3
    num_classes = 2
    learning_rate = 1e-5
    weight_decay = 5e-4
    batch_size = 2
    max_epochs = 100
    dropout = 0.5
    data_dir = '/network/rit/lab/ceashpc/fjie/tmp/ICML2017DVN/FCN/vgg.npy'

    vggfcn = FCN(channel=channel, num_classes=num_classes, learning_rate=learning_rate, weight_decay=weight_decay, data_dir=data_dir)

    # TODO: how to handle data augmentation
    vggfcn.train(train_imgs, train_lbls, test_imgs, test_lbls, dropout=dropout, batch_size=batch_size, max_epochs=max_epochs)
