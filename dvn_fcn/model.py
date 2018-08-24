#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : model
# @Date : 08/21/2018 12:07:49
# @Poject : tf_fcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc : Implementation of FCN baseline methods in DVN paper (Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs)

import os
import sys

proj_root = ''
sys.path.append(os.path.abspath(proj_root)) # add project root into path variable when run this code in terminal

import logging
from logging.config import fileConfig

fileConfig('../logging.conf')

import time
import pickle

import numpy as np
import tensorflow as tf

class FCN(object):
    def __init__(self, channel, num_classes, learning_rate, weight_decay, data_dir):

        self.logger = logging.getLogger('fei')
        self.logger.debug('Initialize model parameters.')

        self.channel = channel
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weights = []
        self.loss = 0.
        self.current_step = 0

        self.build()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def build(self):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('input'):
            self.images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channel], name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None, None, 1], name='labels') # labels, use index of class
            self.dropout = tf.placeholder_with_default(0., shape=(), name='dropout')

        with tf.name_scope('arch'):
            # input, (?, 24, 24, 3)
            # output, (?, 24, 24, 64)
            self.conv1 = self.conv_layer(self.images, shape=[5, 5, 3, 64], strides=[1, 1, 1, 1], name='conv1')

            self.logger.debug('conv1_shape, {}'.format(self.conv1.shape))

            # input, (?, 24, 24, 64)
            # output, (?, 12, 12, 128)
            self.conv2 = self.conv_layer(self.conv1, shape=[5, 5, 64, 128], strides=[1, 2, 2, 1], name='conv2')

            self.logger.debug('conv2_shape, {}'.format(self.conv2.shape))

            # input, (?, 12, 12, 128)
            # output, (?, 6, 6, 128)
            self.conv3 = self.conv_layer(self.conv2, shape=[5, 5, 128, 128], strides=[1, 2, 2, 1], name='conv3')

            self.logger.debug('conv3_shape, {}'.format(self.conv3.shape))

            # TODO: dropout op

            # input, (?, 6, 6, 128)
            # output, (?, 24, 24, ?), (batch_size, height, width, num_classes)
            # difference between tf.shape(x) and x.get_shape()
            # tf.shape() can get dynamic shape, while x.get_shape() will return ? for those which are determined when running. So to specify shape with undermined dimensions, we can use tf.shape(x) instead of x.get_shape()
            conv3_shape = self.conv3.get_shape()
            filter = self.get_filter_norm(shape=[5, 5, self.num_classes, conv3_shape[3].value], name='w_tr')
            bias = self.get_bias_const(shape=[self.num_classes], name='b_tr')
            image_shape = tf.shape(self.images)
            output_shape = [image_shape[0], image_shape[1], image_shape[2], self.num_classes]
            self.logits = self.conv_transpose_layer(self.conv3, filter=filter, bias=bias, output_shape=output_shape, strides=[1, 4, 4, 1], name='conv_tr')
            self.pred_cidx = tf.argmax(self.logits, dimension=3, name='pred_class_idx')

        with tf.name_scope('loss'):
            for weight in self.weights:
                self.loss += tf.nn.l2_loss(weight)
            self.loss *= self.weight_decay

            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.labels, squeeze_dims=[3]), logits=self.logits, name='cross_entropy'))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)


    def conv_layer(self, value, shape, strides, name):
        with tf.name_scope(name):
            filter = self.get_filter_norm(shape=shape, name=name)
            conv = tf.nn.conv2d(value, filter, strides, padding='SAME')

            bias = self.get_bias_const(shape=[shape[3]])

            return tf.nn.relu(tf.nn.bias_add(conv, bias))

    def get_filter_norm(self, shape, stddev=0.02, name=None):
        init = tf.truncated_normal(shape, stddev=stddev)
        filter = tf.get_variable(name, initializer=init)

        self.weights.append(filter)

        return filter


    def get_bias_const(self, shape, name=None):
        init = tf.constant(0., shape=shape)

        if name is None:
            return tf.Variable(init)
        else:
            return tf.get_variable(name, initializer=init)

    def conv_transpose_layer(self, value, filter, bias, output_shape, strides, name):

        conv_tr = tf.nn.conv2d_transpose(value, filter=filter, output_shape=output_shape, strides=strides, padding='SAME')

        return tf.nn.bias_add(conv_tr, bias)

    def train(self, train_images, train_labels, val_images=None, val_labels=None, test_images=None, test_labels=None, dropout=0.3, batch_size=5, max_epochs=0, shuffle=True, val_epoch=1, test_epoch=5):

        # normalization or subtract mean (refer to FCN)
        mean_channels = np.mean(train_images, axis=(0, 1, 2))

        train_images -= mean_channels
        if val_images is not None and val_labels is not None:
            val_images -= mean_channels
        if test_images is not None and test_labels is not None:
            test_images -= mean_channels

        self.logger.debug('Starting training process ...')
        for epoch in range(max_epochs):
            if shuffle:
                self.logger.debug('Shuffle input training dataset.')
                order = np.random.permutation(np.arange(len(train_images)))
                train_images = train_images[order]
                train_labels = train_labels[order]

            epoch_loss = []
            for idx in np.arange(0, len(train_images), batch_size):
                start = time.time()

                batch_images = train_images[idx:min(len(train_images), idx+batch_size)]
                batch_labels = train_labels[idx:min(len(train_labels), idx+batch_size)]

                self.current_step, _, loss = self.sess.run([self.global_step, self.train_op, self.loss], feed_dict={
                    self.images: batch_images,
                    self.labels: batch_labels,
                    self.dropout: dropout
                })
                epoch_loss.append(loss)

                self.logger.debug('Epoch={:d}, step={:d}, batch mean loss={:.5f}, running time={:.5f}.'.format(epoch, self.current_step, loss, time.time() - start))

            if val_images is not None and val_labels is not None and 0 == epoch % val_epoch:
                pred_cidx, loss = self.sess.run([self.pred_cidx, self.loss], feed_dict={
                    self.images: val_images,
                    self.labels: val_labels
                })

                # display IOU results
                self.logger.debug('Validation results.')
                class_IOUs = self.evaluate(pred_cidx, val_labels, num_classes=self.num_classes)
                for cidx in range(self.num_classes):
                    self.logger.debug('Class={:d}, class mean IOU={:.5f}.'.format(cidx, np.mean(class_IOUs[:, cidx])))
                self.logger.debug('Validation mean loss={:.5f}, mean IOU={:.5f}.'.format(loss, np.mean(class_IOUs)))

            if test_images is not None and test_labels is not None and 0 == epoch % test_epoch:
                pred_cidx, loss = self.sess.run([self.pred_cidx, self.loss], feed_dict={
                    self.images: test_images,
                    self.labels: test_labels
                })

                # display IOU results
                self.logger.debug('Testing results.')
                class_IOUs = self.evaluate(pred_cidx, test_labels, num_classes=self.num_classes)
                for cidx in range(self.num_classes):
                    self.logger.debug('Class={:d}, class mean IOU={:.5f}.'.format(cidx, np.mean(class_IOUs[:, cidx])))
                self.logger.debug('Testing mean loss={:.5f}, mean IOU={:.5f}.'.format(loss, np.mean(class_IOUs)))


    def predict(self):
        pass

    def evaluate(self, all_pred_labels, all_gt_labels, num_classes):
        all_pred_labels = np.squeeze(all_pred_labels) # remove the single-dimensinoal entries
        all_gt_labels = np.squeeze(all_gt_labels)

        if 2 == len(all_pred_labels.shape):
            all_pred_labels = all_pred_labels[None]
        if 2 == len(all_gt_labels.shape):
            all_gt_labels = all_gt_labels[None]

        assert (32, 32) == all_pred_labels.shape[1:]
        assert (32, 32) == all_gt_labels.shape[1:]

        class_IOUs = []
        for i in range(len(all_pred_labels)):
            pred_lbls, gt_lbls = all_pred_labels[i], all_gt_labels[i]
            class_IOU, _ = self.get_IOU(pred_lbls, gt_lbls, num_classes=num_classes)
            class_IOUs.append(class_IOU)

        class_IOUs = np.array(class_IOUs, dtype=np.float32)

        return class_IOUs


    def get_IOU(self, pred_labels, gt_labels, num_classes):
        assert (32, 32) == pred_labels.shape
        assert (32, 32) == gt_labels.shape

        print('Predicted mask')
        print(pred_labels)
        print('Gt mask')
        print(gt_labels)

        class_IOU = np.zeros(num_classes)
        class_weight = np.zeros(num_classes)

        for cidx in range(num_classes):
            intersection = np.float32(np.sum((pred_labels == gt_labels) * (gt_labels == cidx)))
            union = np.sum(pred_labels == cidx) + np.sum(gt_labels == cidx) - intersection

            print(cidx, intersection, union)
            if union > 0.:
                class_IOU[cidx] = intersection / union
                class_weight[cidx] = union


        return class_IOU, class_weight


def load_dataset(path, fn):
    with open(os.path.join(path, fn), 'rb') as rfile:
        data = pickle.load(rfile, encoding='latin1')

    images = np.array([img for img in data['imgs']], dtype=np.float32)
    labels = np.array([np.reshape(seg, (32, 32, 1)) for seg in data['segs']], dtype=np.int32)

    return images, labels

if __name__ == '__main__':

    # load dataset
    path = '/network/rit/lab/ceashpc/fjie/tmp/ICML2017DVN/ICML2014/mrseg_data_release/horse'
    train_fn = 'train_raw.pdata'
    val_fn = 'val_raw.pdata'
    test_fn = 'test_raw.pdata'
    train_images, train_labels = load_dataset(path, train_fn)
    val_images, val_labels = load_dataset(path, val_fn)
    test_images, test_labels = load_dataset(path, test_fn)

    np.set_printoptions(threshold=np.nan, linewidth=5000)
    # print(train_images.shape)
    # print(train_labels[0, :, :, 0])

    channel = 3
    num_classes = 2
    learning_rate = 1e-5
    weight_decay = 5e-4
    batch_size = 2
    max_epochs = 300
    dropout = 0.5
    data_dir = ''

    dvnfcn = FCN(channel=channel, num_classes=num_classes, learning_rate=learning_rate, weight_decay=weight_decay, data_dir=data_dir)

    dvnfcn.train(train_images, train_labels, val_images, val_labels, test_images, test_labels, dropout=dropout, batch_size=batch_size, max_epochs=max_epochs)