# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import os
import pickle
from collections import OrderedDict

ConfDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "conf")
DataDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
UNKNOWN = "<OOV>"


class MyCNN(object):
    def __init__(self, configs):
        # parse config
        self.word_max_len = configs["word_max_len"]
        self.char_max_len = configs["char_max_len"]
        self.num_classes = configs["num_classes"]
        self.word_embedding_size = configs["word_embedding_size"]
        self.char_embedding_size = configs["char_embedding_size"]
        self.word_filter_size_list = [int(l) for l in configs["word_filter_sizes"].split(" ")]
        self.word_num_filters = configs["word_num_filters"]
        self.char_filter_size_list = [int(l) for l in configs["char_filter_sizes"].split(" ")]
        self.char_num_filters = configs["char_num_filters"]
        self.l2_reg_lambda = configs["l2_reg_lambda"]
        self.use_char = configs["use_char"]
        serialization_data_dir = os.path.join(DataDirPath, configs["serialization_dir"])
        if configs["word_embedding"] != "None":
            self.word_init_embeddings = pickle.load(open(os.path.join(serialization_data_dir, configs["word_embedding"]), 'rb'))
        else:
            self.word_init_embeddings = None
        self.word_vocab = pickle.load(open(os.path.join(serialization_data_dir, configs["word_to_id_dict"]), 'rb'))
        if configs["char_embedding"] != "None":
            self.char_init_embeddings = pickle.load(open(os.path.join(serialization_data_dir, configs["char_embedding"]), 'rb'))
        else:
            self.char_init_embeddings = None
        self.char_vocab = pickle.load(open(os.path.join(serialization_data_dir, configs["char_to_id_dict"]), 'rb'))

        # create graph
        # input
        self.input = tf.placeholder(tf.int32, [None, self.word_max_len+self.char_max_len], name="model_input")
        self.word_input = tf.slice(self.input, [0, 0], [-1, self.word_max_len])
        self.char_input = tf.slice(self.input, [0, self.word_max_len], [-1, self.char_max_len])
        self.target_input = tf.placeholder(tf.float32, [None, self.num_classes], name="target_input")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # print(self.input)
        # print(self.dropout_keep_prob)
        # embedding layer
        if self.word_init_embeddings is None:
            self.word_embeddings = tf.get_variable(name="word_embedding",
                                                        shape=[len(self.word_vocab), self.word_embedding_size],
                                                        dtype=tf.float32, trainable=True)
        else:
            self.word_embeddings = tf.Variable(self.word_init_embeddings, name="word_embedding",
                                               dtype=tf.float32, trainable=True)

        if self.char_init_embeddings is None:
            self.char_embeddings = tf.get_variable(name="char_embedding",
                                                          shape=[len(self.char_vocab), self.char_embedding_size],
                                                          dtype=tf.float32, trainable=True)
        else:
            self.char_embeddings = tf.Variable(self.char_init_embeddings, name="char_embedding",
                                                 dtype=tf.float32, trainable=True)

        word_inputs = tf.nn.embedding_lookup(self.word_embeddings, self.word_input)
        char_inputs = tf.nn.embedding_lookup(self.char_embeddings, self.char_input)
        word_inputs = tf.expand_dims(word_inputs, -1)
        char_inputs = tf.expand_dims(char_inputs, -1)

        # conv and maxpool layer
        pool_features = []
        # word conv
        for i, filter_size in enumerate(self.word_filter_size_list):
            with tf.name_scope("word_conv_maxpool-%s" % filter_size):
                # conv layer
                filter_shape = [filter_size, self.word_embedding_size, 1, self.word_num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.word_num_filters]), name="b")
                conv = tf.nn.conv2d(
                    word_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.word_max_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                # print("word pool out put shape:", pooled.shape)
                pool_features.append(pooled)
        # char conv
        if self.use_char:
            for i, filter_size in enumerate(self.char_filter_size_list):
                with tf.name_scope("char_conv_maxpool-%s" % filter_size):
                    # conv layer
                    filter_shape = [filter_size, self.char_embedding_size, 1, self.char_num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.char_num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        char_inputs,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.char_max_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pool_features.append(pooled)

        # Dense and dropout
        num_filters_total = self.word_num_filters * len(self.word_filter_size_list)
        if self.use_char:
            num_filters_total += self.char_num_filters * len(self.char_filter_size_list)
        self.h_pool = tf.concat(pool_features, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="dropout_feature")
            # print(self.h_drop)
        # loss
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes], name="b"))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # print(self.predictions)
        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.target_input)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.target_input, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        # saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)


# for test
if __name__ == "__main__":
    configs = json.load(open(os.path.join(ConfDirPath, "train_configs.json")))
    cnn_configs = configs["CNN_configs"]
    train_configs = configs["train_configs"]
    # train
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-train_configs["init_scale"], train_configs["init_scale"])
        with tf.variable_scope("text_classification", reuse=None, initializer=initializer):
            # CNN model
            TextCNN = MyCNN(cnn_configs)