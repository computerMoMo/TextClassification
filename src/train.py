# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import json
import pickle
import datetime
from cnn import MyCNN

ConfDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "conf")
DataDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
UNKNOWN = "<OOV>"


def iterator(x_data, y_data, batch_size):
    data_len = len(x_data)
    batch_len = data_len // batch_size
    xArray = []
    yArray = []
    for i in range(batch_len):
        xArray.append(x_data[i*batch_size:(i+1)*batch_size])
        yArray.append(y_data[i*batch_size:(i+1)*batch_size])
    return xArray, yArray


def create_optimizer(configs):
    lrate = configs["learning_rate"]
    if configs['optimizer'] == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
    elif configs['optimizer'] == 'MSGD':
        optimizer = tf.train.MomentumOptimizer(learning_rate = lrate,
                                        momentum = configs['momentum'], use_nesterov = True)
    elif configs['optimizer'] == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate = lrate,
                                        rho = configs['rho'], epsilon = configs['epsilon'])
    elif configs['optimizer'] == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate = lrate,
                                    beta1 = configs['beta_1'], beta2 = configs['beta_2'],
                                    epsilon = configs['epsilon'])
    elif configs['optimizer'] == 'Rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lrate,
                                    momentum = configs['momentum'],
                                    epsilon = configs['epsilon'])
    else:
        raise ValueError('unrecongised optimizer {}'.format(configs['optimizer']))
    return optimizer


if __name__ == "__main__":
    # load config json
    configs = json.load(open(os.path.join(ConfDirPath, "train_configs.json")))
    cnn_configs = configs["CNN_configs"]
    train_configs = configs["train_configs"]
    # load data
    data_serialization_dir = os.path.join(DataDirPath, train_configs["serialization_dir"])
    x_train_data = np.asarray(pickle.load(open(os.path.join(data_serialization_dir, train_configs["x_train_data"]), 'rb')), dtype=np.int32)
    y_train_data = np.asarray(pickle.load(open(os.path.join(data_serialization_dir, train_configs["y_train_data"]), 'rb')), dtype=np.float32)
    x_valid_data = np.asarray(pickle.load(open(os.path.join(data_serialization_dir, train_configs["x_valid_data"]), 'rb')), dtype=np.int32)
    y_valid_data = np.asarray(pickle.load(open(os.path.join(data_serialization_dir, train_configs["y_valid_data"]), 'rb')), dtype=np.float32)
    xArray, yArray = iterator(x_train_data, y_train_data, train_configs["batch_size"])
    # train
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-train_configs["init_scale"], train_configs["init_scale"])
        with tf.variable_scope("text_classification", reuse=None, initializer=initializer):
            #CNN model
            TextCNN = MyCNN(cnn_configs)
            # optimizer op
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = create_optimizer(configs=train_configs["optimizer_conf"])
            grads_and_vars = optimizer.compute_gradients(TextCNN.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # CheckPoint State
            check_point_dir = os.path.join(DataDirPath, train_configs["check_point_path"])
            if not os.path.exists(check_point_dir):
                os.mkdir(check_point_dir)
            else:
                for _file in os.listdir(check_point_dir):
                    os.remove(os.path.join(check_point_dir, _file))
            # ckpt = tf.train.get_checkpoint_state(check_point_dir)
            # if ckpt:
            #     print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            #     TextCNN.saver.restore(session, tf.train.latest_checkpoint(check_point_dir))
            # else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

            best_dev_acc = 0.0

            # run train op
            for i in range(train_configs["max_epoch"]):
                # forward
                for x_batch, y_batch in zip(xArray, yArray):
                    train_feed_dicts = {
                        TextCNN.input: x_batch,
                        TextCNN.target_input: y_batch,
                        TextCNN.dropout_keep_prob: train_configs["dropout_keep_prob"]
                    }
                    _, step, loss, accuracy = session.run([train_op, global_step, TextCNN.loss, TextCNN.accuracy], train_feed_dicts)
                    print("epoch-{}: step {}, loss {:g}, acc {:g}".format(i, step, loss, accuracy))
                # eval
                dev_feed_dicts = {
                    TextCNN.input: x_valid_data,
                    TextCNN.target_input: y_valid_data,
                    TextCNN.dropout_keep_prob: 1.0
                }
                dev_loss, dev_accuracy = session.run([TextCNN.loss, TextCNN.accuracy], dev_feed_dicts)
                if dev_accuracy > best_dev_acc:
                    best_dev_acc = dev_accuracy
                print("Evaluation: loss {:g}, acc {:g}".format(dev_loss, dev_accuracy))
                print("Save model.")
                model_name = "model_dev_acc_{:.5f}.ckpt".format(dev_accuracy)
                TextCNN.saver.save(session, os.path.join(check_point_dir, model_name), global_step=tf.train.global_step(session, global_step))
            print("best dev acc:", best_dev_acc)
