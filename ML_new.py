import numpy as np
import pandas as pd
import tensorflow as tf

import networkx as nx
from vis_py import *
from itertools import chain
import matplotlib.pyplot as plt
import copy
import glob
import random
from sklearn.preprocessing import MinMaxScaler

path = './features/*.npy'
files = glob.glob(path)
list = []
for file in files:
    list.append(np.load(file))
    print(list[0].shape)


df_all = np.concatenate(list, axis = 0)
#df_all = df_all[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17]]
print(df_all.shape)

df_all[1:, [-6, -5, -4, -3, -2, -1]] = (df_all[1:, [-6, -5, -4, -3, -2, -1]] / df_all[:-1, [-6, -5, -4, -3, -2, -1]]) - 1

df_all = df_all[1:, :]

df_categ = np.zeros((df_all.shape[0], df_all.shape[1] + 2))
df_categ[:, :-2] = df_all
df_categ[: -3] = 0
for k in range(df_all.shape[0]):
    if df_all[k, -1] > 0.0144:
        df_categ[k, -1] = 1
    elif df_all[k, -1] < -0.0129:
        df_categ[k, -3] = 1
    else:
        df_categ[k, -2] = 1

print(np.sum(df_categ, axis = 0))
X_data = df_categ[:, :-3]
y_data = df_categ[:, -3:]

scaler = MinMaxScaler()
scaler.fit(X_data)

X_data = scaler.transform(X_data)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 17])
Y = tf.placeholder(tf.float32, [None,3])

#####################################################
# Define model, loss, update and evaluation metric. #

# initialise weight matrix and bias for fully connected linear layer
W_0 = tf.get_variable("w0", dtype=tf.float32, shape=[17, 10], initializer=tf.contrib.layers.xavier_initializer())
b_0 = tf.get_variable("b0", dtype=tf.float32, shape=[1, 10], initializer=tf.contrib.layers.xavier_initializer())

y_0 = tf.matmul(X, W_0) + b_0

#relu_0 = tf.nn.relu(y_0)

# initialise weight matrix and bias for fully connected linear layer
'''W_1 = tf.get_variable("w1", dtype=tf.float32, shape=[10, 5], initializer=tf.contrib.layers.xavier_initializer())
b_1 = tf.get_variable("b1", dtype=tf.float32, shape=[1, 5], initializer=tf.contrib.layers.xavier_initializer())

y_1 = tf.matmul(relu_0, W_1) + b_1

relu_1 = tf.nn.relu(y_1)'''

# initialise weight matrix and bias for fully connected linear layer
W_2 = tf.get_variable("w2", dtype=tf.float32, shape=[10, 3], initializer=tf.contrib.layers.xavier_initializer())
b_2 = tf.get_variable("b2", dtype=tf.float32, shape=[1, 3], initializer=tf.contrib.layers.xavier_initializer())

relu_2 = tf.matmul(y_0, W_2) + b_2

#relu_2 = tf.nn.softmax(y_2)

class_weight = tf.constant([[y_data.shape[0]/np.sum(y_data, axis = 0)[0], y_data.shape[0]/np.sum(y_data, axis = 0)[1], y_data.shape[0]/np.sum(y_data, axis = 0)[2]]])

weight_per_label = tf.transpose( tf.matmul(Y
                           , tf.transpose(class_weight)) )

loss_temp = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits=relu_2, labels=Y))
loss = tf.reduce_mean(loss_temp) + 0.0001 * (tf.nn.l2_loss(W_0) + tf.nn.l2_loss(W_2))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=relu_2, labels=Y))
#model_softmax = tf.nn.softmax(y_0)  # apply softmax to the linear layer

#loss = tf.losses.mean_squared_error(relu_2, y_)  # compute cross-entropy loss on the linear output (softmax applied internally)


update = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)  # optimise to minimise loss via gradient descent

correct_preds = tf.equal(tf.argmax(tf.nn.softmax(relu_2), 1), tf.argmax(tf.nn.softmax(Y), 1))
#get_accuracy = tf.metrics.mean_squared_error(relu_2, y_)

get_prediction = tf.nn.softmax(relu_2)
get_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))  # calculate the accuracy score

recall = [tf.metrics.recall(labels=tf.equal(tf.argmax(Y, 1), k), predictions=tf.equal(tf.argmax(tf.nn.softmax(relu_2), 1), k)) for k in range(3)]


initialiser = init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(initialiser)
    batch_size = 100

    for i in range(150):
        counter = 0
        print('Epoch {}'.format(i))
        to_sample = random.sample(range(y_data.shape[0]), y_data.shape[0])

        while counter < (y_data.shape[0] + batch_size +1):
            #print(counter)
            # print(counter)

            temp_index = to_sample[counter: min(counter + batch_size, y_data.shape[0] - 1)]
            #print(temp_index)
            X_temp = X_data[temp_index, :]
            y_temp = y_data[temp_index, :]

            # print(X.shape, y.shape)
            # X = np.asarray(X).reshape((batch_size,300, 250))
            # print(X)
            # y = np.asarray(y).reshape((batch_size,))

            # prev_loss = sess.run(loss, feed_dict = {data: X, Y: y})
            # print(y)
            sess.run(update, feed_dict={X: X_temp, Y: y_temp})

            # pred = sess.run(sigmoid, feed_dict = {data: X, Y: y})

            # pred = [x[0] for x in pred]
            # print(np.sum(np.equal(np.round(pred), y))/32)
            # print((y, np.round(pred)))
            counter += batch_size

        pred = sess.run(get_prediction, feed_dict={X: X_data, Y: y_data})
        correct = 0
        tot = 0
        # sess.run(update, feed_dict = {data: X, Y: y})
        #acc_test = sess.run(accuracy, feed_dict={data: X, Y: y})

        #curr_loss = sess.run(loss, feed_dict={data: X, Y: y})

        #recall = sess.run(recalling, feed_dict={data: X, Y: y})

        recall_test = sess.run(recall, feed_dict={X: X_data, Y: y_data})

        print('Train accuracy Recalls {}'.format(recall_test))

