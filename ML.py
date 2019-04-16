import numpy as np
import pandas as pd
import tensorflow as tf

import networkx as nx
from vis_py import *
from itertools import chain
import matplotlib.pyplot as plt
import copy


'''df = np.genfromtxt('./data/italy.csv', delimiter=',')
df = df.reshape(len(df), 1)
time = np.arange(len(df), dtype=int).reshape(len(df), 1)
df = np.concatenate([time, df], axis=1)



for t in range(100, 1000, 100):
    df_temp = df[t - 100:t, :]
    W = visibility_graph(df_temp, directed=False)
    # W = np.heaviside(W, 0).astype(int)

    import matplotlib.pylab as plt
    import scipy.sparse as sps

    M = sps.csr_matrix(W)
    plt.spy(M)
    plt.show()

    plt.plot(df_temp[:, 0], df_temp[:, 1] / max(df_temp[:, 1]))
    plt.plot(df_temp[:, 0], np.sum(W, axis=1) / max(np.sum(W, axis=1)))
    plt.show()

    G_weight = nx.convert_matrix.from_numpy_matrix(W, create_using=nx.Graph)
    W = np.heaviside(W, 0).astype(int)
    G_unweight = nx.convert_matrix.from_numpy_matrix(W, create_using=nx.Graph)

    nx.draw(G_unweight, node_size=1)
    plt.draw()
    plt.show()'''

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.

def get_classes(df):
    shifted = np.roll(df, shift=1)[1:]
    return shifted - df[1:]

class get_data():
    def __init__(self):
        df = np.load('features_nyse.npy')
        print(df[:, -6:])

        df[1:, -6:] = df[1:, -6:] - df[:-1, -6:] #get returns
        df = df[1:, :]

        print(df[:, -6:])

        one_hots = np.zeros(shape = (df.shape[0], 2))

        one_hots[:, 0] = np.heaviside(df[:, -1], 0)
        one_hots[:, 1] = np.heaviside( - df[:, -1], 0)

        print(df)

        print(one_hots)

        df = np.concatenate([df, one_hots], axis = 1)

        print(df[:, -3:])

        for_shuffle = np.copy(df)

        np.random.shuffle(for_shuffle)

        self.feature_matrix = for_shuffle
        self.test_matrix = df
        self.count = 0


    def __call__(self):

        features = self.feature_matrix[ self.count : (self.count+batch_size), :-2]
        labels = np.expand_dims(self.feature_matrix[ self.count : (self.count+batch_size), -2:], axis=1)

        to_return = features, labels
        self.count += 1
        return to_return


    def all(self):

        features = self.test_matrix[:, :-2]
        labels = self.test_matrix[:, -2:]

        to_return = features, labels

        return to_return





    '''TFdataset = tf.data.Dataset.from_tensor_slices((feature_matrix, label_vector))

    dataset = TFdataset.repeat().batch(batch_size)

    iter = dataset.make_one_shot_iterator()
    return iter'''


def get_placeholders():
  x = tf.placeholder(tf.float32, [None, 17])
  y_ = tf.placeholder(tf.float32, [None, 2])
  return x, y_

# Store results of runs with different configurations in a list.
# Use a tuple (num_epochs, learning_rate) as keys, and a tuple (training_accuracy, testing_accuracy)
experiments_task1 = []
settings = [(5, 0.0001)]#, (5, 0.005), (15, 0.1)]
log_period_samples = 100
batch_size = 50

print('Training Model')
# Train Model 1 with the different hyper-parameter settings.
for (num_epochs, learning_rate) in settings:

    # Reset graph, recreate placeholders and dataset.
    tf.reset_default_graph()
    x, y_ = get_placeholders()
    mnist = get_data()

    #####################################################
    # Define model, loss, update and evaluation metric. #

    # initialise weight matrix and bias for fully connected linear layer
    W_0 = tf.get_variable("w0", dtype=tf.float32, shape=[17, 10], initializer=tf.contrib.layers.xavier_initializer())
    b_0 = tf.get_variable("b0", dtype=tf.float32, shape=[1, 10], initializer=tf.contrib.layers.xavier_initializer())

    y_0 = tf.matmul(x, W_0) + b_0

    #relu_0 = tf.nn.relu(y_0)

    # initialise weight matrix and bias for fully connected linear layer
    W_1 = tf.get_variable("w1", dtype=tf.float32, shape=[10, 5], initializer=tf.contrib.layers.xavier_initializer())
    b_1 = tf.get_variable("b1", dtype=tf.float32, shape=[1, 5], initializer=tf.contrib.layers.xavier_initializer())

    y_1 = tf.matmul(y_0, W_1) + b_1

    relu_1 = tf.nn.relu(y_1)

    # initialise weight matrix and bias for fully connected linear layer
    W_2 = tf.get_variable("w2", dtype=tf.float32, shape=[5, 2], initializer=tf.contrib.layers.xavier_initializer())
    b_2 = tf.get_variable("b2", dtype=tf.float32, shape=[1, 2], initializer=tf.contrib.layers.xavier_initializer())

    y_2 = tf.matmul(relu_1, W_2) + b_2

    relu_2 = tf.nn.softmax(y_2)

    loss = tf.losses.softmax_cross_entropy_with_logits(logits=y_2, labels=y_)

    #model_softmax = tf.nn.softmax(y_0)  # apply softmax to the linear layer

    #loss = tf.losses.mean_squared_error(relu_2, y_)  # compute cross-entropy loss on the linear output (softmax applied internally)


    update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)  # optimise to minimise loss via gradient descent

    correct_preds = tf.equal(tf.argmax(relu_2, 1), tf.argmax(y_, 1))
    #get_accuracy = tf.metrics.mean_squared_error(relu_2, y_)

    get_prediction = [relu_2, y_]
    get_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))  # calculate the accuracy score

    #####################################################

    # Train.
    i, train_accuracy, test_accuracy = 0, [], []
    log_period_updates = int(log_period_samples / batch_size)
    with tf.train.MonitoredSession() as sess:
        while i < 1600:

            # Update.
            i += 1
            batch_xs, batch_ys = mnist()

            #################
            # Training step #
            sess.run(update, feed_dict={x: batch_xs, y_: batch_ys})  # run the gradient descent update

            #################

            # Periodically evaluate.
            if i % log_period_updates == 0:
                #####################################
                # Compute and store train accuracy. #

                temp_accuracy = sess.run(get_accuracy,
                                         feed_dict={x: batch_xs, y_: batch_ys})  # get the accuracy

                train_accuracy.append(temp_accuracy[1])  # append it to the list



        batch_xs, batch_ys = mnist.all()

        predictions = sess.run(get_prediction,
                                         feed_dict={x: batch_xs, y_: batch_ys})



                #####################################
        #experiments_task1.append(train_accuracy)
        #experiments_task1.append(
            #((num_epochs, learning_rate), train_accuracy, test_accuracy))
'''train_plot = []
for i in train_accuracy:
    train_plot.append(i[1])

test_plot = []
for i in test_accuracy:
    test_plot.append(i[1])'''

plt.plot(np.arange(len(train_accuracy)), train_accuracy, label = 'train')
plt.legend()

plt.show()

'''pred_0 = []
pred_1 = []
for i in predictions[0]:
    pred_0.append(i)
for j in predictions[1]:
    pred_1.append(j)
plt.plot(np.arange(len(pred_0)), pred_0, label = 'pred')
plt.plot(np.arange(len(pred_1)), pred_1, label = 'label')
plt.legend()
plt.show()'''
plt.plot(np.arange(len(predictions[0])), predictions[0], label = 'pred')
plt.plot(np.arange(len(predictions[1])), predictions[1], label = 'label')
plt.grid()
plt.legend()
plt.show()


plt.plot(np.arange(len(predictions[0])), predictions[0], label = 'pred')
plt.plot(np.arange(len(predictions[1])), predictions[1], label = 'label')
plt.grid()
plt.legend()

plt.savefig('price_chart.eps', format='eps', dpi=1000)

'''    stop = False
    test_epoch.append(temp_accuracy[1])
    if len(test_epoch) > 2:
        if (test_epoch[-1] < test_epoch[-2]) or (test_epoch[-1] < test_epoch[-3]):
            stop = False
        else:
            stop = True




    if stop:
        break'''

'''class get_data():
    def __init__(self):
        df = np.load('features_nyse.npy')
        df = df.reshape(len(df), 1)
        self.T = len(df)
        time = np.arange(len(df), dtype=int).reshape(self.T, 1)

        df = np.concatenate([time, df], axis=1)
        feature_matrix = np.zeros(shape=(self.T - 101, 19))
        #label_vector = np.zeros(shape=(self.T - 100, 1))
        #get visibility graphs
        df_class = get_classes(df[:, 1])
        df_class = np.heaviside(df_class, 0)
        df_class = df_class.astype(int)
        label_prep = np.zeros((self.T - 101, 2), dtype = int)
        print(df_class)
        label_prep[np.arange(self.T - 101, dtype=int), df_class[100:]] = 1

        df = df[1:, :]

        self.T = df.shape[0]


        for t in range(101, self.T, 1):
            df_temp = df[t - 101:t]
            W = visibility_graph(df_temp, directed=False)
            G = nx.convert_matrix.from_numpy_matrix(W, create_using=nx.Graph)

            avg_degree, avg_strength, std_strength, avg_clustering, diameter, degree_correlation, eigen_max, degrees_to_print = get_measures(G, W)
            last_prices = df_class[(t - 5):t]
            to_add = np.concatenate([[avg_degree, avg_strength, std_strength, avg_clustering, diameter, degree_correlation, eigen_max], degrees_to_print, last_prices, label_prep[t, :]])
            feature_matrix[t - 101, :] += to_add
            #print(to_add[:])
            print(t)
            #label_vector[t - 100, :] += df[t, 1]

        for_shuffle = np.copy(feature_matrix)

        np.random.shuffle(for_shuffle)

        self.feature_matrix = for_shuffle
        self.test_matrix = feature_matrix
        self.count = 0


    def __call__(self):

        features = self.feature_matrix[ self.count : (self.count+batch_size), :-2]
        labels = np.expand_dims(self.feature_matrix[ self.count : (self.count+batch_size), -2:], axis=1)

        to_return = features, labels
        self.count += 1
        return to_return


    def all(self):

        features = self.test_matrix[:, :-2]
        labels = self.test_matrix[:, -2:]

        to_return = features, labels

        return to_return
'''