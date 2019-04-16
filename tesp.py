import numpy as np
import pandas as pd
import tensorflow as tf

import networkx as nx
from vis_py import *
from itertools import chain
import matplotlib.pyplot as plt
import copy
for k in range(7, 200, 2):
    print(k)
    df = np.genfromtxt('./data/NYSE_day.csv', delimiter=',', usecols = k, skip_header = 1)

    df = df[np.logical_not(np.isnan(df))]
    #print(df)
    df = df.reshape(len(df), 1)
    T = len(df)
    time = np.arange(len(df), dtype=int).reshape(T, 1)

    df = np.concatenate([time, df], axis=1)
    feature_matrix = np.zeros(shape=(T - 100, 18))
    # label_vector = np.zeros(shape=(self.T - 100, 1))
    # get visibility graphs
    for t in range(100, T, 1):
        print(t, df[t, 1])
        df_temp = df[t - 100:t, :]
        W = visibility_graph(df_temp, directed=False)
        G = nx.convert_matrix.from_numpy_matrix(W, create_using=nx.Graph)

        avg_degree, avg_strength, std_strength, avg_clustering, diameter, degree_correlation, eigen_max, degrees_to_print = get_measures(
            G, W)
        last_prices = df[(t - 5):t, 1]
        to_add = np.concatenate(
            [[avg_degree, avg_strength, std_strength, avg_clustering, diameter, degree_correlation, eigen_max],
             degrees_to_print, last_prices, [df[t, 1]]])
        feature_matrix[t - 100, :] += to_add



    np.save('./features/features_nyse_' + str(k) + '.npy', feature_matrix)
    np.savetxt("./features/features_nyse_" + str(k) + ".csv", feature_matrix, delimiter=",")