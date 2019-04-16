import numpy as np
import pandas as pd

import networkx as nx

def visibility_graph(T, directed = True):

    #VISIBILITY_GRAPH Summary of this function goes here
    #   T: nx2 times series. first column times, second intensities
    n = T.shape[0]
    y = T[:, 1]
    t = T[:, 0]
    W = np.zeros(shape = (n, n)) #make sparse
    id_sort = np.argsort(-y) #use minus for descending as prices positive
    #print(id_sort)

    for node1 in id_sort:
        temp = W[node1, node1:] > 0
        #print(temp, W[node1, node1:])
        r = np.array(range(len(temp)))
        #print(r)
        check_empty = r[temp]

        #check_empty = W[node1, node1:][W[node1, node1:] > 0] #get indices not elements
        if check_empty.size == 0:
            limit_right = n - 1
        else:
            limit_right = (node1 - 1) + check_empty[0]
        #print(check_empty, W, limit_right)
        #limit_right = node1-1+find(W(node1,node1:end)>0,1,'first');
        '''if isempty(limit_right)==1
            limit_right = n;
        end'''
        temp = W[node1, :node1] > 0
        #print(temp, W[node1, node1:])
        r = np.array(range(len(temp)))
        #print(r)
        check_empty = r[temp]

        if check_empty.size == 0:
            limit_left = 0
        else:
            limit_left = check_empty[-1]
        #print('left ', check_empty, W, limit_left)
        search_range_limit = np.arange(limit_left, limit_right + 1, 1)
        for node2 in search_range_limit:
            if node2 == ( node1+1 ):
                w = compute_w(node1, node2, t, y)
                W[node1,node2] = w
                W[node2,node1] = w
            elif node1 == ( node2+1 ):
                w = compute_w(node1, node2, t, y)
                W[node1,node2] = w
                W[node2,node1] = w
            elif node2 != node1:

                temp = search_range_limit  == min(node1,node2)
                r = np.array(range(len(temp)))
                a = int(r[temp])

                temp = search_range_limit  == max(node1,node2)
                r = np.array(range(len(temp)))
                b = int(r[temp])
                #a = search_range_limit[search_range_limit  == min(node1,node2)] #get indices not elements
                #b = search_range_limit[search_range_limit  == max(node1, node2)] #get indices not elements
                #a = find(search_range_limit==min(node1,node2));
                #b = find(search_range_limit==max(node1,node2));
                idx = search_range_limit[(a + 1) : b]
                d = y[node2] + ( y[node1] - y[node2] ) * (t[node2] - t[idx]) / ( t[node2] - t[node1] )

                if np.all( d > y[idx] ):
                    w = compute_w(node1,node2,t,y)
                    W[node1, node2] = w
                    W[node2, node1] = w
    if directed == True:
        W = np.triu(W, k = 0)

    return W

def compute_w(node1,node2,t,y):
    result = np.arctan((y[node2] - y[node1]) / (t[node2] - t[node1])) + np.pi / 2
    return result

#right = np.array([[1, 10], [2, 7.8], [2.01, 1], [3, 8], [4, 1], [50, 4]]).reshape(6, 2)
#right = np.array([[1, 10], [2, 9], [3, 2], [4, 8]]).reshape(4, 2)

def get_measures(G, W):

    nodes = G.nodes()
    '''degrees_in = G.in_degree(weight='weight')  # Dict with Node ID, Degree

    degrees_in_to_print = np.asarray([degrees_in[n] for n in nodes])


    degrees_out = G.out_degree(weight='weight')  # Dict with Node ID, Degree

    degrees_out_to_print = np.asarray([degrees_out[n] for n in nodes])'''

    degrees = G.degree()  # Dict with Node ID, Degree

    degrees_to_print = np.asarray([degrees[n] for n in nodes])

    avg_degree = np.mean(degrees_to_print)

    strength = G.degree(weight='weight')  # Dict with Node ID, Degree

    strength_to_print = np.asarray([strength[n] for n in nodes])

    avg_strength = np.mean(degrees_to_print)

    std_strength = np.std(degrees_to_print)

    clustering = nx.clustering(G)

    clustering_to_print = np.asarray([clustering[n] for n in nodes])

    avg_clustering = np.mean(clustering_to_print)

    diameter = nx.diameter(G)

    degree_correlation = nx.degree_pearson_correlation_coefficient(G)

    L = nx.normalized_laplacian_matrix(G)

    eigen_max = max(np.linalg.eigvals(L.A))

    #degree price rank correlation

    return avg_degree, avg_strength, std_strength, avg_clustering, diameter, degree_correlation, eigen_max, degrees_to_print[-5:]


