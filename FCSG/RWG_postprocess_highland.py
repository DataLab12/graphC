import pandas as pd
import numpy as np
import networkx as nx
import math
from numpy.linalg import matrix_power
import time
import sys
import csv

pos_adj_df = pd.read_csv('/Users/maria/Desktop/data-repos_datalab/graphC/input_data/data-highland/HT_pos_edges_adj.csv', sep=' ', header=None)
neg_adj_df = pd.read_csv('/Users/maria/Desktop/data-repos_datalab/graphC/input_data/data-highland/HT_neg_edges_adj.csv', sep=' ', header=None)
pos_adj = pos_adj_df.to_numpy()
neg_adj = neg_adj_df.to_numpy()
print(pos_adj.shape)
print(neg_adj.shape)

pos_g = nx.convert_matrix.from_numpy_matrix(pos_adj)
neg_g = nx.convert_matrix.from_numpy_matrix(neg_adj)
pos_largest_cc = max(nx.connected_components(pos_g), key=len)
len(pos_largest_cc)
total_expected = len(pos_largest_cc)

all_nodes = pos_g.nodes
kept_nodes = np.array(list(pos_largest_cc))
dropped_nodes = np.setdiff1d(all_nodes, kept_nodes)
print(dropped_nodes)

file1 = open('/Users/maria/Desktop/data-repos_datalab/graphC/GraphC-highland_demo/FCSG/highland_rwg_clusters.txt', 'r')
Lines = file1.readlines()

count = 0
totalClustered = 0
for line in Lines:
    count += 1
    totalClustered += len(np.fromstring(line, sep=','))
    if count < len(Lines):
        print("Cluster {}: {}".format(count, line))
    if(count == 1):
        all_clustered = np.fromstring(line, sep=',')
    else:
        all_clustered = np.append(all_clustered, (np.fromstring(line, sep=',')))
totalClustered = totalClustered-1

print("Total clustered: {}".format(totalClustered))
numClusters = len(Lines)-1
print("Number of clusters: {}".format(numClusters))

singletons = np.setdiff1d(np.arange(0,total_expected), all_clustered)
print("Singletons: {}".format(singletons))

reindexed_clusters = []
count = 0
for line in Lines:
    count += 1
    if count < len(Lines):
        cluster = np.fromstring(line, sep=',')
        for index, item in enumerate(cluster):
            if item+len(dropped_nodes[dropped_nodes <= item]) not in dropped_nodes:
                cluster[index] = item+len(dropped_nodes[dropped_nodes <= item])
                item = item+len(dropped_nodes[dropped_nodes <= item])
            else:
                while(item in dropped_nodes):
                    item += 1
                cluster[index] = item
        reindexed_clusters.append(cluster)
if len(singletons) != 0:
    for index, item in enumerate(singletons):
            if item+len(dropped_nodes[dropped_nodes <= item]) not in dropped_nodes:
                singletons[index] = item+len(dropped_nodes[dropped_nodes <= item])
                item = item+len(dropped_nodes[dropped_nodes <= item])
            else:
                while(item in dropped_nodes):
                    item += 1
                singletons[index] = item
    reindexed_clusters.append(singletons)

reindexed_clusters.append(dropped_nodes)

count = 0
for cluster in reindexed_clusters:
    if count == 0:
        node_vector = cluster
        comm_vector = np.repeat(count, repeats=len(cluster))
    else:
        node_vector = np.append(node_vector, cluster)
        comm_vector = np.append(comm_vector, np.repeat(count, repeats=len(cluster)))
    count += 1
dataset = pd.DataFrame({'node_id': node_vector, 'label': comm_vector}, columns=['node_id', 'label'])
dataset_sorted = dataset.sort_values(by='node_id', ignore_index=True)
dataset_sorted_T = dataset_sorted.T
print(dataset_sorted_T)
dataset_sorted_T.iloc[1:].to_csv('highland_FCSG_labels_dup.csv', header=False, index=False)
