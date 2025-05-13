import pandas as pd
import numpy as np
import networkx as nx
import math
from numpy.linalg import matrix_power
import time
import sys

start_time = time.perf_counter()

L = int(sys.argv[1])

pos_adj_df = pd.read_csv(sys.argv[2], sep=sys.argv[5], header=None)
neg_adj_df = pd.read_csv(sys.argv[3], sep=sys.argv[5], header=None)
pos_adj = pos_adj_df.to_numpy()
neg_adj = neg_adj_df.to_numpy()

pos_g = nx.convert_matrix.from_numpy_matrix(pos_adj)
neg_g = nx.convert_matrix.from_numpy_matrix(neg_adj)
pos_largest_cc = max(nx.connected_components(pos_g), key=len)
pos_sg = pos_g.subgraph(pos_largest_cc)
neg_sg = neg_g.subgraph(pos_largest_cc)
pos_adj = nx.convert_matrix.to_numpy_matrix(pos_sg)
neg_adj = nx.convert_matrix.to_numpy_matrix(neg_sg)

W_prime = pos_adj
W_prime2 = pos_adj + neg_adj
n = pos_adj.shape[0]
theta_prime = np.empty(shape = (n,n))
theta_prime2 = np.empty(shape = (n,n))
for i in range(n):
    for j in range(n):
        if(W_prime[i, j] != 0):
            theta_prime[i, j] = W_prime[i, j]/(np.sum(W_prime[i, ]))
        else:
            theta_prime[i, j] = 0
        if(W_prime2[i, j] != 0):
            theta_prime2[i, j] = W_prime2[i, j]/(np.sum(W_prime2[i, ]))
        else:
            theta_prime2[i, j] = 0

H_G_prime = 0
H_G_prime2 = 0
for i in range(L):
    H_G_prime = H_G_prime + matrix_power(theta_prime, i+1)
    H_G_prime2 = H_G_prime2 + matrix_power(theta_prime2, i+1)

D_star = np.empty(shape = (n,n))
for i in range(n):
    for j in range(n):
        alpha = (H_G_prime2[i,j]-H_G_prime[i,j])/(H_G_prime[i,j])
        if(alpha > 0):
            D_star[i,j] = alpha
        else:
            D_star[i,j] = 0

RWG_mat = np.empty(shape = (n,n))
D_star_T = np.transpose(D_star)
for i in range(n):
    for j in range(n):
        RWG_mat[i,j] = math.exp(-0.5*(D_star[i,j]+D_star_T[i,j]))

W_star = np.empty(shape = (n,n))
W = pos_adj-neg_adj
for i in range(n):
    for j in range(n):
        if(W[i,j] > 0):
            W_star[i,j] = W[i,j]*RWG_mat[i,j]
        elif(W[i,j] < 0):
            W_star[i,j] = W[i,j]
        else:
            W_star[i,j] = 0

w_star_g = nx.convert_matrix.from_numpy_matrix(W_star)
edges = list(w_star_g.edges.data("weight"))
edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)

log_file = sys.argv[4]+"_rwg_log.txt"
a_file = open(log_file, "w")

max_id = max(list(w_star_g.nodes()))+1
print(edges_sorted)
while(edges_sorted[0][2] > 0):

    # Identify edges that must be copied

    l_edges = list(w_star_g.edges(edges_sorted[0][0], "weight"))
    l_edges[1:len(l_edges)]
    l_connecting_nodes = set([a_tuple[1] for a_tuple in l_edges[1:len(l_edges)]])

    r_edges = list(w_star_g.edges(edges_sorted[0][1], "weight"))
    r_edges[1:len(r_edges)]
    r_connecting_nodes = set([a_tuple[1] for a_tuple in r_edges[1:len(r_edges)]])

    both = l_connecting_nodes.intersection(r_connecting_nodes)
    l_edges_only = l_connecting_nodes-both
    r_edges_only = r_connecting_nodes-both

    # Create new combined node

    w_star_g.add_node(max_id)

    # Add l edges to new node

    if(len(list(l_edges_only)) > 0):
        for i in range(len(list(l_edges_only))):
            new_node = max_id
            old_edge = [item for item in l_edges if item[1] == list(l_edges_only)[i]]
            new_weight = old_edge[0][2]
            w_star_g.add_edge(list(l_edges_only)[i], new_node, weight=new_weight)

    # Add r edges to new node

    if(len(list(r_edges_only)) > 0):
        for i in range(len(list(r_edges_only))):
            new_node = max_id
            old_edge = [item for item in r_edges if item[1] == list(r_edges_only)[i]]
            new_weight = old_edge[0][2]
            w_star_g.add_edge(list(r_edges_only)[i], new_node, weight=new_weight)

    # Add l and r edges to new node

    if(len(list(both)) > 0):
        for i in range(len(list(both))):
            new_node = max_id
            old_edge_l = [item for item in l_edges if item[1] == list(both)[i]]
            old_edge_r = [item for item in r_edges if item[1] == list(both)[i]]
            new_weight = old_edge_l[0][2]+old_edge_r[0][2]
            w_star_g.add_edge(list(both)[i], new_node, weight=new_weight)

    # Delete original nodes

    w_star_g.remove_node(edges_sorted[0][0])
    w_star_g.remove_node(edges_sorted[0][1])

    change =  [edges_sorted[0][0], edges_sorted[0][1], max_id]
    change = np.array(change)
    np.savetxt(a_file, [change], fmt='%i', delimiter=',')

    edges = list(w_star_g.edges.data("weight"))
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
    max_id = max_id + 1
    print(len(edges))
    if(len(edges) == 0):
        break;

a_file.close()

print(edges_sorted)

output = pd.read_csv(log_file, sep=',', header=None)
output = output.to_numpy()

clusters = list()
for i in range(len(output)):
    if(len(clusters) == 0):
        clusters.append(output[i])
    else:
        make_new_cluster = 0
        cluster_pair = list()
        for j in range(len(clusters)):
            if(np.intersect1d(output[i], clusters[j]).size > 0):
                new_cluster = np.union1d(clusters[j], output[i])
                clusters[j] = new_cluster
                make_new_cluster = make_new_cluster+1
                cluster_pair.append(j)
        if(make_new_cluster == 0):
            clusters.append(output[i])
        if(make_new_cluster > 1): # Can be at most 2 overlapping clusters
            combined_cluster = np.union1d(clusters[cluster_pair[0]], clusters[cluster_pair[1]])
            clusters[cluster_pair[0]] = combined_cluster
            clusters.pop(cluster_pair[1])

cluster_file = sys.argv[4]+"_rwg_clusters.txt"
a_file = open(cluster_file, "w")
for cluster in clusters:
    cluster = cluster[(n > cluster)]
    np.savetxt(a_file, [cluster], fmt='%i', delimiter=',')

end_time = time.perf_counter()
total_time = end_time-start_time
print(total_time)
np.savetxt(a_file, [total_time])
a_file.close()
