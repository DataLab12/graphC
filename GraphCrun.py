import sys
import pandas as pd
import os
import numpy as np

import random
from math import ceil
from igraph import Graph
from signet.cluster import Cluster
from scipy import sparse as sp
from scipy import io
import networkx as nx
from sklearn import metrics
import seaborn as sns
import time
import GraphC

wd =  os.getcwd()
sys.path.append(wd)

os.chdir(wd)

pos_adj = np.loadtxt('Input/HT_pos_edges_adj.csv', delimiter=' ')
neg_adj = np.loadtxt('Input/HT_neg_edges_adj.csv', delimiter=' ')

pos_adj_sp = sp.csc_matrix(pos_adj)
neg_adj_sp = sp.csc_matrix(neg_adj)

c = Cluster((pos_adj_sp, neg_adj_sp))

L_none = c.spectral_cluster_laplacian(k = 3, normalisation='none')
L_none = pd.DataFrame(L_none).T

L_sym = c.spectral_cluster_laplacian(k = 3, normalisation='sym')
L_sym = pd.DataFrame(L_sym).T

L_sym_sep = c.spectral_cluster_laplacian(k = 3, normalisation='sym_sep')
L_sym_sep = pd.DataFrame(L_sym_sep).T
print(L_sym_sep)

BNC_none = c.spectral_cluster_bnc(k=3, normalisation='none')
BNC_none = pd.DataFrame(BNC_none).T

BNC_sym = c.spectral_cluster_bnc(k=3, normalisation='sym')
BNC_sym = pd.DataFrame(BNC_sym).T

SPONGE_none = c.SPONGE(k = 3)
SPONGE_none = pd.DataFrame(SPONGE_none).T

SPONGE_sym = c.SPONGE_sym(k = 3)
SPONGE_sym = pd.DataFrame(SPONGE_sym).T

labels_all = pd.concat([L_none, L_sym, L_sym_sep, BNC_none, BNC_sym, SPONGE_none, SPONGE_sym])
labs = pd.DataFrame(["lap_none", "lap_sym", "lap_sym_sep", "BNC_none", "BNC_sym", "SPONGE_none", "SPONGE_sym"])
labels_all = labels_all.reset_index(drop=True)
labs = labs.reset_index(drop=True)
labels_formatted = pd.concat([labs, labels_all], axis=1)
pd.DataFrame(labels_formatted).to_csv('cluster_k3.csv', header=None, index=None)

print("Clustering complete on k=3.")

comm_labs_k3 = pd.read_csv('Input/highland_ground_truth_k3.csv')
comm_labs_k3.columns = comm_labs_k3.columns.astype(int)
sorted_comm_labs_k3_pd = comm_labs_k3.sort_index(axis=1)
sorted_comm_labs_k3 = sorted_comm_labs_k3_pd.iloc[0].values

GraphC.combine_signet_GM_SPM_FCSG_BCM_SSSnet('cluster_k3.csv',
                     'GM/HT_GM_k3.csv',
                     'SPM/HT_SPM_k3.csv',
                     'SSSNET/highland_sssnet_k3.csv',
                     'BCM_KM/highland_BCM_KM_K3.csv',
                     'FCSG/highland_FCSG_labels.csv',
                     sorted_comm_labs_k3,
                     'ARI_highland_final_scores_k3.csv')
