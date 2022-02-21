"""
Milestone one module - this script contains all functions needed to complete
the milestone one analysis
"""
import numpy as np
import pandas as pd
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

def mtx_edges_to_pd(mtx_edges):
    """Reads in an mtx file of edges and outputs number of edges, number of
    unique users, and pandas dataframe of formatted edges.

    Parameters
    ----------
    mtx_edges : str
        the name of the .mtx file containing edges
    Returns
    -------
    n_edges_pos : int
        the number of positive edges in the graph
    n_unique_users : int
        the number of unique nodes in the graph
    edges_pos_df : pandas dataframe
        pandas dataframe containing directed edge information, dimensions are
        n_edges_pos x 2 with column names 'source' and 'target'
    unique_users : pandas dataframe
        pandas dataframe containing a list of unique users
    """
    edges = np.loadtxt(mtx_edges)
    edges_pos = np.delete(edges, 0, 0)
    edges_pos = np.delete(edges_pos, 2, 1)
    edges_pos = edges_pos.astype(int)
    edges_pos_df = pd.DataFrame(edges_pos, columns = ['source','target'])
    unique_users = pd.concat([edges_pos_df['source'],edges_pos_df['target']],ignore_index=True).drop_duplicates().reset_index(drop=True)
    n_edges_pos = edges_pos_df.shape[0]
    n_unique_users = len(unique_users)
    print("Edges converted to pandas dataframe...")
    return n_edges_pos, n_unique_users, edges_pos_df, unique_users

def directed_to_undirected_adj(pos_adj_dir_path, neg_adj_dir_path, pos_adj_undir_path, neg_adj_undir_path, delim):
    """Convert directed adjacency matrices to undirected by removing duplicate rows, i.e. taking the union of edges rather than the intersection.

    Parameters
    ----------
    pos_edge_dir_path : str
        filepath to positive adjacency matrix
    neg_edge_dir_path : str
        filepath to negative adjacency matrix
    pos_edge_undir_path : str
        filepath with which to save the positive adjacency matrix
    neg_edge_undir_path : str
        filepath with which to save the negative adjacency matrix
    delim : char
        delimiter in adjacency input files
    """
    pos_adj = np.loadtxt(pos_adj_dir_path, delimiter=delim)
    neg_adj = np.loadtxt(neg_adj_dir_path, delimiter=delim)

    pos_sym = pos_adj + pos_adj.transpose()
    neg_sym = neg_adj + neg_adj.transpose()

    pos_sym[pos_sym > 0] = 1
    neg_sym[neg_sym > 0] = 1

    pd.DataFrame(pos_sym).to_csv(pos_adj_undir_path, header=None, index=None)
    pd.DataFrame(neg_sym).to_csv(neg_adj_undir_path, header=None, index=None)

def parse_communities(community_labels):
    """Import communities, parse, and save community labels and members to a
    dictionary

    Parameters
    ----------
    community_labels : str
        the name of the file containing communities, formatted as one community
        per row,ie:
        'community name' : id1,id2,id3,...,idn
    Returns
    -------
    comm_dict : dictionary
        Dictionary mapping community labels to arrays of user IDs, represented
        as integers
    """
    all_comm = open(community_labels, 'r')
    Comms = all_comm.readlines()
    comm_dict = dict()
    for comm in Comms:
        community = comm.strip()
        comm_name = community.split(": ")[0]
        comm_ids = community.split(": ")[1]
        comm_dict[comm_name] = np.fromstring(comm_ids, dtype=int, sep=',')
    print("Communities parsed...")
    return comm_dict

def neg_edge_aug_block_samp_dir(neg_edge_perc, n_edges_pos, comm_dict, edges_pos_df):
    """Add a set number of directed negative edges (based on positive edge density)
    between communities. Edges are assigned using random block sampling, i.e.
    first two communities are chosen, then two users are chosen from within the
    communities. Generated edges are only kept if there is not already an edge
    between the two chosen nodes

    Parameters
    ----------
    neg_edge_perc : float
        The percent multiplier relative to number of positive edges that
        determines number of negative edges
    n_edges_pos: int
        The number of positive edges
    comm_dict : dictionary
        Dictionary mapping community labels to arrays of user IDs, represented
        as integers
    edges_pos_df : pandas dataframe
        pandas dataframe containing directed edge information, dimensions are
        n_edges_pos x 2 with column names 'source' and 'target'

    Returns
    -------
    comm_dict : dictionary
        Dictionary mapping community labels to arrays of user IDs, represented as integers
    """

    n_edges_neg = ceil(neg_edge_perc*n_edges_pos)
    comm_names = [*comm_dict]
    count = 0
    edges_neg_df = pd.DataFrame(columns=['source', 'target'])
    comm_df = pd.DataFrame(columns=['source', 'target'])
    while count < n_edges_neg:
        comm_pair = random.sample(comm_names, 2)
        user_from = random.sample(list(comm_dict[comm_pair[0]]), 1)[0]
        user_to = random.sample(list(comm_dict[comm_pair[1]]), 1)[0]
        repeat_neg = ((edges_neg_df['source'] == user_from) & (edges_neg_df['target'] == user_to)).any()
        repeat_pos = ((edges_pos_df['source'] == user_from) & (edges_pos_df['target'] == user_to)).any()
        if(repeat_neg == False and repeat_pos == False):
            edges_neg_df = edges_neg_df.append({'source': user_from, 'target': user_to}, ignore_index=True)
            comm_df = comm_df.append({'source': comm_pair[0], 'target': comm_pair[1]}, ignore_index=True)
            count = count + 1
    print("Negative edges generated...")
    return edges_neg_df

def neg_edge_aug_block_samp_undir(neg_edge_perc, n_edges_pos, comm_dict, edges_pos_df):
    """Add a set number of negative edges (based on positive edge density)
    between communities. Edges are assigned using random block sampling, i.e.
    first two communities are chosen, then two users are chosen from within the
    communities. Generated edges are only kept if there is not already an edge
    between the two chosen nodes

    Parameters
    ----------
    neg_edge_perc : float
        The percent multiplier relative to number of positive edges that
        determines number of negative edges
    n_edges_pos: int
        The number of positive edges
    comm_dict : dictionary
        Dictionary mapping community labels to arrays of user IDs, represented
        as integers
    edges_pos_df : pandas dataframe
        pandas dataframe containing directed edge information, dimensions are
        n_edges_pos x 2 with column names 'source' and 'target'

    Returns
    -------
    comm_dict : dictionary
        Dictionary mapping community labels to arrays of user IDs, represented as integers
    """
    edges_pos_df['dup_edge'] = edges_pos_df.apply(lambda row: ''.join(sorted([str(row['source']), str(row['target'])])), axis=1)
    #print(edges_pos_df.shape)
    edges_pos_df = edges_pos_df.drop_duplicates('dup_edge')
    #print(edges_pos_df.shape)
    edges_pos_df = edges_pos_df.drop(['dup_edge'], axis=1)
    #print(edges_pos_df)
    n_edges_neg = ceil(neg_edge_perc*edges_pos_df.shape[0])
    comm_names = [*comm_dict]
    count = 0
    edges_neg_df = pd.DataFrame(columns=['source', 'target'])
    comm_df = pd.DataFrame(columns=['source', 'target'])
    while count < n_edges_neg:
        comm_pair = random.sample(comm_names, 2)
        user_from = random.sample(list(comm_dict[comm_pair[0]]), 1)[0]
        user_to = random.sample(list(comm_dict[comm_pair[1]]), 1)[0]
        repeat_neg_lr = ((edges_neg_df['source'] == user_from) & (edges_neg_df['target'] == user_to)).any()
        repeat_neg_rl = ((edges_neg_df['target'] == user_from) & (edges_neg_df['source'] == user_to)).any()
        repeat_pos_lr = ((edges_pos_df['source'] == user_from) & (edges_pos_df['target'] == user_to)).any()
        repeat_pos_rl = ((edges_pos_df['target'] == user_from) & (edges_pos_df['source'] == user_to)).any()
        if(repeat_neg_lr == False and repeat_pos_lr == False and repeat_neg_rl == False and repeat_pos_rl == False):
            edges_neg_df = edges_neg_df.append({'source': user_from, 'target': user_to}, ignore_index=True)
            comm_df = comm_df.append({'source': comm_pair[0], 'target': comm_pair[1]}, ignore_index=True)
            count = count + 1
    print("Negative edges generated...")
    return edges_pos_df, edges_neg_df



def gen_adj_matrices_from_tuples(edges_pos_df, edges_neg_df, users_filepath, pos_edge_filepath, neg_edge_filepath, directed):
    """convert edge lists to adjacency matrices and save the adjacencies to the
    specified location

    Parameters
    ----------
    edges_pos_df : pandas dataframe
        pandas dataframe containing directed positive edge information, dimensions are
        n_edges_pos x 2 with column names 'source' and 'target'
    edges_neg_df : pandas dataframe
        pandas dataframe containing directed negative edge information, dimensions are
        n_edges_neg x 2 with column names 'source' and 'target'
    users_filepath : str
        File path to a list of unique users
    pos_edge_filepath : str
        filepath with which to save the positive adjacency matrix
    neg_edge_filepath : str
        filepath with which to save the negative adjacency matrix
    directed : bool
        if true, a directed graph is assumed.  Else undirected is assumed.
    """
    unique_users = np.loadtxt(users_filepath). astype(int)
    if(directed == True):
        g_pos = nx.DiGraph()
        g_pos.add_nodes_from(unique_users)
        g_pos.add_edges_from([tuple(x) for x in edges_pos_df.to_numpy()])

        g_neg = nx.DiGraph()
        g_neg.add_nodes_from(unique_users)
        g_neg.add_edges_from([tuple(x) for x in edges_neg_df.to_numpy()])
    else:
        g_pos = nx.Graph()
        g_pos.add_nodes_from(unique_users)
        g_pos.add_edges_from([tuple(x) for x in edges_pos_df.to_numpy()])

        g_neg = nx.Graph()
        g_neg.add_nodes_from(unique_users)
        g_neg.add_edges_from([tuple(x) for x in edges_neg_df.to_numpy()])

    g_pos_adj = nx.to_numpy_matrix(g_pos, nodelist=sorted(unique_users))
    g_neg_adj = nx.to_numpy_matrix(g_neg, nodelist=sorted(unique_users))
    pd.DataFrame(g_pos_adj).to_csv(pos_edge_filepath, header=None, index=None)
    pd.DataFrame(g_neg_adj).to_csv(neg_edge_filepath, header=None, index=None)
    print("Adjacency matrices saved...")

def cluster_comparison(pos_edge_filepath, neg_edge_filepath, k, labels_filepath, labels_filepath_time, delim, ftype):
    """convert edge lists to adjacency matrices and save the adjacencies to the
    specified location

    Parameters
    ----------
    pos_edge_filepath : str
        filepath to positive adjacency matrix
    neg_edge_filepath : str
        filepath to negative adjacency matrix
    k : int
        number of clusters
    labels_filepath : str
        filepath with which to save the formatted labels
    delim : char
        delimiter in adjacency input files
    """
    if(ftype == 'mtx'):
        pos_adj_sp = io.mmread(pos_edge_filepath)
        neg_adj_sp = io.mmread(neg_edge_filepath)
        c = Cluster((pos_adj_sp, neg_adj_sp))
    else:
        pos_adj = np.loadtxt(pos_edge_filepath, delimiter=delim)
        neg_adj = np.loadtxt(neg_edge_filepath, delimiter=delim)

        pos_adj_sp = sp.csc_matrix(pos_adj)
        neg_adj_sp = sp.csc_matrix(neg_adj)

        c = Cluster((pos_adj_sp, neg_adj_sp))

    start_time = time.perf_counter()
    L_none = c.spectral_cluster_laplacian(k = k, normalisation='none')
    L_none = pd.DataFrame(L_none).T
    end_time = time.perf_counter()
    L_none_total_time = end_time-start_time

    start_time = time.perf_counter()
    L_sym = c.spectral_cluster_laplacian(k = k, normalisation='sym')
    L_sym = pd.DataFrame(L_sym).T
    end_time = time.perf_counter()
    L_sym_total_time = end_time-start_time

    start_time = time.perf_counter()
    L_sym_sep = c.spectral_cluster_laplacian(k = k, normalisation='sym_sep')
    L_sym_sep = pd.DataFrame(L_sym_sep).T
    end_time = time.perf_counter()
    L_sym_sep_total_time = end_time-start_time

    start_time = time.perf_counter()
    BNC_none = c.spectral_cluster_bnc(k=k, normalisation='none')
    BNC_none = pd.DataFrame(BNC_none).T
    end_time = time.perf_counter()
    BNC_none_total_time = end_time-start_time

    start_time = time.perf_counter()
    BNC_sym = c.spectral_cluster_bnc(k=k, normalisation='sym')
    BNC_sym = pd.DataFrame(BNC_sym).T
    end_time = time.perf_counter()
    BNC_sym_total_time = end_time-start_time

    start_time = time.perf_counter()
    SPONGE_none = c.SPONGE(k = k)
    SPONGE_none = pd.DataFrame(SPONGE_none).T
    end_time = time.perf_counter()
    SPONGE_none_total_time = end_time-start_time

    start_time = time.perf_counter()
    SPONGE_sym = c.SPONGE_sym(k = k)
    SPONGE_sym = pd.DataFrame(SPONGE_sym).T
    end_time = time.perf_counter()
    SPONGE_sym_total_time = end_time-start_time

    labels_all = pd.concat([L_none, L_sym, L_sym_sep, BNC_none, BNC_sym, SPONGE_none, SPONGE_sym])
    labs = pd.DataFrame(["lap_none", "lap_sym", "lap_sym_sep", "BNC_none", "BNC_sym", "SPONGE_none", "SPONGE_sym"])
    labels_all = labels_all.reset_index(drop=True)
    labs = labs.reset_index(drop=True)
    labels_formatted = pd.concat([labs, labels_all], axis=1)
    pd.DataFrame(labels_formatted).to_csv(labels_filepath, header=None, index=None)

    time_all = np.array([L_none_total_time, L_sym_total_time, L_sym_sep_total_time, BNC_none_total_time, BNC_sym_total_time, SPONGE_none_total_time, SPONGE_sym_total_time])
    time_formatted = pd.DataFrame([time_all], columns=["lap_none", "lap_sym", "lap_sym_sep", "BNC_none", "BNC_sym", "SPONGE_none", "SPONGE_sym"])
    pd.DataFrame(time_formatted).to_csv(labels_filepath_time, header=None, index=None)

    print("Clustering complete on k="+ str(k) + ".")

def compute_ARS(cluster_filepath, comm_dict):
    """Compute the adjusted Rand scores for each clustering method in cluster_filepath

    Parameters
    ----------
    cluster_filepath : str
        filepath to cluster labels
    comm_dict : dictionary
        Dictionary mapping community labels to arrays of user IDs, represented
        as integers
    """
    clusters = pd.read_csv(cluster_filepath, index_col=0, header=None)

    ground_truth = {}
    for comm_lab, node_id in comm_dict.items():
        for x in node_id:
            ground_truth[x] = comm_lab
    sorted_node_ids = []
    sorted_comm_labs = []
    for key in sorted(ground_truth.keys()):
        sorted_node_ids.append(key)
        sorted_comm_labs.append(ground_truth[key])

    for index, row in clusters.iterrows():
        ars = metrics.adjusted_rand_score(sorted_comm_labs, row)
        print("The adjusted rand score using " + index + " is " + str(ars))

def combine_signet_GM_SPM(cluster_filepath_1, cluster_filepath_2, cluster_filepath_3, gt_comm_vec, ARI_matrix_path):
    """Combines community labels from signet methods, GM, and SPM and computes ARI for labels vs ground-truth
    and all pair labels.

    Parameters
    ----------
    cluster_filepath_1 : str
        filepath to cluster labels (SigNet)
    cluster_filepath_2 : str
        filepath to cluster labels (GM)
    cluster_filepath_3 : str
        filepath to cluster labels (SPM)
    gt_comm_vec : vector
        nx2 vector containing node IDs and corresponding ground-truth labels
    ARI_matrix_path: str
        filepath to desired location for CSV output
    """
    # Add new rows to 'py_labels'

    py_labs = pd.read_csv(cluster_filepath_1, index_col=0, header=None)
    GM_labs = np.loadtxt(cluster_filepath_2, delimiter=',')
    SPM_labs = np.loadtxt(cluster_filepath_3, delimiter=',')

    py_labs.loc['GM'] = GM_labs
    py_labs.loc['SPM'] = SPM_labs

    all_labs = py_labs

    # Compute ARI

    ARI_mat = np.zeros((10,10))
    for i in np.arange(0, all_labs.shape[0], 1):
            ARI_mat[0, i+1] = metrics.adjusted_rand_score(gt_comm_vec, all_labs.iloc[[i]].values.flatten())
            for j in np.arange(i, all_labs.shape[0], 1):
                    #print(str(i) + " , " + str(j))
                    ARI_mat[i+1, j+1] = metrics.adjusted_rand_score(all_labs.iloc[[i]].values.flatten(), all_labs.iloc[[j]].values.flatten())
            ARI_mat[i, i] = 0.5

    ARI_full = ARI_mat + ARI_mat.transpose()
    ARI_full[9, 9] = 1
    ARI_pd = pd.DataFrame(ARI_full)
    ARI_pd = ARI_pd.rename(columns={0: 'ground_truth', 1: 'lap_none', 2: 'lap_sym', 3: 'lap_sym_sep', 4: 'BNC_none', 5: 'BNC_sym', 6: 'SPONGE_none', 7: 'SPONGE_sym', 8: 'GM', 9: 'SPM'},
                       index={0: 'ground_truth', 1: 'lap_none', 2: 'lap_sym', 3: 'lap_sym_sep', 4: 'BNC_none', 5: 'BNC_sym', 6: 'SPONGE_none', 7: 'SPONGE_sym', 8: 'GM', 9: 'SPM'})

    #cm = sns.light_palette("green", as_cmap=True)
    #ARI_pd.style.background_gradient(cmap=cm, axis=None)

    ARI_pd.to_csv(ARI_matrix_path)
    print("ARI matrix computation complete.  CSV results stored at "+ ARI_matrix_path + ".")

def combine_signet_GM_SPM_BCM_SSSnet(cluster_filepath_1, cluster_filepath_2, cluster_filepath_3, cluster_filepath_5, cluster_filepath_6, cluster_filepath_7, gt_comm_vec, ARI_matrix_path):
    """Combines community labels from signet methods, GM, and SPM and computes ARI for labels vs ground-truth
    and all pair labels.

    Parameters
    ----------
    cluster_filepath_1 : str
        filepath to cluster labels (SigNet)
    cluster_filepath_2 : str
        filepath to cluster labels (GM)
    cluster_filepath_3 : str
        filepath to cluster labels (SPM)
    cluster_filepath_4 : str
        filepath to cluster labels (FCSG)
    cluster_filepath_5 : str
        filepath to cluster labels (BCM_SC)
    cluster_filepath_6 : str
        filepath to cluster labels (BCM_KM)
    cluster_filepath_7 : str
        filepath to cluster labels (SSSnet)
    gt_comm_vec : vector
        nx2 vector containing node IDs and corresponding ground-truth labels
    ARI_matrix_path: str
        filepath to desired location for CSV output
    """
    # Add new rows to 'py_labels'

    py_labs = pd.read_csv(cluster_filepath_1, index_col=0, header=None)
    GM_labs = np.loadtxt(cluster_filepath_2, delimiter=',')
    SPM_labs = np.loadtxt(cluster_filepath_3, delimiter=',')
    #FCSG_labs = np.loadtxt(cluster_filepath_4, delimiter=',')
    BCM_SC_labs = np.loadtxt(cluster_filepath_5, delimiter=' ')
    BCM_KM_labs = np.loadtxt(cluster_filepath_6, delimiter=' ')
    SSSnet_labs = np.loadtxt(cluster_filepath_7, delimiter=' ')

    py_labs.loc['GM'] = GM_labs
    py_labs.loc['SPM'] = SPM_labs
    #py_labs.loc['FCSG'] = FCSG_labs
    py_labs.loc['BCM_SC'] = BCM_SC_labs
    py_labs.loc['BCM_KM'] = BCM_KM_labs
    py_labs.loc['SSSnet'] = SSSnet_labs

    all_labs = py_labs

    # Compute ARI

    ARI_mat = np.zeros((13,13))
    for i in np.arange(0, all_labs.shape[0], 1):
            ARI_mat[0, i+1] = metrics.adjusted_rand_score(gt_comm_vec, all_labs.iloc[[i]].values.flatten())
            for j in np.arange(i, all_labs.shape[0], 1):
                    #print(str(i) + " , " + str(j))
                    ARI_mat[i+1, j+1] = metrics.adjusted_rand_score(all_labs.iloc[[i]].values.flatten(), all_labs.iloc[[j]].values.flatten())
            ARI_mat[i, i] = 0.5

    ARI_full = ARI_mat + ARI_mat.transpose()
    #ARI_full[9, 9] = 1
    ARI_pd = pd.DataFrame(ARI_full)
    ARI_pd = ARI_pd.rename(columns={0: 'ground_truth', 1: 'lap_none', 2: 'lap_sym', 3: 'lap_sym_sep', 4: 'BNC_none', 5: 'BNC_sym', 6: 'SPONGE_none', 7: 'SPONGE_sym', 8: 'GM', 9: 'SPM', 10: 'BCM_SC', 11: 'BCM_KM', 12: 'SSSnet'},
                       index={0: 'ground_truth', 1: 'lap_none', 2: 'lap_sym', 3: 'lap_sym_sep', 4: 'BNC_none', 5: 'BNC_sym', 6: 'SPONGE_none', 7: 'SPONGE_sym', 8: 'GM', 9: 'SPM', 10: 'BCM_SC', 11: 'BCM_KM', 12: 'SSSnet'})

    cm = sns.light_palette("green", as_cmap=True)
    ARI_pd.style.background_gradient(cmap=cm, axis=None)

    ARI_pd.to_csv(ARI_matrix_path)
    print("ARI matrix computation complete.  CSV results stored at "+ ARI_matrix_path + ".")

def combine_signet_GM_SPM_FCSG_BCM_SSSnet(cluster_filepath_1, cluster_filepath_2, cluster_filepath_3, cluster_filepath_4, cluster_filepath_5, cluster_filepath_6, gt_comm_vec, ARI_matrix_path):
    """Combines community labels from signet methods, GM, and SPM and computes ARI for labels vs ground-truth
    and all pair labels.

    Parameters
    ----------
    cluster_filepath_1 : str
        filepath to cluster labels (SigNet)
    cluster_filepath_2 : str
        filepath to cluster labels (GM)
    cluster_filepath_3 : str
        filepath to cluster labels (SPM)
    cluster_filepath_4 : str
        filepath to cluster labels (SSSnet)
    cluster_filepath_5 : str
        filepath to cluster labels (BCM_KM)
    cluster_filepath_6 : str
        filepath to cluster labels (FCSG)
    gt_comm_vec : vector
        nx2 vector containing node IDs and corresponding ground-truth labels
    ARI_matrix_path: str
        filepath to desired location for CSV output
    """
    # Add new rows to 'py_labels'

    py_labs = pd.read_csv(cluster_filepath_1, index_col=0, header=None)
    GM_labs = np.loadtxt(cluster_filepath_2, delimiter=',')
    SPM_labs = np.loadtxt(cluster_filepath_3, delimiter=',')
    SSSnet_labs = np.loadtxt(cluster_filepath_4, delimiter=' ')
    BCM_KM_labs = np.loadtxt(cluster_filepath_5, delimiter=' ')
    FCSG_labs = np.loadtxt(cluster_filepath_6, delimiter=',')

    py_labs.loc['GM'] = GM_labs
    py_labs.loc['SPM'] = SPM_labs
    py_labs.loc['FCSG'] = FCSG_labs
    py_labs.loc['SSSnet'] = SSSnet_labs
    py_labs.loc['BCM_KM'] = BCM_KM_labs

    all_labs = py_labs

    # Compute ARI

    ARI_mat = np.zeros((13,13))
    for i in np.arange(0, all_labs.shape[0], 1):
            ARI_mat[0, i+1] = metrics.adjusted_rand_score(gt_comm_vec, all_labs.iloc[[i]].values.flatten())
            for j in np.arange(i, all_labs.shape[0], 1):
                    #print(str(i+1) + " , " + str(j+1))
                    ARI_mat[i+1, j+1] = metrics.adjusted_rand_score(all_labs.iloc[[i]].values.flatten(), all_labs.iloc[[j]].values.flatten())
            ARI_mat[i, i] = 0.5
    ARI_mat[12,12] = 0.5
    ARI_full = ARI_mat + ARI_mat.transpose()
    #ARI_full[9, 9] = 1
    ARI_pd = pd.DataFrame(ARI_full)
    ARI_pd = ARI_pd.round(2)
    ARI_pd = ARI_pd.rename(columns={0: 'ground_truth', 1: 'lap_none', 2: 'lap_sym', 3: 'lap_sym_sep', 4: 'BNC_none', 5: 'BNC_sym', 6: 'SPONGE_none', 7: 'SPONGE_sym', 8: 'GM', 9: 'SPM', 10: 'FCSG', 11: 'SSSnet', 12: 'graphB_km'},
                       index={0: 'ground_truth', 1: 'lap_none', 2: 'lap_sym', 3: 'lap_sym_sep', 4: 'BNC_none', 5: 'BNC_sym', 6: 'SPONGE_none', 7: 'SPONGE_sym', 8: 'GM', 9: 'SPM', 10: 'FCSG', 11: 'SSSnet', 12: 'graphB_km'})

    #cm = sns.light_palette("green", as_cmap=True)
    #ARI_pd.style.background_gradient(cmap=cm, axis=None)

    ARI_pd.to_csv(ARI_matrix_path)
    print("ARI matrix computation complete.  CSV results stored at "+ ARI_matrix_path + ".")
