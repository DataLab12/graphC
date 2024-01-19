addpath(genpath('/Users/maria/Desktop/Software/GM/clustering_signed_networks_with_geometric_mean_of_Laplacians'));
addpath(genpath('/Users/maria/Desktop/Software/GM/utils'))
addpath(genpath('/Users/maria/Desktop/Software/SPM/utils'))
addpath(genpath('/Users/maria/Desktop/Software/SPM/subroutines'))
addpath(genpath('/Users/maria/Desktop/Software/SPM/SpectralClusteringOfSignedGraphsViaMatrixPowerMeans'))

%% Highland Tribes: K=2
%{
Wpos = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/highland/highland_tribes_positive_edges_adj.csv'));
Wneg = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/highland/highland_tribes_negative_edges_adj.csv'));
Wcell = {Wpos, Wneg}
numClusters = 2;

HT_GM_k2 = clustering_signed_networks_with_geometric_mean_of_Laplacians(Wpos, Wneg, numClusters);
HT_GM_k2 = transpose(HT_GM_k2);
writematrix(HT_GM_k2,'/Users/maria/Desktop/CS7387_Spring2021/data/highland/HT_GM_k2.csv');

HT_SPM_k2 = clustering_signed_graphs_with_power_mean_laplacian(Wcell, 1, numClusters);
HT_SPM_k2 = transpose(HT_SPM_k2);
writematrix(HT_SPM_k2,'/Users/maria/Desktop/CS7387_Spring2021/data/highland/HT_SPM_k2.csv');

%% Highland Tribes: K=3

numClusters = 3;

HT_GM_k3 = clustering_signed_networks_with_geometric_mean_of_Laplacians(Wpos, Wneg, numClusters);
HT_GM_k3 = transpose(HT_GM_k3);
writematrix(HT_GM_k3,'/Users/maria/Desktop/CS7387_Spring2021/data/highland/HT_GM_k3.csv'); 

HT_SPM_k3 = clustering_signed_graphs_with_power_mean_laplacian(Wcell, 1, numClusters);
HT_SPM_k3 = transpose(HT_SPM_k3);
writematrix(HT_SPM_k3,'/Users/maria/Desktop/CS7387_Spring2021/data/highland/HT_SPM_k3.csv');

%% Sampson: k=4

Wpos = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/sampson/adjacency_matrices/sampson_positive_edges_undir_adj.csv'));
Wneg = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/sampson/adjacency_matrices/sampson_negative_edges_undir_adj.csv'));
Wcell = {Wpos, Wneg}
numClusters = 4;

sampson_GM_k4 = clustering_signed_networks_with_geometric_mean_of_Laplacians(Wpos, Wneg, numClusters);
sampson_GM_k4 = transpose(sampson_GM_k4);
writematrix(sampson_GM_k4,'/Users/maria/Desktop/CS7387_Spring2021/data/sampson/sampson_GM_k4.csv'); 

sampson_SPM_k4 = clustering_signed_graphs_with_power_mean_laplacian(Wcell, 1, numClusters);
sampson_SPM_k4 = transpose(sampson_SPM_k4);
writematrix(sampson_SPM_k4,'/Users/maria/Desktop/CS7387_Spring2021/data/sampson/sampson_SPM_k4.csv');

%% football: k=20

Wpos = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/football/pos_adj.csv'));
Wneg = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/football/neg_adj.csv'));
Wcell = {Wpos, Wneg}
numClusters = 20;

football_GM_k20 = clustering_signed_networks_with_geometric_mean_of_Laplacians(Wpos, Wneg, numClusters);
football_GM_k20 = transpose(football_GM_k20);
writematrix(football_GM_k20,'/Users/maria/Desktop/CS7387_Spring2021/data/football/football_GM_k20.csv'); 

football_SPM_k20 = clustering_signed_graphs_with_power_mean_laplacian(Wcell, 1, numClusters);
football_SPM_k20 = transpose(football_SPM_k20);
writematrix(football_SPM_k20,'/Users/maria/Desktop/CS7387_Spring2021/data/football/football_SPM_k20.csv');

%% olympics: k=28

Wpos = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/olympics/pos_adj.csv'));
Wneg = sparse(readmatrix('/Users/maria/Desktop/CS7387_Spring2021/data/olympics/neg_adj.csv'));
Wcell = {Wpos, Wneg}
numClusters = 28;

olympics_GM_k28 = clustering_signed_networks_with_geometric_mean_of_Laplacians(Wpos, Wneg, numClusters);
olympics_GM_k28 = transpose(olympics_GM_k28);
writematrix(olympics_GM_k28,'/Users/maria/Desktop/CS7387_Spring2021/data/olympics/olympics_GM_k28.csv'); 

olympics_SPM_k28 = clustering_signed_graphs_with_power_mean_laplacian(Wcell, 1, numClusters);
olympics_SPM_k28 = transpose(olympics_SPM_k28);
writematrix(olympics_SPM_k28,'/Users/maria/Desktop/CS7387_Spring2021/data/olympics/olympics_SPM_k28.csv');
%}

%% CoW: k=3

Wpos = sparse(readmatrix('/Users/maria/Desktop/data-repos_datalab/graphC/input_data/data-cow/cow_pos_adj.csv'));
Wneg = sparse(readmatrix('/Users/maria/Desktop/data-repos_datalab/graphC/input_data/data-cow/cow_neg_adj.csv'));
Wcell = {Wpos, Wneg}
numClusters = 3;

cow_GM_3 = clustering_signed_networks_with_geometric_mean_of_Laplacians(Wpos, Wneg, numClusters);
cow_GM_k3 = transpose(cow_GM_k3);
writematrix(cow_GM_k3,'/Users/maria/Desktop/data-repos_datalab/graphC/output_data/results-cow/cow_GM_k3.csv'); 

cow_SPM_k3 = clustering_signed_graphs_with_power_mean_laplacian(Wcell, 1, numClusters);
cow_SPM_k3 = transpose(cow_SPM_k3);
writematrix(cow_SPM_k3,'/Users/maria/Desktop/data-repos_datalab/graphC/output_data/results-cow/cow_SPM_k3.csv');
