


### Data Creation
All needed data is already generated, but you can verify the labelings using the following steps:
- Geometric Means Laplacian Clustering (GM) - https://github.com/melopeo/GM
    - Clone the git repo then use the mercado_methods.m script with the appropriate filepaths added and the adjacency matrices from the *Input* folder.
- Matrix Power Means Clustering (SPM) - https://github.com/melopeo/SPM
    - Clone the git repo then Use the mercado_methods.m script with the appropriate filepaths added and the adjacency matrices from the *Input* folder.
- SSSnet (SSSNET) - https://github.com/SherylHYX/SSSNET_Signed_Clustering
    - Clone the repo and following the installation instructions on the SSSnet git. Use the numpy object file in *Input* as input.  The following definition must be added to the 'preprocess.py' script of the SSSnet package with an appropriately updated filepath:
    ~~~
    def load_highland():
        A = np.load(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../data/highland_dl/highland_dl.npy'))
        N = A.shape[0]
        A_p = np.maximum(A,np.zeros((N,N)))
        A_n = np.maximum(-A,np.zeros((N,N)))
        A_p = sp.csr_matrix(A_p)
        A_n = sp.csr_matrix(A_n)
        labels = np.array([1, 1, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 2, 2, 1, 1])
        return A_p, A_n, labels
    ~~~
    
    Additionally, the following must be added to the appropriate place in the load_data() definition of preprocess.py:
    ~~~
    elif args.dataset[:-1].lower() ==  'highland_dl':
            A_p, A_n, labels = load_highland()
    ~~~
    Finally from the command line of the *SSSnet_Signed_Clustering/src* directory run: 
    ~~~
    python train.py --dataset highland_dl -SP --feature_options None --seed_ratio 0.5 --test_ratio 0.5 --no_validation --samples 200 --num_trials 10 --seeds 31 --epochs 80
    ~~~  
    The results will be available in the *SSSnet_Signed_Clustering/result_arrays/highland_dl/* folder.
- graphB clustering (graphB_KM) - https://github.com/DataLab12/graphB 
- Fast Clustering on Signed Graphs (FCSG)
    - Navigate to the *FCSG* folder in the command line, and run the following with appropriate filepaths in place of the asterisks:
    ~~~
    python RWG.py 5  /*/GraphC-highland_demo/Input/HT_pos_edges_adj.csv /*/GraphC-highland_demo/Input/HT_neg_edges_adj.csv highland ' '
    ~~~
    - Next, run the following to generate the labels:
    ~~~
    python RWG_postprocess_highland.py
    ~~~
### ARI Table Generation
From the command line, run:
~~~
python GraphCrun.py
~~~

Note that the SigNet function *lap_sym_sep* appears to have a convergence issue on this dataset.  During testing, ARI values of 0.26, 0.43, and 0.45 were observed when comparing the *lap_sym_sep* labels with ground-truth.

