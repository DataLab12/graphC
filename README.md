# GraphC: Parameter-Free Optimal Hierarchical Clustering of Signed Graphs

Authors: <em> Muhieddine Shebaro, Martin Burtscher, Lucas Rusnak, Jelena Tešić </em>

![Highland Tribes Execution!](/images/animate.gif "Highland Tribes Clustering")

**graphC** (2024) is a scalable state-of-the-art hierarchical clustering algorithm for signed graphs capable of automatically detecting clusters without a predefined K hyperparameter (number of communities), no matrices, no decomposition spectral solvers, and no ground-truth labels. The algorithm is implemented in C++ and employs an efficient fundamental cycle basis discovery method to balance a connected component, performs Harary cuts, and selects the most optimal split based on the following quality criteria:

![image](https://github.com/DataLab12/graphC/assets/95373719/0c5e23a1-92cb-47f5-bc39-7070016b7d8e =5x20)

𝑝𝑜𝑠_𝑜𝑢𝑡 and 𝑛𝑒𝑔_𝑖𝑛  are normalized, thus resolving the positive–negative imbalance of edge signs. This will ensure that the negative edges are placed between clusters in a priority equal to positive edges between placed within clusters

## Pipeline
![GraphC:Pipeline!](/images/pipeline1.png "GraphC: Pipeline")

1. After a Harary split, the algorithm is going to iterate through newly obtained and old connected components and try to split these CC using the best Harary cuts over and over. 
2. Harary Cut is followed by DFS algorithm to form connected components with their edge signs restored.

3. The algorithm will stop splitting a specific CC if one of the conditions is satisfied:
* Current_clock >= Clock_limit → The entire program will halt, and labels are returned. (-1: Unlimited time)
* CC_size <= γ   → This is mainly for computational efficiency and scalability where algorithm won’t even try and balance a CC and find its Harary cut if it’s small under the assumption that it will yield minimal performance gains. The algorithm will skip it.
* Overall_Loss_Current – Overall_Loss_Previous < ε  → The algorithm will add the CC that had been most recently split into a set “processed”. Any CC in that set won’t be processed and split further because it doesn’t improve the overall loss even if Harary split is performed in it. The split done to this CC will be reversed because it worsens performance.


## Citation
Please cite the following publication: TBD

**BibTeX entry:**
```
TBD
```

## How to Run the Code 

* Simply download index.cpp and GraphBplus_Harary.cpp and compile the former source file like the following:

```
user:~$ g++ -fopenmp index.cpp
```
The signed graph must be in the following format to be compatible with graphC (src,dst,sign).
Preprocessing of the signed graph is embedded and neutral edges are treated as positive.

* To execute the compiled file, graphC utilizes 6 parameters in this order (iteration_count 𝛼 𝛽 ε time_limit γ):
```
user:~$ ./a.out input.txt 0.5 1 0.000001 -1 2
```
You can input -1 in time_limit to allow the algorithm to run the algorithm until it's finished. Minimum of γ is 2. Range of 𝛼 and 𝛽 is [0,1]. Range of ε is [0, infinity].
 
* graphC outputs 2 .txt files:

1. *_labels.txt: This contains the assigned clustering labels for each node (original node ID).
2. *_posin_negout.txt: This contains the history of change of the fraction of positive edges within communities and fraction of negative edges between communities including the overall improvement and unhappy ratio after each Harary split.

**Note:** Do <em> not</em> change the name of "GraphBplus_Harary.cpp" file. And if you run into errors related to the stack memory please run this command before executing the code:
```
user:~$ ulimit -s unlimited
```

[Data Lab @ TXST](DataLab12.github.io)


