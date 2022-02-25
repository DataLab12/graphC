# graphC

**graphC** algorithmm implements the wrapper for the state-of-art (2022) community discovery methods, and measures ARI of the methods on sample data. 

** Automated run on sample data** 
```
>>python run.py
```

The wrapper calls ```graphC.py``` and creates the ARI table for Highland Tribes data. All the Highland signed graph clustering methods are run and outputs saved in the subfolders. 

**Follow [SETUP.md](SETUP.md)** instructions on how to create the ARI table for different dataset. 

# Citation

Please cite this publication:  Tomasso, M., Rusnak, L., Tešić, J. [_Advances in Scaling Community Discovery Methods for Large Signed Graph Networks_](https://arxiv.org/abs/2110.07514). arXiv (2022). 

 **BibTeX entry:**
 
```
@article{2022Survey,
  author    = {Maria Tomasso and Lucas Rusnak and Jelena Tesic},
  title     = {Advances in Scaling Community Discovery Methods for Large Signed Graph Networks},
  journal   = {CoRR},
  volume    = {abs/2110.07514},
  year      = {2022},
  url       = {https://arxiv.org/abs/2110.07514}
}
```

# Advances in Scaling Community Discovery Methods for Large Signed Graph Networks

**Authors:** Maria Tomasso, Lucas Rusnak, Jelena Tešić

**Abstract**
Community detection is a common task in social network analysis (SNA) with applications in a variety of fields including medicine, criminology, and business. Despite the popularity of community detection, there is no clear consensus on the most effective methodology for signed networks. In this paper, we summarize the development of community detection in signed networks and evaluate current state-of-the-art techniques on several real-world data sets. First, we give a comprehensive background of community detection in signed graphs. Next, we compare various adaptations of the Laplacian matrix in recovering ground-truth community labels via spectral clustering in small signed graph data sets. Then, we evaluate the scalability of leading algorithms on small, large, dense, and sparse real-world signed graph networks. We conclude with a discussion of our novel findings and recommendations for extensions and improvements in state-of-the-art techniques for signed graph community discovery in large, sparse, real-world signed graphs.

[Paper](https://arxiv.org/abs/2110.07514)

[Data Lab @ TXST](DataLab12.github.io)


