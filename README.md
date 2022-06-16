# graphC

**graphC** algorithmm implements the wrapper for the state-of-art (2022) community discovery methods, and measures ARI of the methods on sample data. 

** Automated run on sample data** 
```
>>python run.py
```

The wrapper calls ```graphC.py``` and creates the ARI table for Highland Tribes data. All the Highland signed graph clustering methods are run and outputs saved in the subfolders. 

**Follow [SETUP.md](SETUP.md)** instructions on how to create the ARI table for different dataset. 

# Acknowledgments 

Please cite the following publication: **Tomasso, M., Rusnak, L., Tešić, J. [_Advances in Scaling Community Discovery Methods for Signed Graph Networks](https://academic.oup.com/comnet/article-abstract/doi/10.1093/comnet/cnac013/6608828), Journal of Complex Networks, Volume 10, Issue 3, June 2022, cnac013, https://doi.org/10.1093/comnet/cnac013.**

**BibTeX entry:**
```
article{10.1093/comnet/cnac013,
    author = {Tomasso, Maria and Rusnak, Lucas J and Tešić, Jelena},
    title = "{Advances in scaling community discovery methods for signed graph networks}",
    journal = {Journal of Complex Networks},
    volume = {10},
    number = {3},
    year = {2022},
    month = {06},
    issn = {2051-1329},
    doi = {10.1093/comnet/cnac013},
    url = {https://doi.org/10.1093/comnet/cnac013},
    note = {cnac013},
    eprint = {https://academic.oup.com/comnet/article-pdf/10/3/cnac013/44082209/cnac013.pdf},
}
```
[Full Paper](https://arxiv.org/abs/2110.07514)
**Abstract**
Community detection is a common task in social network analysis with applications in a variety of fields including medicine, criminology and business. Despite the popularity of community detection, there is no clear consensus on the most effective methodology for signed networks. In this article, we summarize the development of community detection in signed networks and evaluate current state-of-the-art techniques on several real-world datasets. First, we give a comprehensive background of community detection in signed graphs. Next, we compare various adaptations of the Laplacian matrix in recovering ground-truth community labels via spectral clustering in small signed graph datasets. Then, we evaluate the scalability of leading algorithms on small, large, dense and sparse real-world signed graph networks. We conclude with a discussion of our novel findings and recommendations for extensions and improvements in state-of-the-art techniques for signed graph community discovery in real-world signed graphs.

[Data Lab @ TXST](DataLab12.github.io)


