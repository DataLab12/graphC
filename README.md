# graphC

The code implements comparison presented in  [Advances in Scaling Community Discovery Methods for Signed Graph Networks](https://academic.oup.com/comnet/article-abstract/doi/10.1093/comnet/cnac013/6608828) journal paper. 

* [Full Paper](https://arxiv.org/abs/2110.07514) on arXiv. 

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

## How to Run the Code 
** Automated run on sample data** 
```
>>python run.py
```
The wrapper calls ```graphC.py``` and creates the ARI table for Highland Tribes data. All the Highland signed graph clustering methods are run and outputs saved in the subfolders. Follow [SETUP.md](SETUP.md)** instructions on how to create the ARI table for different dataset. 

[Data Lab @ TXST](DataLab12.github.io)


