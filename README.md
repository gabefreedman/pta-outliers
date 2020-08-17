# pta-outliers
pta-outliers is based on, and largely replicates, the pulsar timing Bayesian-inference pipeline of 
piccard (https://github.com/vhaasteren/piccard).

This code extracts from piccard the hmcLikelihood object 
used for performing single-pulsar outlier analysis (described in detail in arXiv:1609.02144), and restructures it to utilize updated data structures 
from the NANOGrav ENTERPRISE pulsar timing analysis suite.

The outlier analysis itself is performed by running the accompanying Jupyter notebook, which is an exact copy of the outlier analysis notebook 
found in piccard save for replacing piccard's likelihood object with the one developed in this code. Work is in progress to move the outlier analysis 
to a separate module, and allow the user to either run the notebook or import the functions directly.

## Requirements

* Python 3+
* numpy
* scipy
* [libstempo](https://github.com/vallis/libstempo)
* [enterprise](https://github.com/nanograv/enterprise)
