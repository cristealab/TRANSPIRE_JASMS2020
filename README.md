# TRANSPIRE_JASMS2020
**TR**anslocation **AN**alysis for **SP**at**I**al p**R**ot**E**omics

***

## QUICKSTART

#### Installation and depedencies

To install TRANSPIRE, clone or download the [GitHub repository](https://github.com/cristealab/TRANSPIRE_JASMS2020) and install the package using ``pip install .`` from the top-level directory of the package.

**Dependencies**

TRANSPIRE relies heavily on [Pandas]( https://pandas.pydata.org/) for data manipulation, in addition to [GPFlow](https://www.gpflow.org/)[1] and [Tensorflow](https://www.tensorflow.org/)[2] for building and training Gaussian Process Classifiers (GPCs). 

Given the heavy dependence on Pandas, we recommend that users have, at least, a baseline knowledge of this package.

**Hardware requirements (CPU/RAM capacities)**

Fitting and training GPCs can become a very CPU and RAM-intensive task depending on dataset size. In the case of TRANSPIRE analysis, this will be dictated by the number of markers that your dataset has
(both in terms of number of organelles represented and the number of markers per organelle). In our experience, a standard workstation computer with reasonable RAM capacities (~16GB) has been sufficient
for handing model training and post-processing analysis.

### Documentation
TRANSPIRE is documented using [Sphinx](https://www.sphinx-doc.org/en/master/) and its documention is hosted on [Read the Docs](https://transpire.readthedocs.io/en/latest/).

### Tutorials and examples
See the provided [Jupyter notebooks](/docs/notebooks) for examples and tutorials of how to perform and analysis using TRANSPIRE. These notebooks are also available in the documentation on Read the Docs.

### Essential references
[1]: De, A. G.; Matthews, G.; Nickson, T.; Fujii, K.; Boukouvalas, A.; León-Villagrá, P.; Ghahramani, Z.; Hensman, J. GPflow: A Gaussian Process Library Using TensorFlow Mark van Der Wilk; 2017; Vol. 18.

[2]: Abadi, M.; Agarwal, A.; Barham, P.; Brevdo, E.; Chen, Z.; Citro, C.; Corrado, G. S.; Davis, A.; Dean, J.; Devin, M.; Ghemawat, S.; Goodfellow, I.; Harp, A.; Irving, G.; Isard, M.; Jia, Y.; Jozefowicz, R.; Kaiser, L.; Kudlur, M.; Levenberg, J.; Mane, D.; Monga, R.; Moore, S.; Murray, D.; Olah, C.; Schuster, M.; Shlens, J.; Steiner, B.; Sutskever, I.; Talwar, K.; Tucker, P.; Vanhoucke, V.; Vasudevan, V.; Viegas, F.; Vinyals, O.; Warden, P.; Wattenberg, M.; Wicke, M.; Yu, Y.; Zheng, X. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. 2016.


## Citing TRANSPIRE
If you find TRANSPIRE useful in your research, please cite its publication in JASMS:

Kennedy, M. A.; Hofstadter, W. A.; Cristea, I. M. TRANSPIRE: A Computational Pipeline to Elucidate Intracellular Protein Movements from Spatial Proteomics Datasets. J. Am. Soc. Mass Spectrom. 2020, jasms.0c00033. https://doi.org/10.1021/jasms.0c00033.
