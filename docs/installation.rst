TRANSPIRE Installation
======================

Installation from source using pip
----------------------------------

To install TRANSPIRE, clone or download the `GitHub repo`_ and install the package using ``pip install .`` from the top-level directory of the package.

.. _GitHub repo: https://github.com/mak4515/TRANSPIRE

Dependencies
~~~~~~~~~~~~

TRANSPIRE relies heavily on `Pandas`_ for data manipulation, in addition to `GPFlow`_ and `Tensorflow`_ for building and training Gaussian Process Classifiers (GPCs). 

.. _Pandas: https://pandas.pydata.org/
.. _GPFlow: https://www.gpflow.org/
.. _Tensorflow: https://www.tensorflow.org/

Given the heavy dependence on Pandas, we reccommend that users have, at least, a baseline knowledge of this package.

Hardware requirements (CPU/RAM capacities)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitting and training GPCs can become a very CPU and RAM-intensive task depending on dataset size. In the case of TRANSPIRE analysis, this will be dictated by the number of markers that your dataset has
(both in terms of number of organelles represented and the number of markers per organelle). In our experience, a standard workstation computer with reasonable RAM capacities (~16GB) has been sufficient
for handing model training and post-processing analysis.
