Welcome to the documentation for TRANSPIRE
===========================================

TRANSPIRE is a Python package for TRanslocation ANalysis of SPatIal pRotEomics data.

==========
QUICKSTART
==========

Installation and depedencies
----------------------------
To install TRANSPIRE, clone or download the `GitHub repo`_ and install the package using ``pip install .`` from the top-level directory of the package.

.. _GitHub repo: https://github.com/mak4515/TRANSPIRE

Dependencies
~~~~~~~~~~~~
TRANSPIRE relies heavily on `Pandas`_ for data manipulation, in addition to `GPFlow`_ and `Tensorflow`_ for building and training Gaussian Process Classifiers. 

.. _Pandas: https://pandas.pydata.org/
.. _GPFlow: https://www.gpflow.org/
.. _Tensorflow: https://www.tensorflow.org/

Given the heavy dependence on Pandas, we recommend that users have, at least, a baseline knowledge of this package.


======================
Documentation contents
======================


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   installation
   about_transpire
   run_analysis
   
.. toctree::
   :maxdepth: 0
   :glob:
   :caption: Examples and tutorials:

   notebooks/importing and manipulating data
   notebooks/hyperparameter optimization
   notebooks/final model fitting and evaluation
   notebooks/post-processing (GO analysis, co-translocation analysis, etc.)
   
.. toctree::
   :maxdepth: 3
   :glob:
   :caption: API:

   api_index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

