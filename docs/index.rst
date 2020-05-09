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

Given the heavy dependence on Pandas, we reccommend that users have, at least, a baseline knowledge of this package.

To learn how to perform translocation analysis with TRANSPIRE, see our :doc:`notebooks <./notebooks_file>` for examples and tutorials.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   about_transpire
   run_analysis
   
.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Examples and tutorials:

   notebooks_file
   
.. toctree::
   :maxdepth: 3
   :glob:
   :caption: API:

   api_index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
