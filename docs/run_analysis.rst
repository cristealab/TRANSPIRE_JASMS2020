Steps required to run an analysis with TRANSPIRE
================================================

======================
STEP 1: Load your data
======================

Import data from a local file path
----------------------------------

The most important factor when starting to run an analysis with TRANSPIRE is that the input data is formatted correctly.
In order for TRANSPIRE to be able to load and analyze your data, it should be in an Excel, .csv, or .txt (tab separated) file in the below format:

.. csv-table:: 
   :header-rows: 3
   :file: C:\Users\Michelle\Downloads\Copy of TableS2.csv

e.g. experimental conditions and organelle fractions should be listed in separate columns across the top of the document. Each row should represent one protein and begin with its Uniprot accession number, gene name, and localization (optional, only for marker proteins)

Loading data for TRANSPIRE analysis can be accomplished by running::

    data = TRANSPIRE.data.import_data.load_data(myfilepath)


Add organelle markers from included marker sets (optional)
----------------------------------------------------------
The file that you load in may already contain custom localization annotations for organelle markers. However, if it does not, TRANPIRE can load in existing organelle marker sets for you. 

The available marker sets included in TRANSPIRE are:

* human fibroblast cells [1]_
* HEK293T cells [2]_
* Mouse embryonic stem cells [3]_
* HeLa cells [4]_
* U2OS cells [5]_

Markers can be added to the loaded dataset by running::

    marker_data = TRANSPIRE.data.import_data.add_markers(data, 'HEK293T')

Note that if your dataset already has a "localization" level this will raise an error.

=========================================
STEP 2: Generate synthetic translocations
=========================================

TRANSPIRE will generate synthetic translocations by concatenating combinations of organelle marker proteins as defined by the "localization" index level of the input data. 
To generate these translocations, first define which conditions you want to make comparisons between (as a list of tuples), then generate the synthetic translocations for model training::

    comparisons = [('control r1', 'infected r1'), ('control r2', 'infected r2')]
    synthetic_translocations = TRANSPIRE.data.generate_translocations.make_translocations(data, comparisons)

=================================================================
STEP 3: Optimize model hyperparameters (optional, but encouraged)
=================================================================

While there are many hyperparameters that can be optimized for the GPFlow stochastic variational Gaussian process (SVGP) classifier that is used by TRANSPIRE, 
we have implemented a very simple hyperparameter optimization scheme to optimize two of the hyperparameters that we have found to have the greatest impact on model performance: kernel type and number of inducing points. 
Certainly, more complex schemes for hyperparameter optimization and corresponding cross-validation exist, and can be implemented at the user's discretion.

As hyperparameter optimization is a relatively complex and time-intensive task, please see the notebook on 
:doc:`hyperparameter optimization <./notebooks/hyperparameter optimization>` 
for a tutorial and examples of how TRANSPIRE can facilitate this process. As a technical note, we have generally found that the number of inducing points tends to have a greater impact on model performance than the kernel type.

==========================================================
STEP 4: Train final model using optimized hyperparameters
==========================================================

=======================================
STEP 5: Evaluate predictive performance
=======================================

==============================
STEP 6: Predict translocations
==============================

========================================================
STEP 7: Bioinformatic analysis of translocating proteins
========================================================

Gene ontology (GO) enrichment analysis using GOATOOLS
-----------------------------------------------------

Co-translocation analysis
-------------------------




.. [1] Jean Beltran et al. 2016
    Jean Beltran, P. M.; Mathias, R. A.; Cristea, I. M. A Portrait of the Human Organelle Proteome In Space and Time during Cytomegalovirus Infection. Cell Syst. 2016, 3 (4), 361–373. https://doi.org/10.1016/j.cels.2016.08.012.

.. [2] Breckels et al. 2013
    Breckels, L. M.; Gatto, L.; Christoforou, A.; Groen, A. J.; Lilley, K. S.; Trotter, M. W. B. The Effect of Organelle Discovery upon Sub-Cellular Protein Localisation. J. Proteomics 2013, 88, 129–140. https://doi.org/10.1016/j.jprot.2013.02.019.

.. [3] Christoforou et al. 2016
    Christoforou, A.; Mulvey, C. M.; Breckels, L. M.; Geladaki, A.; Hurrell, T.; Hayward, P. C.; Naake, T.; Gatto, L.; Viner, R.; Arias, A. M.; Lilley, K. S. A Draft Map of the Mouse Pluripotent Stem Cell Spatial Proteome. Nat. Commun. 2016, 7, 9992. https://doi.org/10.1038/ncomms9992.

.. [4] Itzhak et al. 2016
    Itzhak, D. N.; Tyanova, S.; Cox, J.; Borner, G. H. Global, Quantitative and Dynamic Mapping of Protein Subcellular Localization. Elife 2016, 5 (JUN2016). https://doi.org/10.7554/eLife.16950.

.. [5] Thul et al. 2017
    Thul, P. J.; Akesson, L.; Wiking, M.; Mahdessian, D.; Geladaki, A.; Ait Blal, H.; Alm, T.; Asplund, A.; Björk, L.; Breckels, L. M.; Bäckström, A.; Danielsson, F.; Fagerberg, L.; Fall, J.; 
    Gatto, L.; Gnann, C.; Hober, S.; Hjelmare, M.; Johansson, F.; Lee, S.; Lindskog, C.; Mulder, J.; Mulvey, C. M.; Nilsson, P.; Oksvold, P.; Rockberg, J.; Schutten, R.; Schwenk, J. M.; 
    Sivertsson, A.; Sjöstedt, E.; Skogs, M.; Stadler, C.; Sullivan, D. P.; Tegel, H.; Winsnes, C.; Zhang, C.; Zwahlen, M.; Mardinoglu, A.; Pontén, F.; Von Feilitzen, K.; Lilley, K. S.; Uhlén, M.; Lundberg, E. 
    A Subcellular Map of the Human Proteome. Science (80-. ). 2017, 356 (6340), eaal3321. https://doi.org/10.1126/science.aal3321.
