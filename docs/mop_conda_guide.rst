How do I make a conda environment for MoP?
===========================================
We have found that when using MoP there can be compilation issues with some of the dependencies.
If you are having trouble installing the dependencies for MoP, the below guide will walk you
through the steps necessary to generate a conda environment that should work on a machine running
Linux.

If you do not have conda installed please check out this
`guide <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

Before running the following commands please make sure your conda is configured to use the
following channels (if you do ::

    conda config --add channels defaults
    conda config --add channels conda-forge
    conda config --add channels bioconda

To install the necessary packages for MoP, please run the following lines of code::

    conda create --name mop_env python=3.7
    conda install numpy pytables pandas
    conda install scipy scikit-learn numba
    conda install louvain python-igraph
    conda install seaborn matplotlib
    conda install umap-learn
    pip install annoy
    pip install cmake
    pip install MulticoreTSNE


