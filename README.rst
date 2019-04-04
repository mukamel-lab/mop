MoP
================

The Multiomics Processing (MoP) package is a collection of functions and wrappers
used by the Mukamel group for the analysis of transcriptomic and epigenomic data. At the
moment MoP is just a collection of functions, but over time it will be packaged into a more
user-friendly configuration. The majority of these functions are designed to work with
data stored in the `loom <http://loompy.org/>`_ file format.

Requirements
------------
* python 3
* loompy
* numpy
* pandas
* scipy
* scikit-learn
* annoy
* fitsne
* louvain
* python-igraph

Usage
-----
Coming at some point in the future.

Installation
------------
The only way to install MoP is by cloning the GitHub repository. Enter the directory
where you would like to download the repository and enter the following commands on
the command line::

    git clone https://github.com/mukamel-lab/mop.git
    cd mop
    python setup.py install

There are some reported issues when running setup.py install related to the installation
of some dependencies. If you are using a conda environment the best way to solve these
issues is to perform the following commands before running setup.py install::


    conda install -c conda-forge gcc cython numpy fftw louvain python-igraph
    pip install --user annoy
    pip install fitsne
    pip install --user loompy


If you are continuing to have trouble installing the dependencies for MoP, a guide to creating
a MoP-specific conda environment can be found `here <docs/mop_conda_guide.rst>`_.

Authors
-------
MoP was developed by `Wayne Doyle <widoyle@ucsd.edu>`_, `Fangming Xie <f7xie@ucsd.edu>`_,
and `Ethan Armand <earmand@ucsd.edu>`_ in the `Mukamel Lab <https://brainome.ucsd.edu/>`_.


Acknowledgments
----------------
We are grateful for support from the Chan-Zuckerberg Initiative and from the NIH
BRAIN Initiative U19 Center for Epigenomics of the Mouse Brain Atlas
(`CEMBA <https://biccn.org/teams/u19-ecker/>`_).

