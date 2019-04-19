FAQs
=====

How do I install all of the dependencies?
------------------------------------------
We have a guide to generating a conda environment for use with
MoP `here <./mop_conda_guide.rst>`_.

Why is my code using so many threads?
--------------------------------------
This is a problem with Numpy (see this `issue <https://github.com/numpy/numpy/issues/11826>`_). You can solve this
issue in two ways.

The easiest way is to run the following lines on your command line before running any Python scripts or notebooks::

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

You can also run the below code in the first cell of a Python notebook or the beginning or a Python script. It must
be run before importing any other packages (including SCF)::

    import os
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["OPENBLAS_NUM_THREADS"] = '1'
    os.environ["MKL_NUM_THREADS"] = '1'
    os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
    os.environ["NUMEXPR_NUM_THREADS"] = '1'

You can specify the maximum number of threads that you want to use in that script or notebook by changing the value
from 1 to your desired integer.

This information came from this `StackOverflow question
<https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy>`_.
