# Pyqcm

Pyqcm is a Python/C++ library that implements a few quantum cluster methods with an exact diagonalization impurity solver. 
Quantum cluster methods are used in the study of strongly correlated electrons to provide an approximate solution to Hubbard-like models.
The methods covered by this library are Cluster Perturbation Theory (CPT), the Variational Cluster Approach (VCA) and Cellular (or Cluster) Dynamical Mean Field Theory (CDMFT).
The impurity solver (the technique used to compute the cluster's interacting Green function) is exact diagonalization from sparse matrices, using the Lanczos algorithm and variants thereof. 
The core library is written in C++ for performance, but the interface is in Python, for ease of use and inter-operability with the numerical Python ecosystem. 
The library is distributed under the GPL license.

A comprehensive paper describing the different methods and version 2.1 of the software is available on [arXiv repository](https://arxiv.org/abs/2305.18643).


## Installation

Pyqcm depends on a BLAS-LAPACK implementation available on your system (such as OpenBLAS) and the Eigen template librairie.
These dependencies can be installed using the `apt` package manager on Debian system as (Ubuntu):

```
sudo apt install libopenblas-dev libeigen3-dev
```

Once installed, Pyqcm can be installed using the `pip` package manager as:

```
pip install pyqcm
```

Note only sources distribution are available and compilation is carried on the target system.
This is for enabling the most appropriate optimizations at building time for faster running time.
By default, a build-capability build of Pyqcm is carried.
Pyqcm is compatible (i.e. have been tested) on various Linux distribution on x86 computers (Intel and AMD) and Apple ARM processors.
Finely tuned installation can be acheived by specifiyng the environment variable `CMAKE_ARGS` before to install Pyqcm via `pip` (see the section [Advanced installation](https://qcm-wed.readthedocs.io/en/latest/intro.html#advanced-installation) on the documentation).


## Examples and documentation

Pyqcm documentation is hosted on ReadTheDocs and is available at [https://qcm-wed.readthedocs.io/](https://qcm-wed.readthedocs.io/en/latest/).

EXAMPLES can be found in the notebooks/ folder, either in plain python files (`*.py`) or in the form of Jupyter notebooks (`*.ipynb`).


## Contact

Questions and comments can be addressed to:

```
david.senechal@usherbrooke.ca
```
