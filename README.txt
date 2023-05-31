Pyqcm is a Python/C++ library that implements a few quantum cluster methods with an exact diagonalization impurity solver. Quantum cluster methods are used in the study of strongly correlated electrons to provide an approximate solution to Hubbard-like models. The methods covered by this library are Cluster Perturbation Theory (CPT), the Variational Cluster Approach (VCA) and Cellular (or Cluster) Dynamical Mean Field Theory (CDMFT).
The impurity solver (the technique used to compute the cluster's interacting Green function) is exact diagonalization from sparse matrices, using the Lanczos algorithm and variants thereof. 
The core library is written in C++ for performance, but the interface is in Python, for ease of use and inter-operability with the numerical Python ecosystem. 
The library is distributed under the GPL license.

A comprehensive paper describing the different methods and version 2.1 of the software is available at:

https://arxiv.org/abs/2305.18643

EXAMPLES can be found in the notebooks/ folder, either in plain python files (*.py) or in the form of Jupyter notebooks (*.ipynb)

Please see the INSTALL file for installation instructions.

Dependencies:
- LAPACK-BLAS or the equivalent
- The CUBA library (https://feynarts.de/cuba/)
- Python3.7 or more recent, with Numpy, Scipy and matplotlib
- Optional : eigen3 (https://eigen.tuxfamily.org)
- Optional : PRIMME eigensolver (https://github.com/primme/)



Questions and comments can be addressed to:

david.senechal@usherbrooke.ca

