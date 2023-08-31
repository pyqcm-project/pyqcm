############
Introduction
############

What is pyqcm?
==============

pyqcm is a python module that interfaces with a library written in C++: **qcm**.
This library provide a collection of functions that help implement quantum cluster methods.
Specifically, it provides an exact diagonalization solver for small clusters on which a Hubbard-like model is defined and provides functions to define infinite-lattice models and to embed the clusters into the lattice via *Cluster Perturbation Theory* (CPT). Methods like the *Variational Cluster Approximation* (VCA) and *Cluster Dynamical Mean Field Theory* (CDMFT) are then implemented from qcm by the pyqcm module, which is written in Python only.

This document is not a review of the above methods; the reader is referred to the appropriate review articles for that. It is strictly a user's manual for the pyqcm module.
Some degree of understanding of the above methods is however necessary to proceed.
This document also provides insights as to the inner working of the code, and therefore consitutes also a embryonic developer's manual.


Installation
============

Dependencies
------------

Pyqcm is written in Python but required the compilation of its core C++ library
(**qcm**) provided with the distribution, and depends on **CUBA** for numerical
integration, **BLAS-LAPACK** for elementary (non sparse) linear algebra, and 
optionally, the **Eigen** for high performance sparse linear algebra (although 
qcm implements one, it is not as performant on many core system) and the 
**PRIMME** eigensolver.


Minimal installation
--------------------

These instructions are for installing Pyqcm, its core library, and automatically
downloading and compiling the numerical integration library CUBA.

The source code can be cloned with the following command::

    git clone https://dsenech@bitbucket.org/dsenechQCM/qcm_wed.git

Compiling can be done with pip::

    cd <path_to_qcm_wed_folder>
    pip install .

The previous instructions, however, does not install an optimized executable and
can be significantly slower. For state of the art performance, see the Advanced
installation instruction below.


Advanced installation
---------------------

These instruction allow to finely tune the configuration of Pyqcm and its optional dependencies. These options are set before installing pyqcm with pip and are configurable through the CMAKE_ARGS environment variable such as:
export CMAKE_ARGS="[BUILD_ARG1]=[VALUE1] [BUILD_ARG2]=[VALUE2] ..."

Optional build arguments and their values include:

* ``-DDOWNLOAD_CUBA=0/1``: Specify to download and compile automatically the CUBA integration library (default not downloaded: ``0``).
* ``-DCUBA_DIR=[path_to_CUBA_root_dir]``: If CUBA not downloaded from above, specify the path to CUBA directory for linking Pyqcm against (must contain compiled Cuba library ``libcuba.a`` along with the header ``cuba.h``, empty by default).
* ``-DBLA_VENDOR=[value]``: BLAS implementation to use. See `CMake vendor documentation <https://cmake.org/cmake/help/latest/module/FindBLAS.html?highlight=bla_vendor#blas-lapack-vendors>`_ for more information (recommended: do not specify or ``FlexiBLAS`` on `Alliance cluster <https://docs.alliancecan.ca/wiki/BLAS_and_LAPACK>`_).
* ``-DEIGEN_HAMILTONIAN=0/1``: Specify to compile with Eigen format for the Hamiltonian for better performance in the diagonalisation solver on multi-core machine (require `Eigen library <https://eigen.tuxfamily.org/index.php?title=Main_Page>`_, enabled by default).
* ``-DWITH_PRIMME=0/1``: Whether to use or not the PRIMME library and its eigensolver for finding ground state of the Hamiltonian (needs ``-DEIGEN_HAMILTONIAN=1``, default not used).
* ``-DDOWNLOAD_PRIMME=0/1``: Specify to download and compile automatically the PRIMME eigensolver library (needs ``-DWITH_PRIMME=1``, not downloaded by default).
* ``-DPRIMME_DIR=[path_to_PRIMME_root_dir]``: Specify the path to the PRIMME root directory for linking qcm_wed library against if not download automatically (default no set).
* ``-DWITH_GF_OPT_KERNEL=0/1``: If true, automatically detect host processor AVX2 capabilities (i.e. Intel processors after 2014 and AMD processors after 2015) and compile the optimized (5 times quicker) kernel for Green function evaluation accordingly (default yes).
* ``-DQCM_DEBUG=0/1``: Enable extra verbose information for debug purpose (default no).
* ``-DCMAKE_C_COMPILER=[gcc/icc/clang/...]``: The C compiler to use to compile qcm and its automatically download dependencies.
* ``-DCMAKE_CXX_COMPILER=[g++/icpc/clang++/...]``: The C++ compiler to use to compile qcm and its automatically download dependencies (must match the C compiler, for example ``gcc`` with ``g++``).
* ``-DAPPLE_LIBOMP_PREFIX=[path]``: Custom path to the OpenMP library ``libomp`` on Apple computer.
* ``-DUSE_OPENMP=0/1``: Enable parallelization using OpenMP (default yes).

For a full-capability build, use::

    export CMAKE_ARGS="-DDOWNLOAD_CUBA=1 -DEIGEN_HAMILTONIAN=1 -DWITH_PRIMME=1 -DDOWNLOAD_PRIMME=1 -DWITH_GF_OPT_KERNEL=1 -DUSE_OPENMP=1"

Lastly, compile with::

    pip install .

See also the file ``INSTALL`` in the root of pyqcm folder which contains more information.
