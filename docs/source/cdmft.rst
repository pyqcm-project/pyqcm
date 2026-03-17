Cluster Dynamical Mean Field Theory
###################################

This submodule provides functions that perform the CDMFT algorithm.
The class `CDMFT` contains the CDMFT algorithm *per se* and implements it.
Some auxiliary classes are also defined:

* `convergence_manager` defines a test for assessing convergence of the CDMFT procedure

* `frequency_grid` defines a grid of imaginary frequencies for computing the CDMFT distance function.

* `hybridization` defines the hybridization function, mostly for import/export.

* `general_bath` defines general cluster models with bath, with automatic bath parameter definitions, etc.


frequency grids
===============

All frequency grids used in CDMFT are defined along the imaginary axis. The basic grid is the set of fermionic Matsubara frequencies :math:`\omega_n` at inverse temperature ``beta`` between 0 and a maximum frequency ``wc``, and all these frequencies have the same weight. This choice is obtained by specifying ``grid_type==sharp`` in the call to ``CDMFT()`` and by specifying ``wc``and ``beta``. Other choices are:
 * ``grid_type==ifreq`` : same grid as ``sharp``, but the weights are proportional to :math:`1/\omega_n`.
 * ``grid_type==legendre``: this is a non uniform grid, defined by the tuple ``(w1,w2,n)``. A Gauss-Legendre grid is defined in the interval :math:`(0,\omega_1)` with :math:`n` points, another one in the interval :math:`(\omega_1,\omega_2)`, and a final one as a function of :math:`1/\omega` in the interval :math:`(0,1/\omega_2)`, for a total of :math:`3n` points.

In addition, the boolean option ``selfnorm`` (false by default) multiplies the weights by the norm of the self-energy matrix, giving more weight to those frequencies at which correlations are more important.

Subbaths
========

It is possible to associate more than one bath system to a given cluster. In defining the cluster, one must simply provide a liste of ``cluster_model`` objects instead of one. This feature is useless unless a bath system is defined in the ``cluster_model`` object. If :math:`N_s` cluster models are associated to a cluster (we call them *systems*), then we are faced with :math:`N_s` sets of bath parameters and as many different self-energies :math:`\Sigma_i`  and hybridization functions :math:`\Gamma_i` . The default behavior is to average these self-energies and hybridization functions when computing the *host*, from the projected Green function. This allows in principle a tighter correspondence between the host and the set of impurity models. This is newer feature (version > 2.19).


.. autoclass:: pyqcm.cdmft.CDMFT
    :members:

.. autoclass:: pyqcm.cdmft.convergence_manager
    :members:

.. autoclass:: pyqcm.cdmft.frequency_grid
    :members:

.. autoclass:: pyqcm.cdmft.hybridization
    :members:

.. autoclass:: pyqcm.cdmft.general_bath
    :members:

