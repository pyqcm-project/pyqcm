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

All frequency grids used in CDMFT are defined along the imaginary axis. The grid is built by passing a ``frequency_grid`` instance to ``CDMFT()`` via its ``grid`` argument; if none is supplied a default grid is constructed. The ``frequency_grid`` constructor takes a ``grid_type`` and a ``specs`` tuple. The available grid types are:

 * ``grid_type='legendre'`` (default): a non-uniform Gauss-Legendre grid, defined by the tuple ``(w1, w2, n1, n2, n3)``. A Gauss-Legendre grid is defined in the interval :math:`(0,\omega_1)` with :math:`n_1` points, another one in the interval :math:`(\omega_1,\omega_2)` with :math:`n_2` points, and a final one as a function of :math:`1/\omega` in the interval :math:`(0,1/\omega_2)` with :math:`n_3` points.
 * ``grid_type='matsubara'`` : the set of fermionic Matsubara frequencies :math:`\omega_n` at inverse temperature ``beta`` between 0 and a maximum frequency ``wc``, defined by the tuple ``(wc, beta)``. All frequencies carry the same integration weight.
 * ``grid_type='regular'`` : a uniform grid defined by the tuple ``(wc, n1, n2)``, with :math:`n_1` points in :math:`(0, \omega_c)` and :math:`n_2` points in the high-frequency complement.

In addition, the constructor takes an ``opt`` argument that modifies the CDMFT distance weights:

 * ``opt='self'`` : the CDMFT weights are scaled like the norm of the self-energy at each frequency (the Hartree-Fock part is subtracted), giving more weight to those frequencies at which correlations are more important.
 * ``opt='ifreq'`` : the CDMFT weights are scaled like :math:`1/\omega_n`.

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

