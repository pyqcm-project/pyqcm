Cluster Dynamical Mean Field Theory
###################################

This submodule provides functions that perform the CDMFT algorithm.
The class `CDMFT` contains the CDMFT algorithm *per se* and implements it.
Some auxiliary classes are also defined:

* `convergence_manager` defines a test for assessing convergence of the CDMFT procedure

* `frequency_grid` defines a grid of imaginary frequencies for computing the CDMFT distance function.

* `hybridization` defines the hybridization function, mostly for import/export.

* `general_bath` defines general cluster models with bath, with automatic bath parameter definitions, etc.

* `observable` defines quantities that can be used to assess the convergence of the CDMFT procedure, if the latter is not base on the bath parameters themselves

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

.. autoclass:: pyqcm.cdmft.observable
    :members:

