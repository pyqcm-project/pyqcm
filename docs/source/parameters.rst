Parameter sets, Hilbert space sectors and model instances
#########################################################

Parameter sets
==============

Each term in the lattice Hamiltonian is associated with a parameter :math:`h_a`, and each cluster of the super unit cell has an equal number of parameters, for the corresponding terms of its Hamiltonian. If a bath is added to a cluster, then even more parameters are necessary to specify the cluster Hamiltonian. By default, the values of the cluster parameters are inherited from those of the lattice parameters. But their values may be different if desired.
In addition, even within the lattice or within a cluster, one may want to impose automatic constraints between parameters, such as between the Coulomb repulsion :math:`U` and the chemical potential :math:`\mu`, or other constraints coming from accidental symmetries.
The large number of parameters and this inheritance of values needs to be managed. This is done via a *parameter set*.

On a cluster, the name of the operator will received a suffix ``_i``, i being the cluster label (from 1 to the number of clusters :math:`N_\mathrm{clus}`). Thus, from a lattice operator ``t``, an operator ``t_1`` will be constructed on cluster #1, and an operator ``t_2`` on cluster #2, and so on. The values of the coefficients of these operators will be designated by the same symbols. This suffix scheme will apply to all operators defined in the model and on the clusters.

The `lattice_model` object contains a single instance of a structure called `parameter_set` that can define equivalences between parameters beyond those imposed by default, or release the default constraints. This structure is defined in pyqcm through the member function ``set_parameters(str)``. 
This may be the most important function of the library. It takes a single, long string argument, for instance::

    model.set_parameters("""
        t = 1
        U = 8
        mu = 0.5*U
        M = 0
        M_1 = 0.1
    """)

The values of the parameters are set by the equal sign. Dependent parameters are specified by a multiple of another parameter and the multiplication (*) sign. Note that a prefactor is always required, even when it is unity. Thus, you should write ``M_2 = 1*M_1`` and not ``M_2=M_1``.
Notice that setting a parameter to ``0`` causes qcm to *not create the operator* in the model.


Hilbert space sectors
=====================

For each cluster in the super unit cell, one must specify in which Hilbert space sector to look for the ground state.
Each sector is specified by a Python string indicating  which sectors of the Hilbert space must be considered in the exact diagonalization procedure. For instance, if the model conserves the number of particles and the z-component of the spin, the string ``R0:N4:S0`` means that the ground state of the cluster must be looked for in the Hilbert space sector with N=4 electrons and S=0 spin projection. R0 means that the ground state should belong to the trivial representation (labeled 0) of the point group. The spin projection is expressed by an integer :math:`S = 2 S_z`. Thus, ``S1`` means :math:`S_z=\frac12` and ``S-2`` means :math:`S_z=-1`. 

The label of the irreducible representation is taken from the character table of the point group. For instance, in the important case of a :math:`C_{2v}` symmetry, each irreducible representation is either even or odd with respect to horizontal and vertical reflections. The binary digits 0 and 1 are associated with even and odd symmetries, respectively.
Thus, the binary number 10 = 2 labels a representation that is odd in y and even in x, 01 = 1 labels a representation that is even in y and odd in x, 11 = 3 is odd in both directions and 00=0 is even in both directions. These possibilities are thus represented by the strings R2, R1, R3 and R0, respectively.

Hilbert space sectors are a crucial element of the use of the library and may be the source of physical errors. Performance issues dictate that not all Hilbert space sectors should be checked for the true ground state for every calculation. Some judgement must be applied as to which sector or subset of sectors contains the true ground state. For a given cluster, a subset of sectors may be provided instead of a single one, by separating the sector keywords by slashes ( / ). For instance, the string indicating that the ground state should be search in the sectors of the trivial representation, with N=3 electrons and spin projection -1/2 or 1/2 is ``R0:N3:S-1/R0:N3:S1``. Such a set of possible sectors is then specified with this call::

    model.set_target_sectors(['R0:N3:S-1/R0:N3:S1'])

Note that the argument to that function is a list of strings, which in the above example contains a single element because the super unit cell contains a single cluster.    

If spin is not conserved because of the presence of spin-flip terms, then the spin label must be omitted. For instance, the string "R0:N4" denotes the sector containing 4 electrons, in the trivial point group representation. An error message will be issued if the user specifies a spin sector in such cases, or inversely if the spin sector is not specified when spin is conserved.

The same is true in cases where particle number is not conserved (superconductivity): the number label must be omitted.
For instance, the string ``R0:S0`` denotes the sector with zero spin, in the trivial point-group representation and an undetermined number of electrons.

When the target Hilbert space sector (or subset of sectors) specified in ``set_target_sectors(...)`` does not contain the true ground state, then the Green function computed thereafter will likely have the wrong normalization and analytic properties, because "excited states" obtained from the pseudo ground state by applying creation or annihilation operators may have a lower energy than the latter.

The ED solver has a global real-valued parameter called ``temperature`` which allows a certain mixtures of Hilbert space sectors in the density matrix of the system. This temperature must be low, as the ED solver is not a finite-temperature impurity solver. Only a few low-lying states may be computed. Within a Hilbert space sector, only the lowest-energy state will be found, unless the Davidson method is used and the global parameter ``Davidson_states`` is set to an integer value greater than one. A finite number of low-energy states will then be computed, and will contribute to the density matrix, provided their Boltzmann weight is larger than some minimum defined by the global parameter ``minimum_weight``. The same applies to states coming from different sectors. This feature allows for a smoother transition between sectors when looping over chemical potential.


Model instances
===============

Once a set of parameters has been recorded in the library's parameter set, one may define an instance of the lattice model by creating an object of type `model_instance` ::

    I = pyqcm.model_instance(model, lab)

Here `model` is the lattice model object and `lab` is a non-negative integer that defines a label for that instance. All computations from the qcm library are performed on a given model instance, associated with particular values of the lattice and cluster parameters. In most cases a single model instance is needed at a given time, so that the label `lab` is 0 by default. Another call to ``model_instance(lab)`` with the same label will erase any previous instance of the model with the same label. Alternately, the member function ``I.reset()`` can be called to reset the instance to a blank state with the current values of the parameters in the `parameter_set`` object.  

Computations on a model instance are **lazy**. For instance, computing the ground state of the cluster is done only when needed, for instance when asking for the ground state averages or the Green function; computing the cluster Green function representation (either Lehmann or through continued fractions) is done only when a Green-function related quantity is needed, etc.

.. autoclass:: pyqcm.model_instance 
    :members:



