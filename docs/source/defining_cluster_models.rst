#######################
Defining cluster models 
#######################

This section explains how to define impurity (or cluster) models through python calls.

Clusters are the building blocks of lattice models. One needs to define them first.
This is done through the construction of an object of type `cluster_model`. For instance::

    import pyqcm
    CM = pyqcm.cluster_model(4, 0, '2x2_C2v', [[3, 4, 1, 2], [2, 1, 4, 3]])

The constructor ``cluster_model(Ns, Nb, name, perm, bath_irrep)`` takes the following arguments:

#. The number :math:`N_s` of physical sites
#. The number :math:`N_b`  of bath sites
#. A name given to the cluster model ('clus' by default). Useful if more than one cluster models are necessary.
#. (optional) A list of permutations of the  :math:`N_o=N_s+N_b`  orbitals that define generators of the symmetries of the cluster.
#. (optional) A boolean flag that, if true, signals that bath orbitals belong to irreducible representations of the symmetry group of the cluster, instead of being part of permutations of the different orbitals of the cluster-bath system.

In the above example, a four-site cluster is defined, without any bath sites. The positions of the sites are not relevant to the impurity solver, and so are not defined at this stage.
However, a plaquette geometry is implicit here, with the following site labels:

.. figure:: 2x2.png
    :align: center
    :height: 100px

    Figure 1


The cluster symmetries (permutations) passed to the function thus correspond to reflections with respect to the horizontal and vertical axes, respectively.
The cluster symmetries will be used by the ED solver to lighten the exact diagonalization task. Large clusters will take less memory, convergence of the Lanczos method will be faster, and computing the Green function at a given frequency will be more efficient.
The different permutations that constitute the generators must commute with each other. They generate an Abelian group with :math:`g` elements. Such a group has an equal number :math:`g` of irreducible representations, numbered from 0 to :math:`g-1`.

Defining operators on the cluster
---------------------------------

Most operators in the model are best defined on the lattice and their restriction to the cluster is defined automatically, so there is no need to define them explicitly on each cluster of the super unit cell. This is not the case if one wants to use the ED solver as a standalone program without reference to a lattice model.
Bath operators, on the other hand, need to be defined explicitly within the cluster model since they do not exist on the lattice model.

The following code defines the cluster and bath operators for the cluster illustrated in the last section, which we reproduce here:

.. figure:: h4-6b.png
    :align: center
    :height: 200px

    Figure 2

Content of the cluster definition file::

    from pyqcm import *

    no = 10
    ns = 4
    CM = pyqcm.cluster_model(ns, no-ns, 'clus', [[1, 3, 4, 2, 6, 7, 5, 9, 10, 8]])


    CM.new_operator('bt1', 'one-body', [
        (2, 5, 1.0),
        (3, 6, 1.0),
        (4, 7, 1.0),
        (2+no, 5+no, 1.0),
        (3+no, 6+no, 1.0),
        (4+no, 7+no, 1.0)
    ])

    CM.new_operator('bt2', 'one-body', [
        (2, 8, 1.0),
        (3, 9, 1.0),
        (4, 10, 1.0),
        (2+no, 8+no, 1.0),
        (3+no, 9+no, 1.0),
        (4+no, 10+no, 1.0)
    ])

    CM.new_operator('be1', 'one-body', [
        (5, 5, 1.0),
        (6, 6, 1.0),
        (7, 7, 1.0),
        (5+no, 5+no, 1.0),
        (6+no, 6+no, 1.0),
        (7+no, 7+no, 1.0)
    ])

    CM.new_operator('be2', 'one-body', [
        (8, 8, 1.0),
        (9, 9, 1.0),
        (10, 10, 1.0),
        (8+no, 8+no, 1.0),
        (9+no, 9+no, 1.0),
        (10+no, 10+no, 1.0)
    ])


Note that the symmetry defined here is a rotation by 120 degrees. This generates the group :math:`C_3`, which has complex representations. **pyqcm** can only deal with Abelian groups (the correct treatment of non-Abelian symmetries is too complex when computing Green functions for the benefits it would provide). In the above example, a better strategy when no complex operators are present would be to define only a :math:`C_2` symmetry based on one of three possible reflections. This would only provide 2 symmetry operations instead of 3, but the representations would be real instead of complex, thus saving more time and memory.

The member function ``new_operator(name, type, elements)`` takes the following arguments:

#. The name of the operator
#. The type of operator; one of 'one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg'
#. An array of real matrix elements. Each element of the array is a 3-tuple giving the labels of the orbitals involved and the value of the matrix element itself. Note that spin-up and spin-down orbital labels are separated by the total number of orbitals on the cluster, here `no=10`.

If a complex-valued operator is needed, then the function ``new_cluster_operator_complex()`` must be used, the only difference being that the actual matrix elements are complex numbers.

General interactions
--------------------

The interactions available on the lattice (Hubbard, Hund, Heisenberg, X, Y, Z) are all of a special form.
An arbitrary two-body interaction can be defined on the cluster instead, with the member function
``general_interaction_operator(name, elements)``. The operator thus defined is

.. math::
    H = \sum_{ijkl}\sum_{\sigma\sigma'} v_{ijkl}\,
        c^\dagger_{i\sigma}c^\dagger_{j\sigma'}c_{k\sigma'}c_{l\sigma}

where :math:`i,j,k,l` label the orbitals of the cluster (from 1 to :math:`N_o=N_s+N_b`, the bath orbitals
coming after the physical sites) and the sum over the spins :math:`\sigma,\sigma'` is performed automatically.
Each matrix element is specified by a 5-tuple ``(i,j,k,l,v)``. For instance, the Hubbard interaction
:math:`U\sum_i n_{i\uparrow}n_{i\downarrow}` on a two-site cluster is defined by::

    CM = pyqcm.cluster_model(2, name='clus')
    CM.general_interaction_operator('U', [(1,1,1,1,0.5), (2,2,2,2,0.5)])

the value of the operator (i.e. of the parameter, see below) then being :math:`U` itself.

Two points require attention:

#. The terms :math:`(i,j,k,l)` and :math:`(j,i,l,k)` are one and the same, because of the sum over the spins
   and of the anticommutation of the creation (and of the annihilation) operators. If both are provided,
   their values are added.
#. The operator must be Hermitian. The Hermitian conjugate of the term :math:`(i,j,k,l)` is the term
   :math:`(l,k,j,i)`, with the complex conjugate value; it is added automatically to the list if it is
   absent, and an error is raised if it is present but inconsistent. Thus a pair-hopping term
   ``(1,1,2,2,v)`` really stands for the pair hopping plus its Hermitian conjugate.

If the interaction is not the same for all spin projections, the option ``spin_dependent=True`` may be used;
the labels then run from 1 to :math:`2N_o` (the spin-down label of an orbital being its spin-up label plus
:math:`N_o`) and no sum over spins is performed. The above Hubbard interaction would then read::

    CM.general_interaction_operator('U', [(1,3,3,1,1.0), (2,4,4,2,1.0)], spin_dependent=True)

Since a general interaction has no counterpart on the lattice model, it is a cluster-specific parameter:
its name carries the label of the cluster, e.g. ``U_1`` for the first cluster, and it must be given a value
as such in :func:`~pyqcm.lattice_model.set_parameters`. Note also that such an interaction exists on the
cluster only : it plays no role in the (Hartree) treatment of the interactions on the lattice. Finally, it
cannot be applied on the fly, i.e. the global option ``Hamiltonian_format`` cannot be set to 'N' when a
general interaction is present (all the other formats are fine).

Class for defining cluster models
---------------------------------

.. autoclass:: pyqcm.cluster_model 
    :members:

Class for defining clusters
---------------------------

When the repeated unit contains more than one cluster, several of them can be based on the same cluster model.
This is an important memory saving point, as operators are stored in memory for each cluster model, not for each
cluster. Moreover, a cluster model has no notion of geometry of site positions. The latter information is 
contained in a different class, `pyqcm.cluster`.

.. autoclass:: pyqcm.cluster 
    :members:
