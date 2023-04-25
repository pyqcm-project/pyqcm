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
#. An array of real matrix elements. Each element of the array is a 3-tuple giving the labels of the orbitals involved and the value of the matrix element itself. Note that spin-up and spin-down orbital labels are separated by the total number of orbitals on the custer, here `no=10`.

If a complex-valued operator is needed, then the function ``new_cluster_operator_complex()`` must be used, the only difference being that the actual matrix elements are complex numbers.

Class for defining cluster models
---------------------------------

.. autoclass:: pyqcm.cluster_model 
    :members:

Class for defining clusters
---------------------------

When the repeated unit contains more than one cluster, several of them can be based on the same cluster model.
This is an important memory saving point, as operators as stored in memory for each cluster model, not for each
cluster. Moreover, a cluster model has no notion of geometry of site positions. The latter information is 
contained in a different class, `pyqcm.cluster`.

.. autoclass:: pyqcm.cluster 
    :members:
