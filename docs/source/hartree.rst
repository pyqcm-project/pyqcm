The Hartree approximation
#########################

Extended interactions require an extension of quantum cluster methods: inter-cluster interactions have to be cut off and replaced by a mean field, in the Hartree approximation. 
An extended interaction on the cluster is then reduced to

.. math::
    \sum_{(ij)}V^c_{ij}n_i n_j + 
    \sum_{(ij)}V^m_{ij}\left(n_i \langle n_j\rangle + n_j \langle n_i\rangle - \langle n_i\rangle\langle n_j\rangle \right) 

where :math:`V_m` stands for the terms of :math:`V_{ij}` which need to be treated in the Hartree approximation, and :math:`V^c` those that do not need to be. :math:`(ij)` stands for a pair of sites (:math:`i\ne j`).

Since :math:`V_m` is real symmetric, the mean-field Hamiltonian may be recast into

.. math::
    H_m = \sum_{ij}V^m_{ij}\left(n_i \langle n_j\rangle - \frac12\langle n_i\rangle\langle n_j\rangle \right)

where now the sum is over independent values of :math:`i` and :math:`j`.
:math:`V^m` can be diagonalized by an orthogonal transformation :math:`L`: :math:`V^m = L\Lambda \tilde L`. 
One then defines eigenoperators :math:`O_a = L_{aj}n_j` such that

.. math::
    H_m = \sum_a \lambda_a\left\{O_a \langle O_a\rangle - \frac12 \langle O_a\rangle^2\right\}

Now, let us consider the mean values :math:`\langle O_a\rangle` as adjustable variational (or mean-field) parameters.
Then, in terms of the coefficients :math:`h_a=\lambda_a\langle O_a\rangle` of the operator :math:`O_a`, the mean-field Hamiltonian takes the following form:

.. math::
    H_m = \sum_a \left\{ h_a O_a - \frac{h_a^2}{2\lambda_a} \right\}

The variation of parameter :math:`h_a` the yields

.. math::
    \langle O_a\rangle = \frac{h_a}{\lambda_a} \rightarrow h_a=\lambda_a\langle O_a\rangle

The Hartree procedure consists in starting with trial values of :math:`_a` and iteratively performing the above assignation until convergence.


.. autoclass:: pyqcm.hartree
    :members: 
