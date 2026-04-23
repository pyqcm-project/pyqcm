Spectral properties, bases, etc.
################################

PyQCM features many functions that compute and plot spectral properties of the model under study. Even though each of these functions is summarily described in the list of functions under the class `model_instance`, in this section we are going to review a few of them in more detail.


Frequencies
===========
Since PyQCM rests on an exact diagonalization solver, it provides a representation for the Green function (GF) that can evaluate the impurity Green function at any complex-valued frequency. 

Some functions plot various quantities on the real frequency axis, in which case one must provide a frequency interval and a small imaginary part `eta`to be added to each frequency. These are

- `spectral_function`
- `cluster_spectral_function`
- `plot_hybridization_function`
- `plot_DoS` 
  
In each case the argument `w`can be either a `float`, in which case the frequency interval is `(-w,w)`, or a 2-tuple, or an explicit array of floats.
In the first two case, the frequency step is set by the function `frequency_array`to be `eta/4`.
In some cases (`cluster_spectral_function`, `plot_hybridization_function`), a boolean argument `imaginary` tells the function to plot the quantity along the imaginary frequency axis instead (the meaning of `w`is changed accordingly`).

The function `plot_host_hybrid` plot quantities along the imaginary frequency axis only, in which case the `w` must be a predefined array of frequencies along that axis.

Wavevectors
===========

In PyQCM a wavevector :math:`\mathbf{k}` can be specified in three different bases:

- The basis of reciprocal lattice vectors (`dual`)
- The basis of reciprocal superlattice vectors (`superdual`)
- The physical basis (`physdual`), which differs from (`dual`) by a change of basis defined in the function `lattice_model.set_basis`.
  
In the case of a square lattice of lattice constant :math:`a`, a wavevector on the Brillouin zone boundary would be, for instance, 
:math:`(\pi/a, 0, 0)`, in the `physdual` basis. In the `dual` basis, this becomes :math:`(0.5, 0, 0)`. In the `superdual` basis, this becomes :math:`(1, 0, 0)` if a :math:`2\times2` cluster is used.

Internally, pyqcm works in the `superdual` basis. All wavevectors are then defined within the unit cube (or square, or segment, depending on dimension).

However, functions computing spectral quantities require wavevectors in the physical basis, divided by :math:`2\pi`.
Hence, if the lattice is square and the lattice constant  :math:`a` is one, this coincides with the `dual` basis: wavevectors are then naturally defined in the unit cube associated with the Brillouin zone.

Real-space bases
================

In real space, we use a working basis in which all position vectors have integer components. If the basis of the lattice is different (e.g. triangular or graphene-like, or a lattice with a basis in multi-orbital models), the `lattice_model.set_basis` must be used to express the working basis in terms of the geometric (physical) basis. 

For instance, if we consider a triangular lattice with lattice vectors at 120 degrees from each other with a unit lattice constant, one should define

`set_basis([(1, 0, 0),(-0.5,np.sqrt(3)/2,0)])`

which correctly expressed the lattice vectors.

A more subtle example is the 3-band Hubbard model for the cuprates. The unit-cell contains one Cu atom and two O atoms, at positions (0,0,0), (1,0,0) and (0,1,0) respectively in the working basis. The lattice vectors are then (2,0,0) and (0,2,0) in the working basis. To compensate this, we set

`set_basis([(0.5, 0, 0), (0, 0.5, 0)])`

which expresses the working basis in terms of the lattice vectors and allows us to use a range of wavevectors identical to the one used for a simple square lattice of lattice constant unity.

Overall, the call to `lattice_model.set_basis` is important in two circumstances:

- When plotting as a function of wavevector, since wavevectors are defined in the `physdual` basis. For instance, in functions such as `mdc`, `dispersion`` and `segment_dispersion`.
- When writing on file an external hybridization function to be re-used by a simpler model. The wavevector grid created in this circumstance should be in the `superdual` basis.



