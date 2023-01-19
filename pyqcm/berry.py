import numpy as np
import matplotlib.pyplot as plt
import pyqcm
import pyqcm.spectral
from pyqcm import qcm

################################################################################
def Berry_curvature(nk=200, eta=0.0, period='G', range=None, label=0, orb=None, subdivide=False, plane='xy', k_perp=0.0, file=None, plt_ax=None, **kwargs):
    """Draws a 2D density plot of the Berry curvature as a function of wavevector, on a square grid going from -pi to pi in each direction.
    
    :param int nk: number of wavevectors on the side of the grid
    :param float eta: imaginary part of the frequency at zero, i.e., w = eta*1j
    :param str period: type of periodization used (e.g. 'G', 'M', 'None')
    :param list range: range of plot [originX, originY, side], in multiples of pi
    :param int label: label of the model instance (0 by default)
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all bands.
    :param int subdivide: True if plaquette subdivision is used.
    :param float k_perp: momentum component in the third direction (x pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str file: Name of the file to save the plot. If None, shows the plot on screen.
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: the contourplot object of matplotlib

    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax
    ax.set_aspect(1)

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    pyqcm.set_global_parameter('eta', eta)
    pyqcm.set_global_parameter('periodization', period)

    if range is not None:
        range[0] *= 0.5
        range[1] *= 0.5
        range[2] *= 0.5
        ext = [2*(range[0]-range[2]), 2*(range[0]+range[2]), 2*(range[1]-range[2]), 2*(range[1]+range[2])]
    else:
        ax.set_xticks((-1, 0, 1))
        ax.set_yticks((-1, 0, 1))
        ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'))
        ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'))
        range = [0.0, 0.0, 0.5]
        ext = [-1, 1,-1,1]

    axis = ''
    k1 = None
    k2 = None
    dir = 3
    if plane in ['z', 'xy', 'yx']:
        dir = 3
        k1 = [range[0]-range[2], range[1]-range[2], 0.5*k_perp]
        k2 = [range[0]+range[2], range[1]+range[2], 0.5*k_perp]
    if plane in ['y', 'xz', 'zx']:
        k1 = [range[0]-range[2], 0.5*k_perp, range[1]-range[2]]
        k2 = [range[0]+range[2], 0.5*k_perp, range[1]+range[2]]
        dir = 2
    elif plane in ['x', 'yz', 'zy']:
        k1 = [0.5*k_perp, range[0]-range[2], range[1]-range[2]]
        k2 = [0.5*k_perp, range[0]+range[2], range[1]+range[2]]
        dir = 1

    B = qcm.Berry_curvature(k1, k2, nk, orb, subdivide, dir, label)
    B *= (2*range[2]/nk)**2

    ax.set_aspect(1)

    # plot per se
    max = np.abs(B).max()
    CS = ax.imshow(np.flip(B,0), vmin=-max, vmax = max, cmap='bwr', extent=ext, **kwargs)
    if plt_ax is None:
        axis = pyqcm.spectral.set_legend_mdc(plane, k_perp)
        plt.colorbar(CS, shrink=0.8, extend='neither')
        ax.set_title(axis, fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()
    
    return CS


################################################################################
def Chern_number(nk=100, eta=0.0, period='G', offset=[0., 0., 0.], label=0, orb=None, subdivide=False):
    """Computes the Chern number by summing the Berry curvature over wavevectors on a square grid going from (0,0) to (pi,pi)

    :param int nk: number of wavevectors on the side of the grid
    :param float eta: imaginary part of the frequency at zero, i.e., w = eta*1j
    :param str period: type of periodization used (e.g. 'G', 'M', 'None')
    :param wavevector offset: wavevector offset of the computation grid
    :param int label: label of the model instance (0 by default)
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all occupied bands.
    :param boolean subdivide: recursivity flag (wavevector grid subdivision)
    :returns float: The Chern number

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    pyqcm.set_global_parameter('eta', eta)
    offset = np.array(offset)
 
    pyqcm.set_global_parameter('dual_basis')
    pyqcm.set_global_parameter('periodization', period)
    B = qcm.Berry_curvature(np.array([0.0, 0.0, 0.0])+offset, np.array([1.0, 1.0, 0.0])+offset, nk, orb, subdivide, 3, label)
    C = B.sum() / (nk * nk)
    if period == 'None':
        C /= pyqcm.model_size()[0]
    return C
         


################################################################################
def monopole(k, a=0.01, nk=20, label=0, orb=None, subdivide=False):
    """computes the topological charge of a node in a Weyl semi-metal

    :param [double] k: wavevector, position of the node
    :param float a: half-side of the cube surrounding the node 
    :param int nk: number of divisions along the side of the cube
    :param int label: label of the model instance (0 by default)
    :param int orb: orbital to compute the charge of (if None, sums over all bands)
    :param booleean subdivide: True if subdivision is allowed (False by default)
    :param int label: label of the model instance (0 by default)
    :return float: the monopole charge

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    return qcm.monopole(0.5*np.array(k), a*0.5, nk, orb, subdivide, label)
    

################################################################################
def Berry_flux(k0, R, nk=40, plane='xy', label=0, orb=None):
    """Computes the integral of the Berry connexion along a closed circle
    
    :param int k0: center of the circle
    :param float R: radius of the circle
    :param int nk: number of wavevectors on the circle
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param int label: label of the model instance (0 by default)
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all bands.
    :param int label: label of the model instance (0 by default)
    :returns float: the flux

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    phi_list = np.linspace(0.0, 2*np.pi, nk, endpoint=False)
    k = np.zeros((nk,3))
    if plane in ['z', 'xy', 'yx']:
        d1 = 0
        d2 = 1
        d3 = 2
    elif plane in ['y', 'xz', 'zx']:
        d1 = 2
        d2 = 0
        d3 = 1
    elif plane in ['x', 'yz', 'zy']:
        d1 = 1
        d2 = 2
        d3 = 0
    else:
        raise ValueError('forbidden value of "plane" in function Berry_flux')

    for i, phi in enumerate(phi_list):
        k[i, d1] = 0.5*(k0[d1] + R*np.cos(phi))
        k[i, d2] = 0.5*(k0[d2] + R*np.sin(phi))
        k[i, d3] = 0.5*k0[d3]

    return qcm.Berry_flux(k, orb, label)

################################################################################
def monopole_map(nk=40, label=0, orb=None, plane='z', k_perp=0.0, file=None, plt_ax = None, **kwargs):
    """Creates a plot of the monopole density (divergence of B) as a function of wavevector

    :param int nk: number of wavevector grid points on each side
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str k_perp: offset in wavevector in the direction perpendicular to the plane (x pi)
    :param int label: label of the model instance (0 by default)
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all bands.
    :param str file: Name of the file to save the plot. If None, shows the plot on screen.
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: the contourplot object of matplotlib

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
        axis = pyqcm.spectral.set_legend_mdc(plane, k_perp)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    K = pyqcm.wavevector_grid(nk, orig=[-1.0, -1.0], side=2, k_perp = k_perp, plane=plane)
    B = np.zeros(nk*nk)
    for i, k in enumerate(K):
        B[i] = monopole(2.0*k, a=2.0/nk, nk=5, orb=orb, label=label)

    ext = [-1, 1,-1,1]
    B = np.reshape(B,(nk,nk)) # transpose because y affects the row number and x the column number
    B2 = np.empty((nk+1,nk+1))
    B2[0:nk, 0:nk] = B
    B2[nk, :] = B2[0, :]
    B2[:, nk] = B2[:, 0]
    B2[nk,nk] = B2[0, 0]
    max = np.abs(B).max()
    CS = ax.imshow(np.flip(B2,0), vmin=-max, vmax = max, cmap='bwr', extent=ext, **kwargs)
    if plt_ax is None:
        plt.xticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
        plt.yticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
        plt.colorbar(CS, shrink=0.8, extend='neither')
        plt.title('monopole map, '+axis, fontsize=9)
    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()
    return CS

################################################################################
def Berry_flux_map(nk=40, plane='z', dir='z', k_perp=0.0, label=0, orb=None, npoints=4, radius=None, file=None, plt_ax = None, **kwargs):
    """Creates a plot of the Berry flux as a function of wavevector

    :param int nk: number of wavevector grid points on each side
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str dir: direction of flux, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str k_perp: offset in wavevector in the direction perpendicular to the plane (x pi)
    :param int label: label of the model instance (0 by default)
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all bands.
    :param int npoints: nombre de points sur chaque boucle
    :param str file: Name of the file to save the plot. If None, shows the plot on screen.
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: the contourplot object of matplotlib

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
        axis = pyqcm.spectral.set_legend_mdc(plane, k_perp)
        plt.title('component : '+dir)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    K = pyqcm.wavevector_grid(nk, orig=[-1.0, -1.0], side=2, k_perp = k_perp, plane=plane)
    B = np.zeros(nk*nk)
    if radius is None:
        radius = 0.8/nk
    for i, k in enumerate(K):
        B[i] = Berry_flux(2.0*k, radius, nk=npoints, plane=dir, orb=orb, label=label)

    B = np.reshape(B,(nk,nk))
    ext = [-1, 1,-1,1]
    B2 = np.empty((nk+1,nk+1))
    B2[0:nk, 0:nk] = B
    B2[nk, :] = B2[0, :]
    B2[:, nk] = B2[:, 0]
    B2[nk,nk] = B2[0, 0]
    max = np.abs(B).max()
    CS = plt.imshow(np.flip(B2,0), vmin=-max, vmax = max, cmap='bwr', extent=ext)
    if plt_ax is None:
        plt.xticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
        plt.yticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
        plt.colorbar(CS, shrink=0.8, extend='neither')
        plt.title(axis, fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

    return CS

################################################################################
def Berry_field_map(nk=40, nsides = 4, plane='z', k_perp=0.0, label=0, orb=None, file=None, plt_ax = None, **kwargs):
    """Creates a plot of the Berry flux as a function of wavevector

    :param int nk: number of wavevector grid points on each side
    :param int nsides: number of sides of the polygon used to compute the circulation of the Berry field.
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str k_perp: offset in wavevector in the direction perpendicular to the plane (x pi)
    :param int label: label of the model instance (0 by default)
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all bands.
    :param str file: Name of the file to save the plot. If None, shows the plot on screen.
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: the contourplot object of matplotlib, the quiver object of matplotlib

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    ax = None
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        axis = pyqcm.spectral.set_legend_mdc(plane, k_perp)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    K = pyqcm.wavevector_grid(nk, orig=[-1.0, -1.0], side=2, k_perp = k_perp, plane=plane)
    Bx = np.zeros(nk*nk)
    By = np.zeros(nk*nk)
    Bz = np.zeros(nk*nk)
    for i, k in enumerate(K):
        Bx[i] = Berry_flux(2.0*k, 2.0/nk, nk=nsides, plane='x', orb=orb, label=label)
        By[i] = Berry_flux(2.0*k, 2.0/nk, nk=nsides, plane='y', orb=orb, label=label)
        Bz[i] = Berry_flux(2.0*k, 2.0/nk, nk=nsides, plane='z', orb=orb, label=label)

    ext = [-1, 1,-1,1]
    Bx = np.reshape(Bx,(nk,nk))
    By = np.reshape(By,(nk,nk))
    Bz = np.reshape(Bz,(nk,nk))
    
    B2x = np.empty((nk+1,nk+1))
    B2x[0:nk, 0:nk] = Bx
    B2x[nk, :] = B2x[0, :]
    B2x[:, nk] = B2x[:, 0]
    B2x[nk,nk] = B2x[0, 0]

    B2y = np.empty((nk+1,nk+1))
    B2y[0:nk, 0:nk] = By
    B2y[nk, :] = B2y[0, :]
    B2y[:, nk] = B2y[:, 0]
    B2y[nk,nk] = B2y[0, 0]

    B2z = np.empty((nk+1,nk+1))
    B2z[0:nk, 0:nk] = Bz
    B2z[nk, :] = B2z[0, :]
    B2z[:, nk] = B2z[:, 0]
    B2z[nk,nk] = B2z[0, 0]

    max = np.abs(Bz).max()

    B = B2z
    max = np.abs(B).max()
    CS = ax.imshow(np.flip(B,0), vmin=-max, vmax = max, cmap='bwr', extent=ext)

    kx = np.linspace(-1,1,nk+1,endpoint=True)
    ky = np.linspace(-1,1,nk+1,endpoint=True)
    SP = ax.streamplot(kx, ky, B2x, B2y, linewidth=0.5, color='k')

    if plt_ax is None:
        plt.colorbar(CS, shrink=0.8)
        ax.set_title(axis, fontsize=9)
        plt.xticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
        plt.yticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()
    
    return CS, SP
