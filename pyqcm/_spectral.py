import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import pyqcm
from . import qcm


####################################################################################################
# Internal functions

#---------------------------------------------------------------------------------------------------
def _color(z):
    x = np.modf(np.angle(z) / (2 * np.pi) + 1.66667)[0]
    return hsv_to_rgb(np.array([x, 0.8, 0.8]))

#---------------------------------------------------------------------------------------------------
def _set_legend_mdc(plane, k_perp):

    ax = plt.gca()
    if plane=='z' or plane=='xy':
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        return '$k_z = {:1.3f}\pi$'.format(k_perp)
    elif plane=='x' or plane=='yz':
        ax.set_xlabel('$k_y$')
        ax.set_ylabel('$k_z$')
        return '$k_x = {:1.3f}\pi$'.format(k_perp)
    elif plane=='y' or plane=='xz':
        ax.set_xlabel('$k_z$')
        ax.set_ylabel('$k_x$')
        return '$k_y = {:1.3f}\pi$'.format(k_perp)

#---------------------------------------------------------------------------------------------------
def __frequency_array(wmax=6.0, eta=0.05, imaginary=False):
    """Returns an array of complex frequencies for plotting spectral quantities

    """

    if type(wmax) is tuple:
        w = np.arange(wmax[0], wmax[1] + 1e-6, eta/4.0)  # defines the array of frequencies
    elif type(wmax) is float or type(wmax) is int:
        w = np.arange(-wmax, wmax + 1e-6, eta/4.0)  # defines the array of frequencies
    elif type(wmax) is np.ndarray and wmax.dtype == float:
        return wmax + eta*1j
    elif type(wmax) is np.ndarray and wmax.dtype == complex:
        return wmax
    else:
        raise TypeError('the type of argument "wmax" in __frequency_array() is wrong')

    if imaginary:
        wc = w*1j
    else:
        wc = np.array([x + eta*1j for x in w], dtype=complex)

    return wc


#---------------------------------------------------------------------------------------------------
def __kgrid(ax, nk, quadrant=False, k_perp=0.0, plane='xy', size=1.0):

    if quadrant:
        orig = np.array([0.5*(1-size), 0.5*(1-size), 0.0])
        k = pyqcm.wavevector_grid(nk, orig, size, k_perp, plane)
        ax.set_xticks((0, 0.5/size, 1/size))
        ax.set_yticks((0, 0.5/size, 1/size))
        ax.set_xticklabels(('$0$', '$\pi/2$', '$\pi$'))
        ax.set_yticklabels(('$0$', '$\pi/2$', '$\pi$'))
        x = np.linspace(0, 1, nk)
    else:
        orig = np.array([-size, -size, 0])
        k = pyqcm.wavevector_grid(nk, orig, size*2, k_perp, plane)
        ax.set_xticks((-1/size, 0, 1/size))
        ax.set_yticks((-1/size, 0, 1/size))
        ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'))
        ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'))
        x = np.linspace(-1, 1, nk)
    return k, x


####################################################################################################
# Additional methods of the model_instance class

def spectral_function(self, wmax=6.0, eta=0.05, path='triangle', nk=32, orb=None, offset=2, opt='A', Nambu_redress=True, inverse_path=False, title=None, file=None, plt_ax=None, threeD = False,  **kwargs):
    """Plots the spectral function :math:`A(\mathbf{k},\omega)` along a wavevector path in the Brillouin zone.
    This version plots the spin-down part with the correct sign of the frequency in the Nambu formalism.

    :param float wmax: the frequency range is from -wmax to wmax if w is a float. If wmax is a tuple then the range is (wmax[0], wmax[1]). wmax can also be an explicit list of real frequencies
    :param float eta: Lorentzian broadening
    :param str path: a keyword that is passed to pyqcm.wavevector_path() to produce a set of wavevectors along a path, or a tuple 
    :param int nk: the number of wavevectors along each segment of the path (passed to pyqcm.wavevector_grid())
    :param int orb: if not None, only plots the spectral function associated with this orbital number (starts at 1). If None, sums over all orbitals.
    :param float offset: vertical offset in the plot between the curves associated to successive wavevectors
    :param str opt: 'A' : spectral function, 'self' : self-energy, 'Sx' : spin (x component), 'Sy' : spin (y component)
    :param boolean Nambu_redress: if True, evaluates the Nambu component at the opposite frequency
    :param boolean inverse_path: if True, inverts the path (k --> -k)
    :param str title: optional title for the plot. If None, a string with the model parameters will be used.
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """

    orbs = pyqcm.orbital_manager(orb, from_zero=True)

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    dim = self.model.dimGF_red
    mix = self.model.mixing

    if orb is not None:
        assert (orb <= self.model.nband and orb > 0), 'The orbital index in plot_spectrum() must vary from 1 to {:d}'.format(self.model.nband)
        orb -= 1

    w = __frequency_array(wmax, eta)

    k, tick_pos, tick_str = pyqcm.wavevector_path(nk, path)  # defines the array of wavevectors
    if inverse_path:
        k *= -1
        for i,s in enumerate(tick_str):
            tick_str[i] = '-'+s


    A = np.zeros((len(w), len(k)))
    A_down = np.zeros((len(w), len(k)))
    plot_down = False
    for i in range(len(w)):
        if opt=='self':
            g = self.self_energy(w[i], k, False)
        else:
            g = self.periodized_Green_function(w[i], k, False)
        if opt=='Sx':
            assert mix&2, 'option Sx in spectral_function() only makes sense if spin-flip terms are present'
        if opt=='Sy':
            assert mix&2, 'option Sy in spectral_function() only makes sense if spin-flip terms are present'
        for j in range(len(k)):
            for l in orbs: 
                A[i, j] += -g[j, l, l].imag

            if mix&2:  
                plot_down = True
                for l in orbs: 
                    A_down[i, j] += -g[j, self.model.nband+l, self.model.nband+l].imag

    if mix == 1:
        # add the contribution to the Nambu channel, but with opposite frequency
        if Nambu_redress:
            k *= -1
        for i in range(len(w)):
            if Nambu_redress:
                W = -np.conj(w[i])
            else:
                W = w[i]
            if opt=='self':
                g = self.self_energy(W, k, False)
            else:
                g = self.periodized_Green_function(W, k, False)
            for j in range(len(k)):
                plot_down = True
                for l in orbs: 
                    A_down[i, j] += -g[j, self.model.nband+l, self.model.nband+l].imag

    if mix == 4:
        plot_down = True
        # add the contribution to the spin-down channel in that case
        for i in range(len(w)):
            if opt=='self':
                g = self.self_energy(w[i], k, True)
            else:
                g = self.periodized_Green_function(w[i], k, True)
            for j in range(len(k)):
                for l in orbs: 
                    A_down[i, j] += -g[j, l, l].imag
    
    ax.set_xlim(np.real(w[0]), np.real(w[-1]))
    ax.set_ylim(0, (1+len(k)) * offset + 1 / eta)
    for j in range(len(k)):
        if threeD:
            ax.plot(np.real(w), A[:, j], 'k-', lw=0.5, zdir='y', zs = offset * j, **kwargs)
        else:
            if plot_down:
                ax.plot(np.real(w), A_down[:, j] + offset * j, 'r-', lw=0.5, **kwargs)
            ax.plot(np.real(w), A[:, j] + offset * j, 'b-', lw=0.5, **kwargs)
    if title is None and plt_ax is None:
        ax.set_title(r'$A(\mathbf{k},\omega)$: '+self.model.parameter_string(), fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    if threeD:
        ax.set_ylim(0, (1+len(k))* offset)
        ax.set_zlim(0, 2*np.max(A))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_zticks([])
        ax.w_zaxis.line.set_lw(0.)
        ax.w_xaxis.line.set_c('b')
        ax.w_yaxis.line.set_c('b')
        ax.grid(False)
    else:
        ax.axvline(0, ls='solid', lw=0.5)
    ax.set_yticks(offset * tick_pos)
    ax.set_yticklabels(tick_str)
    if plt_ax is None:
        ax.set_xlabel(r'$\omega$')
        plt.tight_layout()

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()
        
#---------------------------------------------------------------------------------------------------
def plot_hybridization_function(self, wmax=6, eta=0.01, imaginary=False, clus = 0, realpart=False, file=None, plt_ax=None, **kwargs):
    """This function plots the imaginary part of the hybridization function Gamma as a function of frequency.
    Only the diagonal elements are plotted, but for all clusters if there is more than one.
    The arguments have the same meaning as in `plot_spectrum`, except 'realpart' which, if True, plots
    the real part instead of the imaginary part.

    :param float wmax: the frequency range is from -wmax to wmax if w is a float. If wmax is a tuple then the range is (wmax[0], wmax[1]). wmax can also be an explicit list of real frequencies
    :param float eta: Lorentzian broadening
    :param boolean imaginary: If True, the frequency range is along the imaginary frequency axis
    :param int clus: cluster index (starts at 0)
    :param boolean realpart: if True, the real part of the Green function is shown, not the imaginary part
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    w = __frequency_array(wmax, eta, imaginary)  # defines the array of frequencies
    eta = 0.05j
    d = self.model.dimGF
    A = np.zeros((len(w), d*d))
    for i in range(len(w)):
        g = self.hybridization_function(w[i], clus, False)
        for l1 in range(d):
            for l2 in range(d):
                l = l1 + d*l2
                if realpart:
                    A[i, l] += g[l1, l2].real
                else:
                    A[i, l] += -g[l1, l2].imag

    offset = 2
    if imaginary:
        ax.set_xlim(np.imag(w[0]), np.imag(w[-1]))
        for j in range(d*d):
            ax.plot(np.imag(w), A[:, j] + offset * j, 'b-', lw=0.5, **kwargs)
    else:
        ax.set_xlim(np.real(w[0]), np.real(w[-1]))
        for j in range(d*d):
            ax.plot(np.real(w), A[:, j] + offset * j, 'b-', lw=0.5, **kwargs)
    ax.axvline(0, c='r', ls='solid', lw=0.5)
    if plt_ax is None:
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\Gamma(\omega)$')
        plt.title(r'$\Gamma(\omega)$: '+self.model.parameter_string())

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()


#---------------------------------------------------------------------------------------------------
def cluster_spectral_function(self, wmax=6, eta = 0.05, imaginary=False, clus=0, offset=2, full=False, opt=None, spin_down=False, blocks=False, file=None, plt_ax=None, color = 'b', **kwargs):
    """Plots the spectral function of the cluster in the site basis
    
    :param float wmax: the frequency range is from -wmax to wmax if w is a float. If wmax is a tuple then the range is (wmax[0], wmax[1]). wmax can also be an explicit list of real frequencies
    :param float eta: Lorentzian broadening
    :param boolean imaginary: If True, the frequency range is along the imaginary frequency axis
    :param int clus: label of the cluster within the super unit cell (starts at 0)
    :param float offset: vertical offset in the plot between the curves associated to successive wavevectors
    :param boolean full: if True, plots off-diagonal components as well
    :param boolean self: if True, plots the self-energy instead of the spectral function
    :param str opt: if None, G is computed. Other options are 'self' (self-energy) and 'hyb' (hybridization function)
    :param boolean spin_down: if True, plots the spin down part, if different
    :param boolean blocks: if True, gives the GF in the symmetry basis (block diagonal)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param color: matplotlib color of the curves
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: the array of frequencies, the spectral weight

    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    w = __frequency_array(wmax, eta, imaginary)  # defines the array of frequencies
    # d = self.model.dimGFC[clus]
    d = self.model.clus[clus].nsites
    if full:
        dd = d*d
        # dd = (d*(d+1))//2
        T = []
        for j in range(d):
            for k in range(d):
                T.append('({0:d},{1:d})'.format(j+1,k+1))
        plt.yticks(offset*np.arange(0, dd), T)
    else:
        dd = d
        plt.yticks(offset*np.arange(0, d), [str(i) for i in range(1,d+1)])

    A = np.zeros((len(w), dd))
    for i in range(len(w)):
        if opt is None:
            g = self.cluster_Green_function(w[i], clus, spin_down, blocks) # run of the mill cluster green function
        elif opt == "self":
            g = self.cluster_self_energy(w[i], clus, spin_down) # self-energy functionnal
        elif opt == "hyb":
            g = self.hybridization_function(w[i], clus, spin_down) # hybridization function
        else:
            raise ValueError(f"'{opt}' is not a valid option, must be one of 'self', 'hyb' or None.")
        if full:
            l = 0
            for j in range(d):
                for k in range(d):
                    A[i, l] += -g[j, k].imag
                    l += 1
        else:        
            for j in range(d):
                A[i, j] += -g[j, j].imag

    max = np.max(A)
    plt.ylim(0, dd * offset + max)

    if imaginary:
        ax.set_xlim(np.imag(w[0]), np.imag(w[-1]))
        for j in range(dd):
            plt.plot(np.imag(w), A[:, j] + offset * j, '-', lw=0.5, color=color, **kwargs)
    else:
        ax.set_xlim(np.real(w[0]), np.real(w[-1]))
        for j in range(dd):
            plt.plot(np.real(w), A[:, j] + offset * j, '-', lw=0.5, color=color, **kwargs)

    plt.xlabel(r'$\omega$')
    plt.axvline(0, ls='solid', lw=0.5)
    plt.title(self.model.parameter_string(), fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

    # return w.real, A    # why returning this? check. Could be triggered by optional argument.


#---------------------------------------------------------------------------------------------------
def spectral_function_Lehmann(self, path='triangle', nk=32, orb=1, offset=0.1, lims=None, file=None, plt_ax=None, **kwargs):
    """Plots a Lehmann representation of the spectral function along a wavevector path in the Brillouin zone. Singularities are plotted as impulses with heights proportionnal to the residue.
    
    :param path: if a string, keyword passed to `pyqcm.wavevector_path()` to produce a set of wavevectors; else, explicit list of wavevectors (N x 3 numpy array).
    :param int nk: the number of wavevectors along each segment of the path (passed to pyqcm.wavevector_path())
    :param int orb: only plots the spectral function associated with this orbital number (starts at 1)
    :param float offset: vertical offset in the plot between the curves associated to successive wavevectors
    :param (float,float) lims: limits of the plot in frequency (2-tuple)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """
    
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    d = self.model.dimGF_red

    k, tick_pos, tick_str = pyqcm.wavevector_path(nk, path)  # defines the array of wavevectors

    assert (orb <= self.model.nband and orb > 0), 'The orbital index in spectral_function_Lehmann() must vary from 1 to {:d}'.format(self.model.nband)

    if lims is not None:
        plt.xlim(lims[0], lims[1])
    plt.title(self.model.parameter_string(), fontsize=9)
    plt.yticks(offset * tick_pos, tick_str)
    G = self.Lehmann_Green_function(k, orb)
    for i in range(len(k)):
        plt.vlines(G[i][0], i*offset, G[i][1]+i*offset, lw=0.8, color='b')
        plt.axhline(i*offset, lw=0.25, color='grey')

    plt.xlabel(r'$\omega$')
    plt.axvline(0, ls='solid', lw=0.5)
    plt.tight_layout()

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()


#---------------------------------------------------------------------------------------------------
def gap(self, k, orb = 1, threshold=1e-3):
    """Computes the spectral gap for a series of wavevectors

    :param k : set of wavevectors
    :param int orb : orbital number (starts at 1)
    :param float threshold : weight below which a Lehmann contribution is deemed zero
    returns: an array of gap values
    """

    if len(k.shape)==1:
        k = np.array([k])
    G = self.Lehmann_Green_function(k, orb)
    g = [None]*len(k)
    for i in range(len(k)):
        x1 = -1e6
        x2 = 1e6
        w = G[i][0]
        r = G[i][1]
        for j in range(len(w)):
            if w[j] < 0 and w[j] > x1 and r[j] > threshold:
                x1 = w[j]
            if w[j] > 0 and w[j] < x2 and r[j] > threshold:
                x2 = w[j]
        g[i] = x2-x1
        print('k = ', k[i], '  x2 = ', x2, ', x1 = ', x1, ', gap = ', x2-x1)
        if(np.abs(g[i])> 1e6):
            g[i] = None
    if len(g) == 1: g = g[0]
    return g


#---------------------------------------------------------------------------------------------------
def plot_DoS(self, w, eta = 0.1, sum=False, progress = True, labels=None, colors=None, file=None, data_file='dos.tsv', plt_ax=None, **kwargs):
    """Plots the density of states (DoS) as a function of frequency

    :param float w: the frequency range is from -w to w if w is a float. If w is a tuple then the range is (wmax[0], wmax[1]). w can also be an explicit list of real frequencies, or of complex frequencies (in which case eta is ignored)
    :param float eta: Lorentzian broadening, if w is real
    :param boolean sum: if True, the sum of the DoS of all lattice orbitals is plotted in addition to each orbital individually
    :param boolean progress: if True, prints computation progress
    :param [str] labels: labels of the different curves
    :param [str] colors: colors of the different curves
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None
    
    """
    from cycler import cycler

    plot = True
    if type(plt_ax) == int:
        if plt_ax == 0:
            plot = False
    
    if plot:
        if plt_ax is None:
            plt.figure()
            plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
            ax = plt.gca()
            plt.title('DoS: '+self.model.parameter_string())
        else:
            ax = plt_ax

    w = __frequency_array(w, eta)

    nw = len(w)
    mix = self.model.mixing
    nband = self.model.nband
    d = nband
    if mix != 0: d *=2
    # reserves space for the DoS
    A = np.zeros((nw, d))
    accum = np.zeros((nw, d))
    total = np.zeros(d)

    # computes the DoS
    eps = (w[1]-w[0]).real
    for i in range(nw):
        z = self.dos(w[i])[0:d]
        total += z*eps
        A[i, :] = z
        if i > 0:
            accum[i, :] = accum[i-1, :] + z*eps
        if(i % 20 == 0 and progress):
            print(np.round(w[i],4), A[i, :])

    head = 'w\t'
    
    for i in range(nband):
        head += 'up_{:d}\t'.format(i+1)
    if mix > 0:
        for i in range(nband):
            head += 'down_{:d}\t'.format(i+1)
    for i in range(nband):
        head += 'cumul_up_{:d}\t'.format(i+1)
    if mix > 0:
        for i in range(nband):
            head += 'cumul_down_{:d}\t'.format(i+1)
    np.savetxt(data_file, np.hstack((np.reshape(np.real(w), (nw, 1)), A, accum)), header=head, delimiter='\t', fmt='%1.6g', comments='')
    print('DoS totals: ', total)

    if plot == False: return
    
    if colors != None:
        plt.rc('axes', prop_cycle=cycler(color=colors))
    if labels is None:
        labels = [str(i+1) for i in range(nband)]
    plt.xlim(w[0].real, w[-1].real)
    for i in range(nband):
        plt.plot(np.real(w), A[:, i], '-', label=labels[i], linewidth=1.6, **kwargs)
    if mix == 1:
        for i in range(nband):
            plt.plot(-np.real(w), A[:, i+nband], '-', label=labels[i], linewidth=0.8, **kwargs)
    elif mix>0:
        for i in range(nband):
            plt.plot(np.real(w), A[:, i+nband], '-', label=labels[i]+'$\downarrow$', linewidth=0.8, **kwargs)
    if sum:
        plt.plot(np.real(w), np.sum(A, 1), 'r-', label = 'total', **kwargs)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho(\omega)$')
    plt.ylim(0)
    plt.axvline(0, c='r', ls='solid', lw=0.5)
    plt.legend()

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()


#---------------------------------------------------------------------------------------------------
def mdc(self, nk=200, eta=0.1, orb=None, spin_down=False, quadrant=False, opt='GF', k_perp = 0, freq = 0.0, max=None, plane = 'xy', size=1.0, band_basis=False, sym=None, file=None, plt_ax=None, **kwargs):
    """Plots the spectral weight at zero frequency in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param float eta: Lorentzian broadening
    :param int orb: if None, sums all the orbitals. Otherwise just shows the weight for that orbital (starts at 1)
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param str opt: The quantity to plot. 'GF' = Green function, 'self' = self-energy, 'Z' = quasi-particle weight
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param float freq: frequency at which the spectral function is computed (0 by default)
    :param float max: maximum value of the plotting range (if None, maximum of the data)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param boolean band_basis: uses the band basis instead of the orbital basis (for multiband models)
    :param str sym: symmetrization option for the mdc
    :param float size: size of the plot, in multiple of the default (2 pi on the side)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: the contour plot object
    
    """
    if spin_down and self.model.mixing != 4:
        raise RuntimeError('spin_down can only be True is mixing = 4')
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax
    ax.set_aspect(1)

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane, size=size)

    # reserves space for the spectral function
    A = np.zeros(nk * nk)
    d = self.model.dimGF_red
    nbands = self.model.nband

    orbs = pyqcm.orbital_manager(orb, from_zero=True)

    # computes the spectral function
    if opt == 'self':
        g = self.self_energy(eta * 1j, k)
        for l in orbs: 
            A += -g[:, l, l].imag

    elif opt == 'Z':
        for l in orbs: 
            A = self.QP_weight(k, eta, l+1)

    elif band_basis:
        for l in orbs: 
            A = -self.band_Green_function(freq + eta * 1j, k, spin_down=spin_down)[:, l, l].imag

    else:
        for l in orbs: 
            A += -self.periodized_Green_function_element(l, l, freq + eta * 1j, k, spin_down=spin_down).imag

    A = np.reshape(A, (nk, nk))

    # acting with symmetries
    if sym != None:
        if quadrant:
            raise RuntimeError('quadrant option must be False is symmetrization is done in mdc()')
        if 'R' in sym:
            A = 0.5*(A + A.T)
        if 'X' in sym:
            A = 0.5*(A + np.flip(A,0))
        if 'Y' in sym:
            A = 0.5*(A + np.flip(A,1))

    axis = ''
    ax1 = 'x'
    ax2 = 'y'
    if plane in ['z', 'xy', 'yx'] and k_perp != 0.0:
        axis = r'$k_z = {:1.3f}\times\pi$'.format(k_perp)
        ax1 = 'x'
        ax2 = 'y'
    if plane in ['y', 'xz', 'zx']:
        axis = r'$k_y = {:1.3f}\times\pi$'.format(k_perp)
        ax1 = 'z'
        ax2 = 'x'
    elif plane in ['x', 'yz', 'zy']:
        axis = r'$k_x = {:1.3f}\times\pi$'.format(k_perp)
        ax1 = 'y'
        ax2 = 'z'
    if freq != 0.0:
        axis += ', $\omega = {:1.3f}$'.format(freq)

    title = axis
    if opt == 'self':
        title = r"$\Sigma''(k)$ : "+title
    elif opt == 'Z':
        title = r"$Z(k)$ : "+title
    else:
        title = r"$A(k)$ : "+title


    if max is None:
        max = A.max()
    else:
        print('maximum level = ', A.max()/max)
    # plot per se
    CS = ax.contourf(x, x, A, np.linspace(0, max, 40), extend="max", cmap='jet', **kwargs)

    if plt_ax is None:
        title = _set_legend_mdc(plane, k_perp)
        if freq != 0.0 :
            title += ', $\omega = {:1.3f}$'.format(freq)
        ax.set_title('mdc: '+title, fontsize=9)
        ax.set_xlabel('$k_'+ax1+'$')
        ax.set_ylabel('$k_'+ax2+'$')
        plt.colorbar(CS, shrink=0.8)
        plt.tight_layout()

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()
    
    return CS



#---------------------------------------------------------------------------------------------------
def spin_mdc(self, nk=200, eta=0.1, orb=None, quadrant=False, opt='spin', freq = 0.0, max=None, k_perp = 0, plane = 'xy', band_basis=False, file=None, plt_ax=None, **kwargs):
    """Plots the spin spectral weight at zero frequency in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param float eta: Lorentzian broadening
    :param int orb: if None, sums all the orbitals. Otherwise just shows the weight for that orbital (starts at 1)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param str opt: The quantity to plot. 'spin' = spin texture, 'spins' = spin texture (saturated), 'sz' = z-component, 'spinp' = modulus of xy-component
    :param float freq: frequency at which the spectral function is computed (0 by default)
    :param float max: maximum value of the plotting range (if None, maximum of the data)
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param band_basis: uses the band basis instead of the orbital basis (for multiband models)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: The contour plot object
    
    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax
    ax.set_aspect(1)

    mix = self.model.mixing
    if mix != 2 and mix != 3:
        raise RuntimeError('spin_mdc() makes sense only if spin-flip terms are present')
    
    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    # reserves space for the spectral function
    A = np.zeros(nk * nk)
    d = self.model.dimGF_red

    # computes the spin spectral function
    S = self.spin_spectral_function(freq+eta*1j, k, orb)
    if opt=='sz':
        A = S[:,3]
    elif opt=='spinp':
        A = np.sqrt(S[:,1]*S[:,1] + S[:,2]*S[:,2])
    elif opt == 'spins':
        Sx = S[:,1]
        Sy = S[:,2]
        if opt == 'spins':
            Smod = np.sqrt(Sx*Sx+Sy*Sy)
            Sx /= (Smod + 0.000001)
            Sy /= (Smod + 0.000001)        

    #...............................................................................................

    axis = ''
    ax1 = 'x'
    ax2 = 'y'
    if plane in ['z', 'xy', 'yx'] and k_perp != 0.0:
        axis = r'$k_z = {:1.3f}\times\pi$'.format(k_perp)
        ax1 = 'x'
        ax2 = 'y'
    if plane in ['y', 'xz', 'zx']:
        axis = r'$k_y = {:1.3f}\times\pi$'.format(k_perp)
        ax1 = 'z'
        ax2 = 'x'
    elif plane in ['x', 'yz', 'zy']:
        axis = r'$k_x = {:1.3f}\times\pi$'.format(k_perp)
        ax1 = 'y'
        ax2 = 'z'
    if freq != 0.0:
        axis += ', $\omega = {:1.3f}$'.format(freq)

    title = axis
    if type(orb) is int:
        title = 'orb {:d} : '.format(orb) + title
    if opt == 'self':
        title = r"$\Sigma''(k)$ : "+title
    elif opt == 'Z':
        title = r"$Z(k)$ : "+title

    #...............................................................................................
    if opt == 'spins':
        X, Y = np.meshgrid(x, x)
        CS = ax.quiver(X, Y, Sx, Sy, pivot='mid', angles='xy', scale_units='xy', scale=10, width=0.003, headlength = 4.5, **kwargs)
    elif opt == 'sz':
        A = np.reshape(A, (nk, nk))
        if max is None:
            max = np.abs(A).max()*1.2
        CS = ax.contourf(x, x, A, np.linspace(-max, max, 40), extend="max", cmap='bwr', **kwargs)
    elif opt == 'spinp':
        A = np.reshape(A, (nk, nk))
        if max is None:
            max = A.max()*1.2
        else:
            print('maximum level (spin) = ', A.max()*1.2/max)
        CS = ax.contourf(x, x, A, np.linspace(0, max, 40), extend="max", cmap='jet', **kwargs)

    #...............................................................................................

    if plt_ax is None:
        title = _set_legend_mdc(plane, k_perp)
        if freq != 0.0 :
            title += ', $\omega = {:1.3f}$'.format(freq)
        ax.set_title('spin texture: '+title, fontsize=9)
        ax.set_xlabel('$k_'+ax1+'$')
        ax.set_ylabel('$k_'+ax2+'$')
        if (opt == 'sz') or (opt == 'spinp'):
            plt.colorbar(CS, shrink=0.8)
        plt.tight_layout()

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

    return CS


#---------------------------------------------------------------------------------------------------
def mdc_anomalous(self, nk=200, w=0.1j, orbitals=(1,1), selfenergy=False, im_part=False, quadrant=False, k_perp=0.0, plane='xy', file=None, plt_ax=None, **kwargs):
    """Plots the anomalous Green function or self-energy (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param complex w: complex frequency at which the Green function is computed
    :param int orbitals: shows the weight for orbitals (b1,b2) (starts at 1), or numpy array of spin-Nambu projection
    :param boolean self: if True, plots the anomalous self-energy instead of the spectral function
    :param boolean im_part: if True, plots the imaginary part instead of the real part
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: for 3D models, value of the component of k perpendicular to the plane
    :param str plane: for 3D models, plane of the plot ('z'='xy', 'y'='xz', 'x='yz')
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None
    
    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title('anomalous mdc: '+self.model.parameter_string(), fontsize=9)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    mix = self.model.mixing
    assert (mix == 1 or mix == 3), 'The system is not anomalous! Abort mdc_anomalous().'

    d = self.model.dimGF_red
    d = d//2
    if type(orbitals) is tuple:
        assert (orbitals[0] > 0 and orbitals[0] <= self.model.nband), 'orbitals is out of range in mdc_anomalous()'
        assert (orbitals[1] > 0 and orbitals[1] <= self.model.nband), 'orbitals is out of range in mdc_anomalous()'

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    # reserves space for the spectral function
    A = np.zeros(nk * nk)

    # computes the spectral function
    if selfenergy:
        g = self.self_energy(w, k)
    else:
        g = self.periodized_Green_function(w, k)
    
    if type(orbitals) is np.ndarray:
        if(orbitals.shape != (d,d)):
            print('the size of argument "orbitals" is not square of side ', d)
            return
        A = np.zeros(nk*nk)
        for i in range(nk*nk):
            if im_part:
                A[i] = (-g[i,0:d,d:2*d]*orbitals).sum().imag
            else:
                A[i] = (-g[i,0:d,d:2*d]*orbitals).sum().real
    else:    
        if im_part:
            A = -g[:, orbitals[0]-1, orbitals[1]+d-1].imag
        else:
            A = -g[:, orbitals[0]-1, orbitals[1]+d-1].real

    A = np.reshape(A, (nk, nk))
    max = np.abs(A).max()


    # plot per se
    CS = plt.contourf(x, x, A, np.linspace(-max, max, 40), extend="max", cmap='bwr')
    plt.colorbar(CS, shrink=0.8)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

#---------------------------------------------------------------------------------------------------
def plot_dispersion(self, nk=64, spin_down=False, orb=None, contour=False, datafile=None, quadrant=False, k_perp = 0, plane = 'xy', file=None, plt_ax=None, view_angle=None, **kwargs):
    """Plots the dispersion relation in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int orb: if None, sums all the orbitals. Otherwise just shows the weight for that orbital (starts at 1)
    :param boolean contour: True if a contour plot is produced instead of a 3D plot.
    :param str datafile: if given, name of the data file (no extension please) in which the data is printed, for plotting with an external program. Does not plot. Will produce one file per orbital, with the .tsv extension.
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param tuple view_angle: optional projection angle to pass to view_init() in the format of (elevation, azimuth) in degrees
    :param kwargs: keyword arguments passed to the matplotlib 'plot_surface' function
    :return: None

    """
    orbs = pyqcm.orbital_manager(orb, from_zero=True)

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        if contour:
            ax = plt.gca()
            plt.gca().set_aspect(1)
        else:
            ax = plt.axes(projection='3d')
    else:
        ax = plt_ax
    
    if view_angle is not None:
        ax.view_init(*view_angle) # adjusts viewing angle if specified

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    d = self.model.dimGF_red
    e = self.dispersion(k, spin_down=spin_down)
    k.shape = (nk, nk, 3)
    e.shape = (nk, nk, d)

    if datafile != None:
        for j in range(d):
            np.savetxt(datafile+'_'+str(j)+'.tsv', e[:,:,j-1])
        return

    print('plotting...')

    if contour:
        if len(orbs) > 1:
            print('Contour plots of the dispersion with more than one orbital make no sense visually! first label used only')
        CS = plt.contour(x, x, e[:, :, orbs[0]], linewidths=0.5)
        ax.clabel(CS, inline=True, fontsize=9)
    else:
        x, y = np.meshgrid(x, x)
        for j in orbs:
            ax.plot_surface(x, y, e[:, :, j], rstride=1,cstride=1, linewidth=0.2, antialiased=False, **kwargs)
            
    if plt_ax is None:
        axis = _set_legend_mdc(plane, k_perp)
        plt.title(axis+' '+self.model.parameter_string(), fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

#---------------------------------------------------------------------------------------------------
def segment_dispersion(self, path='triangle', nk=64, file=None, plt_ax=None, orb = None, **kwargs):
    """Plots the dispersion relation in the Brillouin zone along a wavevector path

    :param str path: wavevector path, as used by the function wavevector_path()
    :param int nk: number of wavevectors on each side of the grid
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param orb: orbital (or sequence of orbitals) to plot. None for all.
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """
    
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title(self.model.parameter_string(), fontsize=9)
    else:
        ax = plt_ax

    k, tick_pos, tick_str = pyqcm.wavevector_path(nk, path)  # defines the array of wavevectors
    d = self.model.dimGF_red
    e = self.dispersion(k)

    if orb == None:
        orb = range(1,d+1)

    for i in orb:
        ax.plot(e[:,i-1], label=str(i+1), **kwargs)

    if self.model.mixing == 4:
        e = self.dispersion(k, True)
        for i in orb:
            ax.plot(e[:,i-1], label=str(i+1), **kwargs)

    
    for x in tick_pos:
        ax.axvline(x, ls='solid', lw=0.5)
    ax.axhline(0, ls='solid', lw=0.5, color='r')
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_str)
    ax.set_xlim(0,len(k)-1)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

#---------------------------------------------------------------------------------------------------
def Fermi_surface(self, nk=64, orb=None, quadrant=False, plane='xy', k_perp=0.0, file=None, plt_ax=None, **kwargs):
    """Plots the Fermi surface of the non-interacting model (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param int orb: if None, plots all the orbitals. Otherwise just plots the FS for that orbital (starts at 1)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param float k_perp: momentum component in the third direction (in multiple of :math:`\pi`)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """
    orbs = pyqcm.orbital_manager(orb, from_zero=True)
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax
    ax.set_aspect(1)
    
    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    d = self.model.dimGF_red
    e = self.dispersion(k)
    k = k*2
    k.shape = (nk, nk, 3)
    e.shape = (nk, nk, d)

    for j in orbs:
        plt.contour(x, x, e[:, :, j], levels=[0.0], **kwargs)

    if plt_ax is None:
        axis = _set_legend_mdc('xy', 0.0)
        plt.title('Fermi surface: '+axis+' '+self.model.parameter_string(), fontsize=9)
    
    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

#---------------------------------------------------------------------------------------------------
def G_dispersion(self, nk=64, orb=None, period = 'G', contour=False, inv=False, quadrant=False, datafile=None, max=None, k_perp = 0.0, plane = 'xy', file=None, plt_ax=None, **kwargs):
    """Plots the eigenvalues of the inverse Green function at zero frequency

    :param int nk: number of wavevectors on each side of the grid
    :param int orb: if 0, plots all the orbitals. Otherwise just shows the plot for that orbital (starts at 1)
    :param str period: periodization scheme ('G', 'M')
    :param boolean contour: True for a contour plot; otherwise a 3D plot.
    :param boolean inv: True if the inverse eigenvalues (inverse energies) are plotted instead
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param str datafile: if different from None, just writes the data in a file and does not plot
    :param float max: energy range (from -max to max) 
    :param float k_perp: momentum component in the third direction (in multiple of :math:`pi`)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None
    
    """

    orbs = pyqcm.orbital_manager(orb, from_zero=True)
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        if contour:
            ax = plt.gca()
            plt.gca().set_aspect(1)
        else:
            ax = plt.axes(projection='3d')
        plt.title('G dispersion: '+self.model.parameter_string(), fontsize=9)
    else:
        ax = plt_ax

    freq = 0.1j

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    pyqcm.set_global_parameter('periodization', period)
    if period == 'None':
        d = self.Green_function_dimension()
        g = self.CPT_Green_function(freq, k)
    else:
        d = self.model.dimGF_red
        g = self.periodized_Green_function(freq, k)
        
    print('plotting...')

    k.shape = (nk, nk, 3)
    g.shape = (nk, nk, d, d)
    e = np.empty((nk,nk,d))
    for i in range(nk):
        for j in range(nk):
            e[i,j,:] = np.linalg.eigvalsh(g[i,j,:,:])

    if not inv:
        e = 1.0/e

    if datafile != None:
        np.savetxt(datafile,e[:,:,orb])
        return

    if contour:
        if len(orbs) >1:
            print('Contour plots of the dispersion with more than one orbital make no sense visually! first value used')
        A = e[:, :, orbs[0]]
        CS = plt.contour(x, x, A, **kwargs)
        ax.clabel(CS, inline=True, fontsize=9)
    else:    
        x, y = np.meshgrid(x, x)
        if max != None:
            ax.set_zlim3d(-max,max)
        if quadrant:
            plt.xticks((0, 0.5, 1), ('$0$', '$\pi/2$', '$\pi$'))
            plt.yticks((0, 0.5, 1), ('$0$', '$\pi/2$', '$\pi$'))
        else:
            plt.xticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
            plt.yticks((-1, 0, 1), ('$-\pi$', '$0$', '$\pi$'))
        for j in orbs:
            ax.plot_surface(x, y, e[:, :, j], rstride=1,cstride=1, linewidth=0.2, antialiased=False, **kwargs)

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()


#---------------------------------------------------------------------------------------------------
def Luttinger_surface(self, nk=200, orb=1, quadrant=False, k_perp = 0, plane = 'xy',  file=None, plt_ax=None, **kwargs):
    """Plots the Luttinger surface (zeros of the Green function) in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param int orb: orbital number (starts at 1)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: for 3D models, value of the component of k perpendicular to the plane
    :param str plane: for 3D models, plane of the plot ('z'='xy', 'y'='xz', 'x='yz')
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title('Luttinger surface : '+self.model.parameter_string(), fontsize=9)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)
    
    g = self.periodized_Green_function(0.0, k)
    A = g[:,orb-1,orb-1].real
    A = np.reshape(A, (nk, nk))
    A = 1.0/A
    CS = plt.contour(x, x, A, levels=[0], **kwargs)

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()




#---------------------------------------------------------------------------------------------------
def plot_momentum_profile(self, op, nk=50, quadrant=False, k_perp=0.0, plane='xy', file=None, plt_ax=None, **kwargs):
    """Plots the momentum-resolved average of operator op in the Brillouin zone (2D)

    :param str op: name of the lattice operator
    :param int nk: number of wavevectors on each side of the grid
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None
    
    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title('profile of '+op+' : '+self.model.parameter_string(), fontsize=9)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    # reserves space for the average
    A = self.momentum_profile(op, k)
    A = np.reshape(A, (nk, nk))
    max = np.abs(A).max()

    # plot per se
    CS = plt.contourf(x, x, A, np.linspace(-max, max, 40), extend="max", cmap='bwr')
    plt.colorbar(CS, shrink=0.8)

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()


#---------------------------------------------------------------------------------------------------
def plot_host_hybrid(self, w, e, clus=0, file=None, plt_ax=None, **kwargs):
    """
    Plots a comparison between the host function and the hybridization function

    :param [float] w : array of frequencies used
    :param (int,int) e: matrix element to plot (zero based)
    :param int clus: cluster label (starts at 0)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :return: None

    """

    H = pyqcm.qcm.get_CDMFT_host(clus, self.label)
    assert(H.shape[0] == w.shape[0])

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    hyb = np.empty(H.shape, dtype=complex)
    for i in range(w.shape[0]):
        hyb[i,:,:] = self.hybridization_function(w[i]*1j, clus)

    ax.plot(w, -H[:,e[0],e[1]].real,'bo-',label='host (real)', lw=1, **kwargs)
    ax.plot(w, -H[:,e[0],e[1]].imag,'bo--',label='host (imag)', lw=1, **kwargs)
    ax.plot(w, hyb[:,e[0],e[1]].real,'rs-',label='hyb. (real)', lw=1, ms=4, **kwargs)
    ax.plot(w, hyb[:,e[0],e[1]].imag,'rs--',label='hyb. (imag)', lw=1, ms=4, **kwargs)
    if plt_ax is None:
        ax.legend()
        ax.set_xlabel('$i\omega_n$')
        if file is not None:
            plt.savefig(file)
            plt.close()
        else:
            plt.show()


#---------------------------------------------------------------------------------------------------
def Berry_curvature(self, nk=200, eta=0.0, period='G', range=None, orb=None, subdivide=False, plane='xy', k_perp=0.0, file=None, plt_ax=None, **kwargs):
    """Draws a 2D density plot of the Berry curvature as a function of wavevector, on a square grid going from -pi to pi in each direction.
    
    :param int nk: number of wavevectors on the side of the grid
    :param float eta: imaginary part of the frequency at zero, i.e., w = eta*1j
    :param str period: type of periodization used (e.g. 'G', 'M', 'None')
    :param list range: range of plot [originX, originY, side], in multiples of pi
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

    B = qcm.Berry_curvature(k1, k2, nk, orb, subdivide, dir, self.label)
    B *= (2*range[2]/nk)**2

    ax.set_aspect(1)

    # plot per se
    max = np.abs(B).max()
    CS = ax.imshow(np.flip(B,0), vmin=-max, vmax = max, cmap='bwr', extent=ext, **kwargs)
    if plt_ax is None:
        axis = _set_legend_mdc(plane, k_perp)
        plt.colorbar(CS, shrink=0.8, extend='neither')
        ax.set_title(axis, fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()
    
    return CS


#---------------------------------------------------------------------------------------------------
def Chern_number(self, nk=100, eta=0.0, period='G', offset=[0., 0., 0.], orb=None, subdivide=False):
    """Computes the Chern number by summing the Berry curvature over wavevectors on a square grid going from (0,0) to (pi,pi)

    :param int nk: number of wavevectors on the side of the grid
    :param float eta: imaginary part of the frequency at zero, i.e., w = eta*1j
    :param str period: type of periodization used (e.g. 'G', 'M', 'None')
    :param wavevector offset: wavevector offset of the computation grid
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
    B = qcm.Berry_curvature(np.array([0.0, 0.0, 0.0])+offset, np.array([1.0, 1.0, 0.0])+offset, nk, orb, subdivide, 3, self.label)
    C = B.sum() / (nk * nk)
    if period == 'None':
        C /= pyqcm.model_size()[0]
    return C
         


#---------------------------------------------------------------------------------------------------
def monopole(self, k, a=0.01, nk=20, orb=None, subdivide=False):
    """computes the topological charge of a node in a Weyl semi-metal

    :param [double] k: wavevector, position of the node
    :param float a: half-side of the cube surrounding the node 
    :param int nk: number of divisions along the side of the cube
    :param int orb: orbital to compute the charge of (if None, sums over all bands)
    :param booleean subdivide: True if subdivision is allowed (False by default)
    :return: the monopole charge
    :rtype: float

    """

    if orb is None:
        orb=0 # this is a quick and simple uniformity change across pyqcm functions

    return qcm.monopole(0.5*np.array(k), a*0.5, nk, orb, subdivide, self.label)
    

#---------------------------------------------------------------------------------------------------
def Berry_flux(self, k0, R, nk=40, plane='xy', orb=None):
    """Computes the integral of the Berry connexion along a closed circle
    
    :param int k0: center of the circle
    :param float R: radius of the circle
    :param int nk: number of wavevectors on the circle
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param int orb: the orbital to use in the computation (1 to number of bands). None (default) means a sum over all bands.
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

    return qcm.Berry_flux(k, orb, self.label)

#---------------------------------------------------------------------------------------------------
def monopole_map(self, nk=40, orb=None, plane='z', k_perp=0.0, file=None, plt_ax = None, **kwargs):
    """Creates a plot of the monopole density (divergence of B) as a function of wavevector

    :param int nk: number of wavevector grid points on each side
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str k_perp: offset in wavevector in the direction perpendicular to the plane (x pi)
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
        axis = _set_legend_mdc(plane, k_perp)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    K = pyqcm.wavevector_grid(nk, orig=[-1.0, -1.0], side=2, k_perp = k_perp, plane=plane)
    B = np.zeros(nk*nk)
    for i, k in enumerate(K):
        B[i] = self.monopole(2.0*k, a=2.0/nk, nk=5, orb=orb)

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

#---------------------------------------------------------------------------------------------------
def Berry_flux_map(self, nk=40, plane='z', dir='z', k_perp=0.0, orb=None, npoints=4, radius=None, file=None, plt_ax = None, **kwargs):
    """Creates a plot of the Berry flux as a function of wavevector

    :param int nk: number of wavevector grid points on each side
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str dir: direction of flux, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str k_perp: offset in wavevector in the direction perpendicular to the plane (x pi)
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
        axis = _set_legend_mdc(plane, k_perp)
        plt.title('component : '+dir)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    K = pyqcm.wavevector_grid(nk, orig=[-1.0, -1.0], side=2, k_perp = k_perp, plane=plane)
    B = np.zeros(nk*nk)
    if radius is None:
        radius = 0.8/nk
    for i, k in enumerate(K):
        B[i] = self.Berry_flux(2.0*k, radius, nk=npoints, plane=dir, orb=orb)

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

#---------------------------------------------------------------------------------------------------
def Berry_field_map(self, nk=40, nsides = 4, plane='z', k_perp=0.0, orb=None, file=None, plt_ax = None, **kwargs):
    """Creates a plot of the Berry flux as a function of wavevector

    :param int nk: number of wavevector grid points on each side
    :param int nsides: number of sides of the polygon used to compute the circulation of the Berry field.
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str k_perp: offset in wavevector in the direction perpendicular to the plane (x pi)
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
        axis = _set_legend_mdc(plane, k_perp)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    K = pyqcm.wavevector_grid(nk, orig=[-1.0, -1.0], side=2, k_perp = k_perp, plane=plane)
    Bx = np.zeros(nk*nk)
    By = np.zeros(nk*nk)
    Bz = np.zeros(nk*nk)
    for i, k in enumerate(K):
        Bx[i] = self.Berry_flux(2.0*k, 2.0/nk, nk=nsides, plane='x', orb=orb)
        By[i] = self.Berry_flux(2.0*k, 2.0/nk, nk=nsides, plane='y', orb=orb)
        Bz[i] = self.Berry_flux(2.0*k, 2.0/nk, nk=nsides, plane='z', orb=orb)

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


#---------------------------------------------------------------------------------------------------
def plot_profile(self, n_scale=1, bond_scale=1, current_scale=1, spin_scale=1,
                 spin_angle=0, bond_spin_scale=1, singlet_scale=1, triplet_scale=1, file=None, layer=0):
    """Produces a figure of various local quantities on the repeated unit, from averages computed in the ground state wavefunction

    :param float n_scale: scale factor applied to the density
    :param float bond_scale: scale factor applied to the bond charge density
    :param float current_scale: scale factor applied to the currents on the bonds
    :param float spin_scale: scale factor applied to the spins on the sites
    :param float spin_angle: angle at which to draw the spins, from their nominal direction
    :param float bond_spin_scale: scale factor applied to the spins on the bonds 
    :param float singlet_scale: scale factor applied to the singlet pairing amplitudes
    :param float triplet_scale: scale factor applied to the triplet pairing amplitudes
    :param str file: name of the output file for the plot, if not None
    :param float layer: layer number (z coordinate)
        
    """


    S, B = self.site_and_bond_profile()

    # printing on the screen

    print('\nsite profiles:')
    for x in S:
        # if np.abs(x[2] - z) > 0.01 :
        #     continue
        print('({: 1.1f},{: 1.1f},{: 1.1f}) : n={: 1.4f}    S=({: 1.4f},{: 1.4f})    psi={: 1.4f}'.format(
            x[0], x[1], x[2], x[3], x[4], x[6], x[7] + x[8] * 1j))

    print('\nbond profiles:')
    for x in B:
        # if np.abs(x[2] - z) > 0.01 :
        #     continue
        print("""({: 0.1f},{: 0.1f},{: 1.1f})-({: 0.1f},{: 0.1f},{: 0.1f}) : b0={: 1.3f}    b=({: 1.3f},{: 1.3f},{: 1.3f})\n
                d0={: 1.3f}   d=({: 1.3f},{: 1.3f},{: 1.3f})""".format(
            x[0].real, x[1].real, x[2].real, x[0].imag, x[1].imag, x[2].imag, x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]))

    # plotting

    plt.subplot(221)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlim(np.ma.min(S[:, 0]) - 0.5, np.ma.max(S[:, 0]) + 0.5)
    plt.ylim(np.ma.min(S[:, 1]) - 0.5, np.ma.max(S[:, 1]) + 0.5)

    ns = len(S)
    nb = len(B)
    bond_scale *= 0.7
    t = spin_angle*np.pi/180
    rotation_matrix = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])

    #...............................................................................................
    # normal part

    plt.title('charge and spin', fontsize=9)
    plt.plot(S[:,0], S[:,1], 'ko', ms = 1)

    for i in range(nb):
        if np.abs(B[i,2].real - layer) > 0.01 or np.abs(B[i,2].imag - layer):
            continue
        plt.plot([B[i, 0].real, B[i, 0].imag], [B[i, 1].real, B[i, 1].imag], 'k-', lw=0.5)
        x = 0.5 * (B[i, 0].real + B[i, 0].imag)
        y = 0.5 * (B[i, 1].real + B[i, 1].imag)
        dx = B[i, 0].real - B[i, 0].imag
        dy = B[i, 1].real - B[i, 1].imag
        v = B[i, 3].real * bond_scale
        current = B[i, 3].imag * current_scale
        spin = np.array([B[i, 4].real, B[i, 6].real])
        spin = np.dot(rotation_matrix, spin) * bond_spin_scale
        sx = spin[0]
        sz = spin[1]

        if v > 0:
            ax.add_patch(patches.Ellipse((x,y), v, 0.3*v, angle = np.degrees(np.arctan2(dy,dx)), facecolor='blue', alpha=0.4))
        else:
            ax.add_patch(patches.Ellipse((x,y), v, 0.3*v, angle = np.degrees(np.arctan2(dy,dx)), facecolor='red', alpha=0.4))
        if np.abs(current) > 0.01 :
            ax.add_patch(patches.Arrow(x-0.5*current*dx, y-0.5*current*dy, current*dx, current*dy, width=0.2, fc = 'red', ec = 'black', lw = 0.5))
        if np.sqrt(sx*sx+sz*sz)>0.01 :
            ax.add_patch(patches.Arrow(x-0.5*sx, y-0.5*sz, sx, sz, width=0.2, fc = 'orange', ec = 'black', lw = 0.5))

    for i in range(ns):
        if np.abs(S[i,2] - layer) > 0.01 :
            continue
        x = S[i, 0]
        y = S[i, 1]
        n = S[i, 3]-1
        spin = np.array([S[i, 4], S[i, 6]])
        spin = np.dot(rotation_matrix,spin)*spin_scale
        sx = spin[0]
        sz = spin[1]
        if n>0 :
            ax.add_patch(patches.Circle((x,y), n_scale*n, facecolor='blue', alpha=0.4))
        else:
            ax.add_patch(patches.Circle((x,y), -n_scale*n, facecolor='red', alpha=0.4))
        if np.sqrt(sx*sx+sz*sz)>0.01 :
            ax.add_patch(patches.Arrow(x-0.5*sx, y-0.5*sz, sx, sz, width=0.1, fc = 'red', ec = 'black', lw = 0.5, zorder=100))

    #...............................................................................................
    # singlet part

    plt.subplot(222)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlim(np.ma.min(S[:,0]) - 1, np.ma.max(S[:,0]) + 1)
    plt.ylim(np.ma.min(S[:,1]) - 1, np.ma.max(S[:,1]) + 1)

    plt.title('singlet pairing', fontsize=9)
    plt.plot(S[:,0], S[:,1], 'ko', ms = 1)

    for i in range(nb):
        if np.abs(B[i,2].real - layer) > 0.01 or np.abs(B[i,2].imag - layer):
            continue
        plt.plot([B[i, 0].real, B[i, 0].imag], [B[i, 1].real, B[i, 1].imag], 'k-', lw = 0.5)
        x = 0.5*(B[i, 0].real+B[i, 0].imag)
        y = 0.5*(B[i, 1].real+B[i, 1].imag)
        dx = B[i, 0].real-B[i, 0].imag
        dy = B[i, 1].real-B[i, 1].imag
        v = np.abs(B[i, 7])*singlet_scale
        col = _color(B[i, 7])
        ax.add_patch(patches.Ellipse((x,y), v, 0.3*v, angle = np.degrees(np.arctan2(dy,dx)), facecolor=col, alpha=0.4))

    for i in range(ns):
        if np.abs(S[i,2] - layer) > 0.01 :
            continue
        x = S[i, 0]
        y = S[i, 1]
        psi = S[i, 7]+S[i, 8]*1.0j
        v = np.abs(psi)*singlet_scale
        col = _color(psi)
        ax.add_patch(patches.Circle((x,y), v, facecolor=col, alpha=0.4))

    #...............................................................................................
    # dz part

    plt.subplot(223)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlim(np.ma.min(S[:,0]) - 1, np.ma.max(S[:,0]) + 1)
    plt.ylim(np.ma.min(S[:,1]) - 1, np.ma.max(S[:,1]) + 1)

    plt.title('triplet pairing (d_z)', fontsize=9)
    plt.plot(S[:,0], S[:,1], 'ko', ms = 1)

    for i in range(nb):
        if np.abs(B[i,2].real - layer) > 0.01 or np.abs(B[i,2].imag - layer):
            continue
        plt.plot([B[i, 0].real, B[i, 0].imag], [B[i, 1].real, B[i, 1].imag], 'k-', lw = 0.5)
        x = 0.5*(B[i, 0].real+B[i, 0].imag)
        y = 0.5*(B[i, 1].real+B[i, 1].imag)
        dx = B[i, 0].real-B[i, 0].imag
        dy = B[i, 1].real-B[i, 1].imag
        v = np.abs(B[i, 10])*triplet_scale
        z = B[i, 10]
        if np.abs(np.angle(z)) > np.pi/2:
            z = -z
            v *= -1
        col = _color(z)
        ax.add_patch(patches.Arrow(x-0.5*v*dx, y-0.5*v*dy, v*dx, v*dy, width=0.5, fc = col, ec = 'black', lw = 0.5, zorder=100))

    #...............................................................................................
    # dx, dy part

    plt.subplot(224)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlim(np.ma.min(S[:,0]) - 1, np.ma.max(S[:,0]) + 1)
    plt.ylim(np.ma.min(S[:,1]) - 1, np.ma.max(S[:,1]) + 1)

    plt.title('triplet pairing (d_x and d_y)', fontsize=9)
    plt.plot(S[:,0], S[:,1], 'ko', ms = 1)

    for i in range(nb):
        if np.abs(B[i,2].real - layer) > 0.01 or np.abs(B[i,2].imag - layer):
            continue
        plt.plot([B[i, 0].real, B[i, 0].imag], [B[i, 1].real, B[i, 1].imag], 'k-', lw = 0.5)
        x = 0.5*(B[i, 0].real+B[i, 0].imag)
        y = 0.5*(B[i, 1].real+B[i, 1].imag)
        dx = B[i, 0].real-B[i, 0].imag
        dy = B[i, 1].real-B[i, 1].imag

        spinR = np.array([B[i, 8].real, B[i, 9].real])
        spinI = np.array([B[i, 8].imag, B[i, 9].imag])
        spinR = np.dot(rotation_matrix,spinR)*triplet_scale
        spinI = np.dot(rotation_matrix,spinI)*triplet_scale
        sx = spinR[0]
        sz = spinR[1]
        if np.sqrt(sx*sx+sz*sz) > 0.01 :
            ax.add_patch(patches.Arrow(x-0.5*sx, y-0.5*sz, sx, sz, width=0.2, fc = 'blue', ec = 'black', lw = 0.5, zorder=100))
        sx = spinI[0]
        sz = spinI[1]
        if np.sqrt(sx*sx+sz*sz) > 0.01 :
            ax.add_patch(patches.Arrow(x-0.5*sx, y-0.5*sz, sx, sz, width=0.2, fc = 'red', ec = 'black', lw = 0.5, zorder=100))

    plt.gcf().suptitle(self.model.parameter_string()+', layer '+str(layer), y=1.0, fontsize=9)
    plt.tight_layout()

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
