import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyqcm

################################################################################
def set_legend_mdc(plane, k_perp):

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

################################################################################
def __frequency_array(wmax=6.0, eta=0.05, matsubara=False):
    """Returns an array of complex frequencies for plotting spectral quantities

    """

    if type(wmax) is tuple:
        w = np.arange(wmax[0], wmax[1] + 1e-6, eta/4.0)  # defines the array of frequencies
    elif type(wmax) is float or type(wmax) is int:
        w = np.arange(-wmax, wmax + 1e-6, eta/4.0)  # defines the array of frequencies
    elif type(wmax) is np.ndarray and wmax.dtype == complex:
        return wmax
    else:
        raise TypeError('the type of argument "wmax" in __frequency_array() is wrong')

    if matsubara:
        wc = w*1j
    else:
        wc = np.array([x + eta*1j for x in w], dtype=complex)

    return wc


################################################################################
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

################################################################################
def spectral_function(wmax=6.0, eta=0.05, path='triangle', nk=32, label=0, band=None, offset=2, opt='A', Nambu_redress=True, inverse_path=False, title=None, file=None, plt_ax=None, **kwargs):
    """Plots the spectral function :math:`A(\mathbf{k},\omega)` along a wavevector path in the Brillouin zone.
    This version plots the spin-down part with the correct sign of the frequency in the Nambu formalism.

    :param float wmax: the frequency range is from -wmax to wmax if w is a float. If wmax is a tuple then the range is (wmax[0], wmax[1]). wmax can also be an explicit list of real frequencies
    :param float eta: Lorentzian broadening
    :param str path: a keyword that is passed to pyqcm.wavevector_path() to produce a set of wavevectors along a path, or a tuple 
    :param int nk: the number of wavevectors along each segment of the path (passed to pyqcm.wavevector_grid())
    :param int label: label of the instance of the model
    :param int band: if not None, only plots the spectral function associated with this orbital number (starts at 1). If None, sums over all bands.
    :param float offset: vertical offset in the plot between the curves associated to successive wavevectors
    :param str opt: 'A' : spectral function, 'self' : self-energy, 'Sx' : spin (x component), 'Sy' : spin (y component)
    :param boolean Nambu_redress: if True, evaluates the Nambu component at the opposite frequency
    :param boolean inverse_path: if True, inverts the path (k --> -k)
    :param str title: optional title for the plot. If None, a string with the model parameters will be used.
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    dim, nbands = pyqcm.reduced_Green_function_dimension()
    mix = pyqcm.mixing()

    if band is not None:
        assert (band <= nbands and band > 0), 'The band index in plot_spectrum() must vary from 1 to {:d}'.format(nbands)
        band -= 1

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
            g = pyqcm.self_energy(w[i], k, False, label)
        else:
            g = pyqcm.periodized_Green_function(w[i], k, False, label)
        if opt=='Sx':
            assert mix&2, 'option Sx in spectral_function() only makes sense if spin-flip terms are present'
        if opt=='Sy':
            assert mix&2, 'option Sy in spectral_function() only makes sense if spin-flip terms are present'
        for j in range(len(k)):
            if band is None:
                for l in range(nbands): 
                    A[i, j] += -g[j, l, l].imag
            else:
                A[i, j] += -g[j, band, band].imag

            if mix&2:  
                plot_down = True
                if band is None:
                    for l in range(nbands): 
                        A_down[i, j] += -g[j, nbands+l, nbands+l].imag
                else:
                    A_down[i, j] += -g[j, nbands+band, nbands+band].imag

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
                g = pyqcm.self_energy(W, k, False, label)
            else:
                g = pyqcm.periodized_Green_function(W, k, False, label)
            for j in range(len(k)):
                plot_down = True
                if band is None:
                    for l in range(nbands): 
                        A_down[i, j] += -g[j, nbands+l, nbands+l].imag
                else:
                    A_down[i, j] += -g[j, nbands+band, nbands+band].imag

    if mix == 4:
        plot_down = True
        # add the contribution to the spin-down channel in that case
        for i in range(len(w)):
            if opt=='self':
                g = pyqcm.self_energy(w[i], k, True, label)
            else:
                g = pyqcm.periodized_Green_function(w[i], k, True, label)
            for j in range(len(k)):
                if band is None:
                    for l in range(nbands): 
                        A_down[i, j] += -g[j, l, l].imag
                else:
                    A_down[i, j] += -g[j, band, band].imag
    

    ax.set_xlim(np.real(w[0]), np.real(w[-1]))
    ax.set_ylim(0, (1+len(k)) * offset + 1 / eta)
    for j in range(len(k)):
        if plot_down:
            ax.plot(np.real(w), A_down[:, j] + offset * j, 'r-', lw=0.5, **kwargs)
        ax.plot(np.real(w), A[:, j] + offset * j, 'b-', lw=0.5, **kwargs)
    if title is None and plt_ax is None:
        ax.set_title(r'$A(\mathbf{k},\omega)$: '+pyqcm.parameter_string(), fontsize=9)
    else:
        ax.set_title(title, fontsize=9)    
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
        

################################################################################
def hybridization_function(wmax=6, eta=0.01, matsubara=False, clus = 0, realpart=False, label=0, file=None, plt_ax=None, **kwargs):
    """This function plots the imaginary part of the hybridization function Gamma as a function of frequency.
    Only the diagonal elements are plotted, but for all clusters if there is more than one.
    The arguments have the same meaning as in `plot_spectrum`, except 'realpart' which, if True, plots
    the real part instead of the imaginary part.

    :param float wmax: the frequency range is from -wmax to wmax if w is a float. If wmax is a tuple then the range is (wmax[0], wmax[1]). wmax can also be an explicit list of real frequencies
    :param float eta: Lorentzian broadening
    :param boolean matsubara: If True, the frequency range is along the imaginary frequency axis
    :param int clus: cluster index (starts at 0)
    :param boolean realpart: if True, the real part of the Green function is shown, not the imaginary part
    :param int label: label of the model instance
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    w = __frequency_array(wmax, eta, matsubara)  # defines the array of frequencies
    eta = 0.05j
    info = pyqcm.cluster_info()
    d = info[clus][3]
    d = pyqcm.Green_function_dimension()
    print(info)
    nclus = len(info)
    print('number of clusters = ', nclus)
    print('dimension of Green function matrix = ', d)
    A = np.zeros((len(w), d*d))
    for i in range(len(w)):
        g = pyqcm.hybridization_function(clus, w[i], False, label)
        for l1 in range(d):
            for l2 in range(d):
                l = l1 + d*l2
                if realpart:
                    A[i, l] += g[l1, l2].real
                else:
                    A[i, l] += -g[l1, l2].imag

    offset = 2
    if matsubara:
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
        plt.title(r'$\Gamma(\omega)$: '+pyqcm.parameter_string())

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()


################################################################################
def cluster_spectral_function(wmax=6, eta = 0.05, matsubara=False, clus=0, label=0, offset=2, full=False, opt=None, spin_down=False, blocks=False, file=None, plt_ax=None, realpart=False, color = 'b', **kwargs):
    """Plots the spectral function of the cluster in the site basis
    
    :param float wmax: the frequency range is from -wmax to wmax if w is a float. If wmax is a tuple then the range is (wmax[0], wmax[1]). wmax can also be an explicit list of real frequencies
    :param float eta: Lorentzian broadening
    :param boolean matsubara: If True, the frequency range is along the imaginary frequency axis
    :param int clus: label of the cluster within the super unit cell (starts at 0)
    :param int label: label of the model instance
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
    :returns: the array of frequencies, the spectral weight

    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    w = __frequency_array(wmax, eta, matsubara)  # defines the array of frequencies
    info = pyqcm.cluster_info()
    d = info[clus][3]
    if full:
        dd = (d*(d+1))//2
        T = []
        for j in range(d):
            for k in range(j+1):
                T += ['({0:d},{1:d})'.format(j+1,k+1)]
        plt.yticks(offset*np.arange(0, dd), T)
    else:
        dd = d
        plt.yticks(offset*np.arange(0, d), [str(i) for i in range(1,d+1)])

    A = np.zeros((len(w), dd))
    for i in range(len(w)):
        if opt is None:
            g = pyqcm.cluster_Green_function(clus, w[i], spin_down, label, blocks) # run of the mill cluster green function
        elif opt == "self":
            g = pyqcm.cluster_self_energy(clus, w[i], spin_down, label) # self-energy functionnal
        elif opt == "hyb":
            g = pyqcm.hybridization_function(clus, w[i], realpart, label) # hybridization function
        else:
            raise ValueError(f"'{opt}' is not a valid option, must be one of 'self', 'hyb' or None.")
        if full:
            l = 0
            for j in range(d):
                for k in range(j+1):
                    A[i, l] += -g[j, k].imag
                    l += 1
        else:        
            for j in range(d):
                A[i, j] += -g[j, j].imag

    max = np.max(A)
    plt.ylim(0, dd * offset + max)

    if matsubara:
        ax.set_xlim(np.imag(w[0]), np.imag(w[-1]))
        for j in range(dd):
            plt.plot(np.imag(w), A[:, j] + offset * j, '-', lw=0.5, color=color, **kwargs)
    else:
        ax.set_xlim(np.real(w[0]), np.real(w[-1]))
        for j in range(dd):
            plt.plot(np.real(w), A[:, j] + offset * j, '-', lw=0.5, color=color, **kwargs)

    plt.xlabel(r'$\omega$')
    plt.axvline(0, ls='solid', lw=0.5)
    plt.title(pyqcm.parameter_string(), fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

    # return w.real, A    # why returning this? check. Could be triggered by optional argument.


################################################################################
def spectral_function_Lehmann(path='triangle', nk=32, label=0, band=1, offset=0.1, lims=None, file=None, plt_ax=None, **kwargs):
    """Plots a Lehmann representation of the spectral function along a wavevector path in the Brillouin zone. Singularities are plotted as impulses with heights proportionnal to the residue.
    
    :param path: if a string, keyword passed to `pyqcm.wavevector_path()` to produce a set of wavevectors; else, explicit list of wavevectors (N x 3 numpy array).
    :param int nk: the number of wavevectors along each segment of the path (passed to pyqcm.wavevector_path())
    :param int label: label of the instance of the model
    :param int band: only plots the spectral function associated with this orbital number (starts at 1)
    :param float offset: vertical offset in the plot between the curves associated to successive wavevectors
    :param (float,float) lims: limits of the plot in frequency (2-tuple)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """
    
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    d, nbands = pyqcm.reduced_Green_function_dimension()

    k, tick_pos, tick_str = pyqcm.wavevector_path(nk, path)  # defines the array of wavevectors

    assert (band <= nbands and band > 0), 'The band index in spectral_function_Lehmann() must vary from 1 to {:d}'.format(nbands)

    if lims is not None:
        plt.xlim(lims[0], lims[1])
    plt.title(pyqcm.parameter_string(), fontsize=9)
    plt.yticks(offset * tick_pos, tick_str)
    G = pyqcm.Lehmann_Green_function(k, band, label=label)
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


################################################################################
def gap(k, band = 1, threshold=1e-3):
    """Computes the spectral gap for a series of wavevectors

    :param k : set of wavevectors
    :param int band : band number (starts at 1)
    :param float threshold : weight below which a Lehmann contribution is deemed zero
    returns: an array of gap values
    """

    if len(k.shape)==1:
        k = np.array([k])
    G = pyqcm.Lehmann_Green_function(k, band)
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
    return g


################################################################################
def DoS(w, eta = 0.1, label=0, sum=False, progress = True, labels=None, colors=None, file=None, data_file='dos.tsv', plt_ax=None, **kwargs):
    """Plots the density of states (DoS) as a function of frequency

    :param float w: the frequency range is from -w to w if w is a float. If w is a tuple then the range is (wmax[0], wmax[1]). w can also be an explicit list of real frequencies, or of complex frequencies (in which case eta is ignored)
    :param float eta: Lorentzian broadening, if w is real
    :param int label: label of the model instance 
    :param boolean sum: if True, the sum of the DoS of all bands is plotted in addition to each band individually
    :param boolean progress: if True, prints computation progress
    :param [str] labels: labels of the different curves
    :param [str] colors: colors of the different curves
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None
    
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
            plt.title('DoS: '+pyqcm.parameter_string())
        else:
            ax = plt_ax

    w = __frequency_array(w, eta)

    nw = len(w)
    mix = pyqcm.mixing()
    nsites = pyqcm.model_size()[1]
    d = nsites
    if mix != 0: d *=2
    # reserves space for the DoS
    A = np.zeros((nw, d))
    accum = np.zeros((nw, d))
    total = np.zeros(d)

    # computes the DoS
    eps = (w[1]-w[0]).real
    for i in range(nw):
        z = pyqcm.dos(w[i], label)[0:d]
        total += z*eps
        A[i, :] = z
        if i > 0:
            accum[i, :] = accum[i-1, :] + z*eps
        if(i % 20 == 0 and progress):
            print(np.round(w[i],4), A[i, :])

    head = 'w\t'
    
    for i in range(nsites):
        head += 'up_{:d}\t'.format(i+1)
    if mix > 0:
        for i in range(nsites):
            head += 'down_{:d}\t'.format(i+1)
    for i in range(nsites):
        head += 'cumul_up_{:d}\t'.format(i+1)
    if mix > 0:
        for i in range(nsites):
            head += 'cumul_down_{:d}\t'.format(i+1)
    np.savetxt(data_file, np.hstack((np.reshape(np.real(w), (nw, 1)), A, accum)), header=head, delimiter='\t', fmt='%1.6g', comments='')
    print('DoS totals: ', total)
    mix = pyqcm.mixing()

    if plot == False: return
    
    if colors != None:
        plt.rc('axes', prop_cycle=cycler(color=colors))
    if labels is None:
        labels = [str(i+1) for i in range(nsites)]
    plt.xlim(w[0].real, w[-1].real)
    for i in range(nsites):
        plt.plot(np.real(w), A[:, i], '-', label=labels[i], linewidth=1.6, **kwargs)
    if mix == 1 or mix == 5:
        for i in range(nsites):
            plt.plot(-np.real(w), A[:, i+nsites], '-', label=labels[i], linewidth=0.8, **kwargs)
    elif mix>0:
        for i in range(nsites):
            plt.plot(np.real(w), A[:, i+nsites], '-', label=labels[i]+'$\downarrow$', linewidth=0.8, **kwargs)
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


################################################################################
def mdc(nk=200, eta=0.1, label=0, band=None, spin_down=False, quadrant=False, opt='GF', k_perp = 0, freq = 0.0, max=None, plane = 'xy', size=1.0, band_basis=False, sym=None, file=None, plt_ax=None, **kwargs):
    """Plots the spectral weight at zero frequency in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param float eta: Lorentzian broadening
    :param int label: label of the model instance 
    :param int band: if None, sums all the bands. Otherwise just shows the weight for that band (starts at 1)
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param str opt: The quantity to plot. 'GF' = Green function, 'self' = self-energy, 'Z' = quasi-particle weight
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param float freq: frequency at which the spectral function is computed (0 by default)
    :param float max: maximum value of the plotting range (if None, maximum of the data)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param band_basis: uses the band basis instead of the orbital basis (for multiband models)
    :param str sym: symmetrization option for the mdc
    :param float size: size of the plot, in multiple of the default (2 pi on the side)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: the contour plot object
    
    """
    if spin_down and pyqcm.mixing() != 4:
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
    d, nbands = pyqcm.reduced_Green_function_dimension()

    # computes the spectral function
    if opt == 'self':
        g = pyqcm.self_energy(eta * 1j, k)
        if band is None:
            for l in range(nbands): 
                A += -g[:, l, l].imag
        else:
            A = -g[:, band-1, band-1].imag

    elif opt == 'Z':
        A = pyqcm.QP_weight(k, eta, band)

    else:
        if band is None:
            for l in range(d): 
                A += -pyqcm.periodized_Green_function_element(l, l, freq + eta * 1j, k, spin_down=spin_down, label=label).imag
        elif not band_basis:
            A = -pyqcm.periodized_Green_function_element(band-1, band-1, freq + eta * 1j, k, spin_down=spin_down, label=label).imag
        else:
            A = -pyqcm.band_Green_function(freq + eta * 1j, k, spin_down=spin_down, label=label)[:, band-1, band-1].imag

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
    if band != None:
        title = 'band {:d} : '.format(band) + title
    if opt == 'self':
        title = r"$\Sigma''(k,0)$ : "+title
    elif opt == 'Z':
        title = r"$Z(k,0)$ : "+title

    if max is None:
        max = A.max()
    else:
        print('maximum level = ', A.max()/max)
    # plot per se
    CS = ax.contourf(x, x, A, np.linspace(0, max, 40), extend="max", cmap='jet', **kwargs)

    if plt_ax is None:
        title = set_legend_mdc(plane, k_perp)
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



################################################################################
def spin_mdc(nk=200, eta=0.1, label=0, band=None, quadrant=False, opt='spin', freq = 0.0, max=None, k_perp = 0, plane = 'xy', band_basis=False, file=None, plt_ax=None, **kwargs):
    """Plots the spin spectral weight at zero frequency in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param float eta: Lorentzian broadening
    :param int label: label of the model instance 
    :param int band: if None, sums all the bands. Otherwise just shows the weight for that band (starts at 1)
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
    :returns: The contour plot object
    
    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax
    ax.set_aspect(1)

    mix = pyqcm.mixing()
    if mix != 2 and mix != 3:
        raise RuntimeError('spin_mdc() makes sense only if spin-flip terms are present')
    
    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    # reserves space for the spectral function
    A = np.zeros(nk * nk)
    d, nbands = pyqcm.reduced_Green_function_dimension()

    # computes the spin spectral function
    S = pyqcm.spin_spectral_function(freq+eta*1j, k, band, label)
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

#------------------------------------------------------------------

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
    if band != None:
        title = 'band {:d} : '.format(band) + title
    if opt == 'self':
        title = r"$\Sigma''(k,0)$ : "+title
    elif opt == 'Z':
        title = r"$Z(k,0)$ : "+title

#------------------------------------------------------------------
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

#------------------------------------------------------------------

    if plt_ax is None:
        title = set_legend_mdc(plane, k_perp)
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


################################################################################
def mdc_anomalous(nk=200, w=0.1j, label=0, bands=(1,1), self=False, im_part=False, quadrant=False, k_perp=0.0, plane='xy', file=None, plt_ax=None, **kwargs):
    """Plots the anomalous Green function or self-energy (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param complex w: complex frequency at which the Green function is computed
    :param int label: label of the model instance 
    :param int bands: shows the weight for (b1,b2) (starts at 1), or numpy array of spin-Nambu projection
    :param boolean self: if True, plots the anomalous self-energy instead of the spectral function
    :param boolean im_part: if True, plots the imaginary part instead of the real part
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: for 3D models, value of the component of k perpendicular to the plane
    :param str plane: for 3D models, plane of the plot ('z'='xy', 'y'='xz', 'x='yz')
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None
    
    """
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title('anomalous mdc: '+pyqcm.parameter_string(), fontsize=9)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    mix = pyqcm.mixing()
    assert (mix == 1 or mix == 3), 'The system is not anomalous! Abort mdc_anomalous().'

    d, nbands = pyqcm.reduced_Green_function_dimension()
    d = d//2
    if type(bands) is tuple:
        assert (bands[0] > 0 and bands[0] <= nbands), 'bands is out of range in mdc_anomalous()'
        assert (bands[1] > 0 and bands[1] <= nbands), 'bands is out of range in mdc_anomalous()'

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    # reserves space for the spectral function
    A = np.zeros(nk * nk)

    # computes the spectral function
    if self:
        g = pyqcm.self_energy(w, k, label=label)
    else:
        g = pyqcm.periodized_Green_function(w, k, label=label)
    
    if type(bands) is np.ndarray:
        if(bands.shape != (d,d)):
            print('the size of argument "bands" is not square of side ', d)
            return
        A = np.zeros(nk*nk)
        for i in range(nk*nk):
            if im_part:
                A[i] = (-g[i,0:d,d:2*d]*bands).sum().imag
            else:
                A[i] = (-g[i,0:d,d:2*d]*bands).sum().real
    else:    
        if im_part:
            A = -g[:, bands[0]-1, bands[1]+d-1].imag
        else:
            A = -g[:, bands[0]-1, bands[1]+d-1].real

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

################################################################################
def plot_dispersion(nk=64, label=0, spin_down=False, band=None, contour=False, datafile=None, quadrant=False, k_perp = 0, plane = 'xy', file=None, plt_ax=None, view_angle=None, **kwargs):
    """Plots the dispersion relation in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param int label: label of the model instance 
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int band: if None, sums all the bands. Otherwise just shows the weight for that band (starts at 1)
    :param boolean contour: True if a contour plot is produced instead of a 3D plot.
    :param str datafile: if given, name of the data file (no extension please) in which the data is printed, for plotting with an external program. Does not plot. Will produce one file per band, with the .tsv extension.
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param tuple view_angle: optional projection angle to pass to view_init() in the format of (elevation, azimuth) in degrees
    :param kwargs: keyword arguments passed to the matplotlib 'plot_surface' function
    :returns: None

    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        if contour:
            ax = plt.gca()
            plt.gca().set_aspect(1)
        else:
            ax = plt.gca(projection='3d')
    else:
        ax = plt_ax
    
    if view_angle is not None:
        ax.view_init(*view_angle) # adjusts viewing angle if specified

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    d, nbands = pyqcm.reduced_Green_function_dimension()
    e = pyqcm.dispersion(k, label=label, spin_down=spin_down)
    k.shape = (nk, nk, 3)
    e.shape = (nk, nk, d)

    if datafile != None:
        for j in range(d):
            np.savetxt(datafile+'_'+str(j)+'.tsv', e[:,:,j-1])
        return

    print('plotting...')

    if contour:
        if band is None:
            print('Contour plots of the dispersion with more than one band make no sense visually! band set to 1')
            band=1
        CS = plt.contour(x, x, e[:, :, band-1], linewidths=0.5)
        ax.clabel(CS, inline=True, fontsize=9)
    else:
        x, y = np.meshgrid(x, x)
        if band is None:
            for j in range(d):
                ax.plot_surface(x, y, e[:, :, j], rstride=1,cstride=1, linewidth=0.2, antialiased=False, **kwargs)
        else:
            ax.plot_surface(x, y, e[:, :, band-1], rstride=1,cstride=1, linewidth=0.2, antialiased=False, **kwargs)
            
    if plt_ax is None:
        axis = set_legend_mdc(plane, k_perp)
        plt.title(axis+' '+pyqcm.parameter_string(), fontsize=9)

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

################################################################################
def segment_dispersion(path='triangle', nk=64, label=0, file=None, plt_ax=None, **kwargs):
    """Plots the dispersion relation in the Brillouin zone along a wavevector path

    :param str path: wavevector path, as used by the function wavevector_path()
    :param int nk: number of wavevectors on each side of the grid
    :param int label: label of the model instance 
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """
    
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title(pyqcm.parameter_string(), fontsize=9)
    else:
        ax = plt_ax

    k, tick_pos, tick_str = pyqcm.wavevector_path(nk, path)  # defines the array of wavevectors
    d, nbands = pyqcm.reduced_Green_function_dimension()
    e = pyqcm.dispersion(k, label=label)

    for i in range(d):
        ax.plot(e[:,i], **kwargs)

    if pyqcm.mixing() == 4:
        e = pyqcm.dispersion(k, True)
        for i in range(d):
            ax.plot(e[:,i], **kwargs)

    
    for x in tick_pos:
        ax.axvline(x, ls='solid', lw=0.5)
    ax.axhline(0, ls='solid', lw=0.5, color='r')
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_str)
    ax.set_xlim(0,len(k))

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

################################################################################
def Fermi_surface(nk=64, label=0, band=None, quadrant=False, plane='xy', k_perp=0.0, file=None, plt_ax=None, **kwargs):
    """Plots the Fermi surface of the non-interacting model (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param int label: label of the model instance 
    :param int band: if None, plots all the bands. Otherwise just plots the FS for that band (starts at 1)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param float k_perp: momentum component in the third direction (in multiple of :math:`\pi`)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax
    ax.set_aspect(1)
    
    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    d, nbands = pyqcm.reduced_Green_function_dimension()
    e = pyqcm.dispersion(k, label=label)
    k = k*2
    k.shape = (nk, nk, 3)
    e.shape = (nk, nk, d)

    if band is None:
        for j in range(d):
            plt.contour(x, x, e[:, :, j], levels=[0.0], **kwargs)
    else:
        plt.contour(x, x, e[:, :, band-1], levels=[0.0], **kwargs)

    if plt_ax is None:
        axis = set_legend_mdc('xy', 0.0)
        plt.title('Fermi surface: '+axis+' '+pyqcm.parameter_string(), fontsize=9)
    
    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()

################################################################################
def G_dispersion(nk=64, label=0, band=None, period = 'G', contour=False, inv=False, quadrant=False, datafile=None, max=None, k_perp = 0.0, plane = 'xy', file=None, plt_ax=None, **kwargs):
    """Plots the eigenvalues of the inverse Green function at zero frequency

    :param int nk: number of wavevectors on each side of the grid
    :param int label: label of the model instance 
    :param int band: if 0, plots all the bands. Otherwise just shows the plot for that band (starts at 1)
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
    :returns: None
    
    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        if contour:
            ax = plt.gca()
            plt.gca().set_aspect(1)
        else:
            ax = plt.gca(projection='3d')
        plt.title('G dispersion: '+pyqcm.parameter_string(), fontsize=9)
    else:
        ax = plt_ax

    freq = 0.1j

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    pyqcm.set_global_parameter('periodization', period)
    if period == 'None':
        d = pyqcm.Green_function_dimension()
        g = pyqcm.CPT_Green_function(freq, k, label=label)
    else:
        d, nbands = pyqcm.reduced_Green_function_dimension()
        g = pyqcm.periodized_Green_function(freq, k, label=label)
        
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
        np.savetxt(datafile,e[:,:,band])
        return

    if contour:
        if band is None:
            print('Contour plots of the dispersion with more than one band make no sense visually! band set to 1')
            band=1
        A = e[:, :, band-1]
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
        if band is None:
            for j in range(d):
                ax.plot_surface(x, y, e[:, :, j], rstride=1,cstride=1, linewidth=0.2, antialiased=False, **kwargs)
        else:
            ax.plot_surface(x, y, e[:, :, band], rstride=1,cstride=1, linewidth=0.2, antialiased=False, **kwargs)

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()


################################################################################
def Luttinger_surface(nk=200, label=0, band=1, quadrant=False, k_perp = 0, plane = 'xy',  file=None, plt_ax=None, **kwargs):
    """Plots the Luttinger surface (zeros of the Green function) in the Brillouin zone (2D)

    :param int nk: number of wavevectors on each side of the grid
    :param int label: label of the model instance 
    :param int band: band number (starts at 1)
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: for 3D models, value of the component of k perpendicular to the plane
    :param str plane: for 3D models, plane of the plot ('z'='xy', 'y'='xz', 'x='yz')
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title('Luttinger surface : '+pyqcm.parameter_string(), fontsize=9)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)
    
    g = pyqcm.periodized_Green_function(0.0, k, label=label)
    A = g[:,band-1,band-1].real
    A = np.reshape(A, (nk, nk))
    A = 1.0/A
    CS = plt.contour(x, x, A, levels=[0], **kwargs)

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()




################################################################################
def momentum_profile(op, nk=50, label=0, quadrant=False, k_perp=0.0, plane='xy', file=None, plt_ax=None, **kwargs):
    """Plots the momentum-resolved average of operator op in the Brillouin zone (2D)

    :param str op: name of the lattice operator
    :param int nk: number of wavevectors on each side of the grid
    :param int label: label of the model instance 
    :param boolean quadrant: if True, plots the first quadrant of a square Brillouin zone only
    :param float k_perp: momentum component in the third direction (in multiple of pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None
    
    """

    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(14/2.54, 14/2.54)
        ax = plt.gca()
        plt.title('profile of '+op+' : '+pyqcm.parameter_string(), fontsize=9)
    else:
        ax = plt_ax
    ax.set_aspect(1)

    k, x = __kgrid(ax, nk, quadrant=quadrant, k_perp=k_perp, plane=plane)

    # reserves space for the average
    A = pyqcm.momentum_profile(op, k, label=label)
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