################################################################################
# This file contains functions implementing 
# Cluster dynamical mean-field theory (CDMFT)
################################################################################

from multiprocessing.sharedctypes import Value
import numpy as np
import pyqcm
from pyqcm import qcm
import timeit

#-------------------------------------------------------------------------------
# global variables

var = []
w = []
wr = []
weight = []
first_time = True
first_time2 = True
Hyb = []
Hyb_down = []
hybrid_to_param = None

Gdim = 0
nclus = 0
nmixed = 1
_mixing = 0
clusters = None
maxfev = 5000000
E0_VMC = 0.0
E0_err_VMC = 0.0
min_iter_E0 = 5

################################################################################
# PRIVATE FUNCTIONS
################################################################################

def moving_std(x, min):
    """finds the subsequence with the smallest standard deviation, with minimum size min
    
    :param [float] x: sequence 
    :param int min: minimum length of subsequence

    """
    n = len(x)
    if n<min :
        raise ValueError('the sequence given to moving_std is too short (should be >= {:d})'.format(min))
    if min<2 :
        raise ValueError('the minimum value of argument "min" in moving_std() is 2')

    std = np.empty(n-1) # stores the standard deviations of the subsequences
    for i in range(n-1):
        std[i] = np.std(x[n-i-2:n])/np.sqrt(i+2) # the length of the subsequence is i+2
    I = np.argmin(std[min-2:]) + min - 2
    return std[I], np.mean(x[n-I-2:n]), I+2

################################################################################
# CLASSES
######################################################################
class observable:
    def __init__(self, name, tol, min_length=2):
        self.name = name
        self.tol = tol
        self.val = np.zeros(32)
        self.ave = 999
        self.length = 0
        self.min_length = min_length

    def __repr__(self):
        return self.name + '(tol = {:.2g})'.format(self.tol)

    def test_convergence(self, iter, x):
        if iter > len(self.val):
            self.val.resize(len(self.val) + 32)
        self.val[iter] = x
        if iter+1 < self.min_length:
            return False
        std, self.ave, self.length = moving_std(self.val[0:iter+1], self.min_length)
        if std < self.tol:
            return True
        else:
            return False
        


######################################################################
class solution:    
    def __init__(self, x, f, nfev, success, mess):
        self.x = x
        self.fun = f
        self.nfev = nfev
        self.success = success
        self.message = mess

    def __str__(self):
        S = ''
        if not self.success:
            S += 'optimization failure!\n'
        S += self.x.__str__()
        S += '\nf = {:f}\n'.format(self.fun)
        S += '{:d} function evaluations\n'.format(self.nfev)
        S += self.message
        return S


######################################################################
def __check_bounds(x, B=100, v=None):
    """Checks whether one of the variables in the array x is out of bounds

    :param [float] x: array of parameters
    :param float B: maximum value of all parameters
    :param [str] v: list of corresponding names, if available
    :return: None
    """
    for i in range(len(x)):
        if np.abs(x[i]) > B:
            if v != None:
                print('parameter ', v[i], ' is out of bounds!!!')
            else:
                print('parameter no ', i+1, ' is out of bounds!!!')
            raise pyqcm.OutOfBoundsError(i)


######################################################################
def __set_Hyb(spin_down=False):
    """Computes the hybridization function

    :param boolean spin_down: if True, considers the spin down sector (when mixing=4)
    :returns: an array of arrays of matrices. Hyb[i], for cluster #i, is a (nw,d,d) Numpy array. with nw frequencies, and d sites.

    """
    global w, Gdim, nclus, nmixed
    nw = len(w)
    Hyb = []
    for j in range(nclus):
        d = clusters[j]*nmixed
        Hyb.append(np.zeros((nw, d, d), dtype=np.complex128))

    for i in range(nw):
        for j in range(nclus):
            Hyb[j][i, :, :] = pyqcm.hybridization_function(j, w[i], spin_down)

    return Hyb

def __print_Hyb(Hyb):
    global w, Gdim, nclus, nmixed
    print('\nHost function:')
    for i,f in enumerate(w):
        print('w = ', f)
        print(Hyb[0][i,:,:])

######################################################################
def __frequency_grid(type='sharp', beta='50', wc=2):
    """
    constructs a grid of frequencies along the imaginary axis for the distance function

    :param str type: type of grid ('sharp', 'ifreq', 'self')
    :param float beta : inverse temperature
    :param float wc : cutoff
    :returns str dist_function

    """
    global w, wr, weight

    wr = np.arange((np.pi / beta), wc + 1e-6, 2 * np.pi / beta)
    w = np.ones(len(wr), dtype=np.complex128)
    w = w * 1j
    w *= wr
    nw = len(w)
    if type == 'sharp':
        weight = np.ones(nw)
        weight *= 1.0 / nw
        dist_function = 'sharp_wc_{0:.1f}_b_{1:d}'.format(wc, int(beta))
    elif type == 'ifreq':
        weight = 1.0/wr
        weight *= 1.0 / weight.sum()
        dist_function = 'ifreq_wc_{0:.1f}_b_{1:d}'.format(wc, int(beta))
    elif type == 'self':
        weight = np.zeros(nw)
        Sig_inf = pyqcm.cluster_self_energy(0, 1.0e6j)
        for i, x in enumerate(w):
            Sig = pyqcm.cluster_self_energy(0, x) - Sig_inf
            weight[i] = np.linalg.norm(Sig)
        weight *= 1.0 / weight.sum()
        dist_function = 'self_wc_{0:.1f}_b_{1:d}'.format(wc, int(beta))
    else:
        raise pyqcm.WrongArgumentError(f"frequency_grid set as type `{type}`")
    return dist_function

######################################################################
# hybridization function array
def __diff_hybrid(hyb1, hyb2):
    """
    Computes a difference in hybridization functions between the current iteration and the previous one

    :param [ndarray] hyb1: first set of hybridization matrices (one per cluster)
    :param [ndarray] hyb2: second set of hybridization matrices (one per cluster)
    :returns float: the difference in hybridization arrays
    
    """
    global nclus, nmixed, w, weight, _mixing
    nw = len(w)

    diff = 0.0
    for i in range(nw):
        for j in range(nclus):
            diffH = hyb1[j][i,:,:] - hyb2[j][i,:,:]
            norm = np.linalg.norm(diffH)
            diff += weight[i]*norm*norm

    if _mixing == 0:
        diff *= 2
    elif _mixing == 3:
        diff /= 2
    diff /= nw  
    return np.sqrt(diff)


######################################################################
# optimization of the bath parameters

def __optimize(F, x, method='Nelder-Mead', initial_step=0.1, accur = 1e-4, accur_dist = 1e-8):
    """

    :param F: function to be optimized
    :param (float) x: array of variables
    :param str method: method to use, as used in scipy.optimize.minimize()
    :param float initial_step: initial step in the minimization procedure
    :param float accur: requested accuracy in the parameters 
    :param float accur_dist: requested accuracy in the distance function

    """
    import scipy.optimize
    global maxfev
    displaymin = False
    nvar = len(x)
    if method == 'Nelder-Mead':
        initial_simplex = np.zeros((nvar+1,nvar))
        for i in range(nvar+1):
            initial_simplex[i, :] = x
        for i in range(nvar):
            initial_simplex[i+1, i] += initial_step
        sol = scipy.optimize.minimize(F, x, method='Nelder-Mead', options={'disp':displaymin, 'maxfev':maxfev, 'xatol': accur, 'fatol': accur_dist, 'initial_simplex': initial_simplex})
        iter_done = sol.nit

    elif method == 'Powell':
        sol = scipy.optimize.minimize(F, x, method='Powell', tol = accur)
        iter_done = sol.nit

    elif method == 'CG':
        sol = scipy.optimize.minimize(F, x, method='CG', jac=False, tol = accur, options={'gtol':1e-6, 'eps':1e-6})
        iter_done = sol.nit

    elif method == 'BFGS':
        sol = scipy.optimize.minimize(F, x, method='BFGS', jac=False, tol = accur, options={'eps':accur})
        iter_done = sol.nit

    elif method == 'COBYLA':
        sol = scipy.optimize.minimize(F, x, method='COBYLA', options={'disp':displaymin, 'rhobeg':initial_step, 'maxiter':maxfev, 'tol': accur_dist})
        iter_done = sol.nfev
        
    elif method == 'ANNEAL':
        sol = scipy.optimize.basinhopping(F, x, minimizer_kwargs = {'method':'COBYLA', 'options':{'disp':displaymin, 'rhobeg':initial_step, 'maxiter':maxfev, 'tol': accur_dist}})
        iter_done = sol.nfev

    else:
        raise ValueError(f'unknown method specified for minimization: {method}')

    if not sol.success:
        print(sol.message)
        raise pyqcm.MinimizationError()
    return sol, iter_done

################################################################################
# PUBLIC FUNCTIONS
################################################################################

######################################################################
# main CDMFT function

def cdmft(
    varia=None, 
    beta=50, 
    wc=2.0, 
    maxiter=32, 
    miniter=0, 
    accur=1e-3, 
    accur_hybrid=1e-4, 
    accur_dist=1e-10,
    accur_E0=1e-10,
    alpha = 0.0,
    displaymin=False, 
    method='Nelder-Mead', 
    file='cdmft.tsv', 
    skip_averages=False, 
    eps_algo=0, 
    initial_step = 0.1, 
    hartree=None, 
    check_sectors=None, 
    grid_type = 'sharp', 
    counterterms=None, 
    SEF=False, 
    observables=None,
    verb=False,
    max_function_eval = 5000000
):
    """Performs the CDMFT procedure

    :param [str] varia: list of variational parameters 
        OR tuple of two lists : bath energies and bath hybridizations
        OR function that returns dicts of bath energies and bath hybridizations given numeric arrays
    :param float beta: inverse fictitious temperature (for the frequency grid)
    :param float wc: cutoff frequency (for the frequency grid)
    :param int maxiter: maximum number of CDMFT iterations
    :param int miniter: minimum number of CDMFT iterations
    :param float accur: the procedure converges if parameters do not change by more than accur
    :param float accur_hybrid: the procedure converges on the hybridization function with this accuracy
    :param float accur_dist: convergence criterion when minimizing the distance function.
    :param float accur_E0: convergence criterion on the impurity ground state energy.
    :param float alpha: damping parameter (fraction of the previous iteration in the new one) OR (float,int) with number of iterations where damping is used (at the beginning if positive, at the end if negative)
    :param boolean displaymin: displays the minimum distance function when minimized
    :param str method: method to use, as used in scipy.optimize.minimize()
    :param str file: name of the file where the solution is written
    :param boolean skip_averages: if True, does NOT compute the lattice averages of the converged solution
    :param int eps_algo: number of elements in the epsilon algorithm convergence accelerator = 2*eps_algo + 1 (0 = no acceleration)
    :param float initial_step: initial step in the minimization routine
    :param [class hartree] hartree: mean-field hartree couplings to incorportate in the convergence procedure
    :param boolean check_sectors: the ground state is checked against the ground states of the sectors contained in target_sectors
    :param str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self'
    :param [str] counterterms: list of counterterms names (cluster operators that should strive to have zero average)
    :param boolean SEF: if True, computes the Potthoff functional at the end
    :param [class observable]: list of observables used to assess convergence
    :param boolean verb: If True, prints debugging information
    :param int max_function_eval: maximum number of distance function evaluations when minimizing distance
    :returns: None

    """
    global w, wr, weight, var, _mixing, first_time, first_time2, Gdim, nclus, nmixed, clusters, maxfev, Hyb, Hyb_down, E0_VMC, E0_err_VMC, min_iter_E0

    maxfev = max_function_eval
    if type(hartree) is not list and hartree is not None:
        hartree = [hartree] # quick fix to protect against hartree=`obj` behavior

    nvar_E = 0
    if type(varia) is tuple:
        nvar_E = len(varia[0])
        nvar_H = len(varia[1])
        var = varia[0] + varia[1]
    else:
        var = varia
        
    begin_with_damping = False
    n_damping = 1
    if type(alpha) is tuple:
        n_damping = alpha[1]
        alpha = alpha[0]
        if n_damping > 0:
            begin_with_damping = True
        else:
            n_damping = -n_damping


    if hartree is None:
        pyqcm.banner('CDMFT procedure', '*', skip=1)
    else:
        pyqcm.banner('CDMFT procedure (combined with Hartree procedure)', '*', skip=1)
        if eps_algo:
            for C in hartree:
                C.init_epsilon(maxiter, eps_algo)

    # identifying the variational parameters
    nvar = len(var)
    qcm.CDMFT_variational_set(var)
    var_data = np.empty((nvar, maxiter+1))
    if nvar == 0:
        raise ValueError('CDMFT requires at least one variational parameter...Aborting.')

    print('minimization method = ', method)
    _mixing = pyqcm.mixing()
    print('mixing state = ', _mixing)
    Gdim = pyqcm.Green_function_dimension()
    nsites, nbands, clusters, bath_size, ref = pyqcm.model_size()
    clusters = np.array(clusters)
    bath_size = np.array(bath_size)
    nclus = len(clusters)
    nmixed = Gdim//nsites
    observable_series_length = 0

    # counterterms
    CT_converged = True
    
    # storing the GS energy and error (the error is relevant for the DVMC solver)
    E0 = np.zeros(maxiter+1)
    E0_err = np.ones(maxiter+1)
    moving_ave = np.zeros(maxiter+1)


    # convergence criterion in the bath parameters
    superiter = 0
    diff_hartree = 0.0
    hartree_converged = True

    # checks on the observables
    if observables != None:
        for x in observables:
            if x.name[-2:] != '_1':
                raise ValueError('observables must be cluster operators associated with cluster 1 only')

    # first define the frequency grid for the distance function
    print('frequency grid type = ', grid_type)
    print('fictitious inverse temperature = ', beta)
    print('frequency cutoff = ', wc)
    print('-'*80)

    params_array = np.zeros(len(var))
    
    # CDMFT loop
    converged = False
    diffH=1e6
    diff_E0 = 0.0
    time_ED = 0.0
    time_MIN = 0.0
    while True:
        pyqcm.new_model_instance()
        params = pyqcm.parameters() # params is a dict

        # puts the values only of the parameters into array params_array
        for i in range(nvar):
            params_array[i] = params[var[i]]
        var_data[:, superiter] = params_array
        __check_bounds(params_array, 1000.0, v=var)


        # check convergence in the DVMC case
        if pyqcm.solver == 'dvmc':
            print('E0_VMC = ', E0_VMC, '\tE0_err_VMC = ', E0_err_VMC)
            E0[superiter] = E0_VMC
            E0_err[superiter] = E0_err_VMC
        else:
            gs = pyqcm.ground_state()
            for x in gs:
                E0[superiter] += x[0]
        if superiter >= min_iter_E0-1:
            moving_ave[superiter] = 0.0
            tmp_norm = 0.0
            for i in range(min_iter_E0):
                tmp_norm += 1.0/E0_err[superiter-i]
                moving_ave[superiter] += E0[superiter-i]/E0_err[superiter-i]
            moving_ave[superiter] /= tmp_norm
        if superiter >= min_iter_E0 and superiter >= miniter:
            diff_E0 = np.abs(moving_ave[superiter]-moving_ave[superiter-1])
            if diff_E0 < accur_E0:
                converged = True
                # print('moving averages of E0 : ', moving_ave[0:superiter+1])
                pyqcm.banner('CDMFT converged on the ground state energy', '=')
                break

        # initializes G_host
        t1 = timeit.default_timer()
        if superiter > 0:
            Hyb0 = Hyb
            if _mixing == 4:
                Hyb_down0 = Hyb_down
        dist_function = __frequency_grid(grid_type, beta, wc)
        qcm.CDMFT_host(wr, weight)
        Hyb = __set_Hyb()
        # if verb: __print_Hyb(Hyb)
        if _mixing == 4:
            Hyb_down = __set_Hyb(True)

        t2 = timeit.default_timer()
        time_ED += t2 - t1
        if type(method) == tuple:
            sol = __dual_minimization(params_array, nvar_E, method, initial_step=initial_step, accur=accur, accur_dist=accur_dist, nsteps=12)
            iter_done = sol.nfev
        else:
            sol, iter_done = __optimize(lambda x : qcm.CDMFT_distance(x, 0), params_array, method, initial_step, accur, accur_dist)

        t3 = timeit.default_timer()
        time_MIN += t3 - t2

        if method != 'ANNEAL' and not sol.success:
            print(sol)
            raise pyqcm.MinimizationError()

        if sol.nfev > maxfev:
            print(sol)
            print('number of function evaluations exceeds preset maximum of ', maxfev)
            raise pyqcm.MinimizationError()

        dist_value = sol.fun
        # updating the bath parameters (replace old by new)

        if alpha > 0.0 and ((superiter < n_damping and begin_with_damping is True) or (superiter > n_damping and begin_with_damping is False)):
            print('applying damping with alpha = ', alpha)
            for i in range(nvar):
                pyqcm.set_parameter(var[i], (1-alpha)*sol.x[i]+alpha*params_array[i])
        else:
            for i in range(nvar):
                pyqcm.set_parameter(var[i], sol.x[i])


        diff_param = np.linalg.norm(params_array - sol.x)/np.sqrt(nvar)
        initial_step = diff_param
        if superiter > 0:
            diffH = __diff_hybrid(Hyb, Hyb0)
            print('\nCDMFT iteration {:d}, distance = {: #.2e}, diff param = {: #.2e}, diff hybrid = {: #.2e}, diff E0 = {: #.2g}\n{:d} minimization steps, time(MIN)/time(ED)={:.5f}'.format(superiter+1, dist_value, diff_param, diffH, diff_E0, iter_done, time_MIN/time_ED), flush=True)
        else:
            print('\nCDMFT iteration {:d}, distance = {: #.2e}, diff param = {: #.2e}\n{:d} minimization steps, time(MIN)/time(ED)={:.5f}'.format(superiter+1, dist_value, diff_param, iter_done, time_MIN/time_ED), flush=True)

        #--------------------------------- Hartree step ---------------------------------
        if hartree != None:
            hartree_converged = True
            hartree_ave = np.zeros(nclus)
            diff_hartree = 0
            diff_hartree_rel = 0
            for C in hartree:
                C.update(pr=True)
                diff_hartree += np.abs(C.diff)
                hartree_converged = hartree_converged and C.converged()
        #--------------------------------------------------------------------------------

        #--------------------------------- counterterms ---------------------------------
        if counterterms != None:
            CT_converged = True
            for C in counterterms:
                C.update(first_time2)
                CT_converged = CT_converged and C.converged()
        #--------------------------------------------------------------------------------


        # writing the parameters in a progress file
        des = 'distance\tdiff_param\tdiff_hybrid\t'
        val = '{: #.2e}\t{: #.2e}\t{: #.2e}\t'.format(dist_value, diff_param, diffH)
        pyqcm.write_summary('cdmft_iter.tsv', suppl_descr = des, suppl_values = val, first_of_series=first_time2)
        first_time2 = False


        #______________________________________________________________________
                
        # checking convergence on the parameters (note the sqrt(nvar) factor in order not to punish large parameter sets)
        if (diff_param < accur) and hartree_converged and CT_converged and superiter > miniter:
            converged = True
            pyqcm.banner('CDMFT converged on the parameters', '=')
            break

        # checking convergence on the hybridization matrix
        if superiter > 0:
            diffH = __diff_hybrid(Hyb, Hyb0)
            if _mixing == 4:
                diffH += __diff_hybrid(Hyb_down, Hyb_down0)
            if (diffH < accur_hybrid) and hartree_converged and superiter > miniter:
                pyqcm.banner('CDMFT converged on the hybridization function', '=')
                converged = True
                break

        # checking convergence on the observables
        # for the moment, this works only for observables belonging to the first cluster
        obs_converged = True
        if observables != None:
            ave = pyqcm.cluster_averages()
            observable_series_length = 0
            for x in observables:
                val = ave[x.name[0:-2]][0]
                Conv = x.test_convergence(superiter, val)
                if x.length > observable_series_length:
                    observable_series_length = x.length
                obs_converged = obs_converged and Conv
                print('observable <{:s}> = {:.6g}'.format(x.name, x.ave))
            if obs_converged and superiter > miniter:
                converged = True
                pyqcm.banner('CDMFT converged on the observables (length of series : {:d})'.format(observable_series_length), '=')
                break

        superiter += 1
        var_val = pyqcm.__varia_table(var,sol.x)
        print(var_val)
        if superiter > maxiter:
            raise pyqcm.TooManyIterationsError(maxiter)

        #--------------------------- convergence acceleration ---------------------------
        eps_length = 2*eps_algo + 1
        if eps_algo and superiter>=2*eps_length and superiter%(2*eps_length) == 0:
            pyqcm.banner('applying the epsilon algorithm')
            for i in range(nvar):
                z = pyqcm.epsilon(var_data[i,superiter-eps_length:superiter])
                var_data[i,superiter] = z
                pyqcm.set_parameter(var[i], z)
            var_val = pyqcm.__varia_table(var,var_data[:,superiter])
            print(var_val)
        #-------------------------------------------------------------------------------


    # here we have converged
    if converged:

        var_val = pyqcm.__varia_table(var,sol.x)
        print(var_val)

        GS0 = pyqcm.ground_state()
        if check_sectors:
            GS = pyqcm.ground_state()
            if GS != GS0:
                raise ValueError(f'Ground state inconsistent : {GS} from the converged parameters and {GS0} initially')

        if skip_averages is False:
            ave = pyqcm.averages()
            pyqcm.print_averages(ave)    

        if SEF:
            omegaH=pyqcm.Potthoff_functional(hartree)

        if file != None:
            des = 'iterations\tdist_function\tdistance\tdiff_hybrid\t'
            val = '{:d}\t{:s}\t{: #.2e}\t{: #.2e}\t'.format(superiter, dist_function, dist_value, diffH)
            if SEF : 
                des += 'omegaH\t'
                val += '{: #.8g}\t'.format(omegaH)
            if observables != None:
                for x in observables:
                    des += 'ave_'+x.name+'_obs\t'
                    val += '{: #.6e}\t'.format(x.ave)
                des += 'series_length\t'
                val += '{:d}\t'.format(observable_series_length)
            pyqcm.write_summary(file, suppl_descr = des, suppl_values = val, first_of_series=first_time)
            first_time = False
            first_time2 = True

        # writing the frequency grid to a file
        if 'self' in dist_function:
            pyqcm.write_summary('cdmft_grid.tsv', suppl_descr = des, suppl_values = val, first_of_series=first_time)
            first_time = False
            with open('cdmft_grid.tsv','a') as gridfile:
                np.savetxt(gridfile, np.stack((w.imag, weight),axis=-1),header='w\tweight', fmt='%.8f', delimiter='\t')
                gridfile.close()


        pyqcm.banner('CDMFT completed successfully', '*')
    else:
        pyqcm.banner('Failure of the CDMFT algorithm', '*')
    
    return superiter, observable_series_length




######################################################################
# perform a sequece of forcing with an external field in CDMFT

def cdmft_forcing(field_name, seq, beta_seq=None, **kwargs):
    """performs a sequence of CDMFT runs with the external field 'field_name' takes the successive values in 'seq'

    :param str field_name: name of the forcing field
    :param [float] seq: sequence of values to be taken by the forcing field
    :param [float] beta_seq: an optional sequence of fictitious inverse temperatures (same length as seq)
    :param kwargs: named parameters passed to the CDMFT function
    :return: None

    """
    if beta_seq:
        assert len(beta_seq) == len(seq), 'the length of beta_seq must be the same as that of varia_seq'
    for i, x in enumerate(seq):
        pyqcm.set_parameter(field_name, x)
        print('--------> ', field_name, ' = ', x)
        if beta_seq != None:
            cdmft(beta = beta_seq[i], **kwargs)
        else:
            cdmft(**kwargs)

######################################################################
# performs a sequence of CDMFT runs with an increasing variational set

def cdmft_variational_sequence(basic_params, varia_seq, **kwargs):
    """performs a sequence of CDMFT runs with an increasing variational set.

    :param str basic_params: specifies non variational parameters, in the format used by set_parameters()
    :param [str] varia_seq: a sequence of strings specifying additional variational parameters and their initial values
    :param kwargs: named parameters passed to the CDMFT function
    :return: None

    """

    P = basic_params
    variaset = []
    for x in varia_seq:
        P += x
        q = pyqcm.set_parameters(x, dump=False)
        for y in q:
            if len(y)==2:
                variaset += [y[0]]

        pyqcm.set_parameters(P)
        cdmft(varia= variaset, **kwargs)

######################################################################
def forcing_sequence(f1, f2, beta1, beta2, n=6):
    """generates logarithmic sequences of fields and temperatures, for use with cdmft_forcing

    :param float f1: high value of the field
    :param float f2: low value of the field
    :param float beta1: low value of the inverse temperature
    :param float beta2: high value of the inverse temperature
    :param int n: number of values in the sequence
    :return ([float],[float]): lists of field and beta values

    """

    l1 = np.log(f1)
    l2 = np.log(f2)
    l = np.linspace(l1,l2,n)
    l = np.around(np.exp(l), 4)
    lb1 = np.log(beta1)
    lb2 = np.log(beta2)
    lb = np.linspace(lb1,lb2,n)
    lb = np.rint(np.exp(lb))
    l = np.append(l,1e-8)
    lb = np.append(lb,lb[-1])
    return list(l), list(lb)





######################################################################
# hybrid minimization routine : the set of variational parameters is separated into two subsets:
# energies and hybridizations

def __dual_minimization(params_array, N, method, initial_step=1e-3, accur=1e-4, accur_dist=1e-8, nsteps=4):
    """
    This is a distance function minimization method that alternates between minimizing for the bath energies and for the 
    bath hybridizations separately, alternately, until convergence. It is sometimes useful in problems with a large
    number of bath parameters.

    :param [float] params_array: initial value of the bath parameters. The first N parameters are bath energies, the remaining ones are hybridizations
    :param int N: number of bath energies (first section of params_array)
    :param (str,str) method: the two scipy.optimize() methods use to optimize for bath energies (method[0]) and bath hybridization (method[1])
    :param float initial_step: initial step size in the minimization procedure
    :param float accur_dist: required tolerance in the distance function to be minimize (stops when the change in distance is lower than accur_dist)
    :param int nsteps: maximum number of alternations between the two minimization procedures (for energies and for hybridizations)
    
    """

    if N == 0:
        raise ValueError('dual minimization requires that the variational parameters be split into energy and hybridization parameters')

    if(len(method) != 2):
        raise ValueError('dual minimization requires that two methods be specified')

    global var, maxfev
    x0 = params_array
    x1 = params_array[0:N]
    x2 = params_array[N:]
    nfev = 0
    tol = accur_dist*30
    nvar2 = len(x2)
    nvar1 = len(x1)
    for i in range(nsteps):
        if tol < accur:
            tol = accur
        else:
            tol /= 2


        # second set of parameters : bath hybridizations ------------------------------------------
        sol, iter = __optimize(lambda x : qcm.CDMFT_distance(np.concatenate((x1, x))), x2, method[1], initial_step, accur, accur_dist)
        try:
            __check_bounds(sol.x, 1000.0)
        except pyqcm.OutOfBoundsError(i):
            raise pyqcm.MinimizationError(i)

        if not sol.success:
            break
        x2 = sol.x
        nfev += sol.nfev

        # first set of parameters : bath energies --------------------------------------------------
        sol, iter = __optimize(lambda x : qcm.CDMFT_distance(np.concatenate((x, x2)),0), x1, method[0], initial_step, accur, accur_dist)
        try:
            __check_bounds(sol.x, 1000.0)
        except pyqcm.OutOfBoundsError():
            raise pyqcm.MinimizationError()

        if not sol.success:
            break
        x1 = sol.x
        nfev += sol.nfev

        xp = np.concatenate((x1, x2))
        diff = np.linalg.norm(x0 - xp)/np.sqrt(len(params_array))
        x0 = xp
        if diff < accur and i>0:
            print('-----> exit after ', i+1, ' internal iterations between the two sets of parameters')
            break

        if i == nsteps-1:
            print('-----> exit without closure : diff = ', diff)

    sol2 = solution(np.concatenate((x1, x2)), sol.fun, nfev, sol.success, sol.message)
    return sol2

################################################################################
# PUBLIC CLASSES
################################################################################
# definition of a general bath for a new cluster model
class general_bath:    

    def  __init__(self, name, ns, nb, spin_dependent=False, spin_flip=False, singlet=False, triplet=False, complex=False, sites=None):
        """Defines a general bath (constructor)

        :param str name: name of the cluster-bath model to be defined
        :param int ns: number of sites in the cluster
        :param int nb: number of bath orbitals in the cluster
        :param boolean spin_dependent: if True, the parameters are spin dependent
        :param boolean spin_flip: if True, spin-flip hybridizations are present
        :param boolean singlet: if True, defines anomalous singlet hybridizations
        :param boolean triplet: if True, defines anomalous triplet hybridizations
        :param boolean complex: if True, defines imaginary parts as well, when appropriate
        :param [[int]] sites: 2-level list of sites to couple to the bath orbitals (labels from 1 to ns). Format resembles [[site labels to bind to orbital 1], ...] . 

        """
        from pyqcm import new_cluster_model, new_cluster_operator, new_cluster_operator_complex
        new_cluster_model(name, ns, nb)
        self.ns = ns
        self.nb = nb
        self.name = name
        self.var_E = []
        self.var_H = []
        self.spin_dependent = spin_dependent
        self.spin_flip = spin_flip
        self.singlet = singlet
        self.triplet = triplet
        self.complex = complex
        no = ns+nb
        if sites is None:
            self.sites = [range(1,ns+1) for i in range(nb)]
        else:
            if len(sites) != nb :
                print('the format of the argument "sites" is incorrect : it should be a list of ', ns, ' lists')
                raise ValueError(f'the format of the argument "sites" is incorrect : it should be a list of {ns} lists')
            self.sites = sites

        self.nmixed = 1
        if spin_flip:
            self.nmixed *= 2
        if singlet or triplet:
            self.nmixed *= 2

        # bath energies
        if spin_dependent or spin_flip:
            for x in range(1,nb+1):
                param_name = 'eb{:d}u'.format(x)
                new_cluster_operator(name, param_name, 'one-body', [(x+ns, x+ns, 1)])
                self.var_E += [param_name]
                param_name = 'eb{:d}d'.format(x)
                new_cluster_operator(name, param_name, 'one-body', [(x+ns+no, x+ns+no, 1)])
                self.var_E += [param_name]
        else:
            for x in range(1,nb+1):
                param_name = 'eb{:d}'.format(x)
                new_cluster_operator(name, param_name, 'one-body', [(x+ns, x+ns, 1), (x+ns+no, x+ns+no, 1)])
                self.var_E += [param_name]

        # hybridizations
        if spin_dependent or spin_flip:
            for x in range(1,nb+1):

                for y in self.sites[x-1]:
                    param_name = 'tb{:d}u{:d}u'.format(x,y)
                    new_cluster_operator(name, param_name, 'one-body', [(y, x+ns, 1)])
                    self.var_H += [param_name]
                    param_name = 'tb{:d}d{:d}d'.format(x,y)
                    new_cluster_operator(name, param_name, 'one-body', [(y+no, x+ns+no, 1)])
                    self.var_H += [param_name]
                    if spin_flip:
                        param_name = 'tb{:d}u{:d}d'.format(x,y)
                        new_cluster_operator(name, param_name, 'one-body', [(x+ns, y+no, 1)])
                        self.var_H += [param_name]
                        param_name = 'tb{:d}d{:d}u'.format(x,y)
                        new_cluster_operator(name, param_name, 'one-body', [(y+no, x+ns+no, 1)])
                        self.var_H += [param_name]

            
                if complex:
                    for y in self.sites[x-1][1:]:
                        param_name = 'tb{:d}u{:d}ui'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'one-body', [(y, x+ns, 1j)])
                        self.var_H += [param_name]
                        param_name = 'tb{:d}d{:d}di'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'one-body', [(y+no, x+ns+no, 1j)])
                        self.var_H += [param_name]
                    if spin_flip:
                        for y in self.sites[x-1]:
                            param_name = 'tb{:d}u{:d}di'.format(x,y)
                            new_cluster_operator_complex(name, param_name, 'one-body', [(x+ns, y+no, 1j)])
                            self.var_H += [param_name]
                            param_name = 'tb{:d}d{:d}ui'.format(x,y)
                            new_cluster_operator_complex(name, param_name, 'one-body', [(y+no, x+ns+no, 1j)])
                            self.var_H += [param_name]
        
        else:
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'tb{:d}{:d}'.format(x,y)
                    new_cluster_operator(name, param_name, 'one-body', [(y, x+ns, 1), (y+no, x+ns+no, 1)])
                    self.var_H += [param_name]

                if complex:
                    for y in self.sites[x-1][1:]:
                        param_name = 'tb{:d}{:d}i'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'one-body', [(y, x+ns, 1j), (y+no, x+ns+no, 1j)])
                        self.var_H += [param_name]

        if singlet:    
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'sb{:d}{:d}'.format(x,y)
                    new_cluster_operator(name, param_name, 'anomalous', [(y, x+ns+no, 1), (x+ns, y+no, 1)])
                    self.var_H += [param_name]
                if complex:
                    for y in self.sites[x-1]:
                        if x==1 and y==1:
                            continue
                        param_name = 'sb{:d}{:d}i'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'anomalous', [(y, x+ns+no, 1j), (x+ns, y+no, 1j)])
                        self.var_H += [param_name]

        if triplet:    
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'pb{:d}{:d}'.format(x,y)
                    new_cluster_operator(name, param_name, 'anomalous', [(y, x+ns+no, 1), (x+ns, y+no, -1)])
                    self.var_H += [param_name]
                if complex:
                    for y in self.sites[x-1]:
                        if x==1 and y==1:
                            continue
                        param_name = 'pb{:d}{:d}i'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'anomalous', [(y, x+ns+no, 1j), (x+ns, y+no, -1j)])
                        self.var_H += [param_name]

    #---------------------------------------------------------------------
    def  __str__(self):
        S = 'cluster model "' + self.name + '"'
        if self.spin_flip :
            S += ', spin flip'
        if self.spin_dependent :
            S += ', spin dependent'
        if self.complex :
            S += ', complex-valued'
        if self.singlet :
            S += ', singlet SC'
        if self.singlet :
            S += ', triplet SC'
        S += '\nbath energies : '
        for x in self.var_E:
            S += x + ', '
        S = S[0:-2] + '\nhybridizations : '
        for x in self.var_H:
            S += x + ', '
        return S[0:-2]

    #---------------------------------------------------------------------
    def  varia_E(self, c=1):
        """returns a list of parameter names from the bath energies with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath energies with the suffix appropriate for cluster c

        """
        v = []
        for x in self.var_E:
            v += [x+'_'+str(c)] 
        return v

    #---------------------------------------------------------------------
    def  varia_H(self, c=1):
        """returns a list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        """
        v = []
        for x in self.var_H:
            v += [x+'_'+str(c)] 
        return v


    #---------------------------------------------------------------------
    def  varia(self, H=None, E=None, c=1, spin_down=False):
        """creates a dict of variational parameters to values taken from the hybridization matrix H and the energies E, for cluster c
        
        :param ndarray H: matrix of hybridization values
        :param ndarray E: array of energy values
        :param boolean spin_down: True for the spin-down values
        :return {str,float}: dict of variational parameters to values
        
        """
        nb = self.nb
        ns = self.ns
        no = ns+nb
        nn = no*self.nmixed//2

        if H.shape != (self.nmixed*self.nb, self.nmixed*self.ns):
            raise ValueError('shape of hybridization matrix does not match model in general bath back propagation')
        
        D = {}
        # bath energies
        if self.spin_flip:
            for x in range(1,nb+1):
                param_name = 'eb{:d}u_{:d}'.format(x, c)
                D[param_name] =  E[x-1]
                param_name = 'eb{:d}d_{:d}'.format(x, c)
                D[param_name] = E[x-1+nb]

        elif self.spin_flip and not spin_down:
            for x in range(1,nb+1):
                param_name = 'eb{:d}u_{:d}'.format(x, c)
                D[param_name] = E[x-1]
        elif self.spin_flip and spin_down == True:
            for x in range(1,nb+1):
                param_name = 'eb{:d}d_{:d}'.format(x, c)
                D[param_name] = E[x-1]
        else:
            for x in range(1,nb+1):
                param_name = 'eb{:d}_{:d}'.format(x, c)
                D[param_name] = E[x-1]

        # hybridizations
        if self.spin_flip:
            for x in range(1,nb+1):

                for y in self.sites:
                    param_name = 'tb{:d}u{:d}u_{:d}'.format(x,y,c)
                    D[param_name] = H[x-1, y-1].real
                    param_name = 'tb{:d}d{:d}d_{:d}'.format(x,y,c)
                    D[param_name] = H[x+nb-1, y-1].real
                    param_name = 'tb{:d}u{:d}d_{:d}'.format(x,y,c)
                    D[param_name] = H[x-1, y+ns-1].real
                    param_name = 'tb{:d}d{:d}u_{:d}'.format(x,y,c)
                    D[param_name] = H[x+nb-1, y+ns-1].real

            
                if self.complex:
                    for y in self.sites[1:]:
                        param_name = 'tb{:d}u{:d}ui_{:d}'.format(x,y,c)
                        D[param_name] = H[x-1, y-1].imag
                        param_name = 'tb{:d}d{:d}di_{:d}'.format(x,y,c)
                        D[param_name] = H[x+nb-1, y-1].imag
                        param_name = 'tb{:d}u{:d}di_{:d}'.format(x,y,c)
                        D[param_name] = H[x-1, y+ns-1].imag
                        param_name = 'tb{:d}d{:d}ui_{:d}'.format(x,y,c)
                        D[param_name] = H[x+nb-1, y+ns-1].imag

        elif self.spin_dependent:
            for x in range(1,nb+1):

                for y in self.sites:
                    if spin_down:
                        param_name = 'tb{:d}d{:d}d_{:d}'.format(x,y,c)
                    else:
                        param_name = 'tb{:d}u{:d}u_{:d}'.format(x,y,c)
                    D[param_name] = H[x-1, y-1].real
            
                if self.complex:
                    for y in self.sites[1:]:
                        if spin_down:
                            param_name = 'tb{:d}d{:d}di_{:d}'.format(x,y,c)
                        else:
                            param_name = 'tb{:d}u{:d}ui_{:d}'.format(x,y,c)
                        D[param_name] = H[x-1, y-1].imag
        
        else:
            for x in range(1,nb+1):
                for y in self.sites:
                    param_name = 'tb{:d}{:d}_{:d}'.format(x,y,c)
                    D[param_name] = H[x-1, y-1].real

                if self.complex:
                    for y in self.sites[1:]:
                        param_name = 'tb{:d}{:d}i_{:d}'.format(x,y,c)
                        D[param_name] = H[x-1, y-1].imag

        if self.singlet:    
            for y in self.sites:
                param_name = 'sb{:d}{:d}_{:d}'.format(x,y,c)
                D[param_name] = H[x+nn-1, y-1].real # à modifier
            if self.complex:
                for y in self.sites:
                    if x==1 and y==1:
                        continue
                    param_name = 'sb{:d}{:d}i_{:d}'.format(x,y,c)
                    D[param_name] = H[x+nn-1, y-1].imag # à modifier

        if self.triplet:    
            for y in self.sites:
                param_name = 'pb{:d}{:d}_{:d}'.format(x,y,c)
                D[param_name] = H[x+nn-1, y-1].real # à modifier
            if self.complex:
                for y in self.sites:
                    if x==1 and y==1:
                        continue
                    param_name = 'pb{:d}{:d}i_{:d}'.format(x,y,c)
                    D[param_name] = H[x+nn-1, y-1].imag # à modifier
        return D

    #---------------------------------------------------------------------
    def  starting_values(self, c=1, e = (0.5, 1.5), hyb = (0.5, 0.2), shyb = (0.1, 0.05), pr=False):
        """returns an initialization string for the bath parameters

        :param int c: cluster label (starts at 1)
        :param (float,float) e: bounds of the values for the bath energies (absolute value)
        :param (float,float) hyb: average and deviation of the normal hybridization parameters
        :param (float,float) shyb: average and deviation of the anomalous hybridization parameters
        :param boolean pr: prints the starting values if True
        :return str: initialization string

        """
        S = ''
        fac = 1
        E = np.linspace(e[0], e[1], len(self.var_E))
        for i, x in enumerate(self.var_E):
            bn = int(x[2])
            fac = 2*(bn%2)-1
            S += x + '_' + str(c)+ ' = ' + str(fac*E[i])+'\n'
        for x in self.var_H:
            if x[0:2] == 'sb' or x[0:2] == 'pb':
                S += x + '_' + str(c)+ ' = ' + str(shyb[0] + shyb[1]*(2*np.random.random()-1))+'\n'
            else:
                S += x + '_' + str(c)+ ' = ' + str(hyb[0] + hyb[1]*(2*np.random.random()-1))+'\n'
        if pr:
            print('starting values:\n', S)
        return S

    #---------------------------------------------------------------------
    def  starting_values_PH(self, c=1, e = (1, 0.5), hyb = (0.5, 0.2), phi=None, pr=False):
        """returns an initialization string for the bath parameters, in the particle-hole symmetric case.

        :param int c: cluster label
        :param (float) e: range of bath energies
        :param (float,float) hyb: average and deviation of the normal hybridization parameters
        :param [int] phi: PH phases of the cluster sites proper
        :param boolean pr: if True, prints info
        :return str: initialization string

        """
        S = ''
        fac = 1
        if self.nb%2:
            raise ValueError('the number of bath orbitals must be even for starting_values_PH() to apply')
        if phi is None:
            raise ValueError('The PH phases of the sites must be specified')
        NE = self.nb//2

        if self.spin_dependent or self.spin_flip:
            for i in range(NE):
                S += 'eb{:d}u_{:d} = -1.0*eb{:d}d_{:d}\n'.format(2*i+2, c, 2*i+1, c)
                S += 'eb{:d}d_{:d} = -1.0*eb{:d}u_{:d}\n'.format(2*i+2, c, 2*i+1, c)
                for s in self.sites[2*i]:
                    S += 'tb{:d}u{:d}u_{:d} = {:2.1f}*tb{:d}d{:d}d_{:d}\n'.format(2*i+2, s, c,  phi[s-1], 2*i+1, s, c)
                    S += 'tb{:d}d{:d}d_{:d} = {:2.1f}*tb{:d}u{:d}u_{:d}\n'.format(2*i+2, s, c,  phi[s-1], 2*i+1, s, c)
                if self.complex:
                    for s in self.sites[2*i]:
                        S += 'tb{:d}u{:d}ui_{:d} = {:2.1f}*tb{:d}d{:d}di_{:d}\n'.format(2*i+2, s, c, -phi[s-1], 2*i+1, s, c)
                        S += 'tb{:d}d{:d}di_{:d} = {:2.1f}*tb{:d}u{:d}ui_{:d}\n'.format(2*i+2, s, c, -phi[s-1], 2*i+1, s, c)
                if self.spin_flip:
                    for s in self.sites[2*i]:
                        S += 'tb{:d}u{:d}d_{:d} = {:2.1f}*tb{:d}u{:d}d_{:d}\n'.format(2*i+2, s, c, -phi[s-1], 2*i+1, s, c)
                        S += 'tb{:d}d{:d}u_{:d} = {:2.1f}*tb{:d}d{:d}u_{:d}\n'.format(2*i+2, s, c, -phi[s-1], 2*i+1, s, c)
                    if self.complex:
                        for s in self.sites[2*i]:
                            S += 'tb{:d}u{:d}di_{:d} = {:2.1f}*tb{:d}u{:d}di_{:d}\n'.format(2*i+2, s, c,  phi[s-1], 2*i+1, s, c)
                            S += 'tb{:d}d{:d}ui_{:d} = {:2.1f}*tb{:d}d{:d}ui_{:d}\n'.format(2*i+2, s, c,  phi[s-1], 2*i+1, s, c)

        else:
            for i in range(NE):
                S += 'eb{:d}_{:d} = -1.0*eb{:d}_{:d}\n'.format(2*i+2, c, 2*i+1, c)
                for s in self.sites[2*i]:
                    S += 'tb{:d}{:d}_{:d} = {:2.1f}*tb{:d}{:d}_{:d}\n'.format(2*i+2, s, c,  phi[s-1], 2*i+1, s, c)
                if self.complex:
                    for s in self.sites[2*i]:
                        S += 'tb{:d}{:d}i_{:d} = {:2.1f}*tb{:d}{:d}i_{:d}\n'.format(2*i+2, s, c, -phi[s-1], 2*i+1, s, c)

        var_E = self.var_E
        self.var_E = []
        for x in var_E:
            for i in range(NE):
                if 'eb{:d}'.format(2*i+1) in x:
                    self.var_E += [x]
        var_H = self.var_H
        self.var_H = []
        for x in var_H:
            for i in range(NE):
                if 'tb{:d}'.format(2*i+1) in x:
                    self.var_H += [x]

        for i, x in enumerate(self.var_E):
            S += x + '_' + str(c)+ ' = ' + str(e[0] + e[1]*(2*np.random.random()-1))+'\n'
        for x in self.var_H:
            S += x + '_' + str(c)+ ' = ' + str(hyb[0] + hyb[1]*(2*np.random.random()-1))+'\n'
        if pr:
            print('starting values:\n', S)
        return S


######################################################################
# DEBUG HELPER

def cdmft_distance_debug(varia=None, vset=None, beta=50, wc=2.0, grid_type = 'sharp', counterterms=None):
    """Debugs the CDMFT distance function

    :param [str] varia: list of variational parameters 
    :param [[float]] vset: sets of bath parameters
    :param float beta: inverse fictitious temperature (for the frequency grid)
    :param float wc: cutoff frequency (for the frequency grid)
    :param str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self'
    :param [str] counterterms: list of counterterms names (cluster operators that should strive to have zero average)
    :param boolean SEF: if True, computes the Potthoff functional at the end
    :param [class] observable: list of observables used to assess convergence
    :returns: None

    """
    global w, wr, weight, var, _mixing, first_time, first_time2, Gdim, nclus, nmixed, clusters, maxfev, Hyb, Hyb_down

    # identifying the variational parameters
    var = varia
    nvar = len(var)
    qcm.CDMFT_variational_set(var)
    if nvar == 0:
        raise ValueError('CDMFT requires at least one variational parameter...Aborting.')

    pyqcm.new_model_instance()
    _mixing = pyqcm.mixing()
    print('mixing state = ', _mixing)
    Gdim = pyqcm.Green_function_dimension()
    nsites, nbands, clusters, bath_size, ref = pyqcm.model_size()
    clusters = np.array(clusters)
    bath_size = np.array(bath_size)
    nclus = len(clusters)
    nmixed = Gdim//nsites
        
    pyqcm.new_model_instance()
    dist_function = __frequency_grid(grid_type, beta, wc)
    qcm.CDMFT_host(wr, weight)

    # puts the values only of the parameters into array params_array
    x = np.zeros(nvar)
    first = True
    for s in vset:
        assert len(s) == nvar
        x = np.array(s)
        d = qcm.CDMFT_distance(x, 0)
        if first:
            S = 'dist\t'
            for i in range(nvar):
                S += var[i] + '\t'
            fout = open('cdmft_distance.tsv', 'a')
            fout.write(S+'\n')
            fout.close()
            first = False
        S = '{:g}\t'.format(d)
        for i in range(nvar):
            S += '{:g}\t'.format(x[i])
        fout = open('cdmft_distance.tsv', 'a')
        fout.write(S+'\n')
        fout.close()
