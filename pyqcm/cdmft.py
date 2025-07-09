"""This file contains functions implementing
Cluster dynamical mean-field theory (CDMFT).
"""

import nlopt
import timeit
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import pyqcm
from pyqcm import qcm


class convergence_manager:
    """
    class managing a convergence test. One or more instances of this class must be used by the CDMFT
    procedure. One instance corresponds to a quantity that is evaluated at each iteration and then
    compared with the corresponding quantities at a number (`depth`) of previous iterations.
    A difference function (`diff_func`) is applied to the current value of the quantity and each of
    its predecessors, and convergence is declared when the result of this difference (a float) is smaller than
    the tolerance (`tol`) for each of the `depth` previous iterations.
    The possible values of the 'name' argument are:
    * 'parameters' : the set of bath parameters is the quantity used.
    * 'GS energy' : the ground state energy of the impurity problem is used. In the case of more than one cluster, the sum is used.
    * 'hybridization' : the hybridization function (or set thereof, in the case of many clusters). The test is based on the norm of the matrix differences, summed over grid frequencies.
    * 'self-energy' : same as above, but using the impurity self-energy instead.
    * any one-body or anomalous lattice operator name. The lattice average is used as test value.
    """

    def __init__(self, name, diff_func, tol, depth=2, stdev=False):
        """
        :param str name: name of the convergence test
        :param function diff_func: difference function applied to the objects tested
        :param float tol: tolerance for convergence
        :param int depth: number of previous iterations the comparison is made with
        :param boolean stdev: indicates that the convergence is tested using the standard deviation
        """

        self.name = name
        self.diff_func = diff_func
        self.tol = tol
        self.depth = depth
        self.diff = np.zeros(depth)
        self.iter = 0
        self.X = []  # list of test objects
        self.op = None
        self.converged = False
        self.stdev = stdev
        if stdev :
            self.val = np.zeros(32)
            self.std = None
            self.ave = None

    def test(self, x):
        """
        tests for convergence and stores the test object x in an array for comparisons with 'depth' future iterations

        :param object x: the object containing the current test quantity
        :return: True if converged, False otherwise
        """
        self.iter += 1
        if self.stdev:
            return self.stdev_test(x)
        converged = True
        if self.iter <= self.depth:
            self.X.append(x)
            return False
        else:
            for i in range(self.depth):
                self.diff[i] = self.diff_func(x, self.X[i])
            for i in range(self.depth):
                if self.diff[i] > self.tol:
                    converged = False
                    break
            for i in range(self.depth-1):
                self.X[i] = self.X[i+1]
            self.X[-1] = x
            if converged:
                self.converged = True
                return True
            else:
                self.converged = False
                return False

    def stdev_test(self, x):
        print(self.name + ' = {:1.5g} (added to sequence)'.format(x))
        if self.iter > len(self.val):
            self.val.resize(len(self.val) + 32)
        self.val[self.iter-1] = x
        self.std = np.std(self.val[0:self.iter])/np.sqrt(self.iter)
        self.ave = np.mean(self.val[0:self.iter])
        if self.std < self.tol:
            return True
        else:
            return False

    def print(self):
        """
        Prints the status of the convergence (differences are printed, with a number "depth" of previous iterations)
        """
        if self.iter <= self.depth: return
        if self.stdev:
            S = '---> ave({:s}) = {:1.7g} +/- {:1.4g}'.format(self.name, self.ave, self.std)
        else:
            S = 'differences in ' + self.name + ' : '
            for i in range(self.depth): S += '{:1.3g}\t'.format(self.diff[i])
        if self.converged: S += ' * converged * '
        print(S, flush=True)


class CDMFT:
    """
    class containing the elements of a CDMFT computation. The constructor executes the computation.

    :param [str] varia: list of variational parameters
    :param [(str,str,float)] phase_constraint: phase constraint between real and imaginary variational parameters
    :param float beta: inverse fictitious temperature (for the frequency grid)
    :param float wc: cutoff frequency (for the frequency grid)
    :param str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self', 'adapt'
    :param int maxiter: maximum number of CDMFT iterations
    :param int miniter: minimum number of CDMFT iterations
    :param float accur_bath: the x-tolerance for distance function optimization
    :param [str] convergence: the convergence tests (sequence of strings or single string)
    :param [float] accur: the tolerance for the various convergence tests (sequence of floats or single float)
    :param [float] accur_dist: relative tolerance of the distance function when minimizing it
    :param [float] accur_dist_outer: convergence criteria based on the relative change of the distance function
    :param boolean converge_with_stdev: If True, checks convergence using the standard deviation of the convergence tests, not the difference
    :param str iteration: method of iteration of parameters ('fixed_point' or 'Broyden')
    :param float alpha: if iteration='fixed_point', damping parameter (fraction of the previous iteration in the new one). If iteration='Broyden', 1+alpha is the inverse initial Jacobian (or alpha can literally be a matrix, the inverse Jacobian from a previous run).
    :param boolean displaymin: displays the minimum distance function when minimized
    :param str method: method to use, as used in scipy.optimize.minimize(). Choices: 'Nelder-Mead' (default), 'Powell', 'CG', 'BFGS', 'ANNEAL', or a choice of NLopt methods : 'NELDERMEAD', 'COBYLA', 'BOBYQA', 'PRAXIS', 'SUBPLEX'
    :param str file: name of the file where the solution is written
    :param str iter_file: name of the file where the CDMFT iterations are recorded
    :param int eps_algo: number of elements in the epsilon algorithm convergence accelerator = 2*eps_algo + 1 (0 = no acceleration)
    :param float initial_step: initial step in the minimization routine
    :param [hartree] hartree: mean-field hartree couplings to incorportate in the convergence procedure
    :param boolean SEF: if True, computes the Potthoff functional at the end
    :param boolean check_ground_state: if True, checks the ground state consistency and raises exception if inconsistent
    :param int max_function_eval: maximum number of distance function evaluations when minimizing distance
    :param boolean compute_potential_energy: If True, computes Tr(Sigma*G) along with the averages
    :param ndarray host_function: if not None, function that computes the host array and passes it to qcm
    :param function pre_host: function to be executed before computing the host. Takes a model instance as argument
    :param float max_value: maximum absolute value of variational parameters
    :param boolean fallback: if True, falls back to the other iteration method (Broyden or fixed_point) if the current one fails
    :ivar lattice_model model: (unique) model on which the computation is based
    :ivar ndarray Hyb: host function
    :ivar ndarray Hyb_down: host function for the spin down component in the case of mixing=4
    :ivar I: current model instance (changes in the course of the computation)

    """

    def __init__(self,
        model,
        varia,
        phase_constraint=None,
        beta=50,
        wc=2.0,
        grid_type = 'sharp',
        maxiter=32,
        miniter=0,
        convergence='parameters',
        depth=2,
        accur_bath=1e-3,
        accur=1e-4,
        accur_dist=1e-8,
        accur_dist_outer=1e-3,
        converge_with_stdev = False,
        iteration = 'Broyden', # or 'fixed_point'
        alpha = 0.0,
        method='Nelder-Mead',
        file='cdmft.tsv',
        iter_file='cdmft_iter.tsv',
        eps_algo=0,
        initial_step = 0.1,
        hartree=None,
        SEF=False,
        check_ground_state = False,
        max_function_eval = 500000,
        compute_potential_energy = False,
        host_function = None,
        pre_host = None,
        max_value = 100,
        fallback=False,
    ):

        self.model =model
        self.Hyb = None # internal : hybridization function
        self.Hyb_down = None # internal : hybridization function (spin downs, when mixing=4)
        self.sigma = None # internal : self-energy
        self.sigma_down = None # internal : self-energy (spin downs, when mixing=4)
        self.hartree = hartree
        self.pre_host = pre_host
        self.host_function = host_function
        self.grid_type = grid_type
        self.beta = beta
        self.wc = wc
        self.method = method
        self.initial_step = initial_step
        self.accur_bath = accur_bath
        self.accur_dist = accur_dist
        self.accur_dist_outer = accur_dist_outer
        self.max_function_eval = max_function_eval
        self.max_value = max_value
        self.alpha = alpha
        self.dist = 1e6
        self.delta_dist = 1e6
        self.check_ground_state = check_ground_state
        self.iter_file = iter_file

        if pyqcm.is_sequence(accur) == False:
            accur = (accur,)

        if pyqcm.is_sequence(hartree) == False and hartree is not None:
            hartree = (hartree,)
        else: hartree = hartree

        if iteration not in ['Broyden', 'fixed_point']:
            raise ValueError("the argument 'iteration' of CDMFT() should be one of 'Broyden', 'fixed_point'")

        # variational parameters
        varia_set = set(varia) # makes sure there are no duplicates
        if phase_constraint:
            constraint_set = set()
            for c in phase_constraint:
                constraint_set.add(c[0])
                constraint_set.add(c[1])
                assert constraint_set <= varia_set # both real and imaginary parts must be variational parameters
            varia_set = varia_set - constraint_set
        self.var = [] # constructs a list of variational parameters with the complex pairs at the end
        for x in varia_set : self.var.append(x)
        self.var.sort()
        self.nconstr = 0
        if phase_constraint:
            self.nconstr = len(phase_constraint)
            for x in phase_constraint: 
                self.var.append(x[0])
                self.var.append(x[1])

        self.phase_constraint = phase_constraint
        self.nvar_comp = len(self.var)
        self.nvar = self.nvar_comp - self.nconstr  # net number of variational parameters
        if self.nvar <= 0:
            raise ValueError('CDMFT requires at least one variational parameter...Aborting.')
        qcm.CDMFT_variational_set(self.var)
        self.var_data = np.empty((self.nvar, maxiter+1))

        #------------------------- convergence test initialization ------------------------
        if pyqcm.is_sequence(convergence) == False:
            convergence = (convergence,)
        if len(convergence) != len(accur):
            raise ValueError('The number of convergence tests must be the same as the length of "accur"')
        n_convergence_test = len(convergence)
        self.convergence_test = []
        convergence_test_string = ''
        for i, C in enumerate(convergence):
            convergence_test_string += C + '/'
            if C in model.parameters():
                conv_manager = convergence_manager(C, lambda x,y : np.abs(x-y), accur[i], depth, stdev=converge_with_stdev)
                conv_manager.op = C
            else:
                if converge_with_stdev and C != 'GS energy':
                    raise ValueError('In CDMFT, converge_with_stdev=True is only possible when "convergence" is the name of an operator or GS energy')
                op = None
                if C == 'parameters':
                    conv_manager = convergence_manager(C, lambda x,y : np.linalg.norm(x-y)/x.shape[0], accur[i], depth)
                elif C == 'hybridization':
                    conv_manager = convergence_manager(C, lambda x,y : self.diff_matrix(x,y), accur[i], depth)
                elif C == 'self-energy':
                    if grid_type == 'adapt':
                        raise ValueError('cannot use convergence on self-energy with adaptive frequency grid')
                    conv_manager = convergence_manager(C, lambda x,y : self.diff_matrix(x,y), accur[i], depth)
                elif C == 'GS energy':
                    conv_manager = convergence_manager(C, lambda x,y : np.abs(x-y), accur[i], depth, stdev=converge_with_stdev)
                else:
                    raise ValueError('type of convergence test "' + C + '" in CDMFT does not exist')
            self.convergence_test.append(conv_manager)
        convergence_test_string = convergence_test_string[:-1]

        #-------------------------- Hartree mean field parameters -------------------------
        self.nhartree=0
        if self.hartree is None:
            pyqcm.banner('CDMFT procedure', '*', skip=1)
            self.hartree = []
        else:
            pyqcm.banner('CDMFT procedure (combined with Hartree procedure)', '*', skip=1)
        self.nhartree = len(self.hartree)

        #-------------- first define the frequency grid for the distance function ---------
        print('frequency grid type = ', grid_type)
        print('fictitious inverse temperature = ', beta)
        print('frequency cutoff = ', wc)
        if alpha is float : print('damping factor = ', self.alpha)
        print('-'*100)

        # ------------------------------------- CDMFT loop --------------------------------
        converged = False

        # initializaing the array of parameters for optimization (size nvar + nhartree)
        self.CDMFT_params = np.zeros(self.nvar + self.nhartree)
        vartot = self.var + [x.Vm for x in self.hartree]
        V = model.parameters(vartot)
        self.CDMFT_params[0:self.nvar] = self.to_varia_array(V[0:self.nvar_comp])
        for i,x in enumerate(self.hartree):
            self.CDMFT_params[self.nvar+i] = model.parameters()[x]

        #------------------------------- CDMFT main iteration loop ------------------------
        self.niter = 0
        def F(x):
            self.CDMFT_params = np.copy(x)
            try:
                self.CDMFT_step()
            except : raise
            else: return x - self.CDMFT_params

        def G():
            if self.delta_dist < self.accur_dist_outer: return True
            else: return self.check_convergence()

        CDMFT_params0 = self.CDMFT_params

        if iteration == 'Broyden':
            try:
                actual_method = 'Broyden'
                self.CDMFT_params, self.niter, self.alpha = pyqcm.broyden(F, self.CDMFT_params, self.alpha, maxiter=maxiter, miniter=miniter, xtol=1e-6, convergence_test=G)
            except Exception as E:
                if fallback:
                    pyqcm.banner('restarting CDMFT with fixed-point method', '+', skip=1)
                    model.set_parameter(vartot, CDMFT_params0, pr=True)
                    try:
                        actual_method = 'fixed_point'
                        self.CDMFT_params, self.niter = pyqcm.fixed_point_iteration(F, self.CDMFT_params, xtol=1e-6, convergence_test=G, maxiter=maxiter, miniter=miniter, alpha=self.alpha, eps_algo=eps_algo)
                    except Exception as E:
                        print(E)
                        raise pyqcm.SolverError('Failure of the CDMFT method')
                else:
                    print(E)
                    raise pyqcm.SolverError('Failure of the CDMFT method')

        elif iteration == 'fixed_point':
            try:
                actual_method = 'fixed_point'
                self.CDMFT_params, self.niter = pyqcm.fixed_point_iteration(F, self.CDMFT_params, xtol=1e-6, convergence_test=G, maxiter=maxiter, miniter=miniter, alpha=self.alpha, eps_algo=eps_algo)
            except Exception as E:
                if fallback:
                    pyqcm.banner('restarting CDMFT with Broyden method', '+', skip=1)
                    model.set_parameter(vartot, CDMFT_params0, pr=True)
                    try:
                        actual_method = 'Broyden'
                        self.CDMFT_params, self.niter, self.alpha = pyqcm.broyden(F, self.CDMFT_params, self.alpha, maxiter=maxiter, miniter=miniter, xtol=1e-6, convergence_test=G)
                    except Exception as E:
                        print(E)
                        raise pyqcm.SolverError('Failure of the CDMFT method')
                else:
                    print(E)
                    raise pyqcm.SolverError('Failure of the CDMFT method')

        else: raise ValueError('unknown iteration method in CDMFT. must be either "Broyden" or "fixed_point". Check spelling.')

        # Here we have converged
        self.I = pyqcm.model_instance(self.model)  # a last instance with the converged parameters

        # check consistency
        GS_cons = self.I.GS_consistency(self.check_ground_state)
        var_val = pyqcm.varia_table(self.var,self.expand_varia_array(self.CDMFT_params))
        pyqcm.banner('converged variational parameters ({:d} iterations)'.format(self.niter), '-')
        print(var_val)

        ave = self.I.averages(pr=True)
        if compute_potential_energy :
            self.I.potential_energy()
        if SEF:
            omega=self.I.Potthoff_functional(hartree)

        if file != None:
            self.I.props['opt_method'] = method
            self.I.props['GS_consistency'] = GS_cons
            self.I.props['CDMFT_method'] = actual_method
            self.I.props['CDMFT_iterations'] = self.niter
            self.I.props['dist_function'] = self.grid.dist_function
            self.I.props['convergence'] = convergence_test_string
            self.I.props['min_dist'] = self.dist
            self.I.write_summary(file)

        pyqcm.banner('CDMFT completed successfully', '*')


    def to_varia_array(self, x):
        """
        Produces the numerical array of variational parameters passed to the optimizer
        Takes into account phase constraints
        """
        y = np.empty(self.nvar)
        y[0:(self.nvar-self.nconstr)] = x[0:(self.nvar-self.nconstr)]
        for i in range(self.nconstr):
        	y[self.nvar-self.nconstr+i] = np.sqrt(x[self.nvar-self.nconstr+2*i]**2+x[self.nvar-self.nconstr+2*i+1]**2)
        return y

    def expand_varia_array(self, x):
        """
        Expand the array x to replace amplitude data with real and imaginary parts
        """
        y = np.empty(self.nvar_comp)
        y[0:(self.nvar-self.nconstr)] = x[0:(self.nvar-self.nconstr)]
        for i in range(self.nconstr):
            y[self.nvar-self.nconstr+2*i] = np.cos(self.phase_constraint[i][2])*x[self.nvar-self.nconstr+i]
            y[self.nvar-self.nconstr+2*i+1] = np.sin(self.phase_constraint[i][2])*x[self.nvar-self.nconstr+i]
        return y

    #-----------------------------------------------------------------------------------------------
    def CDMFT_step(self):
        """
        Performs a CDMFT step that brings the system from the current values (self.CDMFT_params) of the bath and Hartree
        parameters and updates it to the next set of values
        """

        try:
            check_bounds(self.CDMFT_params, self.max_value, v=self.var)
        except pyqcm.OutOfBoundsError as error:
            raise error
        except:
            raise ValueError

        y = self.expand_varia_array(self.CDMFT_params[0:self.nvar])
        self.model.set_parameter(self.var, y)
        self.I = pyqcm.model_instance(self.model)

        if self.grid_type == 'adapt':  # find the maximum value of variational (bath) parameters
            m = 2.0
            self.wc = max([2.0, max(self.CDMFT_params[0:self.nvar])])

        self.grid = frequency_grid(self.I, self.grid_type, self.beta, self.wc)


        # solve the impurity problem

        # initializes G_host
        t1 = timeit.default_timer()
        if self.pre_host != None:
            self.pre_host(self.I)

        # computing or transferring the host array --------------------------------------

        if self.host_function == None:
            qcm.CDMFT_host(self.grid.wr, self.grid.weight, self.I.label)
        else:
            self.host_function(self.I)
        #--------------------------------------------------------------------------------

        self.I.GS_consistency(self.check_ground_state)

        self.set_Hyb()
        t2 = timeit.default_timer()
        time_ED = t2 - t1

        x_new = np.empty(self.nvar + self.nhartree)
        # updates the Hartree mean-field parameters
        for C in self.hartree:
            C.update(self.I, pr=True)
        P = self.model.parameters()
        for i,h in enumerate(self.hartree):
            x_new[self.nvar + i] = P[h.Vm]

        gs = self.I.ground_state()

        # optimization of the bath parameters
        def DIST(y, grad=None):
            # HERE WE COULD TRANSLATE y in terms of model parameters
            ym = self.expand_varia_array(y)
            d = qcm.CDMFT_distance(ym, self.I.label)
            return d

        dist0 = self.dist
        
        x0 = self.CDMFT_params[0:self.nvar]
        sol = optimize(DIST, x0, self.method, self.initial_step, self.accur_bath, self.accur_dist, self.max_function_eval)
        opt_x, opt_iter_done, opt_success, opt_fun = sol

        self.dist = opt_fun
        self.delta_dist = np.abs((self.dist - dist0)/dist0)
        t3 = timeit.default_timer()
        time_MIN = t3 - t2

        if self.method != 'ANNEAL' and not opt_success:
            print(sol)
            pyqcm.banner('Error in the scipy minimization procedure within CDMFT ', '!')
            raise pyqcm.MinimizationError()

        if opt_iter_done > self.max_function_eval:
            print(sol)
            pyqcm.banner('number of function evaluations exceeds preset maximum of {:d}'.format(self.max_function_eval), '!')
            raise pyqcm.MinimizationError()

        if np.any(np.isnan(opt_x)):
            print(sol)
            pyqcm.banner('NaN found in optimization of bath parameters', '!')
            raise pyqcm.MinimizationError('NaN found in optimization of bath parameters')

        # push back into array
        x_new[0:self.nvar] = np.copy(opt_x)

        try:
            check_bounds(x_new[0:self.nvar], self.max_value, v=self.var)
        except pyqcm.OutOfBoundsError as error:
            raise error
        except:
            raise ValueError

        # writing the parameters in a progress file
        self.I.props['opt_method'] = self.method
        self.I.props['dist_function'] = self.grid.dist_function
        self.I.props['min_dist'] = self.dist

        self.I.write_summary(self.iter_file)

        print('GS sector : ', [X[1] for X in gs])
        print('{:d} minimization steps, time(MIN)/time(ED)={:.3g}, distance = {:1.4g}, delta_dist = {:0.3g}%'.format(opt_iter_done, time_MIN/time_ED, opt_fun, 100*self.delta_dist), flush=True)

        var_val = pyqcm.varia_table(self.var, self.expand_varia_array(x_new))
        print('updated bath parameters:\n{:s}'.format(var_val))

        self.CDMFT_params = np.copy(x_new)

        return

    #-----------------------------------------------------------------------------------------------
    def check_convergence(self):
        """
        Performs the CDMFT convergence tests
        returns True if all are converged, otherwise False
        """

        converged = True
        for C in self.convergence_test:
            if C.name == 'GS energy':
                gs = self.I.ground_state()
                E0 = 0.0
                for x in gs:
                    E0 += x[0]
                converged = converged and C.test(E0)

            elif C.name == 'self-energy':
                self.set_sigma()
                if self.model.mixing == 4:
                    T = C.test((self.sigma,self.sigma_down))
                    converged = converged and T
                else:
                    T = C.test(self.sigma)
                    converged = converged and T

            elif C.name == 'hybridization':
                if self.model.mixing == 4:
                    T = C.test((self.Hyb,self.Hyb_down))
                    converged = converged and T
                else:
                    T = C.test(self.Hyb)
                    converged = converged and T

            elif C.name == 'parameters':
                T = C.test(np.copy(self.CDMFT_params))
                converged = converged and T

            elif C.op != None:
                T = C.test(self.I.averages()[C.op])
                converged = converged and T

        for C in self.convergence_test:
            C.print()

        if converged: pyqcm.banner('CDMFT has converged', '=')
        return converged

    #-----------------------------------------------------------------------------------------------
    def set_Hyb(self):
        """Computes the hybridization function, i.e.
        an array of arrays of matrices.
        Hyb[i], for cluster #i, is a (nw,d,d) Numpy array. with nw frequencies, and d sites
        Hyb_down[i] is the same, for the spin-down part, if mixing=4

        :returns None

        """
        self.Hyb0 = self.Hyb
        self.Hyb = []
        for j in range(self.model.nclus):
            d = self.model.dimGFC[j]
            self.Hyb.append(np.zeros((self.grid.nw, d, d), dtype=np.complex128))

        for i in range(self.grid.nw):
            for j in range(self.model.nclus):
                self.Hyb[j][i, :, :] = self.I.hybridization_function(self.grid.w[i], j, spin_down=False)

        if self.model.mixing == 4:
            self.Hyb0_down = self.Hyb_down
            self.Hyb_down = []
            for j in range(self.model.nclus):
                d = self.model.dimGFC[j]
                self.Hyb_down.append(np.zeros((self.grid.nw, d, d), dtype=np.complex128))

            for i in range(self.grid.nw):
                for j in range(self.model.nclus):
                    self.Hyb_down[j][i, :, :] = self.I.hybridization_function(self.grid.w[i], j, spin_down=True)


    #-----------------------------------------------------------------------------------------------
    def diff_matrix(self, X, Y):
        """
        Computes a difference in hybridization functions between the current iteration (Hyb) and the previous one (Hyb0)
        :param object X : the current test object
        :param object Y : any past test object (same structure as X)
        :returns float: the difference in hybridization arrays

        """

        diff = 0.0
        g = self.grid
        if self.model.mixing!=4:
            for i in range(g.nw):
                for j in range(self.model.nclus):
                    delta = X[j][i,:,:] - Y[j][i,:,:]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i]*norm*norm

        if self.model.mixing==4:
            for i in range(g.nw):
                for j in range(self.model.nclus):
                    delta = X[0][j][i,:,:] - Y[0][j][i,:,:]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i]*norm*norm
            for i in range(g.nw):
                for j in range(self.model.nclus):
                    delta = X[1][j][i,:,:] - Y[1][j][i,:,:]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i]*norm*norm

        if self.model.mixing == 0:
            diff *= 2
        elif self.model.mixing == 3:
            diff /= 2
        diff /= g.nw
        return np.sqrt(diff)

    #-----------------------------------------------------------------------------------------------
    def set_sigma(self):
        """Computes the self-energy on the frequency grid
        an array of arrays of matrices. sigma[i], for cluster #i, is a (nw,d,d) Numpy array. with nw frequencies, and d sites

        :returns None

        """
        self.sigma0 = self.sigma
        self.sigma = []
        for j in range(self.model.nclus):
            d = self.model.dimGFC[j]
            self.sigma.append(np.zeros((self.grid.nw, d, d), dtype=np.complex128))

        for i in range(self.grid.nw):
            for j in range(self.model.nclus):
                self.sigma[j][i, :, :] = self.I.cluster_self_energy(self.grid.w[i], j, spin_down=False)

        if self.model.mixing == 4:
            self.sigma0_down = self.sigma_down
            self.sigma_down = []
            for j in range(self.model.nclus):
                d = self.model.dimGFC[j]
                self.sigma_down.append(np.zeros((self.grid.nw, d, d), dtype=np.complex128))

            for i in range(self.grid.nw):
                for j in range(self.model.nclus):
                    self.sigma_down[j][i, :, :] = self.I.cluster_self_energy(self.grid.w[i], j, spin_down=True)

    #-----------------------------------------------------------------------------------------------
    def diff_sigma(self):
        """
        Computes a difference in self-energy between the current iteration (sigma) and the previous one (sigma0)

        :returns float: the difference in self-energy arrays

        """

        diff = 0.0
        g = self.grid
        for i in range(g.nw):
            for j in range(self.model.nclus):
                delta = self.sigma[j][i,:,:] - self.sigma0[j][i,:,:]
                norm = np.linalg.norm(delta)
                diff += g.weight[i]*norm*norm

        if self.model.mixing==4:
            for i in range(g.nw):
                for j in range(self.model.nclus):
                    delta = self.sigma_down[j][i,:,:] - self.sigma0_down[j][i,:,:]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i]*norm*norm

        if self.model.mixing == 0:
            diff *= 2
        elif self.model.mixing == 3:
            diff /= 2
        diff /= g.nw
        return np.sqrt(diff)

    #-----------------------------------------------------------------------------------------------
    def plot_iterations(self):
        """
        Plots the variational parameters as a function of iteration, for diagnostics purposes
        """

        ncols = 3
        nrows = 1+(self.nvar-1)//ncols
        fig, ax = plt.subplots(nrows, ncols, sharex=True)
        fig.set_size_inches(24/2.54, nrows*6/2.54)
        niter = self.var_data.shape[0]
        for i,x in enumerate(self.var):
            if nrows==1: plt.sca(ax[i])
            else: plt.sca(ax[i//ncols,i%ncols])
            plt.plot(range(self.niter), self.var_data[i,0:self.niter], 'o-', ms=3, lw=0.5)
            plt.title(self.var[i])
        plt.savefig('iterations.pdf')

####################################################################################################
class frequency_grid:
    """
    This class contains the imaginary frequency grid data, including weights

    :param model_instance I: current model instance
    :param str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self'
    :param float beta: inverse fictitious temperature (for the frequency grid)
    :param float wc: cutoff frequency (for the frequency grid)

    :ivar float beta: inverse fictitious temperature
    :ivar float wc: cutoff frequency (for the frequency grid)
    :ivar str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self'
    :ivar wr: array of frequencies (real array, i.e., imaginary part of the frequencies)
    :ivar [float] weight: array of weights for the different frequencies of the grid
    :ivar int nw: number of frequencies in the grid
    """

    def __init__(self, I=None, grid_type='sharp', beta=50, wc=2):
        self.beta = beta
        self.wc = wc
        self.grid_type = grid_type

        self.wr = np.arange((np.pi / self.beta), self.wc + 1e-6, 2 * np.pi / self.beta)
        self.w = np.ones(len(self.wr), dtype=np.complex128)
        self.w = self.w * 1j
        self.w *= self.wr
        self.nw = len(self.w)
        if self.grid_type == 'sharp' or self.grid_type == 'adapt':
            self.weight = np.ones(self.nw)
            self.weight *= 1.0 /self.nw
            self.dist_function = '{:s}_wc_{:.3g}_b_{:d}'.format(self.grid_type, self.wc, int(self.beta))
        elif self.grid_type == 'ifreq':
            self.weight = 1.0/self.wr
            self.weight *= 1.0 / self.weight.sum()
            self.dist_function = 'ifreq_wc_{0:.1f}_b_{1:d}'.format(self.wc, int(self.beta))
        elif self.grid_type == 'self':
            self.weight = np.zeros(self.nw)
            Sig_inf = I.cluster_self_energy(1.0e6j)
            for i, x in enumerate(self.w):
                Sig = I.cluster_self_energy(x) - Sig_inf
                self.weight[i] = np.linalg.norm(Sig)
            self.weight *= 1.0 / self.weight.sum()
            self.dist_function = 'self_wc_{0:.1f}_b_{1:d}'.format(self.wc, int(self.beta))
        else:
            raise ValueError(f"unknown frequency grid type `{grid_type}`")

####################################################################################################
class general_bath:
    """
    Class that construct a cluster model with nb bath orbitals, in the most general way possible, with possible restrictions.

    :param str name: name of the cluster-bath model to be defined
    :param int ns: number of sites in the cluster
    :param int nb: number of bath orbitals in the cluster
    :param boolean spin_dependent: if True, the parameters are spin dependent
    :param boolean spin_flip: if True, spin-flip hybridizations are present
    :param boolean singlet: if True, defines anomalous singlet hybridizations
    :param boolean triplet: if True, defines anomalous triplet hybridizations
    :param boolean complex: if True, defines imaginary parts as well, when appropriate
    :param [[int]] sites: 2-level list of sites to couple to the bath orbitals (labels from 1 to ns). Format resembles [[site labels to bind to orbital 1], ...] .

    :ivar int ns: number of physical sites in the cluster
    :ivar int nb: number of bath orbitals in the cluster
    :ivar str name: name of the cluster model
    :ivar [str] var_E: list of bath parameters of the type "energy level"
    :ivar [str] var_H: list of bath parameters of the type "hybridization"
    :ivar boolean spin_dependent: True if the bath parameter conserve spin, but are different for spins up and down
    :ivar boolean spin_flip: True if we include spin-flip hybridizations terms
    :ivar boolean singlet: True if we include anomalous hybridizations of the singlet type
    :ivar boolean triplet: True if we include anomalous hybridizations of the triplet type
    :ivar boolean complex: True if the model is complex-valued and thus hybridization operators have both real and imaginary components

    """

    def  __init__(self, ns, nb, name = 'clus', spin_dependent=False, spin_flip=False, singlet=False, triplet=False, complex=False, sites=None):
        self.CM = pyqcm.cluster_model(ns, n_bath=nb, name=name)
        self.CM.var_E = []
        self.CM.var_H = []
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
                print('the format of the argument "sites" is incorrect : it should be a sequence of ', ns, ' sequences')
                raise ValueError(f'the format of the argument "sites" is incorrect : it should be a sequence of {ns} sequences')
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
                self.CM.new_operator(param_name, 'one-body', [(x+ns, x+ns, 1)])
                self.CM.var_E.append(param_name)
                param_name = 'eb{:d}d'.format(x)
                self.CM.new_operator(param_name, 'one-body', [(x+ns+no, x+ns+no, 1)])
                self.CM.var_E.append(param_name)
        else:
            for x in range(1,nb+1):
                param_name = 'eb{:d}'.format(x)
                self.CM.new_operator(param_name, 'one-body', [(x+ns, x+ns, 1), (x+ns+no, x+ns+no, 1)])
                self.CM.var_E.append(param_name)

        # hybridizations
        if spin_dependent or spin_flip:
            for x in range(1,nb+1):

                for y in self.sites[x-1]:
                    param_name = 'tb{:d}u{:d}u'.format(x,y)
                    self.CM.new_operator(param_name, 'one-body', [(y, x+ns, 1)])
                    self.CM.var_H.append(param_name)
                    param_name = 'tb{:d}d{:d}d'.format(x,y)
                    self.CM.new_operator(param_name, 'one-body', [(y+no, x+ns+no, 1)])
                    self.CM.var_H.append(param_name)
                    if spin_flip:
                        param_name = 'tb{:d}u{:d}d'.format(x,y)
                        self.CM.new_operator(param_name, 'one-body', [(x+ns, y+no, 1)])
                        self.CM.var_H.append(param_name)
                        param_name = 'tb{:d}d{:d}u'.format(x,y)
                        self.CM.new_operator(param_name, 'one-body', [(y+no, x+ns+no, 1)])
                        self.CM.var_H.append(param_name)


                if complex:
                    for y in self.sites[x-1][1:]:
                        param_name = 'tb{:d}u{:d}ui'.format(x,y)
                        self.CM.new_operator_complex(param_name, 'one-body', [(y, x+ns, 1j)])
                        self.CM.var_H.append(param_name)
                        param_name = 'tb{:d}d{:d}di'.format(x,y)
                        self.CM.new_operator_complex(param_name, 'one-body', [(y+no, x+ns+no, 1j)])
                        self.CM.var_H.append(param_name)
                    if spin_flip:
                        for y in self.sites[x-1]:
                            param_name = 'tb{:d}u{:d}di'.format(x,y)
                            self.CM.new_operator_complex(param_name, 'one-body', [(x+ns, y+no, 1j)])
                            self.CM.var_H.append(param_name)
                            param_name = 'tb{:d}d{:d}ui'.format(x,y)
                            self.CM.new_operator_complex(param_name, 'one-body', [(y+no, x+ns+no, 1j)])
                            self.CM.var_H.append(param_name)

        else:
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'tb{:d}{:d}'.format(x,y)
                    self.CM.new_operator(param_name, 'one-body', [(y, x+ns, 1), (y+no, x+ns+no, 1)])
                    self.CM.var_H.append(param_name)

                if complex:
                    for y in self.sites[x-1][1:]:
                        param_name = 'tb{:d}{:d}i'.format(x,y)
                        self.CM.new_operator_complex(param_name, 'one-body', [(y, x+ns, 1j), (y+no, x+ns+no, 1j)])
                        self.CM.var_H.append(param_name)

        if singlet:
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'sb{:d}{:d}'.format(x,y)
                    self.CM.new_operator(param_name, 'anomalous', [(y, x+ns+no, 1), (x+ns, y+no, 1)])
                    self.CM.var_H.append(param_name)
                if complex:
                    for y in self.sites[x-1]:
                        if x==1 and y==1:
                            continue
                        param_name = 'sb{:d}{:d}i'.format(x,y)
                        self.CM.new_operator_complex(param_name, 'anomalous', [(y, x+ns+no, 1j), (x+ns, y+no, 1j)])
                        self.CM.var_H.append(param_name)

        if triplet:
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'pb{:d}{:d}'.format(x,y)
                    self.CM.new_operator(param_name, 'anomalous', [(y, x+ns+no, 1), (x+ns, y+no, -1)])
                    self.CM.var_H.append(param_name)
                if complex:
                    for y in self.sites[x-1]:
                        if x==1 and y==1:
                            continue
                        param_name = 'pb{:d}{:d}i'.format(x,y)
                        self.CM.new_operator_complex(param_name, 'anomalous', [(y, x+ns+no, 1j), (x+ns, y+no, -1j)])
                        self.CM.var_H.append(param_name)

    #-----------------------------------------------------------------------------------------------
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
        for x in self.CM.var_E:
            S += x + ', '
        S = S[0:-2] + '\nhybridizations : '
        for x in self.CM.var_H:
            S += x + ', '
        return S[0:-2]

    #-----------------------------------------------------------------------------------------------
    def  varia_E(self, c=1):
        """returns a list of parameter names from the bath energies with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath energies with the suffix appropriate for cluster c

        """
        v = []
        for x in self.CM.var_E:
            v.append(x+'_'+str(c))
        return v

    #-----------------------------------------------------------------------------------------------
    def  varia_H(self, c=1):
        """returns a list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        """
        v = []
        for x in self.CM.var_H:
            v.append(x+'_'+str(c))
        return v


    #-----------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------
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
        E = np.linspace(e[0], e[1], len(self.CM.var_E))
        for i, x in enumerate(self.CM.var_E):
            bn = int(x[2])
            fac = 2*(bn%2)-1
            S += x + '_' + str(c)+ ' = ' + str(fac*E[i])+'\n'
        for x in self.CM.var_H:
            if x[0:2] == 'sb' or x[0:2] == 'pb':
                S += x + '_' + str(c)+ ' = ' + str(shyb[0] + shyb[1]*(2*np.random.random()-1))+'\n'
            else:
                S += x + '_' + str(c)+ ' = ' + str(hyb[0] + hyb[1]*(2*np.random.random()-1))+'\n'
        if pr:
            print('starting values:\n', S)
        return S

    #-----------------------------------------------------------------------------------------------
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

        var_E = self.CM.var_E
        self.CM.var_E = []
        for x in var_E:
            for i in range(NE):
                if 'eb{:d}'.format(2*i+1) in x:
                    self.CM.var_E.append(x)
        var_H = self.CM.var_H
        self.CM.var_H = []
        for x in var_H:
            for i in range(NE):
                if 'tb{:d}'.format(2*i+1) in x:
                    self.CM.var_H.append(x)

        for i, x in enumerate(self.CM.var_E):
            S += x + '_' + str(c)+ ' = ' + str(e[0] + e[1]*(2*np.random.random()-1))+'\n'
        for x in self.CM.var_H:
            S += x + '_' + str(c)+ ' = ' + str(hyb[0] + hyb[1]*(2*np.random.random()-1))+'\n'
        if pr:
            print('starting values:\n', S)
        return S


######################################################################
class hybridization:
    """
    Defines hybridization data from a data file.
    the frequency is purely imaginary; it is its imaginary part that appears in the file

    :param str file: name of the file or string. Format : each line starts with a frequency and then has N*N columns for Delta_{ij}(w)

    """


    def  __init__(self, file):
        NN = [1,4,9,16,25,36,49,64,81,100,121,144]
        # data = np.genfromtxt(file, dtype=np.complex128)
        # data = np.genfromtxt(file, invalid_raise=True, loose=False, dtype=None)
        data = np.genfromtxt(file, invalid_raise=True, loose=False, dtype=np.complex128)
        # print(data)
        self.nw = data.shape[0]
        LL = data.shape[1]-1
        try:
            self.n = NN.index(LL)+1
        except ValueError:
            raise ValueError('The number of columns in the hybridization table should be 1 + n^2')

        self.w = np.copy(np.real(data[:,0]))
        self.Delta = np.zeros((self.nw,self.n,self.n), dtype=np.complex128)
        for i in range(self.nw):
            self.Delta[i,:,:] = np.reshape(data[i, 1:], (self.n,self.n))

    #-----------------------------------------------------------------------------------------------
    def distance(self, I):
        """
        Evaluates the distance function bewteen the data and the hybridization function of an impurity model instance of label I

        :param [model_instance] I: model instance
        """
        M = np.zeros((self.nw, self.n, self.n), dtype=np.complex128)
        for i in range(self.nw):
            M[i,:,:] = I.hybridization_function(self.w[i]*1j, spin_down=False)

        # print('M : '); print(M)
        # print('Delta : '); print(self.Delta)

        dist = np.linalg.norm(M-self.Delta)
        return dist*dist/self.nw

    #-----------------------------------------------------------------------------------------------
    def __str__(self):
        S = ""
        for i in range(self.nw):
            S += 'w = ' + self.w[i].__str__() + '\n' + self.Delta[i].__str__() + '\n\n'
        return S

    #-----------------------------------------------------------------------------------------------
    def optimize_bath(self, model, varia, x, method='Nelder-Mead', accur = 1e-4, accur_dist = 1e-8):
        """
        Optimizes the bath parameters to fit the hybridization function delta
        :param [str] varia: list of variational parameters (bath parameters) from PyQCM
        :param [float] x: starting values of the parameters
        """

        nvar = len(varia)
        def tmp_distance(x):
            for i in range(nvar):
                model.set_parameter(varia[i], x[i])
            I = pyqcm.model_instance(model)
            return self.distance(I)

        return optimize(tmp_distance, x, method=method, accur = accur, accur_dist = accur_dist)



####################################################################################################
# PRIVATE FUNCTIONS

#---------------------------------------------------------------------------------------------------
def check_bounds(x, B=100, v=None):
    """Checks whether one of the variables in the array x is out of bounds

    :param [float] x: array of parameters
    :param float B: maximum value of all parameters
    :param [str] v: list of corresponding names, if available
    :return: None
    """
    for i in range(len(x)):
        if np.abs(x[i]) > B:
            if v != None:
                S = 'parameter {:s} = {:g} is out of bounds!'.format(v[i], x[i])
            else:
                S = 'parameter no {:d}  = {:g} is out of bounds!'.format(i+1, x[i])
            print('Try setting `max_value` to a bigger value within the CDMFT procedure!')
            raise pyqcm.OutOfBoundsError(S)

#---------------------------------------------------------------------------------------------------
# optimization of the bath parameters

def optimize(F, x, method='Nelder-Mead', initial_step=0.1, accur = 1e-4, accur_dist = 1e-8, maxfev=5000000):
    """

    :param F: function to be optimized
    :param (float) x: array of variables
    :param str method: method to use, as used in scipy.optimize.minimize()
    :param float initial_step: initial step in the minimization procedure
    :param float accur: requested accuracy in the parameters
    :param float accur_dist: requested accuracy in the distance function

    """

    displaymin, nvar = False, len(x)
    nlopt_methods = {
        'COBYLA': nlopt.LN_COBYLA,
        'BOBYQA': nlopt.LN_BOBYQA,
        'PRAXIS': nlopt.LN_PRAXIS,
        'NELDERMEAD': nlopt.LN_NELDERMEAD,
        'SUBPLEX': nlopt.LN_SBPLX
    }
    if method == 'Nelder-Mead':
        initial_simplex = np.zeros((nvar + 1, nvar))
        for i in range(nvar + 1):
            initial_simplex[i, :] = x
        for i in range(nvar):
            initial_simplex[i + 1, i] += initial_step
        sol = scipy.optimize.minimize(F, x, method='Nelder-Mead', options={'disp':displaymin, 'maxfev':maxfev, 'xatol': accur, 'fatol': accur_dist, 'initial_simplex': initial_simplex, 'adaptive':True, 'disp': True})
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == 'Powell':
        sol = scipy.optimize.minimize(F, x, method='Powell', tol=accur)
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == 'CG':
        sol = scipy.optimize.minimize(F, x, method='CG', jac=False, tol=accur, options={'gtol': 1e-6, 'eps': 1e-6})
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == 'BFGS':
        sol = scipy.optimize.minimize(F, x, method='BFGS', jac=False, tol=accur, options={'eps': accur})
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == 'ANNEAL':
        sol = scipy.optimize.basinhopping(F, x, minimizer_kwargs={'method': 'COBYLA', 'options': {'disp': displaymin, 'rhobeg': initial_step, 'maxiter': maxfev, 'rtol': accur_dist}})
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method in nlopt_methods.keys():
        optimizer = nlopt.opt(nlopt_methods[method], nvar)

        # Set objective function and parameters bounds (same as in `check_bounds` method)
        optimizer.set_min_objective(F)
        optimizer.set_lower_bounds(np.array([-np.inf for _ in x]))
        optimizer.set_upper_bounds(np.array([np.inf for _ in x]))

        # Set function values (ftol) and parameters (xtol) tolerances
        optimizer.set_ftol_rel(accur_dist)
        optimizer.set_xtol_rel(accur)

        optimizer.set_maxeval(maxfev)
        optimizer.set_initial_step(initial_step)

        # Minimizing...
        sol = optimizer.optimize(x)
        opt_x, iter_done, success, fun = sol, optimizer.get_numevals(), optimizer.last_optimize_result(), optimizer.last_optimum_value()

    else:
        raise ValueError(f'unknown method specified for minimization: {method}')

    if not success:
        raise pyqcm.MinimizationError()

    return opt_x, iter_done, success, fun


if __name__ == "__main__":
    pass
