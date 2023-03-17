####################################################################################################
# This file contains functions implementing 
# Cluster dynamical mean-field theory (CDMFT)
####################################################################################################

import numpy as np
import pyqcm
from pyqcm import qcm
import timeit

####################################################################################################
class CDMFT:
    first_time = True
    first_time2 = True

    def __init__(self, model, varia,
        beta=50,
        wc=2.0,
        grid_type = 'sharp',
        maxiter=32,
        miniter=0,
        accur=1e-3,
        accur_hybrid=1e-4,
        accur_dist=1e-10,
        accur_E0=1e-10,
        alpha = 0.0,
        method='Nelder-Mead',
        file='cdmft.tsv',
        averages=False,
        eps_algo=0,
        initial_step = 0.1,
        hartree=None,
        SEF=False,
        check_ground_state = False,
        observables=None,
        verb=False,
        max_function_eval = 5000000,
        compute_potential_energy = False,
        double_counting_correct = None
    ):
        """
        :param [str] varia: list of variational parameters 
            OR tuple of two lists : bath energies and bath hybridizations
            OR function that returns dicts of bath energies and bath hybridizations given numeric arrays
        :param float beta: inverse fictitious temperature (for the frequency grid)
        :param float wc: cutoff frequency (for the frequency grid)
        :param str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self'
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
        :param boolean averages: if True, computes the lattice averages after each iteration. Computes them at the end anyway.
        :param int eps_algo: number of elements in the epsilon algorithm convergence accelerator = 2*eps_algo + 1 (0 = no acceleration)
        :param float initial_step: initial step in the minimization routine
        :param [class hartree] hartree: mean-field hartree couplings to incorportate in the convergence procedure
        :param boolean SEF: if True, computes the Potthoff functional at the end
        :param boolean check_ground_state: if True, checks the ground state consistency and raises exception if inconsistent
        :param [class observable]: list of observables used to assess convergence
        :param boolean verb: If True, prints debugging information
        :param int max_function_eval: maximum number of distance function evaluations when minimizing distance
        :param boolean compute_potential_energy: If True, computes Tr(Sigma*G) along with the averages
        :param [(str,str,str,float,float)] double_counting_correct: list of recipes for double counting corrections: (kinetic operator, interaction operator, density operator, coefficient, value of the kinetic operator without interaction)    :returns: None
        """

        self.model =model
        self.Hyb = None # internal : hybridization function
        self.Hyb_down = None # internal : hybridization function (spin downs, when mixing=4)

        max_function_eval
        if pyqcm.is_sequence(hartree) == False and hartree is not None:
            hartree = (hartree,)
        else: hartree = hartree

        # variational parameters
        var = varia
        nvar = len(var)
        if nvar == 0:
            raise ValueError('CDMFT requires at least one variational parameter...Aborting.')
        qcm.CDMFT_variational_set(var)
        var_data = np.empty((nvar, maxiter+1))
            
        # damping
        begin_with_damping = False
        n_damping = 1
        if type(alpha) is tuple:
            n_damping = alpha[1]
            alpha = alpha[0]
            if n_damping > 0:
                begin_with_damping = True
            else:
                n_damping = -n_damping

        # Hartree mean field parameters
        if hartree is None:
            pyqcm.banner('CDMFT procedure', '*', skip=1)
        else:
            pyqcm.banner('CDMFT procedure (combined with Hartree procedure)', '*', skip=1)
            if eps_algo:
                for C in hartree:
                    C.init_epsilon(maxiter, eps_algo)

        min_iter_E0 = 5

        observable_series_length = 0

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
            self.I = pyqcm.model_instance(self.model)
            params = self.I.instance_parameters() # params is a dict

            # puts the values only of the parameters into array params_array
            for i in range(nvar):
                params_array[i] = params[var[i]]
            var_data[:, superiter] = params_array
            check_bounds(params_array, 1000.0, v=var)


            # check convergence in the DVMC case
            # if pyqcm.solver == 'dvmc':
            #     print('E0_VMC = ', E0_VMC, '\tE0_err_VMC = ', E0_err_VMC)
            #     E0[superiter] = E0_VMC
            #     E0_err[superiter] = E0_err_VMC
            # else:
            #     gs = self.I.ground_state()
            #     for x in gs:
            #         E0[superiter] += x[0]
            gs = self.I.ground_state()
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
            self.grid = frequency_grid(self.I, grid_type, beta, wc)
            if double_counting_correct != None:
                self.I.double_counting_correct(double_counting_correct)
            qcm.CDMFT_host(self.grid.wr, self.grid.weight, self.I.label)
            self.set_Hyb()
            t2 = timeit.default_timer()
            time_ED += t2 - t1

            # optimization of the bath parameters
            sol, iter_done = optimize(lambda x : qcm.CDMFT_distance(x, 0), params_array, method, initial_step, accur, accur_dist, max_function_eval)
            t3 = timeit.default_timer()
            time_MIN += t3 - t2

            if method != 'ANNEAL' and not sol.success:
                print(sol)
                raise pyqcm.MinimizationError()

            if sol.nfev > max_function_eval:
                print(sol)
                print('number of function evaluations exceeds preset maximum of ', max_function_eval)
                raise pyqcm.MinimizationError()

            dist_value = sol.fun
            # updating the bath parameters (replace old by new)

            if alpha > 0.0 and ((superiter < n_damping and begin_with_damping is True) or (superiter > n_damping and begin_with_damping is False)):
                print('applying damping with alpha = ', alpha)
                for i in range(nvar):
                    self.model.set_parameter(var[i], (1-alpha)*sol.x[i]+alpha*params_array[i])
            else:
                for i in range(nvar):
                    self.model.set_parameter(var[i], sol.x[i])


            diff_param = np.linalg.norm(params_array - sol.x)/np.sqrt(nvar)
            initial_step = diff_param
            gs = self.I.ground_state()
            print('\nGS sector : ', [x[1] for x in gs])
            if superiter > 0:
                diffH = self.diff_hybrid()
                print('CDMFT iteration {:d}, distance = {: #.2e}, diff param = {: #.2e}, diff hybrid = {: #.2e}, diff E0 = {: #.2g}\n{:d} minimization steps, time(MIN)/time(ED)={:.5f}'.format(superiter+1, dist_value, diff_param, diffH, diff_E0, iter_done, time_MIN/time_ED), flush=True)
            else:
                print('CDMFT iteration {:d}, distance = {: #.2e}, diff param = {: #.2e}\n{:d} minimization steps, time(MIN)/time(ED)={:.5f}'.format(superiter+1, dist_value, diff_param, iter_done, time_MIN/time_ED), flush=True)

            #--------------------------------- Hartree step ---------------------------------
            if hartree != None:
                hartree_converged = True
                hartree_ave = np.zeros(self.model.nclus)
                diff_hartree = 0
                diff_hartree_rel = 0
                for C in hartree:
                    C.update(self.I, pr=True)
                    diff_hartree += np.abs(C.diff)
                    hartree_converged = hartree_converged and C.converged()
            #--------------------------------------------------------------------------------

            if averages:
                ave = self.I.averages(pr=True)
                if compute_potential_energy : 
                    self.I.potential_energy()
                    self.I.Potthoff_functional()

            # writing the parameters in a progress file
            des = 'distance\tdiff_param\tdiff_hybrid\t'
            val = '{: #.2e}\t{: #.2e}\t{: #.2e}\t'.format(dist_value, diff_param, diffH)
            self.I.write_summary('cdmft_iter.tsv', suppl_descr = des, suppl_values = val, first_of_series=CDMFT.first_time2)
            CDMFT.first_time2 = False


            #______________________________________________________________________
                    
            # checking convergence on the parameters (note the sqrt(nvar) factor in order not to punish large parameter sets)
            if (diff_param < accur) and hartree_converged and superiter > miniter:
                converged = True
                pyqcm.banner('CDMFT converged on the parameters', '=')
                break

            # checking convergence on the hybridization matrix
            if superiter > 0:
                diffH = self.diff_hybrid()
                if (diffH < accur_hybrid) and hartree_converged and superiter > miniter:
                    pyqcm.banner('CDMFT converged on the hybridization function', '=')
                    converged = True
                    break

            # checking convergence on the observables
            # for the moment, this works only for observables belonging to the first cluster
            obs_converged = True
            if observables != None:
                ave = self.I.cluster_averages()  ### WARNING CHECK
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
            var_val = pyqcm.varia_table(var,sol.x)
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
                    self.model.set_parameter(var[i], z)
                var_val = pyqcm.__varia_table(var,var_data[:,superiter])
                print(var_val)
            #-------------------------------------------------------------------------------


        # here we have converged
        if converged:

            # check consistency
            GS_consistent = True
            GS_cons = ''
            for i in range(self.model.nclus):
                if self.model.clus[i].ref != None: continue
                ave = self.I.cluster_averages(i)
                diffGS = np.abs(ave['mu'][0] - self.I.Green_function_density(i))
                if np.abs(ave['mu'][0] - self.I.Green_function_density(i)) > 1e-6:
                    GS_consistent = False
                    GS_cons += 'N'
                    pyqcm.banner("GROUND STATE INCONSISTENCY FOR CLUSTER {:d}, DENSITY DIFFERENCE = {:1.5f}".format(i+1,diffGS), '+', skip=1)
                    if check_ground_state:
                        raise ValueError("failed GS consistency for cluster {:d} in CDMFT".format(i+1))
                else: GS_cons += 'Y'
            
            var_val = pyqcm.varia_table(var,sol.x)
            print(var_val)

            ave = self.I.averages(pr=True)
            if compute_potential_energy : self.I.potential_energy()

            if SEF:
                omegaH=self.I.Potthoff_functional(hartree)

            if file != None:
                des = 'GS_consistency\titerations\tdist_function\tdistance\tdiff_hybrid\t'
                val = '{:s}\t{:d}\t{:s}\t{: #.2e}\t{: #.2e}\t'.format(GS_cons, superiter, self.grid.dist_function, dist_value, diffH)
                if SEF : 
                    des += 'omegaH\t'
                    val += '{: #.8g}\t'.format(omegaH)
                if observables != None:
                    for x in observables:
                        des += 'ave_'+x.name+'_obs\t'
                        val += '{: #.6e}\t'.format(x.ave)
                    des += 'series_length\t'
                    val += '{:d}\t'.format(observable_series_length)
                self.I.write_summary(file, suppl_descr = des, suppl_values = val, first_of_series=CDMFT.first_time)
                CDMFT.first_time = False
                CDMFT.first_time2 = True

            # writing the frequency grid to a file
            if 'self' in self.grid.dist_function:
                self.I.write_summary('cdmft_grid.tsv', suppl_descr = des, suppl_values = val, first_of_series=True)
                with open('cdmft_grid.tsv','a') as gridfile:
                    np.savetxt(gridfile, np.stack((self.grid.w.imag, self.grid.weight),axis=-1),header='w\tweight', fmt='%.8f', delimiter='\t')
                    gridfile.close()


            pyqcm.banner('CDMFT completed successfully', '*')
        else:
            pyqcm.banner('Failure of the CDMFT algorithm', '*')
    

    #-----------------------------------------------------------------------------------------------
    def set_Hyb(self):
        """Computes the hybridization function, i.e.
        an array of arrays of matrices. Hyb[i], for cluster #i, is a (nw,d,d) Numpy array. with nw frequencies, and d sites

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
    def diff_hybrid(self):
        """
        Computes a difference in hybridization functions between the current iteration (Hyb) and the previous one (Hyb0)

        :returns float: the difference in hybridization arrays
        
        """

        diff = 0.0
        g = self.grid
        for i in range(g.nw):
            for j in range(self.model.nclus):
                diffH = self.Hyb[j][i,:,:] - self.Hyb0[j][i,:,:]
                norm = np.linalg.norm(diffH)
                diff += g.weight[i]*norm*norm

        if self.model.mixing==4:
            for i in range(g.nw):
                for j in range(self.model.nclus):
                    diffH = self.Hyb_down[j][i,:,:] - self.Hyb0_down[j][i,:,:]
                    norm = np.linalg.norm(diffH)
                    diff += g.weight[i]*norm*norm

        if self.model.mixing == 0:
            diff *= 2
        elif self.model.mixing == 3:
            diff /= 2
        diff /= g.nw  
        return np.sqrt(diff)


####################################################################################################
class frequency_grid:
    """
    This class contains the imaginary frequency grid data, including weights

    """

    def __init__(self, I, grid_type='sharp', beta=50, wc=2):
        """
        :param model_instance I: current model instance 
        :param str grid_type: type of frequency grid along the imaginary axis : 'sharp', 'ifreq', 'self'
        :param float beta: inverse fictitious temperature (for the frequency grid)
        :param float wc: cutoff frequency (for the frequency grid)
        """
        self.beta = beta
        self.wc = wc
        self.grid_type = grid_type

        self.wr = np.arange((np.pi / self.beta), self.wc + 1e-6, 2 * np.pi / self.beta)
        self.w = np.ones(len(self.wr), dtype=np.complex128)
        self.w = self.w * 1j
        self.w *= self.wr
        self.nw = len(self.w)
        if self.grid_type == 'sharp':
            self.weight = np.ones(self.nw)
            self.weight *= 1.0 /self.nw
            self.dist_function = 'sharp_wc_{0:.1f}_b_{1:d}'.format(self.wc, int(self.beta))
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
            raise pyqcm.WrongArgumentError(f"unknown frequency grid type `{grid_type}`")


####################################################################################################
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
                new_cluster_operator(name, param_name, 'one-body', [(x+ns, x+ns, 1)])
                self.var_E.append(param_name)
                param_name = 'eb{:d}d'.format(x)
                new_cluster_operator(name, param_name, 'one-body', [(x+ns+no, x+ns+no, 1)])
                self.var_E.append(param_name)
        else:
            for x in range(1,nb+1):
                param_name = 'eb{:d}'.format(x)
                new_cluster_operator(name, param_name, 'one-body', [(x+ns, x+ns, 1), (x+ns+no, x+ns+no, 1)])
                self.var_E.append(param_name)

        # hybridizations
        if spin_dependent or spin_flip:
            for x in range(1,nb+1):

                for y in self.sites[x-1]:
                    param_name = 'tb{:d}u{:d}u'.format(x,y)
                    new_cluster_operator(name, param_name, 'one-body', [(y, x+ns, 1)])
                    self.var_H.append(param_name)
                    param_name = 'tb{:d}d{:d}d'.format(x,y)
                    new_cluster_operator(name, param_name, 'one-body', [(y+no, x+ns+no, 1)])
                    self.var_H.append(param_name)
                    if spin_flip:
                        param_name = 'tb{:d}u{:d}d'.format(x,y)
                        new_cluster_operator(name, param_name, 'one-body', [(x+ns, y+no, 1)])
                        self.var_H.append(param_name)
                        param_name = 'tb{:d}d{:d}u'.format(x,y)
                        new_cluster_operator(name, param_name, 'one-body', [(y+no, x+ns+no, 1)])
                        self.var_H.append(param_name)

            
                if complex:
                    for y in self.sites[x-1][1:]:
                        param_name = 'tb{:d}u{:d}ui'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'one-body', [(y, x+ns, 1j)])
                        self.var_H.append(param_name)
                        param_name = 'tb{:d}d{:d}di'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'one-body', [(y+no, x+ns+no, 1j)])
                        self.var_H.append(param_name)
                    if spin_flip:
                        for y in self.sites[x-1]:
                            param_name = 'tb{:d}u{:d}di'.format(x,y)
                            new_cluster_operator_complex(name, param_name, 'one-body', [(x+ns, y+no, 1j)])
                            self.var_H.append(param_name)
                            param_name = 'tb{:d}d{:d}ui'.format(x,y)
                            new_cluster_operator_complex(name, param_name, 'one-body', [(y+no, x+ns+no, 1j)])
                            self.var_H.append(param_name)
        
        else:
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'tb{:d}{:d}'.format(x,y)
                    new_cluster_operator(name, param_name, 'one-body', [(y, x+ns, 1), (y+no, x+ns+no, 1)])
                    self.var_H.append(param_name)

                if complex:
                    for y in self.sites[x-1][1:]:
                        param_name = 'tb{:d}{:d}i'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'one-body', [(y, x+ns, 1j), (y+no, x+ns+no, 1j)])
                        self.var_H.append(param_name)

        if singlet:    
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'sb{:d}{:d}'.format(x,y)
                    new_cluster_operator(name, param_name, 'anomalous', [(y, x+ns+no, 1), (x+ns, y+no, 1)])
                    self.var_H.append(param_name)
                if complex:
                    for y in self.sites[x-1]:
                        if x==1 and y==1:
                            continue
                        param_name = 'sb{:d}{:d}i'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'anomalous', [(y, x+ns+no, 1j), (x+ns, y+no, 1j)])
                        self.var_H.append(param_name)

        if triplet:    
            for x in range(1,nb+1):
                for y in self.sites[x-1]:
                    param_name = 'pb{:d}{:d}'.format(x,y)
                    new_cluster_operator(name, param_name, 'anomalous', [(y, x+ns+no, 1), (x+ns, y+no, -1)])
                    self.var_H.append(param_name)
                if complex:
                    for y in self.sites[x-1]:
                        if x==1 and y==1:
                            continue
                        param_name = 'pb{:d}{:d}i'.format(x,y)
                        new_cluster_operator_complex(name, param_name, 'anomalous', [(y, x+ns+no, 1j), (x+ns, y+no, -1j)])
                        self.var_H.append(param_name)

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
        for x in self.var_E:
            S += x + ', '
        S = S[0:-2] + '\nhybridizations : '
        for x in self.var_H:
            S += x + ', '
        return S[0:-2]

    #-----------------------------------------------------------------------------------------------
    def  varia_E(self, c=1):
        """returns a list of parameter names from the bath energies with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath energies with the suffix appropriate for cluster c

        """
        v = []
        for x in self.var_E:
            v.append(x+'_'+str(c)) 
        return v

    #-----------------------------------------------------------------------------------------------
    def  varia_H(self, c=1):
        """returns a list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        """
        v = []
        for x in self.var_H:
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

        var_E = self.var_E
        self.var_E = []
        for x in var_E:
            for i in range(NE):
                if 'eb{:d}'.format(2*i+1) in x:
                    self.var_E.append(x)
        var_H = self.var_H
        self.var_H = []
        for x in var_H:
            for i in range(NE):
                if 'tb{:d}'.format(2*i+1) in x:
                    self.var_H.append(x)

        for i, x in enumerate(self.var_E):
            S += x + '_' + str(c)+ ' = ' + str(e[0] + e[1]*(2*np.random.random()-1))+'\n'
        for x in self.var_H:
            S += x + '_' + str(c)+ ' = ' + str(hyb[0] + hyb[1]*(2*np.random.random()-1))+'\n'
        if pr:
            print('starting values:\n', S)
        return S


######################################################################
class hybridization:    

    def  __init__(self, file):
        """Defines hybridization data from a data file

        :param str file: name of the file or string. Format : each line starts with a frequency and then has N*N
        columns for Delta_{ij}(w)
        the frequency is purely imaginary; it is its imaginary part that appears in the file

        """
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

        :param int I: label of the impurity model instance
        """
        M = np.zeros((self.nw, self.n, self.n), dtype=np.complex128)
        for i in range(self.nw):
            M[i,:,:] = pyqcm.hybridization_function(0, self.w[i]*1j, spin_down=False, label=I)
        
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
    def optimize_bath(self, varia, x, method='Nelder-Mead', accur = 1e-4, accur_dist = 1e-8):
        """
        Optimizes the bath parameters to fit the hybridization function delta
        :param [str] varia: list of variational parameters (bath parameters) from PyQCM
        :param [float] x: starting values of the parameters
        """

        nvar = len(varia)
        def tmp_distance(x):
            for i in range(nvar):
                self.model.set_parameter(varia[i], x[i])
            pyqcm.new_model_instance(1)
            return self.distance(1)

        return optimize(tmp_distance, x, method=method, accur = accur, accur_dist = accur_dist)



####################################################################################################
# PRIVATE FUNCTIONS

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
                print('parameter ', v[i], ' is out of bounds!!!')
            else:
                print('parameter no ', i+1, ' is out of bounds!!!')
            raise pyqcm.OutOfBoundsError(i)


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
    import scipy.optimize

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

