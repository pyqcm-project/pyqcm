"""This file contains functions implementing Cluster Dynamical Mean-Field Theory (CDMFT)."""

import timeit

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import scipy.optimize

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
    * 'distance' : the distance function (difference between successive iterations)
    * any one-body or anomalous lattice operator name. The lattice average is used as test value.
    """

    def __init__(self, name, diff_func, tol, depth=2, stdev=False):
        """
        :param str name: name of the convergence test
        :param function diff_func: difference function applied to the objects tested
        :param float tol: tolerance for convergence
        :param int depth: number of previous iterations the comparison is made with
        :param bool stdev: indicates that the convergence is tested using the standard deviation
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
        if stdev:
            self.val = np.zeros(32)
            self.std = None
            self.ave = None

    def test(self, x):
        """
        Tests for convergence and stores the test object x in an array for comparisons with 'depth' future iterations

        :param object x: the object containing the current test quantity
        :returns: True if converged, False otherwise
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
            for i in range(self.depth - 1):
                self.X[i] = self.X[i + 1]
            self.X[-1] = x
            if converged:
                self.converged = True
                return True
            else:
                self.converged = False
                return False

    def stdev_test(self, x):
        """
        Tests for convergence using the standard deviation of the accumulated sequence of values.
        Appends x to the internal sequence and computes the standard deviation of the mean;
        convergence is declared when this standard deviation falls below self.tol.

        :param float x: the new value to add to the convergence sequence
        :returns: True if the standard deviation of the mean is below the tolerance, False otherwise
        :rtype: bool
        """
        print(self.name + " = {:1.5g} (added to sequence)".format(x))
        if self.iter > len(self.val):
            self.val.resize(len(self.val) + 32)
        self.val[self.iter - 1] = x
        self.std = np.std(self.val[0 : self.iter]) / np.sqrt(self.iter)
        self.ave = np.mean(self.val[0 : self.iter])
        if self.std < self.tol:
            return True
        else:
            return False

    def print(self):
        """
        Prints the status of the convergence (differences are printed, with a number "depth" of previous iterations)
        """
        if self.iter <= self.depth:
            return
        if self.stdev:
            S = "---> ave({:s}) = {:1.7g} +/- {:1.4g}".format(
                self.name, self.ave, self.std
            )
        else:
            S = "differences in " + self.name + " : "
            for i in range(self.depth):
                S += "{:g}\t".format(self.diff[i])
        if self.converged:
            S += " * converged * "
        print(S, flush=True)


class bias_field:
    """
    class containing the information about the procedure of applying an external field that breaks a symmetry and whose
    amplitude decreases by a given factor at each CDMFT iteration.
    """

    def __init__(self, op, value=1.0, factor=0.5, maxiter=32, miniter=3):
        """
        @param str name: name of the operator
        @param float value: initial value of the bias field
        @param float factor: decreasing factor between iterations (between 0.1 and 0.9)
        @param int maxiter: maximum iteration number, after which the field is set to 1e-9
        @param int miniter: minimum iteration number, before which the field is not attenuated
        """
        self.op = op
        self.value0 = value
        self.value = value
        self.factor = factor
        self.maxiter = maxiter
        self.miniter = miniter
        if self.maxiter < 5:
            raise ValueError("'maxiter' in bias_field should be >= 5")
        if self.miniter < 1 or self.miniter > 32:
            raise ValueError("'miniter' in bias_field should be from 1 to 32")
        if factor > 0.9 or factor < 0.1:
            raise ValueError(
                "factor in bias_field should be in the interval [0.1, 0.9]"
            )


class CDMFT:
    """
    class containing the elements of a CDMFT computation. The constructor executes the computation.

    :param [str] varia: list of variational parameters.
    :param str grid: frequency_grid object. If None, constructs a default frequency_grid object.
    :param int maxiter: maximum number of CDMFT iterations
    :param int miniter: minimum number of CDMFT iterations
    :param float accur_bath: the x-tolerance for distance function optimization
    :param [str] convergence: the convergence tests (sequence of strings or single string)
    :param [float] accur: the tolerance for the various convergence tests (sequence of floats or single float)
    :param [float] accur_dist: relative tolerance of the distance function when minimizing it
    :param [hartree] hartree: mean-field hartree couplings to incorportate in the convergence procedure
    :param bool converge_with_stdev: If True, checks convergence using the standard deviation of the convergence tests, not the difference
    :param str iteration: method of iteration of parameters ('fixed_point' or 'broyden')
    :param float/iterable alpha: if iteration='fixed_point', damping parameter (fraction of the previous iteration in the new one). If iteration='broyden', 1+alpha is the inverse initial Jacobian (or alpha can literally be a matrix, the inverse Jacobian from a previous run).
    :param str method: minimization method. Derivative-free choices: 'Nelder-Mead' (default), 'Powell', 'CG', 'ANNEAL', NLopt methods 'NELDERMEAD', 'COBYLA', 'BOBYQA', 'PRAXIS', 'SUBPLEX'. Analytical-Jacobian choices (the Jacobian ``qcm.CDMFT_gradient`` is activated automatically): 'trf' (Trust Region Reflective via scipy.least_squares), 'BFGS', 'L-BFGS-B'. The finite-difference step for the Jacobian is ``cdmft_jacobian_delta`` (default 1e-5, tunable via ``pyqcm.set_global_parameter``).
    :param int lm_max_nfev: maximum number of function/gradient evaluations for the jac-capable methods (default 2000; ignored for derivative-free methods)
    :param str file: name of the file where the solution is written
    :param str iter_file: name of the file where the CDMFT iterations are recorded
    :param int eps_algo: number of elements in the epsilon algorithm convergence accelerator = 2*eps_algo + 1 (0 = no acceleration)
    :param float initial_step: initial step in the minimization routine
    :param bool SEF: if True, computes the Potthoff functional at the end
    :param bool check_ground_state: if True, checks the ground state consistency and raises exception if inconsistent
    :param int max_function_eval: maximum number of distance function evaluations when minimizing distance
    :param bool compute_potential_energy: If True, computes Tr(Sigma*G) along with the averages
    :param ndarray host_function: if not None, function that computes the host array and passes it to qcm
    :param function pre_host: function to be executed before computing the host. Takes a model instance as argument
    :param float max_value: maximum absolute value of variational parameters
    :param bias_field bias: bias field (for spontaneous symmetry breaking) that decreases with iterations
    :param function post_min: function to be executed before writing to the progress file
    :ivar lattice_model model: (unique) model on which the computation is based
    :ivar ndarray Hyb: host function
    :ivar ndarray Hyb_down: host function for the spin down component in the case of mixing=4
    :ivar I: current model instance (changes in the course of the computation)

    """

    def __init__(
        self,
        model,
        varia,
        grid=None,
        maxiter=32,
        miniter=0,
        convergence="parameters",
        depth=2,
        accur_bath=1e-6,
        accur=1e-4,
        accur_dist=1e-8,
        hartree=None,
        converge_with_stdev=False,
        iteration="broyden",  # or 'fixed_point'
        alpha=0.0,  # or (damping factor, iterations) for fixed_point
        method="Nelder-Mead",
        lm_max_nfev=2000,
        file="cdmft.tsv",
        iter_file="cdmft_iter.tsv",
        eps_algo=0,
        initial_step=0.1,
        SEF=False,
        check_ground_state=False,
        max_function_eval=1_000_000,
        compute_potential_energy=False,
        host_function=None,
        pre_host=None,
        max_value=100,
        bias=None,
        post_min=None,
    ):

        self.accur_bath = accur_bath
        self.accur_dist = accur_dist
        self.alpha = alpha
        self.check_ground_state = check_ground_state
        self.dist = 1e6
        self.hartree = hartree
        self.host_function = host_function
        self.Hyb = None  # internal : hybridization function
        self.Hyb_down = (
            None  # internal : hybridization function (spin downs, when mixing=4)
        )
        self.initial_step = initial_step
        self.iter_file = iter_file
        self.max_function_eval = max_function_eval
        self.max_value = max_value
        self.lm_max_nfev = lm_max_nfev
        self.method = method
        self.jac = method.lower() in {"trf", "bfgs", "l-bfgs-b"}
        self.model = model
        self.pre_host = pre_host
        self.sigma = None  # internal : self-energy
        self.sigma_down = None  # internal : self-energy (spin downs, when mixing=4)
        self.varia = varia
        self.bias = bias
        self.post_min = post_min

        if pyqcm.is_sequence(accur) == False:
            accur = (accur,)

        if pyqcm.is_sequence(varia) == False:
            raise ValueError(
                "The argument 'varia' of CDMFT should be a list of variational parameters"
            )

        if iteration not in ["broyden", "fixed_point"]:
            raise ValueError(
                "the argument 'iteration' of CDMFT() should be one of 'broyden', 'fixed_point'"
            )

        # ----------- Managing the variational parameters ---------

        if pyqcm.is_sequence(hartree) == False and hartree is not None:
            self.hartree = (hartree,)
        else:
            self.hartree = hartree
        if self.hartree is None:
            self.hartree = []

        var = [[] for i in range(model.nclus + 1)]
        for v in varia:
            p = v.partition("_")
            if len(p) == 1:
                raise ValueError(
                    "{:s} cannot be a CDMFT variational parameter".format(v)
                )
            elif len(p) == 3:
                var[model.sys_clus[int(p[2]) - 1] + 1].append(v)
            else:
                raise ValueError("parameter name" + v + " has wrong format")
        var[0] = [h.Vm for h in self.hartree]
        self.nvar = np.zeros(model.nclus + 1, int)
        for c in range(model.nclus + 1):
            self.nvar[c] = len(var[c])

        qcm.CDMFT_variational_set(var[1:])

        self.varia = []
        for v in var:
            self.varia += v
        self.nvaria = len(self.varia)
        self.x = np.empty(self.nvaria)
        self.var_data = np.empty((self.nvaria, maxiter + 1))

        P = model.parameters()
        for i, v in enumerate(self.varia):
            self.x[i] = P[v]

        # ------------------------- convergence test initialization ------------------------
        if pyqcm.is_sequence(convergence) == False:
            convergence = (convergence,)
        if len(convergence) != len(accur):
            raise ValueError(
                'The number of convergence tests must be the same as the length of "accur"'
            )
        self.convergence_test = []
        convergence_test_string = ""
        for i, C in enumerate(convergence):
            convergence_test_string += C + "/"
            if C in model.parameters():
                conv_manager = convergence_manager(
                    C,
                    lambda x, y: np.abs(x - y),
                    accur[i],
                    depth,
                    stdev=converge_with_stdev,
                )
                conv_manager.op = C
            else:
                if converge_with_stdev and C != "GS energy":
                    raise ValueError(
                        'In CDMFT, converge_with_stdev=True is only possible when "convergence" is the name of an operator or GS energy'
                    )
                op = None
                if C == "parameters":
                    conv_manager = convergence_manager(
                        C,
                        lambda x, y: np.linalg.norm(x - y) / x.shape[0],
                        accur[i],
                        depth,
                    )
                elif C == "hybridization":
                    conv_manager = convergence_manager(
                        C, lambda x, y: self.diff_matrix(x, y), accur[i], depth
                    )
                elif C == "self-energy":
                    conv_manager = convergence_manager(
                        C, lambda x, y: self.diff_matrix(x, y), accur[i], depth
                    )
                elif C == "GS energy":
                    conv_manager = convergence_manager(
                        C,
                        lambda x, y: np.abs(x - y),
                        accur[i],
                        depth,
                        stdev=converge_with_stdev,
                    )
                elif C == "distance":
                    conv_manager = convergence_manager(
                        C, lambda x, y: np.abs(x - y) / y, accur[i], depth
                    )
                else:
                    raise ValueError(
                        'type of convergence test "' + C + '" in CDMFT does not exist'
                    )
            self.convergence_test.append(conv_manager)
        convergence_test_string = convergence_test_string[:-1]

        # ----------- Banner (depending on the presence of hartree parameters) -------------
        if hartree:
            pyqcm.banner(
                "CDMFT procedure (combined with Hartree procedure)", "*", skip=1
            )
        else:
            pyqcm.banner("CDMFT procedure", "*", skip=1)

        # -------------- first define the frequency grid for the distance function ---------
        if grid == None:
            self.grid = frequency_grid()
        else:
            self.grid = grid
        print("frequency grid = ", self.grid.name)

        if isinstance(self.alpha, float) and self.alpha > 0.0:
            # Catch if custom float given. Works for both iteration methods.
            print(f"Constant damping factor of {self.alpha}")
        elif isinstance(self.alpha, tuple):
            # Catch if custom tuple is given. Works only for fixed point.
            print(
                f"Applying a damping factor {self.alpha[0]} after {self.alpha[1]} iterations "
            )
        elif iteration == "fixed_point":
            # Set default 'low-convergence' damping for fixed point iterations
            self.alpha = (0.3, 16)
            print(
                f"Applying a damping factor {self.alpha[0]} after {self.alpha[1]} iterations "
            )

        print("-" * 100)

        # -------------------------------------- bias field --------------------------------
        if self.bias != None:
            model.set_parameter(self.bias.op, self.bias.value0)

        # ------------------------------- CDMFT main iteration loop ------------------------
        self.niter, self.iter = 0, 0

        def F(x):
            self.x = np.copy(x)
            self.CDMFT_step()
            return x - self.x

        def G():
            return self.check_convergence()

        if iteration == "broyden":
            try:
                actual_method = "broyden"
                self.x, self.niter, self.alpha = pyqcm.broyden(
                    F,
                    self.x,
                    self.alpha,
                    maxiter=maxiter,
                    miniter=miniter,
                    xtol=1e-6,
                    convergence_test=G,
                )
            except Exception as E:
                raise pyqcm.SolverError("Failure of the CDMFT method (broyden)") from E

        elif iteration == "fixed_point":
            try:
                actual_method = "fixed_point"
                self.x, self.niter = pyqcm.fixed_point_iteration(
                    F,
                    self.x,
                    xtol=1e-6,
                    convergence_test=G,
                    maxiter=maxiter,
                    miniter=miniter,
                    damping=self.alpha,
                    eps_algo=eps_algo,
                )
            except Exception as E:
                raise pyqcm.SolverError(
                    "Failure of the CDMFT method (fixed-point)"
                ) from E
        else:
            raise ValueError(
                'unknown iteration method in CDMFT. must be one of "broyden", "fixed_point". Check spelling.'
            )

        # Here we have converged
        self.I = pyqcm.model_instance(
            self.model
        )  # a last instance with the converged parameters

        # check consistency
        self.I.GS_consistency(self.check_ground_state)
        var_val = pyqcm.varia_table(self.varia, self.x)
        pyqcm.banner(
            "converged variational parameters ({:d} iterations)".format(self.niter), "-"
        )
        print(var_val)

        ave = self.I.averages(pr=True)
        if compute_potential_energy:
            self.I.potential_energy()
        if SEF:
            omega = self.I.Potthoff_functional(self.hartree)

        if file != None:
            self.I.props["opt_method"] = method
            self.I.props["CDMFT_method"] = actual_method
            self.I.props["CDMFT_iterations"] = self.niter
            self.I.props["dist_function"] = self.grid.name
            self.I.props["convergence"] = convergence_test_string
            self.I.props["min_dist"] = self.dist
            self.I.write_summary(file)

        pyqcm.banner("CDMFT completed successfully", "*")

    # -----------------------------------------------------------------------------------------------
    def CDMFT_step(self):
        """
        Performs a CDMFT step that brings the system from the current values (self.x) of the bath and Hartree
        parameters and updates it to the next set of values
        """

        check_bounds(self.x, self.max_value, v=self.varia)

        self.model.set_parameter(self.varia, self.x)
        self.I = pyqcm.model_instance(self.model)
        self.grid.udpate(self.I)

        # solve the impurity problem

        # initializes G_host
        t1 = timeit.default_timer()
        if self.pre_host != None:
            self.pre_host(self.I)

        # computing or transferring the host array --------------------------------------

        if self.host_function == None:
            qcm.CDMFT_host(self.grid.wr, self.grid.cdmft_weight, self.I.label)
        else:
            self.host_function(self.I)
        # --------------------------------------------------------------------------------

        self.I.GS_consistency(self.check_ground_state)

        self.set_Hyb()
        t2 = timeit.default_timer()
        time_ED = t2 - t1

        # updates the Hartree mean-field parameters
        if self.nvar[0]:
            for C in self.hartree:
                C.update(self.I, pr=True)
            P = self.model.parameters()
            for i, h in enumerate(self.hartree):
                self.x[i] = P[h.Vm]

        gs = self.I.ground_state()

        # --------------------------------------------------------------------------------
        # optimization of the bath parameters, one cluster at a time

        self.dist = 0.0
        ic = self.nvar[0]
        for c in range(1, self.model.nclus + 1):
            if self.nvar[c] == 0:
                continue
            x0 = self.x[ic : ic + self.nvar[c]]
            _clus = c - 1
            _label = self.I.label
            # NOTE: CDMFT_residuals returns r such that ‖r‖² ∝ D (same minimiser,
            # different scale). CDMFT_distance normalises by dw/(dim²) for reporting.
            if self.jac:
                # All jac-capable methods receive the residual vector r(x) and the
                # full Jacobian J(x) = ∂r/∂x.  optimize() selects the appropriate
                # scipy.optimize.least_squares sub-algorithm (trf / lm / dogbox).
                F = lambda x, _c=_clus, _l=_label: np.asarray(
                    qcm.CDMFT_residuals(x, _c, _l)
                )
                jac_fn = lambda x, _c=_clus, _l=_label: np.asarray(
                    qcm.CDMFT_gradient(x, _c, _l)
                )
                maxfev = self.lm_max_nfev
            else:
                F = lambda x, grad=None: qcm.CDMFT_distance(x, _clus, _label)
                jac_fn = False
                maxfev = self.max_function_eval
            opt_x, opt_iter_done, opt_success, _ = optimize(
                F,
                x0,
                self.method,
                self.initial_step,
                self.accur_bath,
                self.accur_dist,
                maxfev,
                jac=jac_fn,
            )
            opt_fun = qcm.CDMFT_distance(opt_x, _clus, _label)

            self.dist += opt_fun

            if opt_iter_done > maxfev:
                print(opt_x)
                pyqcm.banner(
                    "number of function evaluations exceeds preset maximum of {:d}".format(
                        self.max_function_eval
                    ),
                    "!",
                )
                raise pyqcm.MinimizationError()

            if np.any(np.isnan(opt_x)):
                print(opt_x)
                pyqcm.banner("NaN found in optimization of bath parameters", "!")
                raise pyqcm.MinimizationError(
                    "NaN found in optimization of bath parameters"
                )

            # push back into array
            self.x[ic : ic + self.nvar[c]] = np.copy(opt_x)
            self.var_data[ic : ic + self.nvar[c], self.iter] = np.copy(opt_x)

            check_bounds(opt_x, self.max_value, v=self.varia[ic : ic + self.nvar[c]])

            ic += self.nvar[c]

        # --------------------------------------------------------------------------------
        # decreasing the bias, if any

        if self.bias != None:
            if self.iter > self.bias.maxiter:
                self.bias.value = 1e-9
            elif self.iter < self.bias.miniter:
                pass
            else:
                self.bias.value *= self.bias.factor
            if self.bias.value < 1e-9:
                self.bias.value = 1e-9
            self.model.set_parameter(self.bias.op, self.bias.value)
            print("====> bias field : ", self.bias.op, " = ", self.bias.value)

        # --------------------------------------------------------------------------------
        t3 = timeit.default_timer()
        time_MIN = t3 - t2

        if self.post_min != None:
            self.post_min(self.I)

        # writing the parameters in a progress file
        self.I.props["opt_method"] = self.method
        self.I.props["dist_function"] = self.grid.name
        self.I.props["min_dist"] = self.dist
        self.I.write_summary(self.iter_file)

        print("GS sector : ", [X[1] for X in gs])
        print(
            "{:d} minimization steps, time(MIN)/time(ED)={:.3g}, distance = {:1.4g}".format(
                opt_iter_done, time_MIN / time_ED, opt_fun
            ),
            flush=True,
        )

        var_val = pyqcm.varia_table(self.varia, self.x)
        print("updated variational parameters:\n{:s}".format(var_val))
        self.iter += 1

        return

    # -----------------------------------------------------------------------------------------------
    def check_convergence(self):
        """
        Performs the CDMFT convergence tests
        returns True if all are converged, otherwise False
        """

        converged = True

        if self.bias != None and self.iter <= self.bias.miniter:
            converged = False

        for C in self.convergence_test:
            if C.name == "GS energy":
                gs = self.I.ground_state()
                E0 = 0.0
                for x in gs:
                    E0 += x[0]
                converged = converged and C.test(E0)

            elif C.name == "self-energy":
                self.set_sigma()
                if self.model.mixing == 4:
                    T = C.test((self.sigma, self.sigma_down))
                    converged = converged and T
                else:
                    T = C.test(self.sigma)
                    converged = converged and T

            elif C.name == "hybridization":
                if self.model.mixing == 4:
                    T = C.test((self.Hyb, self.Hyb_down))
                    converged = converged and T
                else:
                    T = C.test(self.Hyb)
                    converged = converged and T

            elif C.name == "parameters":
                T = C.test(np.copy(self.x))
                converged = converged and T

            elif C.name == "distance":
                T = C.test(self.dist)
                converged = converged and T

            elif C.op != None:
                T = C.test(self.I.averages()[C.op])
                converged = converged and T

        for C in self.convergence_test:
            C.print()

        if converged:
            pyqcm.banner("CDMFT has converged", "=")
            for C in self.convergence_test:
                C.print()

        return converged

    # -----------------------------------------------------------------------------------------------
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
            for c in range(self.model.nclus):
                self.Hyb[c][i, :, :] = self.I.hybridization_function(
                    self.grid.wr[i] * 1j, c, spin_down=False
                )

        if self.model.mixing == 4:
            self.Hyb0_down = self.Hyb_down
            self.Hyb_down = []
            for c in range(self.model.nclus):
                d = self.model.dimGFC[c]
                self.Hyb_down.append(
                    np.zeros((self.grid.nw, d, d), dtype=np.complex128)
                )

            for i in range(self.grid.nw):
                for c in range(self.model.nclus):
                    self.Hyb_down[c][i, :, :] = self.I.hybridization_function(
                        self.grid.wr[i] * 1j, c, spin_down=True
                    )

    # -----------------------------------------------------------------------------------------------
    def diff_matrix(self, X, Y):
        """
        Computes a difference in hybridization functions between the current iteration (Hyb) and the previous one (Hyb0)
        :param object X: the current test object
        :param object Y: any past test object (same structure as X)
        :returns float: the difference in hybridization arrays

        """

        diff = 0.0
        g = self.grid
        if self.model.mixing != 4:
            for i in range(g.nw):
                for c in range(self.model.nclus):
                    delta = X[c][i, :, :] - Y[c][i, :, :]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i] * norm * norm

        if self.model.mixing == 4:
            for i in range(g.nw):
                for c in range(self.model.nclus):
                    delta = X[0][c][i, :, :] - Y[0][c][i, :, :]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i] * norm * norm / X[0][c].shape[1] ** 2
            for i in range(g.nw):
                for c in range(self.model.nclus):
                    delta = X[1][c][i, :, :] - Y[1][c][i, :, :]
                    norm = np.linalg.norm(delta)
                    diff += g.weight[i] * norm * norm / X[1][c].shape[1] ** 2

        if self.model.mixing == 0:
            diff *= 2
        elif self.model.mixing == 3:
            diff /= 2
        # diff /= g.nw
        diff /= self.model.nclus
        return np.sqrt(diff)

    # -----------------------------------------------------------------------------------------------
    def set_sigma(self):
        """Computes the self-energy on the frequency grid
        an array of arrays of matrices. sigma[i], for cluster #i, is a (nw,d,d) Numpy array. with nw frequencies, and d sites

        :returns None

        """
        self.sigma0 = self.sigma
        self.sigma = []
        for c in range(self.model.nclus):
            d = self.model.dimGFC[c]
            self.sigma.append(np.zeros((self.grid.nw, d, d), dtype=np.complex128))

        for i in range(self.grid.nw):
            for c in range(self.model.nclus):
                self.sigma[c][i, :, :] = self.I.cluster_self_energy(
                    self.grid.wr[i] * 1j, c, spin_down=False
                )

        if self.model.mixing == 4:
            self.sigma0_down = self.sigma_down
            self.sigma_down = []
            for c in range(self.model.nclus):
                d = self.model.dimGFC[c]
                self.sigma_down.append(
                    np.zeros((self.grid.nw, d, d), dtype=np.complex128)
                )

            for i in range(self.grid.nw):
                for c in range(self.model.nclus):
                    self.sigma_down[c][i, :, :] = self.I.cluster_self_energy(
                        self.grid.wr[i] * 1j, c, spin_down=True
                    )

    # -----------------------------------------------------------------------------------------------
    def plot_iterations(self):
        """
        Plots the variational parameters as a function of iteration, for diagnostics purposes
        """

        ncols = 3
        nrows = 1 + (self.varia.nvar_tot - 1) // ncols
        fig, ax = plt.subplots(nrows, ncols, sharex=True)
        fig.set_size_inches(24 / 2.54, nrows * 6 / 2.54)
        niter = self.var_data.shape[1]
        for i, x in enumerate(self.var):
            if nrows == 1:
                plt.sca(ax[i])
            else:
                plt.sca(ax[i // ncols, i % ncols])
            plt.plot(range(self.niter), self.var_data[i, 0:niter], "o-", ms=3, lw=0.5)
            plt.title(self.varia[i])
        plt.savefig("iterations.pdf")


####################################################################################################
class frequency_grid:
    """
    This class contains the imaginary frequency grid data, including weights

    :param str grid_type: type of frequency grid along the imaginary axis : 'legendre', 'matsubara', 'regular'
    :param tuple specs: specific parameters for each grid type. For 'legendre', specs=(w1, w2, n1, n2, n3) where w1 and w2 define 3 regions along the imaginary frequency axis. Below w1, n1 grid points are used; between w1 and w2, n2 grid points are used, and n3 grid points are used between w2 and infinity. For 'matsubara', specs=(wc, beta), where wc is the frequency cutoff and beta the inverse effective temperature. For 'regular', specs=(wc, n1, n2), where wc is the boundary between the low- and high-frequency regions, and n1 and n2 the number of points in each region, respectively.
    :param str opt: if opt='self', the cdmft weights (not the integration weights) are not the same as the integration weights, but scale like the norm of the self-energy at the corresponding frequency (the Hartree-Fock part of the self-energy is subtracted). If opt = 'ifreq', the cdmft weights are rather scaled like 1/frequency.
    :ivar wr: (real array) the frequencies along the imaginary axis
    :ivar weight: the weight associated to each frequency in an integral
    :ivar cdmft_weight: the weight associated to each frequency in the CDMFT distance function
    :ivar name: the name of the frequency array scheme chosen (for reference)
    """

    def __init__(self, grid_type="legendre", specs=(1, 10, 5, 10, 5), opt=""):
        if grid_type == "legendre" or grid_type == "Legendre":
            self.grid_type = "legendre"
            w1, w2, n1, n2, n3 = specs
            self.wr, self.weight = pyqcm.legendre_frequency_grid(w1, w2, n1, n2, n3)
            self.name = "legendre({:g}-{:g}-{:d}-{:d}-{:d})".format(w1, w2, n1, n2, n3)

        elif grid_type == "Matsubara" or grid_type == "matsubara":
            wc, beta = specs
            self.wr = np.arange((np.pi / beta), wc + 1e-6, 2 * np.pi / beta)
            self.weight = (2 * np.pi / beta) * np.ones_like(self.wr)
            self.cdmft_weight = self.weight
            self.name = "matsubara({:g}-{:g})".format(wc, beta)

        elif grid_type == "regular":
            wc, n1, n2 = specs
            self.wr, self.weight = pyqcm.regular_frequency_grid(wc, n1, n2)
            self.name = "regular({:g}-{:d}-{:d})".format(wc, n1, n2)

        else:
            raise ValueError("Unknown type of frequency grid " + grid_type)

        self.nw = self.wr.shape[0]
        self.cdmft_weight = np.ones_like(self.wr)

        self.self_norm = False
        if opt == "":
            pass
        elif opt == "ifreq":
            tmp = np.ones(1.0 / self.wr)
            self.cdmft_weight = tmp / tmp.sum()
            self.name += "_ifreq"
        elif opt == "self":
            self.self_norm = True
            self.name += "_self"
        else:
            raise ValueError("Unknown option " + opt + " in frequency_grid()")

    def udpate(self, I):
        """
        Updates the cdmft_weight array depending on the model_instance

        :param model_instance I: current model instance
        """
        if self.self_norm:
            Sig_inf = I.cluster_self_energy(1.0e6j)
            for i, x in enumerate(self.wr):
                self.cdmft_weight[i] = np.linalg.norm(
                    I.cluster_self_energy(x * 1j) - Sig_inf
                )
            self.cdmft_weight = self.cdmft_weight / self.cdmft_weight.sum()
        else:
            pass


####################################################################################################
class general_bath:
    """
    Class that construct a cluster model with nb bath orbitals, in the most general way possible, with possible restrictions.

    :param str name: name of the cluster-bath model to be defined
    :param int ns: number of sites in the cluster
    :param int nb: number of bath orbitals in the cluster
    :param bool spin_dependent: if True, the parameters are spin dependent
    :param bool spin_flip: if True, spin-flip hybridizations are present
    :param bool singlet: if True, defines anomalous singlet hybridizations
    :param bool triplet: if True, defines anomalous triplet hybridizations
    :param bool complex: if True, defines imaginary parts as well, when appropriate
    :param list[list[int]] sites: 2-level list of sites to couple to the bath orbitals (labels from 1 to ns). Format resembles [[site labels to bind to orbital 1], ...] .

    :ivar int ns: number of physical sites in the cluster
    :ivar int nb: number of bath orbitals in the cluster
    :ivar str name: name of the cluster model
    :ivar [str] var_E: list of bath parameters of the type "energy level"
    :ivar [str] var_H: list of bath parameters of the type "hybridization"
    :ivar bool spin_dependent: True if the bath parameter conserve spin, but are different for spins up and down
    :ivar bool spin_flip: True if we include spin-flip hybridizations terms
    :ivar bool singlet: True if we include anomalous hybridizations of the singlet type
    :ivar bool triplet: True if we include anomalous hybridizations of the triplet type
    :ivar bool complex: True if the model is complex-valued and thus hybridization operators have both real and imaginary components

    """

    def __init__(
        self,
        ns,
        nb,
        name="clus",
        spin_dependent=False,
        spin_flip=False,
        singlet=False,
        triplet=False,
        complex=False,
        sites=None,
    ):
        self.CM = pyqcm.cluster_model(ns, n_bath=nb, name=name)
        self.CM.var_E = []
        self.CM.var_H = []
        self.spin_dependent = spin_dependent
        self.spin_flip = spin_flip
        self.singlet = singlet
        self.triplet = triplet
        self.complex = complex
        no = ns + nb
        if sites is None:
            self.sites = [range(1, ns + 1) for i in range(nb)]
        else:
            if len(sites) != nb:
                print(
                    'the format of the argument "sites" is incorrect : it should be a sequence of ',
                    ns,
                    " sequences",
                )
                raise ValueError(
                    f'the format of the argument "sites" is incorrect : it should be a sequence of {ns} sequences'
                )
            self.sites = sites

        self.nmixed = 1
        if spin_flip:
            self.nmixed *= 2
        if singlet or triplet:
            self.nmixed *= 2

        # bath energies
        if spin_dependent or spin_flip:
            for x in range(1, nb + 1):
                param_name = "eb{:d}u".format(x)
                self.CM.new_operator(param_name, "one-body", [(x + ns, x + ns, 1)])
                self.CM.var_E.append(param_name)
                param_name = "eb{:d}d".format(x)
                self.CM.new_operator(
                    param_name, "one-body", [(x + ns + no, x + ns + no, 1)]
                )
                self.CM.var_E.append(param_name)
        else:
            for x in range(1, nb + 1):
                param_name = "eb{:d}".format(x)
                self.CM.new_operator(
                    param_name,
                    "one-body",
                    [(x + ns, x + ns, 1), (x + ns + no, x + ns + no, 1)],
                )
                self.CM.var_E.append(param_name)

        # hybridizations
        if spin_dependent or spin_flip:
            for x in range(1, nb + 1):
                for y in self.sites[x - 1]:
                    param_name = "tb{:d}u{:d}u".format(x, y)
                    self.CM.new_operator(param_name, "one-body", [(y, x + ns, 1)])
                    self.CM.var_H.append(param_name)
                    param_name = "tb{:d}d{:d}d".format(x, y)
                    self.CM.new_operator(
                        param_name, "one-body", [(y + no, x + ns + no, 1)]
                    )
                    self.CM.var_H.append(param_name)
                    if spin_flip:
                        param_name = "tb{:d}u{:d}d".format(x, y)
                        self.CM.new_operator(
                            param_name, "one-body", [(x + ns, y + no, 1)]
                        )
                        self.CM.var_H.append(param_name)
                        param_name = "tb{:d}d{:d}u".format(x, y)
                        self.CM.new_operator(
                            param_name, "one-body", [(y + no, x + ns + no, 1)]
                        )
                        self.CM.var_H.append(param_name)

                if complex:
                    for y in self.sites[x - 1][1:]:
                        param_name = "tb{:d}u{:d}ui".format(x, y)
                        self.CM.new_operator_complex(
                            param_name, "one-body", [(y, x + ns, 1j)]
                        )
                        self.CM.var_H.append(param_name)
                        param_name = "tb{:d}d{:d}di".format(x, y)
                        self.CM.new_operator_complex(
                            param_name, "one-body", [(y + no, x + ns + no, 1j)]
                        )
                        self.CM.var_H.append(param_name)
                    if spin_flip:
                        for y in self.sites[x - 1]:
                            param_name = "tb{:d}u{:d}di".format(x, y)
                            self.CM.new_operator_complex(
                                param_name, "one-body", [(x + ns, y + no, 1j)]
                            )
                            self.CM.var_H.append(param_name)
                            param_name = "tb{:d}d{:d}ui".format(x, y)
                            self.CM.new_operator_complex(
                                param_name, "one-body", [(y + no, x + ns + no, 1j)]
                            )
                            self.CM.var_H.append(param_name)

        else:
            for x in range(1, nb + 1):
                for y in self.sites[x - 1]:
                    param_name = "tb{:d}{:d}".format(x, y)
                    self.CM.new_operator(
                        param_name,
                        "one-body",
                        [(y, x + ns, 1), (y + no, x + ns + no, 1)],
                    )
                    self.CM.var_H.append(param_name)

                if complex:
                    for y in self.sites[x - 1][1:]:
                        param_name = "tb{:d}{:d}i".format(x, y)
                        self.CM.new_operator_complex(
                            param_name,
                            "one-body",
                            [(y, x + ns, 1j), (y + no, x + ns + no, 1j)],
                        )
                        self.CM.var_H.append(param_name)

        if singlet:
            for x in range(1, nb + 1):
                for y in self.sites[x - 1]:
                    param_name = "sb{:d}{:d}".format(x, y)
                    self.CM.new_operator(
                        param_name,
                        "anomalous",
                        [(y, x + ns + no, 1), (x + ns, y + no, 1)],
                    )
                    self.CM.var_H.append(param_name)
                if complex:
                    for y in self.sites[x - 1]:
                        if x == 1 and y == 1:
                            continue
                        param_name = "sb{:d}{:d}i".format(x, y)
                        self.CM.new_operator_complex(
                            param_name,
                            "anomalous",
                            [(y, x + ns + no, 1j), (x + ns, y + no, 1j)],
                        )
                        self.CM.var_H.append(param_name)

        if triplet:
            for x in range(1, nb + 1):
                for y in self.sites[x - 1]:
                    param_name = "pb{:d}{:d}".format(x, y)
                    self.CM.new_operator(
                        param_name,
                        "anomalous",
                        [(y, x + ns + no, 1), (x + ns, y + no, -1)],
                    )
                    self.CM.var_H.append(param_name)
                if complex:
                    for y in self.sites[x - 1]:
                        if x == 1 and y == 1:
                            continue
                        param_name = "pb{:d}{:d}i".format(x, y)
                        self.CM.new_operator_complex(
                            param_name,
                            "anomalous",
                            [(y, x + ns + no, 1j), (x + ns, y + no, -1j)],
                        )
                        self.CM.var_H.append(param_name)

    # -----------------------------------------------------------------------------------------------
    def __str__(self):
        S = 'cluster model "' + self.name + '"'
        if self.spin_flip:
            S += ", spin flip"
        if self.spin_dependent:
            S += ", spin dependent"
        if self.complex:
            S += ", complex-valued"
        if self.singlet:
            S += ", singlet SC"
        if self.singlet:
            S += ", triplet SC"
        S += "\nbath energies : "
        for x in self.CM.var_E:
            S += x + ", "
        S = S[0:-2] + "\nhybridizations : "
        for x in self.CM.var_H:
            S += x + ", "
        return S[0:-2]

    # -----------------------------------------------------------------------------------------------
    def varia_E(self, c=1):
        """returns a list of parameter names from the bath energies with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath energies with the suffix appropriate for cluster c

        """
        v = []
        for x in self.CM.var_E:
            v.append(x + "_" + str(c))
        return v

    # -----------------------------------------------------------------------------------------------
    def varia_H(self, c=1):
        """returns a list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        :param int c: label of the cluster (starts at 1)
        :return [str]: list of parameter names from the bath hybridization with the suffix appropriate for cluster c

        """
        v = []
        for x in self.CM.var_H:
            v.append(x + "_" + str(c))
        return v

    # -----------------------------------------------------------------------------------------------
    def varia(self, H=None, E=None, c=1, spin_down=False):
        """creates a dict of variational parameters to values taken from the hybridization matrix H and the energies E, for cluster c

        :param ndarray H: matrix of hybridization values, shape (nmixed*nb, nmixed*ns)
        :param ndarray E: array of energy values for the bath orbitals
        :param int c: label of the cluster (starts at 1), used as suffix in parameter names
        :param bool spin_down: True for the spin-down values
        :return {str,float}: dict of variational parameters to values

        """
        nb = self.nb
        ns = self.ns
        no = ns + nb
        nn = no * self.nmixed // 2

        if H.shape != (self.nmixed * self.nb, self.nmixed * self.ns):
            raise ValueError(
                "shape of hybridization matrix does not match model in general bath back propagation"
            )

        D = {}
        # bath energies
        if self.spin_flip:
            for x in range(1, nb + 1):
                param_name = "eb{:d}u_{:d}".format(x, c)
                D[param_name] = E[x - 1]
                param_name = "eb{:d}d_{:d}".format(x, c)
                D[param_name] = E[x - 1 + nb]

        elif self.spin_flip and not spin_down:
            for x in range(1, nb + 1):
                param_name = "eb{:d}u_{:d}".format(x, c)
                D[param_name] = E[x - 1]
        elif self.spin_flip and spin_down == True:
            for x in range(1, nb + 1):
                param_name = "eb{:d}d_{:d}".format(x, c)
                D[param_name] = E[x - 1]
        else:
            for x in range(1, nb + 1):
                param_name = "eb{:d}_{:d}".format(x, c)
                D[param_name] = E[x - 1]

        # hybridizations
        if self.spin_flip:
            for x in range(1, nb + 1):
                for y in self.sites:
                    param_name = "tb{:d}u{:d}u_{:d}".format(x, y, c)
                    D[param_name] = H[x - 1, y - 1].real
                    param_name = "tb{:d}d{:d}d_{:d}".format(x, y, c)
                    D[param_name] = H[x + nb - 1, y - 1].real
                    param_name = "tb{:d}u{:d}d_{:d}".format(x, y, c)
                    D[param_name] = H[x - 1, y + ns - 1].real
                    param_name = "tb{:d}d{:d}u_{:d}".format(x, y, c)
                    D[param_name] = H[x + nb - 1, y + ns - 1].real

                if self.complex:
                    for y in self.sites[1:]:
                        param_name = "tb{:d}u{:d}ui_{:d}".format(x, y, c)
                        D[param_name] = H[x - 1, y - 1].imag
                        param_name = "tb{:d}d{:d}di_{:d}".format(x, y, c)
                        D[param_name] = H[x + nb - 1, y - 1].imag
                        param_name = "tb{:d}u{:d}di_{:d}".format(x, y, c)
                        D[param_name] = H[x - 1, y + ns - 1].imag
                        param_name = "tb{:d}d{:d}ui_{:d}".format(x, y, c)
                        D[param_name] = H[x + nb - 1, y + ns - 1].imag

        elif self.spin_dependent:
            for x in range(1, nb + 1):
                for y in self.sites:
                    if spin_down:
                        param_name = "tb{:d}d{:d}d_{:d}".format(x, y, c)
                    else:
                        param_name = "tb{:d}u{:d}u_{:d}".format(x, y, c)
                    D[param_name] = H[x - 1, y - 1].real

                if self.complex:
                    for y in self.sites[1:]:
                        if spin_down:
                            param_name = "tb{:d}d{:d}di_{:d}".format(x, y, c)
                        else:
                            param_name = "tb{:d}u{:d}ui_{:d}".format(x, y, c)
                        D[param_name] = H[x - 1, y - 1].imag

        else:
            for x in range(1, nb + 1):
                for y in self.sites:
                    param_name = "tb{:d}{:d}_{:d}".format(x, y, c)
                    D[param_name] = H[x - 1, y - 1].real

                if self.complex:
                    for y in self.sites[1:]:
                        param_name = "tb{:d}{:d}i_{:d}".format(x, y, c)
                        D[param_name] = H[x - 1, y - 1].imag

        if self.singlet:
            for y in self.sites:
                param_name = "sb{:d}{:d}_{:d}".format(x, y, c)
                D[param_name] = H[x + nn - 1, y - 1].real  # à modifier
            if self.complex:
                for y in self.sites:
                    if x == 1 and y == 1:
                        continue
                    param_name = "sb{:d}{:d}i_{:d}".format(x, y, c)
                    D[param_name] = H[x + nn - 1, y - 1].imag  # à modifier

        if self.triplet:
            for y in self.sites:
                param_name = "pb{:d}{:d}_{:d}".format(x, y, c)
                D[param_name] = H[x + nn - 1, y - 1].real  # à modifier
            if self.complex:
                for y in self.sites:
                    if x == 1 and y == 1:
                        continue
                    param_name = "pb{:d}{:d}i_{:d}".format(x, y, c)
                    D[param_name] = H[x + nn - 1, y - 1].imag  # à modifier
        return D

    # -----------------------------------------------------------------------------------------------
    def starting_values(
        self, c=1, e=(0.5, 1.5), hyb=(0.5, 0.2), shyb=(0.1, 0.05), pr=False
    ):
        """returns an initialization string for the bath parameters

        :param int c: cluster label (starts at 1)
        :param (float,float) e: bounds of the values for the bath energies (absolute value)
        :param (float,float) hyb: average and deviation of the normal hybridization parameters
        :param (float,float) shyb: average and deviation of the anomalous hybridization parameters
        :param bool pr: prints the starting values if True
        :return str: initialization string

        """
        S = ""
        fac = 1
        E = np.linspace(e[0], e[1], len(self.CM.var_E))
        for i, x in enumerate(self.CM.var_E):
            bn = int(x[2])
            fac = 2 * (bn % 2) - 1
            S += x + "_" + str(c) + " = " + str(fac * E[i]) + "\n"
        for x in self.CM.var_H:
            if x[0:2] == "sb" or x[0:2] == "pb":
                S += (
                    x
                    + "_"
                    + str(c)
                    + " = "
                    + str(shyb[0] + shyb[1] * (2 * np.random.random() - 1))
                    + "\n"
                )
            else:
                S += (
                    x
                    + "_"
                    + str(c)
                    + " = "
                    + str(hyb[0] + hyb[1] * (2 * np.random.random() - 1))
                    + "\n"
                )
        if pr:
            print("starting values:\n", S)
        return S

    # -----------------------------------------------------------------------------------------------
    def starting_values_PH(self, c=1, e=(1, 0.5), hyb=(0.5, 0.2), phi=None, pr=False):
        """returns an initialization string for the bath parameters, in the particle-hole symmetric case.

        :param int c: cluster label
        :param (float) e: range of bath energies
        :param (float,float) hyb: average and deviation of the normal hybridization parameters
        :param [int] phi: PH phases of the cluster sites proper
        :param bool pr: if True, prints info
        :return str: initialization string

        """
        S = ""
        fac = 1
        if self.nb % 2:
            raise ValueError(
                "the number of bath orbitals must be even for starting_values_PH() to apply"
            )
        if phi is None:
            raise ValueError("The PH phases of the sites must be specified")
        NE = self.nb // 2

        if self.spin_dependent or self.spin_flip:
            for i in range(NE):
                S += "eb{:d}u_{:d} = -1.0*eb{:d}d_{:d}\n".format(
                    2 * i + 2, c, 2 * i + 1, c
                )
                S += "eb{:d}d_{:d} = -1.0*eb{:d}u_{:d}\n".format(
                    2 * i + 2, c, 2 * i + 1, c
                )
                for s in self.sites[2 * i]:
                    S += "tb{:d}u{:d}u_{:d} = {:2.1f}*tb{:d}d{:d}d_{:d}\n".format(
                        2 * i + 2, s, c, phi[s - 1], 2 * i + 1, s, c
                    )
                    S += "tb{:d}d{:d}d_{:d} = {:2.1f}*tb{:d}u{:d}u_{:d}\n".format(
                        2 * i + 2, s, c, phi[s - 1], 2 * i + 1, s, c
                    )
                if self.complex:
                    for s in self.sites[2 * i]:
                        S += "tb{:d}u{:d}ui_{:d} = {:2.1f}*tb{:d}d{:d}di_{:d}\n".format(
                            2 * i + 2, s, c, -phi[s - 1], 2 * i + 1, s, c
                        )
                        S += "tb{:d}d{:d}di_{:d} = {:2.1f}*tb{:d}u{:d}ui_{:d}\n".format(
                            2 * i + 2, s, c, -phi[s - 1], 2 * i + 1, s, c
                        )
                if self.spin_flip:
                    for s in self.sites[2 * i]:
                        S += "tb{:d}u{:d}d_{:d} = {:2.1f}*tb{:d}u{:d}d_{:d}\n".format(
                            2 * i + 2, s, c, -phi[s - 1], 2 * i + 1, s, c
                        )
                        S += "tb{:d}d{:d}u_{:d} = {:2.1f}*tb{:d}d{:d}u_{:d}\n".format(
                            2 * i + 2, s, c, -phi[s - 1], 2 * i + 1, s, c
                        )
                    if self.complex:
                        for s in self.sites[2 * i]:
                            S += "tb{:d}u{:d}di_{:d} = {:2.1f}*tb{:d}u{:d}di_{:d}\n".format(
                                2 * i + 2, s, c, phi[s - 1], 2 * i + 1, s, c
                            )
                            S += "tb{:d}d{:d}ui_{:d} = {:2.1f}*tb{:d}d{:d}ui_{:d}\n".format(
                                2 * i + 2, s, c, phi[s - 1], 2 * i + 1, s, c
                            )

        else:
            for i in range(NE):
                S += "eb{:d}_{:d} = -1.0*eb{:d}_{:d}\n".format(
                    2 * i + 2, c, 2 * i + 1, c
                )
                for s in self.sites[2 * i]:
                    S += "tb{:d}{:d}_{:d} = {:2.1f}*tb{:d}{:d}_{:d}\n".format(
                        2 * i + 2, s, c, phi[s - 1], 2 * i + 1, s, c
                    )
                if self.complex:
                    for s in self.sites[2 * i]:
                        S += "tb{:d}{:d}i_{:d} = {:2.1f}*tb{:d}{:d}i_{:d}\n".format(
                            2 * i + 2, s, c, -phi[s - 1], 2 * i + 1, s, c
                        )

        var_E = self.CM.var_E
        self.CM.var_E = []
        for x in var_E:
            for i in range(NE):
                if "eb{:d}".format(2 * i + 1) in x:
                    self.CM.var_E.append(x)
        var_H = self.CM.var_H
        self.CM.var_H = []
        for x in var_H:
            for i in range(NE):
                if "tb{:d}".format(2 * i + 1) in x:
                    self.CM.var_H.append(x)

        for i, x in enumerate(self.CM.var_E):
            S += (
                x
                + "_"
                + str(c)
                + " = "
                + str(e[0] + e[1] * (2 * np.random.random() - 1))
                + "\n"
            )
        for x in self.CM.var_H:
            S += (
                x
                + "_"
                + str(c)
                + " = "
                + str(hyb[0] + hyb[1] * (2 * np.random.random() - 1))
                + "\n"
            )
        if pr:
            print("starting values:\n", S)
        return S


######################################################################
class hybridization:
    """
    Defines hybridization data from a data file.
    the frequency is purely imaginary; it is its imaginary part that appears in the file

    :param str file: name of the file or string. Format : each line starts with a frequency and then has N*N columns for Delta_{ij}(w)

    """

    def __init__(self, file):
        NN = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
        # data = np.genfromtxt(file, dtype=np.complex128)
        # data = np.genfromtxt(file, invalid_raise=True, loose=False, dtype=None)
        data = np.genfromtxt(file, invalid_raise=True, loose=False, dtype=np.complex128)
        # print(data)
        self.nw = data.shape[0]
        LL = data.shape[1] - 1
        try:
            self.n = NN.index(LL) + 1
        except ValueError:
            raise ValueError(
                "The number of columns in the hybridization table should be 1 + n^2"
            ) from None

        self.w = np.copy(np.real(data[:, 0]))
        self.Delta = np.zeros((self.nw, self.n, self.n), dtype=np.complex128)
        for i in range(self.nw):
            self.Delta[i, :, :] = np.reshape(data[i, 1:], (self.n, self.n))

    # -----------------------------------------------------------------------------------------------
    def distance(self, I):
        """
        Evaluates the distance function bewteen the data and the hybridization function of an impurity model instance of label I

        :param [model_instance] I: model instance
        :returns: the squared Frobenius norm of the difference matrix, averaged over frequencies
        :rtype: float
        """
        M = np.zeros((self.nw, self.n, self.n), dtype=np.complex128)
        for i in range(self.nw):
            M[i, :, :] = I.hybridization_function(self.w[i] * 1j, spin_down=False)

        # print('M : '); print(M)
        # print('Delta : '); print(self.Delta)

        dist = np.linalg.norm(M - self.Delta)
        return dist * dist / self.nw

    # -----------------------------------------------------------------------------------------------
    def __str__(self):
        S = ""
        for i in range(self.nw):
            S += "w = " + self.w[i].__str__() + "\n" + self.Delta[i].__str__() + "\n\n"
        return S

    # -----------------------------------------------------------------------------------------------
    def optimize_bath(
        self, model, varia, x, method="Nelder-Mead", accur=1e-4, accur_dist=1e-8
    ):
        """
        Optimizes the bath parameters to fit the hybridization function delta
        :param lattice_model model: the lattice model whose bath parameters are to be optimized
        :param [str] varia: list of variational parameters (bath parameters) from PyQCM
        :param [float] x: starting values of the parameters
        :param str method: minimization method passed to optimize() (default 'Nelder-Mead')
        :param float accur: requested accuracy in the parameters (x-tolerance)
        :param float accur_dist: requested accuracy in the distance function (f-tolerance)
        """

        nvar = len(varia)

        def tmp_distance(x, grad=None):
            for i in range(nvar):
                model.set_parameter(varia[i], x[i])
            I = pyqcm.model_instance(model)
            return self.distance(I)

        return optimize(
            tmp_distance, x, method=method, accur=accur, accur_dist=accur_dist
        )


####################################################################################################
# PRIVATE FUNCTIONS


# ---------------------------------------------------------------------------------------------------
def check_bounds(x, B=100, v=None):
    """Checks whether one of the variables in the array x is out of bounds

    :param [float] x: array of parameters
    :param float B: maximum value of all parameters
    :param [str] v: list of corresponding names, if available
    :returns: None
    """
    for i in range(len(x)):
        if np.abs(x[i]) > B:
            if v != None:
                S = "parameter {:s} = {:g} is out of bounds!".format(v[i], x[i])
            else:
                S = "parameter no {:d}  = {:g} is out of bounds!".format(i + 1, x[i])
            print(
                "Try setting `max_value` to a bigger value within the CDMFT procedure!"
            )
            raise pyqcm.OutOfBoundsError(S)


# ---------------------------------------------------------------------------------------------------
# optimization of the bath parameters


def optimize(
    F,
    x,
    method="Nelder-Mead",
    initial_step=0.1,
    accur=1e-4,
    accur_dist=1e-8,
    maxfev=5000000,
    jac=None,
):
    """

    :param F: function to be optimized.
        When ``jac`` is callable: ``F(x) -> array`` (residual vector ``r(x)``); applies to
        ``trf``, ``BFGS``, and ``L-BFGS-B``.
        When ``jac=False``: ``F(x) -> float`` (scalar objective); ``F`` must also accept an
        optional ``grad`` keyword: ``F(x, grad=None)``.
    :param (float) x: array of variables
    :param str method: minimization method. Scipy methods: ``Nelder-Mead``, ``Powell``, ``CG``,
        ``BFGS``, ``ANNEAL``. NLopt derivative-free: ``COBYLA``, ``BOBYQA``, ``PRAXIS``,
        ``NELDERMEAD``, ``SUBPLEX``. NLopt gradient-based (forward finite differences): ``L-BFGS-B``.
        Jacobian-based least-squares (require ``jac`` callable): ``trf`` (trust-region reflective),
        ``BFGS`` (Levenberg-Marquardt, no bounds), ``L-BFGS-B`` (dogbox, supports bounds).
        All three jac-capable methods call ``scipy.optimize.least_squares`` and therefore
        share the same xtol/ftol stopping criteria, guaranteeing equivalent bath-parameter
        precision and consistent outer CDMFT convergence.
    :param float initial_step: initial step in the minimization procedure
    :param float accur: absolute tolerance on the parameters (x-tolerance)
    :param float accur_dist: absolute tolerance on the objective function value (f-tolerance)
    :param int maxfev: maximum number of function evaluations allowed
    :param jac: ``False`` (default, derivative-free) or a callable ``jac(x) -> 2-D array``
        returning the full Jacobian matrix J = ∂r/∂x (shape ``Nresiduals × Nparams``).
    :param bounds: ``None`` (unbounded, ±∞) or a positive scalar ``B`` that applies a ``[-B, B]``
        box to all parameters. Used by ``trf`` and ``L-BFGS-B`` when ``jac`` is callable;
        ignored by ``BFGS`` (which uses Levenberg-Marquardt, an unbounded algorithm).
    :returns: a 4-tuple (opt_x, iter_done, success, fun) with the optimal parameters, number of evaluations, success flag, and optimal function value
    :rtype: tuple

    """

    _JAC_SUPPORTED = {"trf", "bfgs", "l-bfgs-b"}
    if callable(jac) and method.lower() not in _JAC_SUPPORTED:
        raise ValueError(
            f"method '{method}' does not support a Jacobian; "
            f"supported methods: {', '.join(sorted(_JAC_SUPPORTED))}"
        )
    if method.lower() == "trf" and not callable(jac):
        raise ValueError('"trf" requires a callable jac')

    nvar = len(x)
    nlopt_df_methods = {
        "cobyla": nlopt.LN_COBYLA,
        "bobyqa": nlopt.LN_BOBYQA,
        "praxis": nlopt.LN_PRAXIS,
        "neldermead": nlopt.LN_NELDERMEAD,
        "subplex": nlopt.LN_SBPLX,
    }
    if method.lower() == "trf":
        sol_ls = scipy.optimize.least_squares(
            F,
            x,
            jac=jac,
            method="trf",
            max_nfev=maxfev,
            ftol=accur_dist,
            xtol=accur,
            gtol=1e-12,
        )
        opt_x, iter_done, success, fun = (
            sol_ls.x,
            sol_ls.nfev,
            sol_ls.success,
            sol_ls.cost,
        )

    elif method == "Nelder-Mead":
        initial_simplex = np.zeros((nvar + 1, nvar))
        for i in range(nvar + 1):
            initial_simplex[i, :] = x
        for i in range(nvar):
            initial_simplex[i + 1, i] += initial_step
        sol = scipy.optimize.minimize(
            F,
            x,
            method="Nelder-Mead",
            options={
                "maxfev": maxfev,
                "xatol": accur,
                "fatol": accur_dist,
                "initial_simplex": initial_simplex,
                "adaptive": True,
                "disp": False,
            },
        )
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == "Powell":
        sol = scipy.optimize.minimize(F, x, method="Powell", tol=accur)
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == "CG":
        sol = scipy.optimize.minimize(
            F,
            x,
            method="CG",
            jac=False,
            tol=accur,
            options={
                "gtol": accur,
                "eps": pyqcm.get_global_parameter("cdmft_jacobian_delta"),
            },
        )
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == "BFGS":
        if callable(jac):
            # Route through Levenberg-Marquardt (least-squares, no bounds).
            # BFGS with a gradient J^T r is suboptimal for ‖r‖²: it ignores the
            # Gauss-Newton structure and has no direct x-tolerance stopping criterion.
            # LM uses J directly (Gauss-Newton step) and stops on xtol, matching TRF.
            sol_ls = scipy.optimize.least_squares(
                F,
                x,
                jac=jac,
                method="lm",
                max_nfev=maxfev,
                ftol=accur_dist,
                xtol=accur,
                gtol=1e-12,
            )
            opt_x, iter_done, success, fun = (
                sol_ls.x,
                sol_ls.nfev,
                sol_ls.success,
                sol_ls.cost,
            )
        else:
            sol = scipy.optimize.minimize(
                F,
                x,
                method="BFGS",
                jac=False,
                tol=accur,
                options={"eps": pyqcm.get_global_parameter("cdmft_jacobian_delta")},
            )
            opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method == "ANNEAL":
        sol = scipy.optimize.basinhopping(
            F,
            x,
            minimizer_kwargs={
                "method": "COBYLA",
                "options": {
                    "rhobeg": initial_step,
                    "maxiter": maxfev,
                    "catol": accur_dist,
                },
            },
        )
        opt_x, iter_done, success, fun = sol.x, sol.nfev, sol.success, sol.fun

    elif method.lower() in nlopt_df_methods.keys():
        optimizer = nlopt.opt(nlopt_df_methods[method.lower()], nvar)

        optimizer.set_min_objective(F)
        optimizer.set_lower_bounds(np.full(nvar, -np.inf))
        optimizer.set_upper_bounds(np.full(nvar, np.inf))

        optimizer.set_ftol_abs(accur_dist)
        optimizer.set_xtol_abs(accur)

        optimizer.set_maxeval(maxfev)
        optimizer.set_initial_step(initial_step)

        sol = optimizer.optimize(np.asarray(x, dtype=float))
        opt_x, iter_done, success, fun = (
            sol,
            optimizer.get_numevals(),
            optimizer.last_optimize_result() > 0,
            optimizer.last_optimum_value(),
        )

    elif method == "L-BFGS-B":
        if callable(jac):
            # Route through dogbox (least-squares, supports bounds).
            # NLopt LD_LBFGS with relative tolerances stalls when ‖r‖ is near
            # machine precision; dogbox uses xtol/ftol (absolute) and exploits J
            # directly, matching TRF's convergence precision.
            sol_ls = scipy.optimize.least_squares(
                F,
                x,
                jac=jac,
                method="dogbox",
                max_nfev=maxfev,
                ftol=accur_dist,
                xtol=accur,
                gtol=1e-12,
            )
            opt_x, iter_done, success, fun = (
                sol_ls.x,
                sol_ls.nfev,
                sol_ls.success,
                sol_ls.cost,
            )
        else:
            _fd_eps = pyqcm.get_global_parameter("cdmft_jacobian_delta")

            def _F_with_grad(x, grad):
                f0 = F(x)
                if grad.size > 0:
                    for i in range(nvar):
                        x_fwd = x.copy()
                        x_fwd[i] += _fd_eps
                        grad[i] = (F(x_fwd) - f0) / _fd_eps
                return f0

            optimizer = nlopt.opt(nlopt.LD_LBFGS, nvar)
            optimizer.set_min_objective(_F_with_grad)
            optimizer.set_lower_bounds(np.full(nvar, -np.inf))
            optimizer.set_upper_bounds(np.full(nvar, np.inf))
            optimizer.set_ftol_abs(accur_dist)
            optimizer.set_xtol_abs(accur)
            optimizer.set_maxeval(maxfev)
            optimizer.set_vector_storage(0)
            optimizer.set_exceptions_enabled(False)
            sol = optimizer.optimize(np.asarray(x, dtype=float))
            opt_x, iter_done, success, fun = (
                sol,
                optimizer.get_numevals(),
                optimizer.last_optimize_result() > 0,
                optimizer.last_optimum_value(),
            )

    else:
        raise ValueError(f"unknown method specified for minimization: {method}")

    if not success:
        raise pyqcm.MinimizationError(
            "Failure of the optimization of the bath parameters"
        )

    return opt_x, iter_done, success, fun


if __name__ == "__main__":
    pass
