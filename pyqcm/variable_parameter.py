import numpy as np
import pyqcm

class variable_parameter:
    """This class contains the elements needed to perform computations with parameters that depend on some observable (typically the density). The values of the parameters are adjusted as a function of the observable until some self-consistency is reached.

    attributes:
        - param (str) : name of the parameter that depends on the observable
        - F : function that expresses the parameter as a function of the observable (supplied by the user)
        - obs (str) : name of the observable (a parameter). Default is 'mu'
        - lattice (boolean) : True if lattice averages are used
        - diff : difference between successive values of the parameter
        - ave : current value of the observable
        - accur : desired accuracy on the parameter

    """

    size0 = 0

    def __init__(self, param, F, obs='mu', lattice=True, accur=1e-3):
        """

        :param str param: name of the original parameter
        :param F: function that returns the value of the parameter as a function of the observable
        :param str obs: name of the parameter whose average is the observable
        :param boolean lattice: if True, the lattice average is used, otherwise the cluster average
        :param float accur: required accuracy of the self-consistent procedure

        """

        self.param = param
        self.value = 0.0
        self.F = F
        self.obs = obs
        self.lattice = lattice
        self.diff = 1e6
        self.ave = 0
        self.accur = accur
        self.iter = 0    

    def update(self, pr=False):
        """Updates the value of the parameter based on the value of the observable
        
        :param boolean pr: if True, progress is printed on the screen

        """

        if not self.lattice:
            self.ave = pyqcm.cluster_averages()[self.obs][0]
        else:
            self.ave = pyqcm.averages()[self.obs]
        
        P0 = pyqcm.parameters()[self.param]
        P = self.F(self.ave)
        pyqcm.set_parameter(self.param, P)
        self.diff = P-P0
        self.iter += 1
        if pr:
            print('<{:s}> = {:1.3g},  {:s} --> {:1.3g}, diff = {:1.3g}'.format(self.obs, self.ave, self.param, P, self.diff))


    def compute_obs(self):
        """Computes the observable
        """

        if not self.lattice:
            self.ave = pyqcm.cluster_averages()[self.obs][0]
        else:
            self.ave = pyqcm.averages()[self.obs]
        

    def converged(self):
        """Tests whether the self-consistent procedure has converged

        :return boolean: True if the mean-field procedure has converged
        
        """

        if np.abs(self.diff) < self.accur:
            return True
        else:
            return False


    def __str__(self):
        return 'parameter: '+self.param+', observable: <'+self.obs+'>'


################################################################################
def variable_parameter_self_consistency(F, var_params, maxiter=10, file='var_param.tsv'):
    """Performs a task with parameters that depend on an observable (like density) and self-consistently
    finds the correct value of the parameters that is self-consistent with the observable

    :param F: task to perform within the loop
    :param [class variable_parameter] var_params: list of variable_parameters (or single variable_parameters)
    :param int maxiter: maximum number of iterations in the self-consistent procedure

    """

    global first_time

    if type(var_params) is not list:
        var_params = [var_params]

    pyqcm.banner('Variable parameter self-consistency', c='*', skip=1)
    SC_converged = False
    diff_tot = 1e6

    iter = 0
    while True:
        pyqcm.new_model_instance()
        F()
        iter += 1
        pyqcm.banner('self-consistent iteration {:d}'.format(iter), '-')
        diff_tot = 0
        SC_converged = True
        for i,C in enumerate(var_params):
            C.update(pr=True)
            diff_tot += np.abs(C.diff)
            SC_converged = SC_converged and C.converged()

        print('total difference = {:g}'.format(diff_tot))

        if SC_converged:
            pyqcm.write_summary(file)
            first_time = False
            break

        if iter > maxiter :
            raise RuntimeError('Maximum number of iterations exceeded in self-consistency for variable parameters! Aborting...')

    pyqcm.banner('Variable parameter procedure has converged', c='*')
    

################################################################################
def variable_parameter_search(F, var_param, bracket, file='var_param.tsv'):
    """Performs a root search (Brent algorithm) to find the correct value of the variable parameter
    that matches the observable

    :param F: task to perform within the loop
    :param class variable_parameter var_param: variable_parameter
    :param [float] bracket: list of two values of var_param that bracket the solution

    """

    from scipy.optimize import brentq

    pyqcm.banner('Variable parameter search (Brent method)', c='*', skip=1)

    def G(x):
        pyqcm.set_parameter(var_param.param, x)
        pyqcm.new_model_instance()
        F()
        var_param.compute_obs()
        return x - var_param.F(var_param.ave)

    x0, r = brentq(G, bracket[0], bracket[1], xtol=var_param.accur, maxiter=100, full_output=True, disp=True)

    print(r.flag)
    if not r.converged:
        raise RuntimeError('the root finding routine could not find a solution!')
    else:
        pyqcm.write_summary(file)
        pyqcm.banner('Variable parameter procedure has converged to {:s} = {:4g}'.format(var_param.param, x0), c='*')
        pyqcm.set_parameter(var_param.param, x0)
        pyqcm.new_model_instance()
    