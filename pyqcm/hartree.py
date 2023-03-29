import numpy as np
import pyqcm

class hartree:
    """This class contains the elements needed to perform the Hartree approximation for the inter-cluster components of an
    extended interaction. The basic self-consistency relation is

    .. math:: v_m = ve\langle V_m\\rangle    
    
    where *v* is the coefficient of the operator *V* and :math:`v_m` that of the operator :math:`V_m`, and *e* is an eigenvalue specific to the cluster shape and the interaction. :math:`\langle V_m\\rangle` is the average of the operator :math:`V_m`, taken
    as a lattice of as a cluster average.

    attributes:
        - Vm (str) : mean-field operator
        - V (str) : extended interaction
        - eig (float) : eigenvalue *e* of the mean-field operator in the self-consistency relation
        - lattice (boolean) : True if lattice averages are used
        - diff : difference between successive values of :math:`v_m``
        - ave : averasge of the operator :math:`V_m`
        - accur : desired accuracy

    """

    size0 = 0

    def __init__(self, Vm, V, eig, accur=1e-4, lattice=False):
        """

        :param str Vm: name of the mean-field operator
        :param str V: name of the interaction operator
        :param float eig: eigenvalue
        :param float accur: required accuracy of the self-consistent procedure
        :param boolean lattice: if True, the lattice average is used, otherwise the cluster average

        """
        # if len(pyqcm.the_model.clusters) > 1:
        #     raise ValueError('For the moment, Hartree couplings are only possible in a model with a single cluster')

        self.Vm = Vm
        self.V = V
        self.eig = eig
        self.lattice = lattice
        self.diff = 1e6
        self.ave = 0
        self.accur = accur
        self.epsilon = False
        self.iter = 0

        # par = pyqcm.parameters()
        # assert V in par, 'Hartree : '+V+' is not a parameter in the model!'
        # assert Vm in par, 'Hartree : '+Vm+' is not a parameter in the model!'

        if lattice:
            self.size0 = pyqcm.the_model.sites.shape[0]
        else:
            self.size0 = pyqcm.model_size()[2][0]
    

    def update(self, pr=False):
        """Updates the value of the mean-field operator based on its average
        
        :param boolean pr: if True, progress is printed on the screen

        """

        par = pyqcm.parameters()
        v = par[self.V]
        vm0 = par[self.Vm]
        if not self.lattice:
            self.ave = pyqcm.cluster_averages()[self.Vm][0]*self.size0
        else:
            self.ave = pyqcm.averages()[self.Vm]*self.size0
        self.vm = self.eig*v*self.ave
        pyqcm.set_parameter(self.Vm, self.vm)
        self.diff = self.vm-vm0
        meta_pr = ''
        if self.epsilon:
            eps_length = 2*self.epsilon + 1
            self.data[self.iter] = self.vm
            if self.iter >= 2*eps_length and self.iter%(2*eps_length) == 0:
                self.vm = pyqcm.epsilon(self.data[self.iter-eps_length:self.iter])
                meta_pr = ' (epsilon algo)'
        self.iter += 1
        if pr:
            print('delta {:s} = {:1.3g})'.format(self.Vm, self.diff), meta_pr)

    def omega(self):
        """returns the constant contribution, added to the Potthoff functional
        
        """

        par = pyqcm.parameters()
        v = par[self.V]
        vm0 = par[self.Vm]
        if not self.lattice:
            self.ave = pyqcm.cluster_averages()[self.Vm][0]
        else:
            self.ave = pyqcm.averages()[self.Vm]
        return -0.5*self.eig*v*self.ave*self.ave

    def omega_var(self):
        """returns the constant contribution, added to the Potthoff functional
        
        """

        par = pyqcm.parameters()
        v = par[self.V]
        vm0 = par[self.Vm]
        return -0.5*vm0*vm0/(self.eig*v)

    def converged(self):
        """Tests whether the mean-field procedure has converged

        :return boolean: True if the mean-field procedure has converged
        
        """

        if np.abs(self.diff) < self.accur:
            return True
        else:
            return False


    def __str__(self):
        return 'extended interaction '+self.V+', mean-field operator '+self.Vm+', coupling = {:f}'.format(self.eig)


    def print(self):
        print('<{:s}> = {:g}\t{:s} = {:g} (diff = {:g})'.format(self.Vm, self.ave, self.Vm, self.vm, self.diff))

    def init_epsilon(self, n, eps_length):
        self.data = np.empty(n+1)
        self.epsilon = eps_length
        self.iter = 0



################################################################################
class counterterm:
    """That class contains information about operators that are added to the cluster 
    Hamiltonian in order to make their average vanish. The coefficient *v* of the operator *V*
    is adjusted so that :math:`\langle V\\rangle=0`. The coefficient is updated using the 
    relation 

    .. math:: \langle O_1\\rangle-\langle O_2\\rangle = S (v_1-v_2)

    where :math:`v_{1,2}` are two values of the coefficient of the operator used in succession,
    :math:`\langle v_{1,2}\\rangle` are the corresponding averages and *S* is the (varying) slope.

    attributes:
        - name (str) : name of the countererm operator
        - fullname (str) : name, including the cluster label
        - clus (int) : label of the cluster
        - S (float) : slope
        - accur (float) : accuracy to which the average must vanish.
        - v (float) : value of the coefficient of the operator
        - v0 (float) : previous value of the coefficient of the operator (previous iteration)
    
    """


    size0 = 0

    def __init__(self, name, clus, S, accur=1e-4):
        """Constructor

        :param str name: name of the counterterm (previously defined in the model)
        :param int clus: cluster label to which it is applied
        :param float S: initial value of the slope
        :param float accur: accuracy

        """

        self.name = name
        self.fullname = '{:s}_{:d}'.format(name, clus)
        self.clus = clus
        self.S = S
        self.accur = accur
        self.ave = 0.0

        par = pyqcm.parameters()
        if self.fullname not in par:
            raise ValueError(f'Counterterm {self.fullname} is not a parameter in the model!')

        self.v = par[self.fullname]
        self.v0 = 1e-9

    def update(self, first=False):
        """updates the value of the operator given its averaged

        :param boolean first: True if this is the first time it is called for a given iterative sequence

        """
        self.ave0 = self.ave
        self.ave = pyqcm.cluster_averages(self.clus-1)[self.name][0]
        if not first:
            self.S = (self.ave-self.ave0)/(self.v-self.v0)
        self.v0 = self.v
        self.v -= self.ave/self.S
        pyqcm.set_parameter(self.fullname, self.v)
        print(self)


    def converged(self):
        """Performs a convergence test

        :return boolean: True if converged
        
        """

        if np.abs(self.ave) < self.accur :
            return True
        else:
            return False


    def __str__(self):
        return 'counterterm {:s} = {:1.6g}, average = {:1.6g}, slope = {:1.6g}'.format(self.fullname, self.v, self.ave, self.S)


    def print(self):
        print(self.__str__())