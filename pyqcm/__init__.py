
import os
import sys

import numpy as np
import re
import time
import copy
from . import qcm

des_dict = {}  # use to store description lines in output files. filename->current description line

####################################################################################################
# INITIALIZATION

try:
    from . import qcm_git_hash
    git_hash = qcm_git_hash.git_hash
except:
    print('submodule qcm_git_hash not found. Will not be able to record git version in output files')
    git_hash = 'NA'

try:
    from . import qcm_git_hash
    version = qcm_git_hash.version
except:
    print('version number not found. Will not be able to record version number in output files')
    version = 'NA'

np.set_printoptions(precision=6, linewidth=200, suppress=True, sign=' ')

####################################################################################################
# GLOBAL MODULE VARIABLES
solver = 'ED'

# special wavevectors
wavevector_M = (2/3)*np.array(( 1, 0, 0))
wavevector_K = (2/3)*np.array([1,1/np.sqrt(3),0])

####################################################################################################
# EXCEPTIONS

class OutOfBoundsError(Exception):
    def __init__(self, variable):
        self.variable = variable
    def __str__(self):
        if type(self.variable) == str:
            return 'variable {:s} is out of bounds'.format(self.variable)
        elif type(self.variable) == int:
            return 'variable no {:d} is out of bounds'.format(self.variable)
        else:
            return 'a variable is out of bounds'

class SolverError(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'The impurity solver failed to converge to a solution'

class TooManyIterationsError(Exception):
    pass

class VarParamMismatchError(Exception):
    pass

class MissingArgError(ValueError):
    pass

class MinimizationError(Exception):
    pass

class ParseError(Exception):
    pass

class WrongArgumentError(ValueError):
    pass

def script_file():
    return os.path.basename(sys.argv[0])

####################################################################################################
# CLASSES

####################################################################################################
class cluster_model:
    def __init__(self, n_sites, n_bath=0, name='clus', generators=None, bath_irrep=False):
        self.name = name
        self.n_sites = n_sites
        self.n_bath = n_bath
        self.generators = generators
        self.is_closed = False
        qcm.new_model(name, n_sites, n_bath, generators, bath_irrep)

    #-----------------------------------------------------------------------------------------------
    def new_operator(self, op_name, op_type, elem):

        """creates a new operator from its matrix elements
        
        :param str op_name: name of the operator
        :param str op_type: type of operator ('one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg', 'X', 'Y', 'Z')
        :param [(int,int,float)] elem: array of matrix elements (list of tuples)
        :return: None
            
        """

        if self.is_closed:
            raise ValueError('cluster_model.new_operator() cannot be called any more : the model is closed')

        if op_type == 'anomalous':
            for x in elem:  
                if x[0] >= x[1] :
                    raise ValueError(f'anomalous matrix elements of {op_name} must be such that row index < column index')

        qcm.new_operator(self.name, op_name, op_type, elem)

    #-----------------------------------------------------------------------------------------------
    def new_operator_complex(self, op_name, op_type, elem):

        """creates a new operator from its complex-valued matrix elements

        :param str op_name: name of the operator
        :param str op_type: type of operator ('one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg', 'X', 'Y', 'Z')
        :param [(int, int, complex)] elem: array of matrix elements (list of tuples)
        :return: None

        """

        global the_model

        if self.is_closed:
            raise ValueError('cluster_model.new_operator() cannot be called any more : the model is closed')

        if op_type == 'anomalous':
            for x in elem:  
                if x[0] >= x[1] :
                    raise ValueError(f'anomalous matrix elements of {op_name} must be such that row index < column index')

        qcm.new_operator_complex(self.name, op_name, op_type, elem)

    #-----------------------------------------------------------------------------------------------
    def matrix_elements(self, op):
        """
        returns the type and matrix elements defining a Hermitian operator

        :param str op: name of the operator
        :return: a tuple (typ, elem)

        """
        return qcm.matrix_elements(self.name, op)

    #-----------------------------------------------------------------------------------------------
    def new_model_instance(self, values, sec, label=0):
        """Initiates a new instance of the cluster model

        :param {str,float} values: values of the operators
        :param str sec: target Hilbert space sectors
        :param int label: label of model_instance

        :return: None

        """

        qcm.new_model_instanceC(self.name, values, sec, label)

####################################################################################################
class cluster:
    def __init__(self, X, sites, pos=(0,0,0)):
        
        if isinstance(X, cluster_model):
            self.cluster_model = X
            self.ref = None
        elif isinstance(X, cluster):
            self.ref = X
            self.cluster_model = X.cluster_model
        self.pos = pos
        self.sites = sites
        self.index = 0
        self.nsites = len(sites)

    #-----------------------------------------------------------------------------------------------
    def print_graph(self):
        """prints a graphiz (dot) program for the cluster 

        :param str name:  name of the cluster model
        :param [[int]] sites: list of positions of sites (2D array of ints)
        :return: None

        """

        qcm.print_graph(self.cluster_model.name, self.sites)

####################################################################################################
class lattice_model:

    defined = False
    is_closed = False

    def __init__(self, name, clus, superlattice, lattice=None):
        if lattice_model.defined:
            raise ValueError("Only one lattice model can be defined at a time!")
        lattice_model.defined = True
        self.name = name
        if is_sequence(clus) == False: self.clus = (clus,)
        else : self.clus = clus
        self.nclus = len(self.clus)
        self.dim = len(superlattice)
        self.nsites = 0
        for i,x in enumerate(self.clus):
            if isinstance(x, cluster) == False:
                raise ValueError("The argument 'clus' of 'model' should be of type 'cluster' or a sequence thereof")
            x.index = i+1
            self.nsites += x.nsites
            if x.ref != None: ref = x.ref.index
            else: ref = 0
            qcm.add_cluster(x.cluster_model.name, x.pos, x.sites, ref)

        qcm.lattice_model(name, superlattice, lattice)

    #-----------------------------------------------------------------------------------------------
    def hopping_operator(self, name, link, amplitude, orbitals=None, **kwargs):
        """Defines a hopping term or, more generally, a one-body operator

        :param str name: name of operator
        :param [int] link: bond vector (3-component integer array)
        :param float amplitude: hopping amplitude multiplier
        :param (int,int) orbitals: lattice orbital labels (start at 1); if None, tries all possibilities.
        
        :Keyword Arguments:

            * tau (*int*) -- specifies the tau Pauli matrix  (0,1,2,3)
            * sigma (*int*) -- specifies the sigma Pauli matrix  (0,1,2,3)
    
        :return: None

        """

        if link == ( 0, 0, 0):
            if "tau" in kwargs:
                if kwargs["tau"] != 0: kwargs["tau"] = 0
            else:
                kwargs["tau"] = 0

        orb1, orb2 = orbital_pair_manager(orbitals) 

        for orb_no1 in orb1:
            for orb_no2 in orb2:
                qcm.hopping_operator(name, link, amplitude, orb1=orb_no1, orb2=orb_no2, **kwargs)

    #-----------------------------------------------------------------------------------------------
    def anomalous_operator(self, name, link, amplitude, orbitals=None, **kwargs):
        """Defines an anomalous operator

        :param str name: name of operator
        :param [int] link: bond vector (3-component integer array)
        :param complex amplitude: pairing multiplier
        :param (int,int) orbitals: lattice orbital labels (start at 1); if None, tries all possibilities.

        :Keyword Arguments:
        
            * type (*str*) -- one of 'singlet' (default), 'dz', 'dy', 'dx'
    
        :return: None

        """
        
        orb1, orb2 = orbital_pair_manager(orbitals) 

        for orb_no1 in orb1:
            for orb_no2 in orb2:
                qcm.anomalous_operator(name, link, amplitude, orb1=orb_no1, orb2=orb_no2, **kwargs)

    #-----------------------------------------------------------------------------------------------
    def explicit_operator(self, name, elem, **kwargs):
        """
        Defines an explicit operator

        :param str name: name of operator
        :param [(list, list, complex)] elem: List of tuples. Each tuple contains three elements (in order): a list representing position, a list representing link and a complex amplitude.

        :Keyword Arguments:

            * tau (*int*) -- specifies the tau Pauli matrix  (0,1,2,3)
            * sigma (*int*) -- specifies the sigma Pauli matrix  (0,1,2,3)
            * type (*str*) -- one of 'one-body' [default], 'singlet', 'dz', 'dy', 'dx', 'Hubbard', 'Hund', 'Heisenberg', 'X', 'Y', 'Z'

        :return: None

        """
        qcm.explicit_operator(name, elem, **kwargs)

    #-----------------------------------------------------------------------------------------------
    def density_wave(self, name, t, Q, **kwargs):
        """
        Defines a density wave

        :param str name: name of operator
        :param str t: type of density-wave -- one of 'Z', 'X', 'Y', 'N'='cdw', 'singlet', 'dz', 'dy', 'dx'
        :param wavevector Q:  wavevector of the density wave (in multiple of :math:`pi`)

        :Keyword Arguments:

            * link (*[int]*) -- bond vector, for bond density waves
            * amplitude (*complex*) -- amplitude multiplier. **Caution**: A factor of 2 must be used in some situations (see :ref:`density wave theory`)
            * orb (*int*) -- orbital label (0 by default = all orbitals)
            * phase (*float*) -- real phase (as a multiple of :math:`pi`)

        :return: None

        """

        qcm.density_wave(name, t, Q, **kwargs)

    #-----------------------------------------------------------------------------------------------
    def set_basis(self, B):
        """

        :param B: the basis (a (D x 3) real matrix)
        :return: None

        """

        qcm.set_basis(B)

    #-----------------------------------------------------------------------------------------------
    def interaction_operator(self, name, link=None, orbitals=None, **kwargs):
        """
        Defines an interaction operator of type Hubbard, Hund, Heisenberg or X, Y, Z

        :param str name: name of the operator
        :param (int,int) orbitals: lattice orbital labels (start at 1); if None, tries all possibilities.
        :param list link: link of the operator (None by default)

        :Keyword Arguments:

            * amplitude (*float*): amplitude multiplier
            * type (*str*): one of 'Hubbard', 'Heisenberg', 'Hund', 'X', 'Y', 'Z'

        :return: None

        """

        orb1, orb2 = orbital_pair_manager(orbitals) 

        for orb_no1 in orb1:
            for orb_no2 in orb2:
                qcm.interaction_operator(name, orb1=orb_no1, orb2=orb_no2, link=link, **kwargs)

    #-----------------------------------------------------------------------------------------------
    def set_target_sectors(self, sec):
        """Define the Hilbert space sectors in which to look for the ground state

        :param (str) sec: the target sectors

        :return: None

        """
        if is_sequence(sec) == False: sec = (sec,)
        qcm.set_target_sectors(sec)

    #-----------------------------------------------------------------------------------------------
    def set_parameters(self, params):
        """
        Defines a new set of parameters, including dependencies

        :param tuple/str params: the values/dependence of the parameters (array of 2- or 3-tuples), or string containing syntax        
        """
        if self.is_closed: raise ValueError('WARNING : The function set_parameters() can only be called once')

        if type(params) is str:
            elems = []
            param_set = {}
            for p in re.split('[,;\n]', params):
                if len(p.strip()) == 0:
                    continue
                if p[0] == '#':
                    continue
                s = p.split('=')
                if len(s) != 2:
                    raise ParseError(p)
                s2 = s[1].split('*')
                param_name = s[0].strip()
                if param_name in param_set:
                    raise ParseError('parameter '+param_name+' has already been assigned!')
                if len(s2)==1:  # no dependence
                    elem = (param_name, float(s[1].strip()))       
                elif len(s2)==2:
                    elem = (param_name, float(s2[0].strip()), s2[1].strip())
                else:
                    raise ParseError(p)
                elems.append(tuple(elem))
            qcm.set_parameters(elems)
        else:	
            qcm.set_parameters(params)
        
        return params
    
    #-----------------------------------------------------------------------------------------------
    def set_parameter(self, name, value, pr=False):
        """
        sets the value of a parameter within a parameter_set

        :param str name: name of the parameter
        :param float value: its value
        :return: None

        """
        if pr:
            print('-----> ', name, ' = ', value)

        qcm.set_parameter(name, value)

    #-----------------------------------------------------------------------------------------------
    def parameter_string(self, lattice=True, CR=False):
        """
        Returns a string with the model parameters. 
        :param boolean lattice : if True, only indicates the independent lattice parameters.
        :param boolean CR : if True, put each parameter on a line.
        """
        par = qcm.parameter_set()
        S = ''
        sep = ', '
        if CR:
            sep = '\n'
        first = True
        for x in par:
            if par[x][1] != None:
                continue
            if '_' in x and lattice:
                continue
            if first is False:
                S += sep        
            S += x + '={:g}'.format(par[x][0])
            if first:
                first = False
        return S            

    #-----------------------------------------------------------------------------------------------
    def parameters(self):
        """
        returns the values of the parameters in the parameter set

        :return: a dict {string,float}

        """
        return qcm.parameters()

    #-----------------------------------------------------------------------------------------------
    def parameter_set(self, opt='all'):
        """
        returns the content of the parameter set

        :param str opt: governs the action of the function
        :return: depends on opt

        if opt = 'all', all parameters as a dictionary {str,(float, str, float)}. The three components are 

        (1) the value of the parameter, 
        (2) the name of its overlord (or None), 
        (3) the multiplier by which its value is obtained from that of the overlord.

        if opt = 'independent', returns only the independent parameters, as a dictionary {str,float}
        if opt = 'report', returns a string with parameter values and dependencies.

        """
        P = qcm.parameter_set()
        if opt == 'independent':
            P2 = {}
            for x in P:
                if P[x][1] is None:
                    P2[x] = P[x][0]
            return P2
        elif opt == 'report':
            qcm.print_parameter_set()
            return
        else:
            return P
            
    #-----------------------------------------------------------------------------------------------
    def print_parameter_set(self):
        """
        prints the content of the parameter set

        """
        qcm.print_parameter_set()


    #-----------------------------------------------------------------------------------------------
    def print_model(self, filename):
        """Prints a description of the model into a file

        :param str filename: name of the file
        
        :return: None
        """
        qcm.print_model(filename)

    #-----------------------------------------------------------------------------------------------
    def print_parameters(self, P):
        """
        Prints the parameters nicely on the screen
            
        :param dict P: the dictionary of parameters
        
        """
        for x in P:
            print(x, ' = ', P[x])

    #-----------------------------------------------------------------------------------------------
    def set_params_from_file(self, out_file, n=0):
        """
        reads an output file for parameters

        :param str out_file: name of output file from which parameters are read
        :param int n: line number of data in output file (excluding titles)
        :return: nothing

        """
        par = qcm.parameter_set()
        try:
            D = np.genfromtxt(out_file, names=True, dtype=None, encoding='utf8')
        except:
            raise("The file containing the solutions could not be read!")
        if len(D.shape) == 0:
            for x in par:
                if par[x][1] != None:
                    continue
                if x in D.dtype.names:
                    self.set_parameter(x,D[x],pr=True)
        else:
            for x in par:
                if par[x][1] != None:
                    continue
                if x in D.dtype.names:
                    self.set_parameter(x,D[x][n],pr=True)


    #-----------------------------------------------------------------------------------------------
    def fidelity(self, I1, I2, clus=0):
        """
        computes the fidelity between the two instances I1 and I2, i.e. the overlap squared of the ground states (or generalization thereof in case of a mixied state)

        :param model_instance I1: first model instance
        :param model_instance I2: second model instance
        :param int clus: cluster label


        """

        return qcm.fidelity(I1.label*self.nclus + clus, I2.label*self.nclus + clus)
    
    
    #-----------------------------------------------------------------------------------------------
    def finalize(self):
        """
        Sets some data for the model following the first instance declaration
        e.g. sets the mixing state of the system:

        * 0 -- normal.  GF matrix is n x n, n being the number of sites
        * 1 -- anomalous. GF matrix is 2n x 2n
        * 2 -- spin-flip.  GF matrix is 2n x 2n
        * 3 -- anomalous and spin-flip (full Nambu doubling).  GF matrix is 4n x 4n
        * 4 -- up and down spins different.  GF matrix is n x n, but computed twice, with spin_down = false and true

        """
        self.mixing = qcm.mixing()
        self.dimGF_red = qcm.reduced_Green_function_dimension()
        self.dimGF = qcm.Green_function_dimension()
        self.nmixed = self.dimGF//self.nsites
        self.dimGFC = np.zeros(self.nclus, dtype=int)
        self.nband = qcm.model_size()[1]
        for i in range(self.nclus): self.dimGFC[i] = self.clus[i].nsites*self.nmixed
        self.is_closed = True



    from ._loop import loop_from_file, linear_loop, controlled_loop, fixed_density_loop, fade, Hartree_procedure
    from ._draw import draw_operator, draw_cluster_operator

####################################################################################################
class model_instance:
    def __init__(self, model, label=0):
        self.label = label
        self.model = model
        qcm.new_model_instance(label)
        if self.model.is_closed == False: self.model.finalize()
        self.is_complex = qcm.complex_HS(label)


    #-----------------------------------------------------------------------------------------------
    def reset(self):
        """resets the model instance to the current parameters, with the same label

        """
        qcm.new_model_instance(self.label)
        self.is_complex = qcm.complex_HS(self.label)

    #-----------------------------------------------------------------------------------------------
    def susceptibility_poles(self, op_name):
        """computes the dynamic susceptibility of an operator

        :param str name: name of the operator
        :returns [(float,float)]: array of 2-tuple (pole, residue)

        """

        return qcm.susceptibility_poles(op_name, self.label)

    #-----------------------------------------------------------------------------------------------
    def susceptibility(self, op_name, freqs):
        """computes the dynamic susceptibility of an operator

            :param str op_name: name of the operator
            :param [complex] freqs: array of complex frequencies
            :param int label: label of cluster model instance
            :return: array of complex susceptibilities

        """

        return qcm.susceptibility(op_name, freqs, self.label)

    #-----------------------------------------------------------------------------------------------
    def qmatrix(self):
        """Returns the Lehmann representation of the Green function

            :return: 2-tuple made of
                1. the array of M real eigenvalues, M being the number of poles in the representation
                2. a rectangular (L x M) matrix (real of complex), L being the dimension of the Green function

        """	

        return qcm.qmatrix(self.label)

    #-----------------------------------------------------------------------------------------------
    def write(self, filename):
        """Writes the solved cluster model instance to a text file
        
        :param str filename: name of the file
        :return: None

        """

        qcm.write_instance_to_file(filename, self.label)

    #-----------------------------------------------------------------------------------------------
    def parameters(self):
        """
        returns the values of the parameters of the instance

        :return: a dict {string,float}

        """
        return qcm.instance_parameters(self.label)

    #-----------------------------------------------------------------------------------------------
    def averages(self, ops=[], file='averages.tsv', pr=False):
        """
        Computes the lattice averages of the operators present in the model

        :param str file: name of the file in which the information is appended
        :return {str,float}: a dict giving the values of the averages for each parameter

        """
        ave = qcm.averages(ops, self.label)
        self.write_summary(file)

        if pr:
            for x in ave:
                print('<{:s}> = {:f}'.format(x, ave[x]))
            
        return ave

    #-----------------------------------------------------------------------------------------------
    def cluster_Green_function(self, z, clus=0, spin_down=False, blocks=False):
        """Computes the cluster Green function

        :param int clus: label of the cluster (0 to the number of clusters-1)
        :param complex z: frequency
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 	4)
        :param boolean blocks: true if returned in the basis of irreducible representations
        :return: a complex-valued matrix

        """

        return qcm.cluster_Green_function(clus, z, spin_down, self.label*self.model.nclus + clus, blocks)


    #-----------------------------------------------------------------------------------------------
    def cluster_hopping_matrix(self, clus=0, spin_down=False, full=0):
        """
        returns the one-body matrix of cluster no i

        :param clus: label of the cluster (0 to the number of clusters - 1)
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :param boolean full: if True, returns the full hopping matrix, including bath
        :return: a complex-valued matrix
        """
        if full:
            return qcm.hopping_matrix(spin_down, clus, True)
        else:
            return qcm.cluster_hopping_matrix(clus, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def read_cluster_model_instance(self, S, clus=0):
        """reads the solution from a string 

        :param clus: label of the cluster (0 to the number of clusters - 1)
        :param str S: long string containing the solution 

        :return: None

        """
        qcm.read_instance(S, self.label*self.model.nclus + clus)

    #-----------------------------------------------------------------------------------------------
    def cluster_self_energy(self, z, clus=0, spin_down=False):
        """Computes the cluster self-energy

        :param int clus: label of the cluster (0 to the number of clusters -1)
        :param complex z: frequency
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 	4)
        :param int label:  label of the model instance (default 0)
        :return: a complex-valued matrix

        """

        return qcm.cluster_self_energy(clus, z, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def Green_function_average(self, clus=0, spin_down=False):
        """Computes the cluster Green function average (integral over frequencies)

        :param int clus: label of the cluster (0 to the number of clusters-1)
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 	4)
        :return: a complex-valued matrix

        """
        return qcm.Green_function_average(self.label*self.model.nclus + clus, spin_down)

    #-----------------------------------------------------------------------------------------------
    def interactions(self, clus=0):
        """
        returns the density-density interactions in cluster no i for instance 'label'

        :param clus: label of the cluster (0 to the number of clusters - 1)
        :return: a list of matrix elements tuples (i,j,v)
        """
        return qcm.interactions(self.label*self.model.nclus + clus)


    #-----------------------------------------------------------------------------------------------
    def CPT_Green_function(self, z, k, spin_down=False):
        """
        computes the CPT Green function at a given frequency

        :param z: complex frequency
        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :return: a single or an array of complex-valued matrices
        """
        return qcm.CPT_Green_function(z, k, spin_down, self.label)


    #-----------------------------------------------------------------------------------------------
    def dispersion(self, k, spin_down=False, label=0):
        """
        computes the dispersion relation for a single or an array of wavevectors

        :param wavevector k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :return: a single (ndarray(d)) or an array (ndarray(N,d)) of real values (energies). d is the reduced GF dimension.

        """
        return qcm.dispersion(k, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def dos(self, z):
        """
        computes the density of states at a given frequency.

        :param complex z: frequency
        :return: ndarray(d) of real values, d being the reduced GF dimension

        """
        return qcm.dos(z, self.label)
    
    #-----------------------------------------------------------------------------------------------
    def Green_function_solve(self):
        """
        Usually, the Green function representation is computed only when needed, in a just-in-time fashion (i.e. in a lazy way).
        This forces the computation of the Green function representation for the current instance (i.e. non lazy). 

        :return: None

        """
        qcm.Green_function_solve(self.label)


    #-----------------------------------------------------------------------------------------------
    def ground_state(self, file=None, pr=False):
        """
        :return: a list of pairs (float, str) of the ground state energy and sector string, for each cluster of the system

        """

        GS = qcm.ground_state(self.label)
        if file is not None:
            self.write_summary(file) 
        if pr:
            for x in GS:
                print('E0 = {:f}\tsector =  {:s}'.format(x[0], x[1]))
        return GS


    #-----------------------------------------------------------------------------------------------
    def cluster_averages(self, clus=0, pr=False):
        """
        Computes the average and variance of all operators of the cluster model in the cluster ground state.

        :param int clus: label of the cluster
        :return: a dict str : (float, float) with the averages and variances as a function of operator name

        """
        ave = qcm.cluster_averages(self.label*self.model.nclus+clus)
        s = '@' + str(clus+1)
        
        res = [key for key, val in ave.items() if s in key]
        for x in res:
            ave[x[0:-len(s)]] = ave[x]

        if pr:
            for x in ave:
                print('<{:s}> = {:f}\tvar({:s}) = {:f}'.format(x, ave[x][0], x, ave[x][1]))

        return ave

    #-----------------------------------------------------------------------------------------------
    def Lehmann_Green_function(self, k, orb=1, spin_down=False):
        """
        computes the Lehmann representation of the periodized Green function for a set of wavevectors

        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param int orb: orbital index (starts at 1)
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :return: a list of pairs {poles, residues}, each of poles and residues being itself a list.

        """
        return qcm.Lehmann_Green_function(k, orb, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def hybridization_function(self, z, clus=0, spin_down=False):
        """
        returns the hybridization function for cluster 'cluster' and instance 'label'

        :param int clus: label of the cluster (0 to the number of clusters-1)
        :param complex z: frequency
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :return: a complex-valued matrix

        """
        return qcm.hybridization_function(z, spin_down, clus, self.label)

    #-----------------------------------------------------------------------------------------------
    def momentum_profile(self, name, k):
        """
        computes the momentum-resolved average of an operator

        :param str name: name of the lattice operator
        :param k: array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :return: an array of values

        """

        return qcm.momentum_profile(name, k, self.label)


    #-----------------------------------------------------------------------------------------------
    def instance_parameters(self):
        """
        returns the values of the parameters in a given instance

        :return: a dict {string,float}

        """
        return qcm.instance_parameters(self.label)



    #-----------------------------------------------------------------------------------------------
    def periodized_Green_function(self, z, k, spin_down=False):
        """
        computes the periodized Green function at a given frequency and wavevectors

        :param complex z: frequency
        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
        :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

        """
        return qcm.periodized_Green_function(z, k, spin_down, self.label)


    #-----------------------------------------------------------------------------------------------
    def periodized_Green_function_element(self, r, c, z, k, spin_down=False):
        """
        computes the element (r,c) of the periodized Green function at a given frequency and wavevectors (starts at 0)

        :param int r: a row index (starts at 0)
        :param int c: a column index (starts at 0)
        :param complex z: frequency
        :param k: array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
        :param int label:  label of the model instance
        :return: a vector of complex numbers

        """
        return qcm.periodized_Green_function_element(r, c, z, k, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def band_Green_function(self, z, k, spin_down=False):
        """
        computes the periodized Green function at a given frequency and wavevectors, in the band basis (defined
        in the noninteracting model). It only differs from the periodized Green function in multi-band models.

        :param complex z: frequency
        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
        :param int label:  label of the model instance
        :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

        """
        return qcm.band_Green_function(z, k, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def self_energy(self, z, k, spin_down=False):
        """
        computes the self-energy associated with the periodized Green function at a given frequency and wavevectors

        :param complex z: frequency
        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

        """
        return qcm.self_energy(z, k, spin_down, self.label)


    #-----------------------------------------------------------------------------------------------
    def potential_energy(self):
        """
        computes the potential energy for a given instance, as the functional trace of Sigma x G

        :param int label: label of the model instance
        :return: the value of the potential energy

        """
        return qcm.potential_energy(self.label)

    #-----------------------------------------------------------------------------------------------
    def spectral_average(self, name, z):
        """
        returns the contribution of a frequency to the average of an operator

        :param name: name of the operator
        :param z: complex frequency
        :param int label:  label of the model instance
        :return: float

        """
        return qcm.spectral_average(name, z, self.label)
    

    #-----------------------------------------------------------------------------------------------
    def Green_function_density(self, clus=0):
        """
        Computes the density from the Green function average

        :param int clus: label of the cluster (0 to the number of clusters-1)
        return (float) : the density
        """
        return qcm.Green_function_density(self.label*self.model.nclus+clus)

    #-----------------------------------------------------------------------------------------------
    def density_matrix(self, sites, clus=0):
        """Computes the density matrix of subsystem A, defined by the array of site indices "sites"

        :param int clus: label of the cluster (0 to the number of clusters-1)
        :param [int] sites: list of sites defining subsystem A
        :return [complex], [int32], [int32]: the density matrix, the left and right bases (spins up and down)

        """
        rho, basis = qcm.density_matrix(sites, self.label*self.model.nclus+clus)

        L = len(sites)
        rightmask = np.left_shift(1,L)-1
        leftmask = np.left_shift(rightmask,32)
        basis = np.int64(basis)
        basisL = np.bitwise_and(basis,leftmask)
        basisL = np.right_shift(basis,32-L)
        basisL = np.uint(basisL)
        basisR = np.bitwise_and(basis,rightmask)
        basisR = np.uint(basisR)
        basis = np.bitwise_or(basisL,basisR)

        return rho, basis

    #-----------------------------------------------------------------------------------------------
    def Potthoff_functional(self, hartree=None, file='sef.tsv', symmetrized_operator=None):
        """
        computes the Potthoff functional for a given instance

        :param int label: label of the model instance
        :param str file: name of the file to append with the result
        :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
        :param str symmetrized_operator: name of an operator wrt which the functional must be symmetrized
        :return: the value of the self-energy functional

        """
        OM = qcm.Potthoff_functional(self.label)

        if symmetrized_operator is not None:
            try:
                P = self.parameters()
                x = P[symmetrized_operator]
                self.model.set_parameter(symmetrized_operator,-x)
                I = model_instance(label=1)
                OMsym = qcm.Potthoff_functional(I.label)
                I = model_instance(label=1) # effectively clears model instance 1
                OM = 0.5*(OM + OMsym)
                self.model.set_parameter(symmetrized_operator, x)
            except:
                pass
                
        if hartree != None:
            L = qcm.model_size()[0]
            for C in hartree:
                OM += C.omega_var()/L

        self.write_summary(file, suppl_descr='omegaH\t', suppl_values='{:.8g}\t'.format(OM))

        return OM


    #-----------------------------------------------------------------------------------------------
    def properties(self):
        """
        Returns two strings of properties of a model instance
        
        :param int label:  label of the model instance
        :return: a pair of strings (the description line and the data line).

        """
        des, data = qcm.properties(self.label)
        des += 'githash\tversion\t'
        data += git_hash + '\t' + version + '\t'
        return des, data

    #-----------------------------------------------------------------------------------------------
    def site_and_bond_profile(self):
        """
        Computes the site and bond profiles in all clusters of the repeated unit

        :return: A pair of ndarrays

        site profile -- the components are 
        x y z n Sx Sy Sz psi.real psi.imag

        bond profile -- the components are  
        x1 y1 z1 x2 y2 z2 b0 bx by bz d0.real dx.real dy.real dz.real d0.imag dx.imag dy.imag dz.imag

        """
        return qcm.site_and_bond_profile(self.label)


    #-----------------------------------------------------------------------------------------------
    def print_wavefunction(self, clus=0, pr=True):
        """prints the ground state wavefunction(s) on the screen

        :param int clus: label of the cluster (0 to the number of clusters-1)
        :param bool pr: prints wavefunction to screen if pr=True

        :return str: the wavefunction

        """
        wavefunction = qcm.print_wavefunction(self.label*self.model.nclus+clus)

        if pr:
            print(wavefunction)

        return wavefunction


    #-----------------------------------------------------------------------------------------------
    def QP_weight(self, k, eta=0.01, orb=1, spin_down=False):
        """
        computes the k-dependent quasi-particle weight from the self-energy derived from the periodized Green function

        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param float eta: increment in the imaginary axis direction used to computed the derivative of the self-energy
        :param int orb: orbital index (starts at 1)
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :return: a single float or an array of floats, depending on the shape of k

        """

        if np.shape(k) == (3,):
            k = np.array([k]) # this protects the case where k is an ndarray(3)

        sigma1 = qcm.self_energy(-eta*1j, k, spin_down, self.label)
        sigma2 = qcm.self_energy(eta*1j, k, spin_down, self.label)
        if len(sigma1.shape) == 3:
            Z = (sigma1[:,orb-1,orb-1].imag - sigma2[:,orb-1,orb-1].imag)/(2*eta) + np.ones(len(k))
        else:
            Z = (sigma1[orb-1,orb-1].imag - sigma2[orb-1,orb-1].imag)/(2*eta) + 1.0
        Z = 1.0/Z
        return Z

    #-----------------------------------------------------------------------------------------------
    def projected_Green_function(self, z, spin_down=False):
        """
        computes the projected Green function at a given frequency, as used in CDMFT.

        :param complex z: frequency
        :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
        :param int label:  label of the model instance
        :return: the projected Green function matrix (d x d), d being the dimension of the CPT Green function.

        """
        return qcm.projected_Green_function(z, spin_down, self.label)

    #-----------------------------------------------------------------------------------------------
    def V_matrix(self, z, k, spin_down=False):
        """
        Computes the matrix :math:`V=G_0^{-1}-G^{c-1}_0` at a given frequency and wavevectors, where :math:`G_0` is the noninteracting Green function on the infinite lattice and :math:`G^c_0` is the noninteracting Green function on the cluster.

        :param complex z: frequency
        :param wavevector k: wavevector (ndarray(3)) in units of :math:`\pi`
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :param int label:  label of the model instance
        :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

        """
        return qcm.V_matrix(z, k, spin_down, self.label)


    #-----------------------------------------------------------------------------------------------
    def tk(self, k, spin_down=False):
        """
        computes the k-dependent one-body matrix of the lattice model

        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :param int label:  label of the model instance
        :return: a single or an array of complex-valued matrices
        """
        return qcm.tk(k, spin_down, self.label)


    #-----------------------------------------------------------------------------------------------
    def cluster_QP_weight(self, clus=0, eta=0.01, orb=1, spin_down=False):
        """
        computes the cluster quasi-particle weight from the cluster self-energy

        :param int clus: cluster label (starts at 0)
        :param float eta: increment in the imaginary axis direction used to computed the derivative of the self-energy
        :param int orb: orbital index (starts at 1)
        :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
        :param int label:  label of the model instance
        :return: a float

        """
        sigma1 = self.cluster_self_energy(cluster, -eta*1j, spin_down, self.label*self.model.nclus+clus)
        sigma2 = self.cluster_self_energy(cluster, eta*1j, spin_down, self.label*self.model.nclus+clus)
        Z = (sigma1[orb-1,orb-1].imag - sigma2[orb-1,orb-1].imag)/(2*eta) + 1.0
        Z = 1.0/Z
        return Z
        

    #-----------------------------------------------------------------------------------------------
    def spin_spectral_function(self, freq, k, orb=None):
        """
        computes the k-dependent spin-resolved spectral function

        :param freq: complex freqency
        :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
        :param int orb: if None, sums all the orbitals. Otherwise just shows the weight for that orbital (starts at 1)
        :param int label:  label of the model instance
        :return: depending on the shape of k, a nd.array(3) of nd.array(N,3)

        """
        orbs = orbital_manager(orb, from_zero=True)

        
        ds = qcm.reduced_Green_function_dimension()
        if self.model.mixing != 2 and self.model.mixing != 3:
            raise RuntimeError('The function "spin_spectral_function()" makes sense only if spin-flip terms are present')
        if self.model.mixing == 2:
            ds //= 2
        elif self.model.mixing == 3:
            ds //= 4
        
        G = self.periodized_Green_function(freq, k, False)

        if len(G.shape) == 3:
            nk = G.shape[0]
            S = np.zeros((nk,4))
            for l in orbs:
                s1 = G[:, l, l+ds]
                s2 = G[:, l+ds, l]
                S[:,1] += -(s1+s2).imag
                S[:,2] +=  (s1-s2).real
                S[:,3] += -(G[:, l, l] - G[:, l+ds, l+ds]).imag
                S[:,0] += -(G[:, l, l] + G[:, l+ds, l+ds]).imag
        else:
            S = np.zeros(4)
            for l in orbs:
                s1 = G[l, l+ds]
                s2 = G[l+ds, l]
                S[1] += -(s1+s2).imag
                S[2] +=  (s1-s2).real
                S[3] += -(G[l, l] - G[l+ds, l+ds]).imag
                S[0] += -(G[l, l] + G[l+ds, l+ds]).imag

        return 0.5*S

    #-----------------------------------------------------------------------------------------------
    def write_summary(self, f, suppl_descr=None, suppl_values=None, first_of_series=False):
        first_in_file =False
        des, val = self.properties()
        try:
            des_prev = des_dict[f]
        except:
            des_dict[f] = ''
            des_prev = ''
            first_in_file =True
        if des == des_prev and first_of_series == False: 
            first = False
        else: 
            first = True
            des_dict[f] = copy.copy(des)
        if suppl_values != None:
            val = val + suppl_values
        val += script_file() + '\t' + time.strftime("%Y-%m-%d@%H:%M", time.localtime()) # adds the timestamp
        fout = open(f, 'a')
        if first:
            if suppl_descr != None:
                des = des + suppl_descr
            des += 'script\ttime\t'
            if first_in_file is False: fout.write('\n')
            fout.write(des + '\n')
        fout.write(val  + '\n')
        fout.close()

    #-----------------------------------------------------------------------------------------------
    def double_counting_correct(self, DC):
        """Modifies some kinetic parameters in view of the presence of interactions and averages values,
        in order to account minimally for double counting.

        :param [(str,str,str,float,float)] DC: list of recipes for the correction: (kinetic operator, interaction operator, density operator, coefficient, value of the kinetic operator without interaction)

        """
        
        ave = self.averages()
        P = self.parameters()
        if type(DC) is tuple: DC = [DC]
        corr = {}
        for x in DC:
            if len(x) != 5:
                raise ValueError(f'in double_counting_correct(), tuples should have 5 elements')
            if x[0] in corr:
                corr[x[0]] += P[x[1]]*ave[x[2]]*x[3]
            else:
                corr[x[0]] = x[4] + P[x[1]]*ave[x[2]]*x[3]

        for x in corr:
            self.model.set_parameter(x, corr[x], pr=True)
    
    #-----------------------------------------------------------------------------------------------
    # methods from _spectral.py
    
    from ._spectral import spectral_function, plot_hybridization_function, cluster_spectral_function, spectral_function_Lehmann, gap, plot_DoS, mdc, spin_mdc, mdc_anomalous, plot_dispersion, segment_dispersion, Fermi_surface, G_dispersion, Luttinger_surface, plot_momentum_profile, plot_host_hybrid, Berry_field_map, Berry_flux_map, monopole_map, Berry_flux, monopole, Chern_number, Berry_curvature, plot_profile

####################################################################################################
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

    #-----------------------------------------------------------------------------------------------
    def __init__(self, model, Vm, V, eig, accur=1e-4, lattice=False):
        """

        :param lattice_model model: the lattice model
        :param str Vm: name of the mean-field operator
        :param str V: name of the interaction operator
        :param float eig: eigenvalue
        :param float accur: required accuracy of the self-consistent procedure
        :param boolean lattice: if True, the lattice average is used, otherwise the cluster average

        """

        self.model = model
        self.Vm = Vm
        self.V = V
        self.eig = eig
        self.lattice = lattice
        self.diff = 1e6
        self.diff_rel = 1e6
        self.ave = 0
        self.accur = accur
        self.epsilon = False
        self.iter = 0

        self.L = self.model.nsites
        if lattice == False:
            assert model.nclus == 1, 'A Hartree coupling with lattice=False needs a single-cluster model'
    

    #-----------------------------------------------------------------------------------------------
    def update(self, I, pr=False):
        """Updates the value of the mean-field operator based on its average
        
        :param model_instance I: instance of the lattice model
        :param boolean pr: if True, progress is printed on the screen

        """

        par = I.parameters()
        v = par[self.V]
        vm0 = par[self.Vm]
        if not self.lattice:
            A = I.cluster_averages()
            self.ave = A[self.Vm][0]*self.L
        else:
            self.ave = I.averages()[self.Vm]*self.L
        self.vm = self.eig*v*self.ave
        self.model.set_parameter(self.Vm, self.vm)
        self.diff = self.vm-vm0
        self.diff_rel = np.abs(self.diff)/(np.abs(self.vm)+1e-6)
        meta_pr = ''
        if self.epsilon:
            eps_length = 2*self.epsilon + 1
            self.data[self.iter] = self.vm
            if self.iter >= 2*eps_length and self.iter%(2*eps_length) == 0:
                self.vm = epsilon(self.data[self.iter-eps_length:self.iter])
                meta_pr = ' (epsilon algo)'
        self.iter += 1
        if pr:
            print('delta {:s} = {:1.3g} ( {:1.3g}%)'.format(self.Vm, self.diff, 100*self.diff_rel), meta_pr)

    #-----------------------------------------------------------------------------------------------
    def omega(self, I):
        """returns the constant contribution, added to the Potthoff functional
        
        :param I model_instance: the current model_instance
        """

        par = self.model.parameters()
        v = par[self.V]
        vm0 = par[self.Vm]
        if not self.lattice:
            self.ave = I.cluster_averages()[self.Vm][0]
        else:
            self.ave = I.averages()[self.Vm]
        return -0.5*self.eig*v*self.ave*self.ave

    #-----------------------------------------------------------------------------------------------
    def omega_var(self):
        """returns the constant contribution, added to the Potthoff functional
        
        """

        par = self.model.parameters()
        v = par[self.V]
        vm0 = par[self.Vm]
        return -0.5*vm0*vm0/(self.eig*v)

    #-----------------------------------------------------------------------------------------------
    def converged(self):
        """Tests whether the mean-field procedure has converged

        :return boolean: True if the mean-field procedure has converged
        
        """

        if np.abs(self.diff) < self.accur:
            return True
        else:
            return False


    #-----------------------------------------------------------------------------------------------
    def __str__(self):
        return 'extended interaction '+self.V+', mean-field operator '+self.Vm+', coupling = {:f}'.format(self.eig)


    #-----------------------------------------------------------------------------------------------
    def print(self):
        print('<{:s}> = {:g}\t{:s} = {:g} (diff = {:g}, {:g}%)'.format(self.Vm, self.ave, self.Vm, self.vm, self.diff, 100*self.diff_rel))

    #-----------------------------------------------------------------------------------------------
    def init_epsilon(self, n, eps_length):
        self.data = np.empty(n+1)
        self.epsilon = eps_length
        self.iter = 0

####################################################################################################
# OTHER FUNCTIONS

#---------------------------------------------------------------------------------------------------
def cluster_info():
    """
    :return:A list of 4-tuples: (str, int, int, int, int): name of the cluster model, number of physical sites, number of bath sites, dimension of the Green function, number of point-group symmetry operations
    """
    return qcm.cluster_info()
    
####################################################################################################
# FUNCTIONS RELATION TO PARAMETERS

#---------------------------------------------------------------------------------------------------
def set_global_parameter(name, value=None):
    """
    sets the value of a global parameter. 
    If the global parameter is Boolean, then value is True by default and should not be specified.

    :param str name: name of the global option
    :param value: value of that option (None, int, float or str)
    :return: None

    """
    if value is None:
        return qcm.set_global_parameter(name)
    else:
        return qcm.set_global_parameter(name, value)

#---------------------------------------------------------------------------------------------------
def get_global_parameter(name, value=None):
    """
    gets the value of a global parameter. 

    :param str name: name of the global option
    :return: the value, according to type

    """
    return qcm.get_global_parameter(name)

####################################################################################################
# FUNCTIONS RELATED TO SETS OF WAVEVECTORS for PLOTTING

#---------------------------------------------------------------------------------------------------
def __wavevector_line(k1, k2, n=32):
    """
    Builds a wavevector path and associated tick marks for a straight path between k1 and k2 with n points
    
    :param (float) k1 : starting wavevector
    :param (float) k2 : ending wavevector
    :param int n: number of wavevectors per segment
    :returns tuple: 1) a ndarray of wavevectors 2) a list of tick positions 3) a list of tick strings

    """
    ticks = np.array([0, n + 1])
    tick_labels = [str(k1), str(k2)]
    k1 = np.array(k1)
    k2 = np.array(k2)
    k = np.zeros((n + 1, 3))
    n1 = 1.0/n
    for i in range(n + 1):
        k[i,:] = n1*((n-i)*k1 + i*k2)
    return 0.5 * k, ticks, tick_labels
    

#---------------------------------------------------------------------------------------------------
def wavevector_path(n=32, shape='triangle'):
    """
    Builds a wavevector path and associated tick marks
    
    :param int n: number of wavevectors per segment
    :param str shape: the geometry of the path, one of: line, halfline, triangle, graphene, graphene2, diagonal, cubic, cubic2, tetragonal, tetragonal2  OR a tuple with two wavevectors for a straight path between the two OR a filename ending with ".tsv". In the latter case, the file contains a tab-separated list of wavevectors (in units of pi) and tick marks: the first three columns are the x,y,z components of the wavevectors, and the last columns the strings (possibly latex) for the tick marks (write - in that column if you do not want a tick mark for a specific wavevector).
    :returns tuple: 1) a ndarray of wavevectors 2) a list of tick positions 3) a list of tick strings

    """
    if type(shape) is tuple:
        return __wavevector_line(shape[0], shape[1], n=32)

    elif (shape == 'triangle'):
        k = np.zeros((3 * n + 1, 3))
        for i in range(n):
            k[i, 0] = i / n
        for i in range(n):
            k[i + n, 0] = 1.0
            k[i + n, 1] = i / n
        for i in range(n):
            k[i + 2 * n, 0] = 1.0 - i / n
            k[i + 2 * n, 1] = 1.0 - i / n
        k[-1] = k[0]
        ticks = np.array([0, n, 2 * n, 3 * n + 1])
        tick_labels = [r'$(0,0)$', r'$(\pi,0)$', r'$(\pi,\pi)$', r'$(0,0)$']
    elif shape == 'line':
        k = np.zeros((2 * n + 1, 3))
        for i in range(2 * n + 1):
            k[i, 0] = (i - n) / n
        ticks = np.array([0, n, 2 * n + 1])
        tick_labels = [r'$-\pi$', r'$0$', r'$\pi$']
    elif shape == 'diagonal':
        k = np.zeros((n + 1, 3))
        for i in range(n + 1):
            k[i, 0] = i / n
            k[i, 1] = i / n
        ticks = np.array([0, n / 2, n + 1])
        tick_labels = [r'$\Gamma$', r'$(\pi/2,\pi/2)$', r'$M$']
    elif shape == 'halfline':
        k = np.zeros((n + 1, 3))
        for i in range(n + 1):
            k[i, 0] = i / n
        ticks = np.array([0, n / 2, n + 1])
        tick_labels = [r'$0$', r'$\pi/2$', r'$\pi$']
    elif shape == 'graphene':  # honeycomb lattice (2D) gamma-M-K'-gamma
        k = np.zeros((5 * n // 2 + 1, 3))
        for i in range(n):
            k[i, 0] = i * 0.6666667 / n
        for i in range(1 + n // 2):
            k[n + i, 0] = 0.66666667
            k[n + i, 1] = -4 * i * 0.19245 / n
        for i in range(n):
            k[-i - 1, 0] = i * 0.6666667 / n
            k[-i - 1, 1] = -2 * i * 0.19245 / n
        ticks = np.array([0, n, 3 * n // 2, 5 * n // 2])
        tick_labels = [r'$\Gamma$', r'$M$', r'$K^\prime$', r'$\Gamma$']
    elif shape == 'graphene2':  # honeycomb lattice (2D) M-gamma-K-K'
        k = np.zeros((3*n + 1, 3))
        for i in range(n):
            k[i, 0] = (n-i) * 0.6666667 / n
            k[i, 1] = -2 * (n-i) * 0.19245 / n
        for i in range(n):
            k[i+n, 0] = i * 0.6666667 / n
        for i in range(n+1):
            k[2*n+i, 0] = 0.66666667
            k[2*n+i, 1] = -2 * i * 0.19245 / n
        ticks = np.array([0, n, 2*n, 3*n])
        tick_labels = [r'$K$', r'$\Gamma$', r'$M$', r'$K^\prime$']
    elif shape == 'tri':  # triangular lattice (2D) gamma-M-K'-gamma
        k = np.zeros((5 * n // 2 + 1, 3))
        sq3 = np.sqrt(3.0)
        for i in range(n):
            k[i,1] = i * 0.6666667*sq3 / n
        for i in range(1 + n // 2):
            k[n + i, 1] = 0.66666667*sq3
            k[n + i, 0] = -4 * i * 0.19245*sq3 / n
        for i in range(n):
            k[-i - 1, 1] = i * 0.6666667*sq3 / n
            k[-i - 1, 0] = -2 * i * 0.19245*sq3 / n
        ticks = np.array([0, n, 3 * n // 2, 5 * n // 2])
        tick_labels = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
    elif shape == 'cubic':  # cubic lattice (100)-(000)-(111)-(110)-(000)
        k = np.zeros((4*n+1, 3))
        for i in range(n):
            k[i, 0] = 1.0 - i / n
        for i in range(n):
            k[i + n, 0] = i / n
            k[i + n, 1] = i / n
            k[i + n, 2] = i / n
        for i in range(n):
            k[i + 2*n, 0] = 1
            k[i + 2*n, 1] = 1
            k[i + 2*n, 2] = 1.0 - i / n
        for i in range(n):
            k[i + 3*n, 0] = 1.0 - i / n
            k[i + 3*n, 1] = 1.0 - i / n
        ticks = np.array([0, n, 2 * n, 3 * n, 4 * n + 1])
        tick_labels = [r'$(\pi,0,0)$', r'$(0,0,0)$', r'$(\pi,\pi,\pi)$', r'$(\pi,\pi,0)$', r'$(0,0,0)$']
    elif shape == 'tetragonal':  # tetragonal lattice (000)-(200)-(101/2)-(111/2)-(110)
        k = np.zeros((4*n+1, 3))
        for i in range(n):
            k[i, 0] = 2 * i / n
            k[i, 1] = 0
            k[i, 2] = 0
        for i in range(n):
            k[i + n, 0] = 2 - i / (n)
            k[i + n, 1] = 0
            k[i + n, 2] = i / (2 * n)
        for i in range(n):
            k[i + 2*n, 0] = 1 
            k[i + 2*n, 1] = i / n
            k[i + 2*n, 2] = 1 / 2
        for i in range(n):
            k[i + 3*n, 0] = 1 
            k[i + 3*n, 1] = 1 
            k[i + 3*n, 2] = 1 / 2 - i / (2 * n)
        k[-1,0] = 1
        k[-1,1] = 1
        k[-1,2] = 0 
        ticks = np.array([0, n, 2 * n, 3 * n, 4 * n + 1])
        tick_labels = [r'$(0,0,0)$', r'$(2\pi,0,0)$', r'$(\pi,0,\pi/2)$', r'$(\pi,\pi,\pi/2)$', r'$(\pi,\pi,0)$']
    elif shape == 'tetragonal2':  # tetragonal lattice (000)-(100)-(1/201/4)-(1/21/21/4)-(1/21/20)
        k = np.zeros((4*n+1, 3))
        for i in range(n):
            k[i, 0] = i / n
            k[i, 1] = 0
            k[i, 2] = 0
        for i in range(n):
            k[i + n, 0] = 1 - i / (2 * n)
            k[i + n, 1] = 0
            k[i + n, 2] = i / (4 * n)
        for i in range(n):
            k[i + 2*n, 0] = 1 / 2
            k[i + 2*n, 1] = i / (2 * n)
            k[i + 2*n, 2] = 1 / 4
        for i in range(n):
            k[i + 3*n, 0] = 1 / 2
            k[i + 3*n, 1] = 1 / 2
            k[i + 3*n, 2] = 1 / 4 - i / (4 * n)
        ticks = np.array([0, n, 2 * n, 3 * n, 4 * n + 1])
        tick_labels = [r'$(0,0,0)$', r'$(\pi,0,0)$', r'$(\pi/2,0,\pi/4)$', r'$(\pi/2,\pi/2,\pi/4)$', r'$(\pi/2,\pi/2,0)$']
    elif shape == 'cubic2':  # cubic lattice (000)-(010)-(110)-(000)-(111)-(010)-(000)
        k = np.zeros((6*n+1, 3))
        for i in range(n):
            k[i, 1] = i / n
        for i in range(n):
            k[i + n, 0] = i / n
            k[i + n, 1] = 1.0
            k[i + n, 2] = 0
        for i in range(n):
            k[i + 2*n, 0] = 1.0 - i / n
            k[i + 2*n, 1] = 1.0 - i / n
            k[i + 2*n, 2] = 0
        for i in range(n):
            k[i + 3*n, 0] = i / n
            k[i + 3*n, 1] = i / n
            k[i + 3*n, 2] = i / n
        for i in range(n):
            k[i + 4*n, 0] = 1.0 - i / n
            k[i + 4*n, 1] = 1.0
            k[i + 4*n, 2] = 1.0 - i / n
        for i in range(n):
            k[i + 5*n, 0] = 0
            k[i + 5*n, 1] = 1.0 - i / n
            k[i + 5*n, 2] = 0
        k[-1,0] = 0
        k[-1,1] = 0
        k[-1,2] = 0 
        ticks = np.array([0, n, 2 * n, 3 * n, 4 * n, 5 * n, 6 * n + 1])
        tick_labels = [r'$(0,0,0)$', r'$(0,\pi,0)$', r'$(\pi,\pi,0)$', r'$(0,0,0)$', r'$(\pi,\pi,\pi)$' , r'$(0,\pi,0)$', r'$(0,0,0)$']

    else:
        if '.tsv' in shape:
            try:
                k = np.genfromtxt(shape, usecols=(0,1,2))
                T = np.genfromtxt(shape, usecols=(3), dtype='str')
            except:
                raise ValueError('cannot read file '+shape+' properly')
            ticks = []
            tick_labels = []
            for i in range(len(T)):
                if T[i] != '-':
                    ticks.append(i)
                    tick_labels.append(T[i])
            ticks = np.array(ticks)
        else:
            raise ValueError('wavevector path shape '+shape+' unknown')

    return 0.5 * k, ticks, tick_labels


#---------------------------------------------------------------------------------------------------
# produces a set of wavevectors for an mdc
def wavevector_grid(n=100, orig=[-1.0, -1.0], side=2, k_perp = 0, plane='z'):
    """Produces a set of wavevectors for a MDC

    :param int n: number of wavevectors on the side
    :param [float] orig: origin (in multiples of pi)
    :param float side: length of the side (in multiples of pi)
    :param float k_perp: momentum component in the third direction (in multiples of pi)
    :param str plane: momentum plane, 'xy'='z', 'yz'='x'='zy' or 'xz'='zx'='y'
    :return: ndarray of wavevectors (n*n x 3)
    
    """

    c = np.array([0,1,2])
    if plane in ['y', 'xz', 'zx']:
        c[0] = 2
        c[1] = 0
        c[2] = 1
    elif plane in ['x', 'yz', 'zy']:
        c[0] = 1
        c[1] = 2
        c[2] = 0
    orig0 = 0.5*orig[0]
    orig1 = 0.5*orig[1]
    sidep = 0.5*side
    k = np.zeros((n * n, 3))
    step = 1.0 * sidep / (n-1)
    for i in range(n):
        for j in range(n):
            k[i + n * j, c[0]] = orig0 + i * step
            k[i + n * j, c[1]] = orig1 + j * step
            k[i + n * j, c[2]] = 0.5*k_perp
    return k


####################################################################################################
# GENERIC FUNCTIONS

#---------------------------------------------------------------------------------------------------
def orbital_manager(orbitals, from_zero=False):
    """
    """

    if orbitals is None:
        nbands = qcm.model_size()[1]
        orb_list = [i for i in range(1, nbands+1)]

    elif type(orbitals) is int:
        orb_list = [orbitals]
    elif type(orbitals) is list:
        orb_list = orbitals
    elif type(orbitals) is tuple:
        orb_list = orbitals
    else:
        raise ValueError('The list of orbitals does not have the right format (None, int, tuple or list)')
    if from_zero:
        orb_list = [x-1 for x in orb_list]
    return orb_list

#---------------------------------------------------------------------------------------------------
def orbital_pair_manager(orbitals):
    """
    """

    if orbitals is not None:
        return [orbitals[0]], [orbitals[1]]

    else:
        nbands = qcm.model_size()[1]
        orb_list = [i for i in range(1, nbands+1)]
        return orb_list, orb_list
#---------------------------------------------------------------------------------------------------
def print_options(opt=0):
    """Prints the list of global options and parameters on the screen

      :param int opt: 0 -> prints to screen. 1 -> prints to latex. 2 -> prints to RST

    """
    return qcm.print_options(opt)

#---------------------------------------------------------------------------------------------------
def banner(s, c='-', skip=0):
    """Pretty-prints a banner (one line across) on the screen with a message
    
    :param str s: message
    :param char c: character used in the non-text part of the banner
    :param int skip: number of blank lines above and below the banner

    """
    if skip:
        print('\n'*(skip-1))
    n = len(s)
    L = 100
    if n > L-8 :
        print(c[0]*L, flush=True)
        print(s, flush=True)
        print(c[0]*L, flush=True)
    else:
        m = (L - n - 2) // 2
        print(c[0]*m, s, c[0]*m, flush=True)
    if skip:
        print('\n'*(skip-1))
        

#---------------------------------------------------------------------------------------------------
def varia_table(var, val, prefix = ''):
    """
    Pretty prints a list of variational parameters and values in a table form. For screen output in CDMFT and VCA

    :param [str] var: list of parameter names
    :param [float] val: list of associated values
    :param str prefix: prefix string to each line
    """
    s = prefix
    for i,p in enumerate(var):
        s += '{:<9} = {: .4g}\t'.format(p,val[i])
        if (i+1)%5 == 0:
            s += '\n'
            s += prefix
    return s

#---------------------------------------------------------------------------------------------------
def epsilon(y, pr=False):
    """Performs the epsilon algorithm for accelerated convergence (e.g. in CDMFT)

    :param [float] y: sequence to be extrapolated
    :param boolean pr: if True, prints the resulting extrapolation
    :return [float]: the extrapolated values

    """
    
    if len(y)%2 ==0 :
        print("the epsilon algorithm requires an odd-length sequence")
        return 0
    M = np.zeros((len(y),len(y)+1))
    M[:,1] = y
    for i in range(len(y)-2, -1, -1):
        for k in range(2,len(y)-i+1):
            M[i,k] = M[i+1,k-2] + 1.0/(M[i+1,k-1]-M[i,k-1])
    np.set_printoptions(linewidth=1000)
    if pr == True :
        print(M)
    return M[0,-1]

#---------------------------------------------------------------------------------------------------
def general_interaction_matrix_elements(e, n):
    """Translates a list of matrix elements (i,j,k,l,v) for a general interaction into a list of compound elements (I,J,v)
    where I = i+n*j and J = k+n*l and v is the value of the matrix element
    Also checks that only non redundant elements are given
    :param (int,int,int,int,float) e: list of matrix elements
    :param int n: number of orbitals in the impurity model (excluding spin; 2*n with spin)
    Here a sum over spin for the Coulomb interaction est performed
    """

    E = []
    nn = 2*n
    if len(e[0]) != 5: raise ValueError('The general matrix elements do not have the right format')
    for x in e:
        for s in [0,n]:
            for sp in [0,n]:
                I = x[0]+s-1 + nn*(x[1]+sp-1)
                J = x[3]+s-1 + nn*(x[2]+sp-1) 
                print('-----> ', I, J, x[4])
                E.append((I+1,J+1,x[4]))
                # need to add one because indices start at 1 when transmitted via pyqcm (1 is subtracted in the C++ code)

    return E
        

#---------------------------------------------------------------------------------------------------
def switch_cluster_model(name):
    """
    switches cluster model to 'name'. Hack used in DCA (yet to be developped)

    """
    return qcm.switch_cluster_model(name)

#---------------------------------------------------------------------------------------------------
def is_sequence(obj):
    if isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, np.ndarray): return True
    else: return False

#---------------------------------------------------------------------------------------------------
def model_reset():
    banner("RESETTING THE MODEL", c='#', skip=1)
    lattice_model.defined = False
    qcm.great_reset()


