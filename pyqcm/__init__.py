
import os
#Moise: note I don't like set environment variable here, because if someone uses QCM library to built c++ application, the problem will still occur...
#must be before importing qcm
if "OPENBLAS_NUM_THREADS" not in os.environ.keys():
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "BLIS_NUM_THREADS" not in os.environ.keys():
  os.environ["BLIS_NUM_THREADS"] = "1"
#OMP_NUM_THREADS in QCM
#CUBACORES in QCM

import numpy as np
import re
import time
import copy
from . import qcm
from warnings import warn

parameter_set_str = ''
des_dict = {}  # use to store description lines in output files. filename->current description line


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

################################################################################
# GLOBAL MODULE VARIABLES
solver = 'ED'

# special wavevectors
wavevector_M = (2/3)*np.array([1,0,0])
wavevector_K = (2/3)*np.array([1,1/np.sqrt(3),0])

################################################################################
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

################################################################################
# CLASSES

class model:
    cluster_model = []
    clusters = []
    sites = []

    def __init__(self):
        self.record = ''

    def __repr__(self):
        return self.record

    def _finalize(self):
        self.sites = np.array(self.sites, int)

class cluster_model:
    def __init__(self, name, ns, nb=0):
        self.name = name
        self.n_sites = ns
        self.n_bath = nb

class model_instance:
    def __init__(self, model, label):
        self.record = ''
        self.model = model
        self.label = label

    def __repr__(self):
        return self.model.record + self.record

    def write(self, A):
        self.record += str(A)

    def print(self, filename='record.py'):
        global parameter_set
        with open(filename, 'w') as f:
            # f.write('from pyqcm import *\nset_global_parameter("nosym")\n' + self.model.record + parameter_set_str + self.record)
            f.write('from pyqcm import *\n' + self.model.record + parameter_set_str + self.record)


the_model = model()


################################################################################
def model_is_closed():

    """returns True if the model can no longer be modified. False otherwise.

    """

    return qcm.model_is_closed()

################################################################################
def complex_HS(label=0):

    """returns True if the model needs a complex Hilbert space. False otherwise.

    """

    return qcm.complex_HS(label)

################################################################################
def new_cluster_model(name, n_sites, n_bath=0, generators=None, bath_irrep=False):
    """Initiates a new model (no operators yet)

    :param str name: name to be given to the model
    :param int n_sites: number of cluster sites
    :param int n_bath: number of bath sites
    :param [[int]] generators: symmetry generators (2D array of ints)
    :param [boolean] bath_irrep: True if bath orbitals belong to irreducible representations of the symmetry group
    :return: None

    """

    if model_is_closed():
        print('WARNING : new_cluster_model() cannot be called any more : the model is closed')
        return
    global the_model
    the_model.record += "new_cluster_model('"+name+"', "+str(n_sites)+', '+str(n_bath)+', generators='+str(generators)+', bath_irrep='+str(bath_irrep)+')\n'

    the_model.cluster_model += [cluster_model(name, n_sites, n_bath)]
    
    qcm.new_model(name, n_sites, n_bath, generators, bath_irrep)



################################################################################
def new_cluster_operator(name, op_name, op_type, elem):

    """creates a new operator from its matrix elements
    
    :param str name: name of the cluster model to which the operator will belong
    :param str op_name: name of the operator
    :param str op_type: type of operator ('one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg', 'X', 'Y', 'Z')
    :param [(int,int,float)] elem: array of matrix elements (list of tuples)
    :return: None
        
    """

    global the_model

    if model_is_closed():
        print('WARNING : new_cluster_operator() cannot be called any more : the model is closed')
        return

    the_model.record += "new_cluster_operator('"+name+"', '"+op_name+"', '"+op_type+"', "+str(elem)+')\n'

    if op_type == 'anomalous':
        for x in elem:  
            if x[0] >= x[1] :
                raise ValueError(f'anomalous matrix elements of {op_name} must be such that row index < column index')

    qcm.new_operator(name, op_name, op_type, elem)

################################################################################
def new_cluster_operator_complex(name, op_name, op_type, elem):

    """creates a new operator from its complex-valued matrix elements

    :param str name: name of the cluster model to which the operator will belong
    :param str op_name: name of the operator
    :param str op_type: type of operator ('one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg', 'X', 'Y', 'Z')
    :param [(int, int, complex)] elem: array of matrix elements (list of tuples)
    :return: None

    """

    global the_model

    if model_is_closed():
        print('WARNING : new_cluster_operator_complex() cannot be called any more : the model is closed')
        return

    the_model.record += "new_cluster_operator_complex('"+name+"', '"+op_name+"', '"+op_type+"', "+str(elem)+')\n'

    qcm.new_operator_complex(name, op_name, op_type, elem)


################################################################################
def susceptibility_poles(op_name, label=0):
    """computes the dynamic susceptibility of an operator

    :param str name: name of the operator
    :param int label: label of cluster model instance
    :returns [(float,float)]: array of 2-tuple (pole, residue)

    """

    return qcm.susceptibility_poles(op_name, label)

################################################################################
def susceptibility(op_name, freqs, label=0):
    """computes the dynamic susceptibility of an operator

        :param str op_name: name of the operator
        :param [complex] freqs: array of complex frequencies
        :param int label: label of cluster model instance
        :return: array of complex susceptibilities

    """

    return qcm.susceptibility(op_name, freqs, label)

################################################################################
def qmatrix(label=0):
    """Returns the Lehmann representation of the Green function

        :param int label: label of the cluster model instance
        :return: 2-tuple made of
            1. the array of M real eigenvalues, M being the number of poles in the representation
            2. a rectangular (L x M) matrix (real of complex), L being the dimension of the Green function

    """	

    return qcm.qmatrix(label)

################################################################################
def write_cluster_instance_to_file(filename, clus=0):
    """Writes the solved cluster model instance to a text file
    
    :param str filename: name of the file
    :param int clus: label of the cluster model instance
    
    :return: None

    """

    qcm.write_instance_to_file(filename, clus)

############################   WRAPPERS for qcm   ##############################
def add_cluster(name, pos, sites, ref=0):
    """Adds a cluster to the repeated unit

    :param str name:  name of the cluster model
    :param [int] pos: base position of cluster (array of ints)
    :param [[int]] sites: list of positions of sites (2D array of ints)
    :param [int] ref: label of a previous cluster (starts at 1) to which this one is entirely equivalent (0 = no equivalence)
    :return: None

    """

    global the_model

    if model_is_closed():
        print('WARNING : add_cluster() cannot be called any more : the model is closed')
        return

    the_model.record += "add_cluster('"+name+"', "+str(pos)+', '+str(sites)+', ref = '+str(ref)+')\n'
    
    the_model.clusters += [(name, pos, sites)]
    for x in sites:
        the_model.sites += [np.array(pos, int) + np.array(x, int)]
    
    qcm.add_cluster(name, pos, sites, ref)

############################   WRAPPERS for qcm   ##############################
def print_graph(name, sites):
    """prints a graphiz (dot) program for the cluster 

    :param str name:  name of the cluster model
    :param [[int]] sites: list of positions of sites (2D array of ints)
    :return: None

    """

    qcm.print_graph(name, sites)

################################################################################
def averages(ops=[], label=0, file='averages.tsv'):
    """
    Computes the lattice averages of the operators present in the model

    :param int label:  label of the model instance
    :param str file: name of the file in which the information is appended
    :return {str,float}: a dict giving the values of the averages for each parameter

    """
    ave = qcm.averages(ops, label)
    write_summary(file)
    return ave


################################################################################
def cluster_Green_function(cluster, z, spin_down=False, label=0, blocks=False):
    """Computes the cluster Green function

    :param int cluster: label of the cluster (0 to the number of clusters-1)
    :param complex z: frequency
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 	4)
    :param int label:  label of the model instance (default 0)
    :param boolean blocks: true if returned in the basis of irreducible representations
    :return: a complex-valued matrix

    """

    return qcm.cluster_Green_function(cluster, z, spin_down, label, blocks)

################################################################################
def Green_function_average(cluster=0, spin_down=False):
    """Computes the cluster Green function average (integral over frequencies)

    :param int cluster: label of the cluster (0 to the number of clusters-1)
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 	4)
    :return: a complex-valued matrix

    """
    return qcm.Green_function_average(cluster, spin_down)

################################################################################
def cluster_self_energy(cluster, z, spin_down=False, label=0):
    """Computes the cluster self-energy

    :param int cluster: label of the cluster (0 to the number of clusters -1)
    :param complex z: frequency
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 	4)
    :param int label:  label of the model instance (default 0)
    :return: a complex-valued matrix

    """

    return qcm.cluster_self_energy(cluster, z, spin_down, label)

################################################################################
def cluster_info():
    """
    :return:A list of 4-tuples: (str, int, int, int, int): name of the cluster model, number of physical sites, number of bath sites, dimension of the Green function, number of point-group symmetry operations
    """
    return qcm.cluster_info()


################################################################################
def cluster_hopping_matrix(clus=0, spin_down=False, label=0, full=0):
    """
    returns the one-body matrix of cluster no i for instance 'label'

    :param cluster: label of the cluster (0 to the number of clusters - 1)
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :param boolean full: if True, returns the full hopping matrix, including bath
    :return: a complex-valued matrix
    """
    if full:
        return qcm.hopping_matrix(spin_down, clus, True)
    else:
        return qcm.cluster_hopping_matrix(clus, spin_down, label)

################################################################################
def interactions(clus=0):
    """
    returns the one-body matrix of cluster no i for instance 'label'

    :param cluster: label of the cluster (0 to the number of clusters - 1)
    :return: a real-valued matrix
    """
    return qcm.interactions(clus)


################################################################################
def CPT_Green_function(z, k, spin_down=False, label=0):
    """
    computes the CPT Green function at a given frequency

    :param z: complex frequency
    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single or an array of complex-valued matrices
    """
    return qcm.CPT_Green_function(z, k, spin_down, label)

################################################################################
def CPT_Green_function_inverse(z, k, spin_down=False, label=0):
    """
    computes the inverse CPT Green function at a given frequency

    :param z: complex frequency
    :param k: array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single or an array of complex-valued matrices
    """
    if np.shape(k) == (3,):
        k = np.array([k]) # this protects the case where k is an ndarray(3)

    return qcm.CPT_Green_function_inverse(z, k, spin_down, label)


################################################################################
def dispersion(k, spin_down=False, label=0):
    """
    computes the dispersion relation for a single or an array of wavevectors

    :param wavevector k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single (ndarray(d)) or an array (ndarray(N,d)) of real values (energies). d is the reduced GF dimension.

    """
    return qcm.dispersion(k, spin_down, label)

################################################################################
def dos(z, label=0):
    """
    computes the density of states at a given frequency.

    :param complex z: frequency
    :param int label:  label of the model instance
    :return: ndarray(d) of real values, d being the reduced GF dimension

    """
    return qcm.dos(z, label)


################################################################################
def Green_function_dimension():
    """
    returns the dimension of the CPT Green function matrix
    :return: int

    """
    return qcm.Green_function_dimension()

################################################################################
def cluster_Green_function_dimension(clus=0):
    """
    returns the dimension of the cluster Green function matrix
    :param int clus: label of the cluster
    :return: int

    """
    return qcm.Green_function_dimensionC(clus)

################################################################################
def Green_function_solve(label=0):
    """
    Usually, the Green function representation is computed only when needed, in a just-in-time fashion (i.e. in a lazy way).
    This forces the computation of the Green function representation for the current instance (i.e. non lazy). 

    :param int label:  label of the model instance
    :return: None

    """
    return qcm.Green_function_solve(label)

################################################################################
def ground_state(file=None):
    """
    :return: a list of pairs (float, str) of the ground state energy and sector string, for each cluster of the system

    """

    GS = qcm.ground_state()

    if file is not None:
        write_summary(file) 

    return GS

################################################################################
def cluster_averages(label=0):
    """
    Computes the average and variance of all operators of the cluster model in the cluster ground state.

    :param int label: label of the cluster model instance
    :return: a dict str : (float, float) with the averages and variances as a function of operator name

    """
    return qcm.cluster_averages(label)

################################################################################
def Lehmann_Green_function(k, orb=1, spin_down=False, label=0):
    """
    computes the Lehmann representation of the periodized Green function for a set of wavevectors

    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param int orb: orbital index (starts at 1)
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a list of pairs {poles, residues}, each of poles and residues being itself a list.

    """
    return qcm.Lehmann_Green_function(k, orb, spin_down, label)

################################################################################
def hybridization_function(clus, z, spin_down=False, label=0):
    """
    returns the hybridization function for cluster 'cluster' and instance 'label'

    :param int clus: label of the cluster (0 to the number of clusters-1)
    :param complex z: frequency
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a complex-valued matrix

    """
    return qcm.hybridization_function(z, spin_down, clus, label)

################################################################################
def lattice_model(name, superlattice, lattice=None):
    """
    initiates the lattice model.

    :param str name: the name of the model
    :param [[int]] superlattice: array of integers of shape (d,3), d being the dimension
    :param [[int]] lattice: array of integers of shape (d,3), d being the dimension. If None, will be replaced by the unit lattice.
    :return: None

    """
    global the_model

    if model_is_closed():
        print('WARNING : lattice_model() cannot be called any more : the model is closed')
        return

    if lattice_model.called:
        raise RuntimeError('lattice_model() can only be called once.')
    lattice_model.called = True    

    the_model.record += "lattice_model('"+name+"', "+str(superlattice)+', '+str(lattice)+')\n'

    
    qcm.lattice_model(name, superlattice, lattice)
    
lattice_model.called = False

################################################################################
def mixing():
    """
    returns the mixing state of the system:

    * 0 -- normal.  GF matrix is n x n, n being the number of sites
    * 1 -- anomalous. GF matrix is 2n x 2n
    * 2 -- spin-flip.  GF matrix is 2n x 2n
    * 3 -- anomalous and spin-flip (full Nambu doubling).  GF matrix is 4n x 4n
    * 4 -- up and down spins different.  GF matrix is n x n, but computed twice, with spin_down = false and true

    :return: int

    """
    return qcm.mixing()

################################################################################
def model_size():
    """
    :return: a 5-tuple:

    1. the size of the supercell
    2. the number of bands
    3. a tuple containing the sizes of each cluster
    4. a tuple containing the sizes of each cluster's bath
    5. a tuple containing the references of each cluster (label of reference cluster, from 0 to the nunber of clusters-1)

    """

    return qcm.model_size()

################################################################################
def momentum_profile(name, k, label=0):
    """
    computes the momentum-resolved average of an operator

    :param str name: name of the lattice operator
    :param k: array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param int label:  label of the model instance
    :return: an array of values

    """

    return qcm.momentum_profile(name, k, label)

################################################################################
def new_model_instance(label=0, record=False):
    """
    Creates a new instance of the lattice model, with values associated to terms of the Hamiltonian.

    :param int label:  label of the model instance
    :param boolean record: if True, keeps a str-valued record of the instance in memory, containing all the data necessary to read the instance back without solving.
    :return (class model_instance): an instance of the class `model_instance`

    """
    global solver
    the_instance = model_instance(the_model, label)
    qcm.new_model_instance(label)
    if solver=='dvmc':
        import pyqcm.dvmc
        try:
            pyqcm.dvmc.dvmc_solver()
        except:
            raise SolverError()

    if record:
        cinfo = cluster_info()
        params = parameter_set(True)
        for x in params:
            if params[x][1] is None:
                the_instance.record += 'set_parameter("'+str(x)+'", '+str(params[x][0])+')\n'
            
        the_instance.record += '\nnew_model_instance('+str(label)+')\n'
        mod = model_size()
        nclus = len(mod[2])
        the_instance.record += '\nsolution=[None]*'+str(nclus)+'\n'
        for i in range(nclus):
            # if cinfo[i][4] > 1:
            #     raise ValueError('symmetries must be off when recording the model instance!')
            if mod[4][i] != i:
                continue
            clabel = label*nclus+i
            the_instance.record += '\n#--------------------- cluster no '+str(i+1)+' -----------------\n'
            the_instance.record += 'solution['+str(i)+'] = """\n' + qcm.write_instance(clabel) + '\n"""\n'

        for i in range(nclus):
            if mod[4][i] != i:
                continue
            clabel = label*nclus+i
            the_instance.record += 'read_cluster_model_instance(solution['+str(i)+'], '+str(clabel)+')\n'
            

    return the_instance


################################################################################
def new_cluster_model_instance(name, values, sec, label=0):
    """Initiates a new instance of the cluster model

    :param str name: name of the cluster model
    :param {str,float} values: values of the operators
    :param str sec: target Hilbert space sectors
    :param int label: label of model_instance

    :return: None

    """

    qcm.new_model_instanceC(name, values, sec, label)

################################################################################
def read_cluster_model_instance(S, label=0):
    """reads the solution from a string 

    :param str S: long string containing the solution 
    :param int label: label of model_instance

    :return: None

    """
    qcm.read_instance(S, label)

################################################################################
def set_parameters(params, dump=True):
    """
    Defines a new set of parameters, including dependencies

      :param tuple/str params: the values/dependence of the parameters (array of 2- or 3-tuples), or string containing syntax
      :param boolean dump: if True, sets the global str parameter_set_str tothe value
    
    :return [tuple]: list of tuples of the form (str, float) or (str, float, str). The first form gives the parameter name and its value. The second gives the parameter name, a multiplier and the name of the reference parameter. See the documentation on the hierarchy of parameters.
    
    """
    global parameter_set_str
    if set_parameters.called:
        print('WARNING : The function set_parameters() can only be called once')
        return None

    set_parameters.called = True

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
        if dump:
            parameter_set_str = 'set_parameters("""\n'+str(params)+'""")\n'
        qcm.set_parameters(elems)
        new_model_instance() # Prevents user error by instantiating model
        return elems
    else:	
        qcm.set_parameters(params)
        new_model_instance() # Prevents user error by instantiating model
        return params

set_parameters.called = False        

################################################################################
def set_target_sectors(sec):
    """Define the Hilbert space sectors in which to look for the ground state

    :param [str] sec: the target sectors

    :return: None

    """
    if set_target_sectors.called:
        print('WARNING : set_target_sectors() can only be called once.')
        return # prevents a deep c++ error from calling twice
    set_target_sectors.called = True    
    global the_model
    the_model.record += """
try:
    import model_extra
except:
    pass		
"""
    the_model.record += 'set_target_sectors('+str(sec)+')\n'

    qcm.set_target_sectors(sec)

set_target_sectors.called = False

################################################################################
def __sector_string_builder(R, N, S):
    """Takes non-None values of R, N and S and inserts them in the correct string format for set_target_sectors()
    
    :param int R: Symmetry sector
    :param int N: Particle number sector
    :param int S: Spin sector

    :return string: A string formatted for use with set_target_sectors

    """
    R_string = ""
    N_string = ""
    S_string = ""

    if R is not None:
        R_string = f"R{R}"
    if N is not None:
        N_string = f"N{N}"
    if S is not None:
        S_string = f"S{S}"

    return f"{R_string}:{N_string}:{S_string}"


def __cluster_sector_string_builder(R, N, S):
    sector_string = ""
    for i in range(len(R)):
        sector_string += f"{__sector_string_builder(R[i], N[i], S[i])}/"
    return sector_string[:-1] # removes the last caracter (/) from string


def sectors(R=None, N=None, S=None):
    """Alternative method of setting target sectors (see `set_target_sectors(...)`). An example of usage is, R = [[1,2], [3,4]] --> ["R1.../R2...", "R3.../R4..."].
    
    :param int or [int] or [[int]] R: Symmetry sector.
    :param int or [int] or [[int]] N: Particle number.
    :param int or [int] or [[int]] S: Total spin.

    :return: None

    """
    sector_string_list = []
    
    if type(R) is not list and type(N) is not list and type(S) is not list:
        sector_string_list.append(__sector_string_builder(R, N, S)) # if all arguments are int

    elif type(R) is list and type(N) is list and type(S) is list:
        if len(R) != len(N) or len(N) != len(S):
            raise ValueError("The lengths of R, N and S must be the same!")

        if type(R[0]) is list:
            for i in range(len(R)):
                if len(R[i]) != len(N[i]) or len(N[i]) != len(S[i]):
                    raise ValueError("Sublists at same indices must have same length!")
                sector_string_list.append(__cluster_sector_string_builder(R[i], N[i], S[i])) # if all arguments are nested lists
        else:
            sector_string_list.append(__cluster_sector_string_builder(R, N, S)) # if all arguments are lists
         
    else:
        raise ValueError(f"R, N and S are mismatched! Here, type(R)={type(R)}, type(N)={type(N)}, type(S)={type(S)}")

    set_target_sectors(sector_string_list)
    
################################################################################
def parameters(label=0):
    """
    returns the values of the parameters in a given instance

    :param int label:  label of the model instance
    :return: a dict {string,float}

    """
    return qcm.parameters(label)

################################################################################
def cluster_parameters(label=0):
    """
    returns the values of the cluster parameters in a given instance, as well as the cluster model name

    :param int label:  label of the cluster model instance
    :return: a tuple:  dict{string,float}, str

    """
    return qcm.parametersC(label)

################################################################################
def parameter_set(opt='all'):
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
        
################################################################################
def print_parameter_set():
    """
    prints the content of the parameter set

    """
    qcm.print_parameter_set()


################################################################################
def periodized_Green_function(z, k, spin_down=False, label=0):
    """
    computes the periodized Green function at a given frequency and wavevectors

    :param complex z: frequency
    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

    """
    return qcm.periodized_Green_function(z, k, spin_down, label)

################################################################################
def band_Green_function(z, k, spin_down=False, label=0):
    """
    computes the periodized Green function at a given frequency and wavevectors, in the band basis (defined
    in the noninteracting model). It only differs from the periodized Green function in multi-band models.

    :param complex z: frequency
    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

    """
    return qcm.band_Green_function(z, k, spin_down, label)

################################################################################
def periodized_Green_function_element(r, c, z, k, spin_down=False, label=0):
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
    return qcm.periodized_Green_function_element(r, c, z, k, spin_down, label)


################################################################################
def potential_energy(label=0):
    """
    computes the potential energy for a given instance, as the functional trace of Sigma x G

    :param int label: label of the model instance
    :return: the value of the potential energy

    """
    return qcm.potential_energy(label)


################################################################################
def Potthoff_functional(hartree=None, file='sef.tsv', label=0, symmetrized_operator=None):
    """
    computes the Potthoff functional for a given instance

    :param int label: label of the model instance
    :param str file: name of the file to append with the result
    :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
    :param str symmetrized_operator: name of an operator wrt which the functional must be symmetrized
    :return: the value of the self-energy functional

    """
    OM = qcm.Potthoff_functional(label)

    if symmetrized_operator is not None:
        try:
            P = parameters()
            x = P[symmetrized_operator]
            set_parameter(symmetrized_operator,-x)
            new_model_instance()
            OMsym = qcm.Potthoff_functional(label)
            OM = 0.5*(OM + OMsym)
            set_parameter(symmetrized_operator, x)
        except:
            pass
            
    if hartree != None:
        L = model_size()[0]
        for C in hartree:
            OM += C.omega_var()/L

    write_summary(file, suppl_descr='omegaH\t', suppl_values='{:.8g}\t'.format(OM))

    return OM


################################################################################
def properties(label=0):
    """
    Returns two strings of properties of a model instance
    
    :param int label:  label of the model instance
    :return: a pair of strings (the description line and the data line).

    """
    des, data = qcm.properties(label)
    des += 'githash\tversion\t'
    data += git_hash + '\t' + version + '\t'
    return des, data

################################################################################
def print_options(opt=0):
    """Prints the list of global options and parameters on the screen

      :param int opt: 0 -> prints to screen. 1 -> prints to latex. 2 -> prints to RST

    """
    return qcm.print_options(opt)

################################################################################
def print_model(filename):
    """Prints a description of the model into a file

    :param str filename: name of the file
    
    :return: None
    """
    qcm.print_model(filename)

################################################################################
def projected_Green_function(z, spin_down=False, label=0):
    """
    computes the projected Green function at a given frequency, as used in CDMFT.

    :param complex z: frequency
    :param boolean spin_down: true is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: the projected Green function matrix (d x d), d being the dimension of the CPT Green function.

    """
    return qcm.projected_Green_function(z, spin_down, label)

################################################################################<<<<<<<<< eliminer
def read_model(filename):
    """
    Reads the definition of the model from a file

    :param file: name of the file, the same as that of the model, i.e., without the '.model' suffix.
    :return: None

    """
    qcm.read_model(filename)

################################################################################
def reduced_Green_function_dimension():
    """
    returns the dimension of the reduced Green function, i.e. a simple multiple of the
    number of bands n, depending on the mixing state: n, 2n or 4n, and the number of bands

    """

    d = qcm.reduced_Green_function_dimension()
    nb = d
    mix = mixing()
    if mix == 1 or mix == 2:
        nb = d//2
    if mix == 3:
        nb = d//4
    return d, nb


################################################################################
def self_energy(z, k, spin_down=False, label=0):
    """
    computes the self-energy associated with the periodized Green function at a given frequency and wavevectors

    :param complex z: frequency
    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance 
    :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

    """
    return qcm.self_energy(z, k, spin_down, label)

################################################################################
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

################################################################################
def get_global_parameter(name, value=None):
    """
    gets the value of a global parameter. 

    :param str name: name of the global option
    :return: the value, according to type

    """
    return qcm.get_global_parameter(name)

################################################################################
def set_parameter(name, value, pr=False):
    """
    sets the value of a parameter within a parameter_set

    :param str name: name of the parameter
    :param float value: its value
    :return: None

    """
    if pr:
        print('-----> ', name, ' = ', value)
    # if value == 0:
    #     warn(
    #         "`0` carries a specific meaning for QCM (does not create operator). If a trivial operator value is desired, using a small value such as `1e-9` is preferable."
    #     )

    qcm.set_parameter(name, value)

################################################################################
def spatial_dimension():
    """
    returns the spatial dimension (0, 1, 2, 3) of the model

    """
    return qcm.spatial_dimension()

################################################################################
def spectral_average(name, z, label=0):
    """
    returns the contribution of a frequency to the average of an operator

    :param name: name of the operator
    :param z: complex frequency
    :param int label:  label of the model instance
    :return: float

    """
    return qcm.spectral_average(name, z, label)

################################################################################
def variational_parameters():
    """
     
    :return: a list of the names of the variational parameters

    """
    return qcm.variational_parameters()

################################################################################
def set_basis(B):
    """
    
    :param B: the basis (a (D x 3) real matrix)
    :return: None

    """

    global the_model
    the_model.record += "set_basis("+str(B)+')\n'

    qcm.set_basis(B)

################################################################################
def orbital_manager(orbitals, from_zero=False):
    """
    """

    if orbitals is None:
        nbands = model_size()[1]
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

################################################################################
def __orbital_pair_manager(orbitals):
    """
    """

    if orbitals is not None:
        return [orbitals[0]], [orbitals[1]]

    else:
        nbands = model_size()[1]
        orb_list = [i for i in range(1, nbands+1)]
        return orb_list, orb_list

################################################################################
def interaction_operator(name, link=None, orbitals=None, **kwargs):
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

    global the_model
    if type(the_model.sites) is list:
        the_model._finalize()

    orb1, orb2 = __orbital_pair_manager(orbitals) 

    for orb_no1 in orb1:
        for orb_no2 in orb2:
            the_model.record += "interaction_operator('"+name+"'"
            the_model.record += ', orb1='+str(orb_no1)
            the_model.record += ', orb2='+str(orb_no2)
            if link is not None:
                the_model.record += ', link='+str(link)

            for x in kwargs:
                if type(kwargs[x]) is str:
                    the_model.record += ', '+x+"='"+kwargs[x]+"'"
                else:	
                    the_model.record += ', '+x+'='+str(kwargs[x])
            the_model.record += ')\n'	
            qcm.interaction_operator(name, orb1=orb_no1, orb2=orb_no2, link=link, **kwargs)

################################################################################
def hopping_operator(name, link, amplitude, orbitals=None, **kwargs):
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

    global the_model

    if link == [0,0,0]:
        if "tau" in kwargs:
            if kwargs["tau"] != 0:
                warn("***** Setting tau=0 since the link is [0,0,0] (on-site operator). *****")
                kwargs["tau"] = 0
        else:
            kwargs["tau"] = 0

    if type(the_model.sites) is list:
        the_model._finalize()

    orb1, orb2 = __orbital_pair_manager(orbitals) 

    for orb_no1 in orb1:
        for orb_no2 in orb2:
            the_model.record += "hopping_operator('"+name+"', "+str(link)+', '+str(amplitude)
            the_model.record += ', orb1='+str(orb_no1)
            the_model.record += ', orb2='+str(orb_no2)

            for x in kwargs:
                if type(kwargs[x]) is str:
                    the_model.record += ', '+x+"='"+kwargs[x]+"'"
                else:	
                    the_model.record += ', '+x+'='+str(kwargs[x])
            the_model.record += ')\n'

            qcm.hopping_operator(name, link, amplitude, orb1=orb_no1, orb2=orb_no2, **kwargs)

################################################################################
def anomalous_operator(name, link, amplitude, orbitals=None, **kwargs):
    """Defines an anomalous operator

    :param str name: name of operator
    :param [int] link: bond vector (3-component integer array)
    :param complex amplitude: pairing multiplier
    :param (int,int) orbitals: lattice orbital labels (start at 1); if None, tries all possibilities.

    :Keyword Arguments:
    
        * type (*str*) -- one of 'singlet' (default), 'dz', 'dy', 'dx'
  
    :return: None

    """

    global the_model
    
    if type(the_model.sites) is list:
        the_model._finalize() 

    orb1, orb2 = __orbital_pair_manager(orbitals) 

    for orb_no1 in orb1:
        for orb_no2 in orb2:
            the_model.record += "anomalous_operator('"+name+"', "+str(link)+', '+str(amplitude)
            the_model.record += ', orb1='+str(orb_no1)
            the_model.record += ', orb2='+str(orb_no2)

            for x in kwargs:
                if type(kwargs[x]) is str:
                    the_model.record += ', '+x+"='"+kwargs[x]+"'"
                else:	
                    the_model.record += ', '+x+'='+str(kwargs[x])
            the_model.record += ')\n'

            qcm.anomalous_operator(name, link, amplitude, orb1=orb_no1, orb2=orb_no2, **kwargs)

################################################################################
def explicit_operator(name, elem, **kwargs):
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
    global the_model
    the_model.record += "explicit_operator('"+name+"', "+str(elem)
    for x in kwargs:
        if type(kwargs[x]) is str:
            the_model.record += ', '+x+"='"+kwargs[x]+"'"
        else:	
            the_model.record += ', '+x+'='+str(kwargs[x])
    the_model.record += ')\n'	

    qcm.explicit_operator(name, elem, **kwargs)

################################################################################
def density_wave(name, t, Q, **kwargs):
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

    global the_model
    the_model.record += "density_wave('"+name+"', '"+str(t)+"', "+str(Q)
    for x in kwargs:
        if type(kwargs[x]) is str:
            the_model.record += ', '+x+"='"+kwargs[x]+"'"
        else:	
            the_model.record += ', '+x+'='+str(kwargs[x])
    the_model.record += ')\n'	

    qcm.density_wave(name, t, Q, **kwargs)

################################################################################
def site_and_bond_profile():
    """
    Computes the site and bond profiles in all clusters of the repeated unit

    :return: A pair of ndarrays

    site profile -- the components are 
    x y z n Sx Sy Sz psi.real psi.imag

    bond profile -- the components are  
    x1 y1 z1 x2 y2 z2 b0 bx by bz d0.real dx.real dy.real dz.real d0.imag dx.imag dy.imag dz.imag

    """
    return qcm.site_and_bond_profile()

################################################################################
def V_matrix(z, k, spin_down=False, label=0):
    """
    Computes the matrix :math:`V=G_0^{-1}-G^{c-1}_0` at a given frequency and wavevectors, where :math:`G_0` is the noninteracting Green function on the infinite lattice and :math:`G^c_0` is the noninteracting Green function on the cluster.

    :param complex z: frequency
    :param wavevector k: wavevector (ndarray(3)) in units of :math:`\pi`
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.

    """
    return qcm.V_matrix(z, k, spin_down, label)

################################################################################
def tk(k, spin_down=False, label=0):
    """
    computes the k-dependent one-body matrix of the lattice model

    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single or an array of complex-valued matrices
    """
    return qcm.tk(k, spin_down, label)

################################################################################
def QP_weight(k, eta=0.01, orb=1, spin_down=False, label=0):
    """
    computes the k-dependent quasi-particle weight from the self-energy derived from the periodized Green function

    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param float eta: increment in the imaginary axis direction used to computed the derivative of the self-energy
    :param int orb: orbital index (starts at 1)
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a single float or an array of floats, depending on the shape of k

    """

    if np.shape(k) == (3,):
        k = np.array([k]) # this protects the case where k is an ndarray(3)

    sigma1 = qcm.self_energy(-eta*1j, k, spin_down, label)
    sigma2 = qcm.self_energy(eta*1j, k, spin_down, label)
    if len(sigma1.shape) == 3:
        Z = (sigma1[:,orb-1,orb-1].imag - sigma2[:,orb-1,orb-1].imag)/(2*eta) + np.ones(len(k))
    else:
        Z = (sigma1[orb-1,orb-1].imag - sigma2[orb-1,orb-1].imag)/(2*eta) + 1.0
    Z = 1.0/Z
    return Z

################################################################################
def cluster_QP_weight(cluster=0, eta=0.01, orb=1, spin_down=False, label=0):
    """
    computes the cluster quasi-particle weight from the cluster self-energy

    :param int cluster: cluster label (starts at 0)
    :param float eta: increment in the imaginary axis direction used to computed the derivative of the self-energy
    :param int orb: orbital index (starts at 1)
    :param boolean spin_down: True is the spin down sector is to be computed (applies if mixing = 4)
    :param int label:  label of the model instance
    :return: a float

    """
    sigma1 = cluster_self_energy(cluster, -eta*1j, spin_down, label)
    sigma2 = cluster_self_energy(cluster, eta*1j, spin_down, label)
    Z = (sigma1[orb-1,orb-1].imag - sigma2[orb-1,orb-1].imag)/(2*eta) + 1.0
    Z = 1.0/Z
    return Z
    

################################################################################
def spin_spectral_function(freq, k, orb=None, label=0):
    """
    computes the k-dependent spin-resolved spectral function

    :param freq: complex freqency
    :param k: single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3)) in units of :math:`\pi`
    :param int orb: if None, sums all the orbitals. Otherwise just shows the weight for that orbital (starts at 1)
    :param int label:  label of the model instance
    :return: depending on the shape of k, a nd.array(3) of nd.array(N,3)

    """
    orbs = orbital_manager(orb, from_zero=True)

    mix = mixing()
    ds, nbands = reduced_Green_function_dimension()
    if mix != 2 and mix != 3:
        raise RuntimeError('The function "spin_spectral_function()" makes sense only if spin-flip terms are present')
    if mix == 2:
        ds //= 2
    elif mix == 3:
        ds //= 4
    
    G = periodized_Green_function(freq, k, False, label)

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

################################################################################
def update_bath(label=0):
    """
    updates the model parameters without creating a new instance or reseting the instance specified

    :param int label:  label of the model instance
    :return: None

    """
    qcm.update_bath(label)

################################################################################
def print_wavefunction(label=0, pr=True):
    """prints the ground state wavefunction(s) on the screen

    :param int label: label of the model instance
    :param bool pr: prints wavefucntion to screen if pr=True

    :return: the wavefunction

    """
    wavefunction = qcm.print_wavefunction(label)

    if pr:
        print(wavefunction)

    return wavefunction

################################################################################
def matrix_elements(model, op):
    """
    returns the type and matrix elements defining a Hermitian operator

    :param str model: name of the cluster model
    :param str op: name of the operator
    :return: a tuple (typ, elem)

    """
    return qcm.matrix_elements(model, op)



################################################################################


################################################################################
# GENERIC FUNCTIONS
################################################################################
def banner(s, c='-', skip=0):
    if skip:
        print('\n'*(skip-1))
    n = len(s)
    if n > 72 :
        print(c[0]*80, flush=True)
        print(s, flush=True)
        print(c[0]*80, flush=True)
    else:
        m = (80 - n - 2) // 2
        print(c[0]*m, s, c[0]*m, flush=True)
    if skip:
        print('\n'*(skip-1))



def print_averages(ave):
    """
    Prints the averages nicely on the screen

    :param dict ave: the dictionary produced by qcm.averages()

    """
    for x in ave:
        print('   <{:s}> = {:f}'.format(x, ave[x]))

def print_cluster_averages(ave):
    """
    Prints the averages nicely on the screen

    :param dict ave: the dictionary produced by qcm.averages()

    """
    for x in ave:
        print('   <{:s}> = {:f}\tvar({:s}) = {:f}'.format(x, ave[x][0], x, ave[x][1]))
        

def print_parameters(P):
    """
    Prints the parameters nicely on the screen
        
    :param dict P: the dictionary of parameters
    
    """
    for x in P:
        print(x, ' = ', P[x])



def __dependent_parameter_string():
    D = parameter_set()
    S = ''
    for x in D:
        if D[x][1] != None:
            if D[x][2] == 1:
                S += '{:s}={:s};'.format(D[x][1], x)
            elif D[x][2] == -1:
                S += '{:s}=-{:s};'.format(D[x][1], x)
            else:
                S += '{:s}={:1.1g}*{:s};'.format(D[x][1], D[x][2], x)
    if len(S) == 0:
        S = 'none '
    return S[:-1]


def write_summary(f, suppl_descr=None, suppl_values=None, first_of_series=False):
    first_in_file =False
    des, val = properties()
    dependent_parameters = __dependent_parameter_string()
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
        val = suppl_values + val
    val += time.strftime("%Y-%m-%d@%H:%M", time.localtime())
    fout = open(f, 'a')
    if first:
        if suppl_descr != None:
            des = suppl_descr + des
        des += 'time\tdependent_parameters\t'
        if first_in_file is False: fout.write('\n')
        fout.write(des + '\n')
    fout.write(val + '\t' + dependent_parameters + '\n')
    fout.close()


################################################################################
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
    

################################################################################
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
                    ticks += [i]
                    tick_labels += [T[i]]
            ticks = np.array(ticks)
        else:
            raise ValueError('wavevector path shape '+shape+' unknown')

    return 0.5 * k, ticks, tick_labels


################################################################################
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
    if spatial_dimension() < 2:
        raise RuntimeError('calling "wavevector_grid()" makes no sense for a spatial dimension < 2')

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

################################################################################
def set_params_from_file(out_file, n=0):
    """
    reads an output file for parameters

    :param str out_file: name of output file from which parameters are read
    :param int n: line number of data in output file (excluding titles)
    :return: nothing

    """
    par = qcm.parameter_set()
    try:
        D = np.genfromtxt(out_file, names=True)
    except:
        raise("The file containing the solutions could not be read!")
    for x in par:
        if par[x][1] != None:
            continue
        if x in D.dtype.names:
            set_parameter(x,D[x][n],pr=True)

################################################################################
def parameter_string(lattice=True, CR=False):
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


################################################################################
def __varia_table(var, val, prefix = ''):
    s = prefix
    for i,p in enumerate(var):
        s += '{:<9} = {: .4g}\t'.format(p,val[i])
        if (i+1)%5 == 0:
            s += '\n'
            s += prefix
    return s

################################################################################
def switch_cluster_model(name):
    """
    switches cluster model to 'name'. Hack used in DCA.

    """
    return qcm.switch_cluster_model(name)


######################################################################
def epsilon(y, pr=False):
    """Performs the epsilon algorithm for accelerated convergence

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




######################################################################
def density_matrix(sites, label=0):
    """Computes the density matrix of subsystem A, defined by the array of site indices "sites"
    :param [int] sites: list of sites defining subsystem A
    :return [complex], [int32], [int32]: the density matrix, the left and right bases (spins up and down)

    """
    rho, basis = qcm.density_matrix(sites, label)

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

######################################################################
def general_interaction_matrix_elements(e, n):
    """Translates a list of matrix elements (i,j,k,l,v) for a general interaction into a list of compound elements (I,J,v)
    where I = i+n*j and J = k+n*l and v is the value of the matrix element
    Also checks that only non redundant elements are given
    :param (int,int,int,int,float) e: list of matrix elements
    :param int n: number of orbitals in the impurity model (excluding spin; 2*n with spin)
    """

    E = []
    nn = 2*n
    if len(e[0]) == 5:
        for x in e:
            s = 1
            if x[0]<x[1]:
                I = x[0]+nn*x[1]
            else:
                I = x[1]+nn*x[0]
                s *= -1
            if x[2]<x[3]:
                J = x[2]+nn*x[3]
            else:
                J = x[3]+nn*x[2]
                s *= -1
            if I > J:
                continue
            
            E += [(I+1,J+1,s*x[4])]  # need to add one because indices start at 1 when transmitted via pyqcm (1 is subtracted in the C++ code)
    elif len(e[0]) == 9:
        for y in e:
            x = (y[0]+n*y[1], y[2]+n*y[3], y[4]+n*y[5], y[6]+n*y[7], y[8])
            s = 1
            if x[0]<x[1]:
                I = x[0]+nn*x[1]
            else:
                I = x[1]+nn*x[0]
                s *= -1
            if x[2]<x[3]:
                J = x[2]+nn*x[3]
            else:
                J = x[3]+nn*x[2]
                s *= -1
            if I > J:
                continue
            
            E += [(I+1,J+1,s*x[4])]  # need to add one because indices start at 1 when transmitted via pyqcm (1 is subtracted in the C++ code)
    else:
        raise ValueError('The general matrix elements do not have the right format')

    return E
        
######################################################################
def double_counting_correct(DC):
    """Modifies some kinetic parameters in view of the presence of interactions and averages values,
    in order to account minimally for double counting.
    :param [(str,str,str,float,float)] DC: list of recipes for the correction: (kinetic operator, interaction operator, density operator, coefficient, value of the kinetic operator without interaction)
    """
    
    ave = averages()
    P = parameters()
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
        set_parameter(x, corr[x], pr=True)
    


