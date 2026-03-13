import pyqcm
import os
import subprocess
import numpy as np
import pyqcm.cdmft as cdmft

def freq_grid(beta=50,wc=2,real=False):
    """
    Builds a Matsubara frequency grid up to a cutoff frequency.

    :param float beta: inverse fictitious temperature used to define the Matsubara frequency spacing (2*pi/beta)
    :param float wc: cutoff frequency; the grid runs from pi/beta to wc
    :param boolean real: if False (default), returns purely imaginary frequencies (multiplied by 1j); if True, returns real-valued frequencies
    :returns: array of frequencies (complex if real=False, real otherwise)
    :rtype: numpy.ndarray
    """
    wr = np.arange((np.pi / beta), wc + 1e-6, 2 * np.pi / beta)
    w = np.ones(len(wr), dtype=np.complex128)
    if real == False: 
        w = np.ones(len(wr), dtype=np.complex128)
        w = w * 1j
    else:
        w = np.ones(len(wr))
    w *= wr
    return w

def test_ed_solver():
    """
    Tests an external ED (exact diagonalization) solver by writing the impurity problem data
    (hopping matrix, interaction matrix, hybridization function, bath parameters) to a file
    named 'test.tsv', invoking an external Python script 'test_ed_solver.py', and then
    reading back the resulting cluster model instance from 'instance.out'.
    """

    info = pyqcm.cluster_info()[0]
    n = info[1]
    nn = 2*n
    nb = info[2]
    no = n + nb
    T = np.real(pyqcm.cluster_hopping_matrix(clus=0, spin_down=False))
    I = pyqcm.interactions()
    V = np.zeros((n,n))
    for x in I:
        if x[0] < n and x[1] < n: V[x[0], x[1]] = x[2]
    for x in I:
        if x[0] < n and x[1] == x[0]+no: V[x[0], x[0]] = x[2]

    w = freq_grid(beta=100,wc=10,real=True)

    P = pyqcm.parameters()

    with open('test.tsv', 'w') as f:
        f.write('Tij\n')
        np.savetxt(f, T, fmt='%1.6g', delimiter='\t')
        f.write('\n')
        f.write('Vij\n')
        np.savetxt(f, V, fmt='%1.6g', delimiter='\t')
        f.write('\n')
        f.write('Delta\n')
        for x in w:
            f.write(f'{x:1.6g}\t')
            D = pyqcm.hybridization_function(0,x*1j)
            D = D.reshape((1,nn))
            np.savetxt(f, D, fmt='%-1.6g', delimiter='\t')
        f.write('\n')
        f.write('bath_parameters\n')
        for x in cdmft.var:
            f.write(x+'\t{:1.6g}\n'.format(P[x]))
        f.write('\n')

    # now the file is written, one can call the external solver
    out = os.system("python test_ed_solver.py")
    if out != 0:
        raise ValueError('The external solver returned an error')

    # then we need to read the Green function representation to keep going
    # file_Green = 'Green.out'
    # if not os.path.isfile(file_Green):
    #     raise FileNotFoundError('The file ' + file_Green + ' does not exist!')
    # with open(file_Green) as f:
    #     solution = f.read()
    
    
    pyqcm.read_cluster_model_instance('instance.out', 0)


