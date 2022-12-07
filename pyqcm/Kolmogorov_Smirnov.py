import numpy as np
import pyqcm
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, linewidth=512, suppress=True)

def fixed_qmatrix(site):
    Q = pyqcm.qmatrix()
    W = np.abs(Q[1][:,site])**2
    S = np.argsort(Q[0])
    return (Q[0][S], W[S])


def Kolmogorov_Smirnov(F, G, tol=1e-6, plot=False):
    """
    :param ([float],[float]) F: a discrete spectrum. Sequence of poles and  (numpy arrays)
    :param ([float],[float]) G: a discrete spectrum. Sequence of poles and residues (numpy arrays)
    :param float tol : tolerance for the normalization of the distributions
    """
    # checks whether F and G have the right type and length
    err = 'The first two arguments of Kolmogorov_Smirnov() must be 2-tuples of numpy arrays'
    assert type(F) is tuple, err
    assert type(G) is tuple, err
    assert isinstance(F[0],np.ndarray), err 
    assert isinstance(F[1],np.ndarray), err 
    assert isinstance(G[0],np.ndarray), err 
    assert isinstance(G[1],np.ndarray), err 

    assert len(F[0]) == len(F[1])
    assert len(G[0]) == len(G[1])

    # checks whether residues are positive
    assert np.all(F[1] >= 0), 'Residues in 1st argument are not positive'
    assert np.all(G[1] >= 0), 'Residues in 2nd argument are not positive'

    # construct cumulative values
    Fc = np.cumsum(F[1])
    Gc = np.cumsum(G[1])

    # checks normalization
    assert np.abs(Fc[-1]-1) < tol, 'Array of residues in 1st argument is not normalized'
    assert np.abs(Gc[-1]-1) < tol, 'Array of residues in 2nd argument is not normalized'

    # checks whether poles are sorted
    assert np.all(np.diff(F[0]) >= 0) , 'The poles in 1st argument are not sorted'
    assert np.all(np.diff(G[0]) >= 0) , 'The poles in 2nd argument are not sorted'



    # construct the difference array
    I_f = 0 # current index from the F array
    I_g = 0 # current index from the G array
    N_f = len(F[0])
    N_g = len(G[0])
    D = np.zeros(len(F[0]) + len(G[0])) # initialization of the difference
    X = np.zeros(len(F[0]) + len(G[0])) # initialization of the abcissas

    Fcurr = 0
    Gcurr = 0
    i = 0
    while(True):
        if F[0][I_f] < G[0][I_g]:
            X[i] = F[0][I_f]
            Fcurr = Fc[I_f]
            D[i] = Fcurr-Gcurr
            i += 1
            I_f += 1
            # print('F next\tX = ', X[i], '\tD[i]=', D[i], '\tI_f=', I_f, '\tI_g=', I_g)
            if I_f == N_f:
                break
        else:
            X[i] = G[0][I_g]
            Gcurr = Gc[I_g]
            D[i] = Fcurr-Gcurr
            i += 1
            I_g += 1
            # print('G next\tX = ', X[i], '\tD[i]=', D[i], '\tI_f=', I_f, '\tI_g=', I_g)
            if I_g == N_g:
                break

    # completing with the leftovers from one of the arrays
    if I_f == N_f:
        while I_g < N_g:
            X[i] = G[0][I_g]
            D[i] = Fc[I_f-1]-Gc[I_g]
            i += 1
            I_g += 1
    if I_g == N_g:
        while I_f < N_f:
            X[i] = F[0][I_f]
            D[i] = Fc[I_f]-Gc[I_g-1]
            i += 1
            I_f += 1

    # computes sum of intervals times square difference
    delta = np.sqrt(np.sum(np.diff(X)*D[0:-1]**2))

    if plot:
        plt.step(F[0],Fc, label='F', lw=2, where='post')
        plt.step(G[0],Gc, label='G', lw=1, where='post')
        plt.ylabel('accumulated weight')
        plt.xlabel('$\omega$')
        plt.title('Kolmogorov-Smirnov distance = {:1.4g}'.format(delta))
        plt.legend()
        plt.show()
        # plt.step(X,D, label='G', where='post')
        # plt.ylabel('difference')
        # plt.xlabel('$\omega$')
        # plt.show()

    return delta



