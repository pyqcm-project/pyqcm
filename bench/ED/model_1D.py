import pyqcm
import numpy as np

def model1D(L, sym=True):
    if sym:
        g = [[L-i for i in range(L)]]
        print('generators :', g)
        CM = pyqcm.cluster_model(L, 0, generators=g)
    else:
        CM = pyqcm.cluster_model(L, 0)
    pos = [(i,0,0) for i in range(L)]
    clus = pyqcm.cluster(CM, pos)
    model = pyqcm.lattice_model('1D_{:d}'.format(L), clus, ((L, 0, 0),))

    model.interaction_operator('U')
    model.interaction_operator('V', link=( 1, 0, 0))
    model.hopping_operator('t', ( 1, 0, 0), -1)
    model.hopping_operator('tp', ( 2, 0, 0), -1)
    model.hopping_operator('hx', [0, 0, 0], 1, tau=0, sigma=1)
    model.anomalous_operator('D', ( 1, 0, 0), 1)

    if (L%2 == 0) and sym == False:
        model.density_wave('cdw', 'N', ( 1, 0, 0))
        e = np.sqrt(0.5)
        model.explicit_operator('V0m', (
            ((0,0,0), (0,0,0), e),
            ((L-1,0,0), (0,0,0), e),
        ), tau=0, type='one-body')
        model.explicit_operator('V1m', (
            ((0,0,0), (0,0,0), e),
            ((L-1,0,0), (0,0,0),-e),
        ), tau=0, type='one-body')

        model.explicit_operator('Vf', (((L-1,0,0),(1,0,0),1),))
        model.V0m_eig = 1
        model.V1m_eig = -1
        model.Vf_eig = 1/8

    return model

