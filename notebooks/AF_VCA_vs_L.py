####################################################################################################
# Antiferromagnetism with various cluster sizes
# We construct several clusters to study the simple Hubbard model in two dimension, with sizes 2, 4, 6, 8 and 10, with the most symmetric shape possible and appropriate superlattices. Arrange the clusters in a way that a study of Néel antiferromagnetism is possible, i.e., respect the bipartite character of the square lattice in the tiling.
# For each of these models, perform a VCA analysis of antiferromagnetism at half-filling for $U=8$ and $t=1$. How does the Weiss field change according to cluster size?
####################################################################################################
import pyqcm
from pyqcm.vca import VCA

def create_model(L):
    """
    param L : length of the cluster (L x 2)
    """

    CM = pyqcm.cluster_model(2*L, name='clus'.format(L))
    sites = [(i,0,0) for i in range(L)] + [(i,1,0) for i in range(L)]
    clus = pyqcm.cluster(CM, sites)
    model = pyqcm.lattice_model('{:d}x2_AFM'.format(L), clus, ((L, L%2, 0), (0, 2, 0)))
    model.hopping_operator('t', [1,0,0], -1)
    model.hopping_operator('t', [0,1,0], -1)
    model.interaction_operator('U')
    model.density_wave('M', 'Z', [1,1,0])
    model.set_target_sectors(['R0:N{:d}:S0'.format(2*L)])
    return model

for L in [1,2,3,4,5]:
    pyqcm.reset_model()
    model = create_model(L)

    # Setting model parameters
    model.set_parameters("""
        U = 8
        mu = 0.5*U
        t = 1
        M = 0
        M_1 = 0.1
    """)

    # Running the VCA
    V = VCA(model, varia='M_1', steps=0.01, accur=2e-4, max=10.0, method='altNR', file = 'vca_L{:d}.tsv'.format(L)) 
