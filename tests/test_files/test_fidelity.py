import pyqcm
from pyqcm.cdmft import *

def model1D(ns):
    g = [ns-i for i in range(ns)]
    CM = pyqcm.cluster_model(ns, generators=[g])  
    clus = pyqcm.cluster(CM, [(i,0,0) for i in range(ns)]) 
    return pyqcm.lattice_model('1D_{:d}'.format(ns), clus, ((ns,0,0),)) 

urange = np.arange(1e-9, 16.1, 2)
lrange = np.array([2,4,6,8,10], dtype=int)

f = np.empty((len(lrange), len(urange)))

with open('fid.tsv', 'w') as fout:
    for i,L in enumerate(lrange):
        pyqcm.reset_model()
        model = model1D(int(L))
        model.interaction_operator('U')
        model.hopping_operator('t', (1,0,0), -1)

        model.set_target_sectors(['R0:N{:d}:S0'.format(model.nsites)])
        model.set_parameters("""
            U = 1e-9
            mu = 0.5*U
            t = 1
        """)

        I1 = pyqcm.model_instance(model) 
        for j,u in enumerate(urange):
            model.set_parameter('U', u)
            I2 = pyqcm.model_instance(model) # call to constructor of class "model_instance"
            f[i,j] = model.fidelity(I1,I2)
            S = '{:d}\t{:1.2f}\t{:g}'.format(L, u, f[i,j])
            print(S)
            fout.writelines(S+'\n')
        fout.writelines('\n\n')
        

    for j,u in enumerate(urange):
        pol = np.polyfit(1/lrange, f[:,j], 2)
        fex = np.polyval(pol, 0.0)
        S = 'extr\t{:1.2f}\t{:g}'.format(u, fex)
        print(S)
        fout.writelines(S+'\n')
    
