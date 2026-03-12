import pyqcm
#===============================================================================
# defining the model
nb = 4
ns = 4
no = ns+nb
gS = [i for i in range(ns, 0, -1)] + [0, 0, 0, 0]
gA = [i for i in range(ns, 0, -1)] + [1, 1, 1, 1]

model_nameS = 'S_4b'.format(ns)
model_nameA = 'A_4b'.format(ns)
CMS = pyqcm.cluster_model(ns, nb, generators=[gS], bath_irrep=True, name=model_nameS)
CMA = pyqcm.cluster_model(ns, nb, generators=[gA], bath_irrep=True, name=model_nameA)

varia = []
S = [1,1,-1,-1]
for i in range(1,nb+1):
    CMS.new_operator('eb{:d}'.format(i), 'one-body', [(i+ns,i+ns,1), (i+ns+no,i+ns+no,1)])
    CMS.new_operator('tb{:d}'.format(i), 'one-body', [(1,i+ns,1), (1+no,i+ns+no,1),(ns,i+ns,1), (ns+no,i+ns+no,1)])
    CMA.new_operator('eb{:d}'.format(i), 'one-body', [(i+ns,i+ns,1), (i+ns+no,i+ns+no,1)])
    CMA.new_operator('tb{:d}'.format(i), 'one-body', [(1,i+ns,1), (1+no,i+ns+no,1),(ns,i+ns,-1), (ns+no,i+ns+no,-1)])

varia  = ['eb{:d}_1'.format(i) for i in range(1,nb+1)] + ['tb{:d}_1'.format(i) for i in range(1,nb+1)]
varia += ['eb{:d}_2'.format(i) for i in range(1,nb+1)] + ['tb{:d}_2'.format(i) for i in range(1,nb+1)]

clus = pyqcm.cluster((CMS, CMA), [[i,0,0] for i in range(ns)])
model = pyqcm.lattice_model('1D4_6b_2sb', clus, [[ns,0,0]])

model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)  # NN hopping
model.varia = varia

#===============================================================================
# cdmft

model.set_target_sectors(['N8:S0', 'N8:S0'])
model.set_parameters("""
    U=4
    mu=2
    t=1
    eb1_1=1.0
    eb2_1=-1.0
    eb3_1=1.0
    eb4_1=-1.0
    tb1_1=0.5
    tb2_1=0.55
    tb3_1=0.5
    tb4_1=0.55
    eb1_2=1.0
    eb2_2=-1.0
    eb3_2=1.0
    eb4_2=-1.0
    tb1_2=0.5
    tb2_2=0.55
    tb3_2=0.5
    tb4_2=0.55
""")

from pyqcm.cdmft import CDMFT

X = CDMFT(model, varia, convergence='self-energy', accur=1e-3, method='PRAXIS') 

X.I.cluster_spectral_function()
X.I.spectral_function()