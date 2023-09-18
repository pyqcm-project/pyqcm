import pyqcm

CM = pyqcm.cluster_model(8)
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)))
model = pyqcm.lattice_model('2x2x2', clus, ((2,0,0), (0,2,0), (0,0,2)))
model.interaction_operator('U')
model.hopping_operator('t', (0,1,0),-1)
model.hopping_operator('t', (1,0,0),-1)
model.hopping_operator('tz', [0,0,1],-1)
model.hopping_operator('t2', (1,1,0),-1)
model.hopping_operator('t2', (-1,1,0),-1)
model.anomalous_operator('D', (1,0,0), 1)
model.anomalous_operator('D', (0,1,0),-1)
