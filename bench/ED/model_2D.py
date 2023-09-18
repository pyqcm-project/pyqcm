import pyqcm

def model2D(Lx, Ly, sym=True):

    gen1 = []
    for y in range(Ly):
        gen1 += [x+y*Lx for x in range(Lx,0,-1)]
    gen2 = []
    for y in range(Ly-1, -1, -1):
        gen2 += [x+y*Lx for x in range(1, Lx+1)]

    # print(gen1)
    # print(gen2)
    # exit()

    if sym:
        CM = pyqcm.cluster_model(Lx*Ly, 0, generators=(gen1, gen2))
    else:
        CM = pyqcm.cluster_model(Lx*Ly, 0)
       
    pos = [(x,y,0) for y in range(Ly) for x in range(Lx) ]

    clus = pyqcm.cluster(CM, pos)
    model = pyqcm.lattice_model('rect_{:d}x{:d}'.format(Lx,Ly), clus, ((Lx,0,0),(0,Ly,0)))

    model.interaction_operator('U')
    model.hopping_operator('t', (1,0,0), -1)
    model.hopping_operator('t', (0,1,0), -1)

    return model