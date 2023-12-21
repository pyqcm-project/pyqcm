import numpy as np
import matplotlib.pyplot as plt
import re
import pyqcm

####################################################################################################
def draw_operator(self, op_name, show_labels=False, show_orb_labels=True, show_neighbors=False, values=False, offset = 0.05, orb_offset=0.05, z_offset=0.0, alpha_inter=0.2, spin_scale=0.1, plt_ax = None):
    """
    Draws an operator defined on a lattice model to the screen (for debugging purposes)

    :param str op_name: name of the operator
    :param boolean show_labels: if True, shows the labels of the cluster 
    :param boolean show_neighbors: if True, shows the neighbors of the central cluster 
    :param boolean values: if True, prints de values of the elements 
    :param float offset: offset between sites and labels
    :param float orb_offset: offset between sites and orbital labels
    :param float z_offset: offset between sites and labels for sites along the z axis 
    :param float alpha_inter: alpha value of the inter-cluster links
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    """

    file = 'tmp_model.out'
    self.print_model(file)
    fin = open(file, 'r')

    if plt_ax is not None: plt.sca(plt_ax)

    #...............................................................................................
    # reading positions of sites
    L = ''
    while " sites " not in L:
        L = fin.readline()
        if not L:
            raise ValueError('file ended without sites')

    L = fin.readline()
    sites = []
    orb = []
    cluster = []
    while True:
        L = fin.readline()
        if L == '\n': break
        X = re.split("[(,)\t ]+", L)
        sites.append((int(X[4]), int(X[5]), int(X[6])))
        orb.append(int(X[3])-1)
        cluster.append(int(X[1])-1)

    #...............................................................................................
    # reading basis
    L = ''
    while "phys:" not in L:
        L = fin.readline()
        if not L:
            raise ValueError('file ended without physical basis')

    basis = []
    for i in range(3):
        L = fin.readline()
        X = re.split("[(,) ]+", L)
        basis.append((float(X[1]), float(X[2]), float(X[3])))
    B = np.array(basis)
    B = np.linalg.inv(B)

    S = np.zeros((len(sites), 3))
    for i,s in enumerate(sites):
        S[i,:] = (np.array(s)@B).T

    #...............................................................................................
    # reading superlattice
    L = ''
    while "superlattice:" not in L:
        L = fin.readline()
        if not L:
            raise ValueError('file ended without superlattice')

    super = []
    L = fin.readline()
    X = re.split(" ", L)
    dimension = int(X[1])

    for i in range(dimension):
        L = fin.readline()
        X = re.split("[(,) ]+", L)
        super.append((float(X[1]), float(X[2]), float(X[3])))

    #...............................................................................................
    # reading neighbors
    L = ''
    while "neighbors" not in L:
        L = fin.readline()
        if not L:
            raise ValueError('file ended without neighbors')

    neighbors = []
    while True:
        L = fin.readline()
        if L == '\n': break
        X = re.split("[(,): ]+", L)
        neighbors.append((float(X[1]), float(X[2]), float(X[3])))
    Neigh = np.zeros((len(neighbors), 3))
    for i,s in enumerate(neighbors):
        Neigh[i,:] = (np.array(s)@B).T

    #...............................................................................................
    # reading operator

    L = ''
    while "lattice operators" not in L:
        L = fin.readline()
        if not L:
            raise RuntimeError('file ended without specified lattice operators')

    op_namet = op_name+'\t'
    nc = len(op_namet)
    while op_namet != L[0:nc]:
        L = fin.readline()
        if not L:
            raise RuntimeError(f'file ended without the operator {op_name}')

    X = re.split(r"[\t(): ]+", L)
    op_type = X[1]
    
    fin.readline()
    elements = []
    spin = {}
    Si = {}
    hop = {}
    anom = {}

    while True:
        L = fin.readline()
        if L == '\n' or 'elements' in L: break
        X = re.split("[,\;): ]+", L)
        if len(X) == 6:
            complex = True
            X[3] = X[3][1:]
            v = float(X[3]) + float(X[4])*1j
        else:
            X[-1] = X[-1][0:-1]
            v = float(X[3])
        X[0] = X[0][1:]

        I = int(X[0][0:-1])
        J = int(X[1][0:-1])
        neighbor = int(X[2])
        # if neighbor : continue

        K = '{:d},{:d};{:d}'.format(I,J,neighbor)
        Si[K] = (I-1,J-1,neighbor)
        if 'one-body' in op_type:
            if X[0][-1] == '+' and X[1][-1] == '+':
                sz = 1
                if K in hop:
                    hop[K] += v
                else:
                    hop[K] = v
                if K in spin:
                    spin[K] += np.array([0,v])
                else:
                    spin[K] = np.array([0,v])
            elif X[0][-1] == '-' and X[1][-1] == '-': 
                if K in spin:
                    spin[K] += np.array([0,-v])
                else:
                    spin[K] = np.array([0,-v])
            elif (X[0][-1] == '+' and X[1][-1] == '-') or (X[0][-1] == '-' and X[1][-1] == '+'):
                if K in spin:
                    spin[K] += np.array([v,0])
                else:
                    spin[K] = np.array([v,0])

        elif 'Hubbard' in op_type:
            if X[0][-1] == '+' and X[1][-1] == '-':
                sz = 1
                if K in hop:
                    hop[K] += v
                else:
                    hop[K] = v

        if 'singlet' in op_type or 'dz' in op_type:
            if (X[0][-1] == '+' and X[1][-1] == '-') or (X[0][-1] == '-' and X[1][-1] == '+'):
                if K in anom:
                    anom[K] += v
                else:
                    anom[K] = v

        tmp = {}
        for s in spin:
            if np.linalg.norm(spin[s]) > 1e-2 :tmp[s] = spin[s]
        spin = tmp
    fin.close()
    #...............................................................................................
    # plotting the elements
    for e in hop:
        if ';0' not in e: continue
        if np.abs(hop[e])<0.001: continue
        s1 = Si[e][0]
        s2 = Si[e][1]
        if np.abs(S[s1,2]-S[s2,2]) > 0.001:
            pf = 'r--'
        else:
            pf = 'r-'
        if cluster[s1] == cluster[s2]:
            alpha = 1.0
        else:
            alpha = alpha_inter
        if np.linalg.norm(S[s1,0:2] -  S[s2,0:2]) < 0.0001 :
            plt.plot([S[s1,0]], [S[s1,1]], 'o', ms = 18, c='w', mec='r', mew=2, alpha = alpha)
        else:
            plt.plot([S[s1,0], S[s2,0]], [S[s1,1], S[s2,1]], pf, mew=2, alpha = alpha)
            if values:
                plt.text(0.5*(S[s1,0]+S[s2,0]), 0.5*(S[s1,1]+S[s2,1]), f'${np.round(hop[e],5)}$', va='bottom', ha='center', c='r')

    for e in spin:
        if ';0' not in e: continue
        s1 = Si[e][0]
        s2 = Si[e][1]
        if s1 != s2: continue
        if np.abs(S[s1,2]-S[s2,2]) > 0.001:
            pf = 'r--'
        else:
            pf = 'r-'
        plt.arrow(S[s1,0]-spin_scale*spin[e][0], S[s1,1]-spin_scale*spin[e][1], 2*spin_scale*spin[e][0], 2*spin_scale*spin[e][1], color='r', width=0.1*spin_scale, head_width=0.3*spin_scale)

    for e in anom:
        if ';0' not in e: continue
        s1 = Si[e][0]
        s2 = Si[e][1]
        if s1 == s2: continue
        if cluster[s1] == cluster[s2]:
            alpha = 1.0
        else:
            alpha = alpha_inter
        if np.abs(S[s1,2]-S[s2,2]) > 0.001:
            pf = 'r--'
        else:
            pf = 'r-'
            plt.plot([S[s1,0], S[s2,0]], [S[s1,1], S[s2,1]], pf, mew=2, alpha = alpha)


    #...............................................................................................
    # plotting the sites
    bcol = ['k', 'r', 'b', 'g', 'c', 'm', 'y']
    ncol = len(bcol)
    plt.gca().set_aspect(1)
    plt.gca().axis('off')

    zmin = np.min(S[:,2])
    zmax = np.max(S[:,2])
    for i in range(S.shape[0]):
        if S[i,2]-0.001 < zmin:
            pass
        else:
            plt.plot(S[i,0], S[i,1], 'o', ms = 6 + 6*(S[i,2]-zmin)/(zmax-zmin+0.1), mfc='w', c=bcol[orb[i]%ncol])
            if show_labels: 
                plt.text(S[i,0], S[i,1]+offset, f'${i+1}$', va='bottom', ha='center', color='b', fontsize=8)
            if show_orb_labels: 
                plt.text(S[i,0]+orb_offset, S[i,1]+ z_offset*(S[i,2]-zmin), f'${orb[i]+1}$', va='center', ha='left', color='g', fontsize=8)
    for i in range(S.shape[0]):
        if S[i,2]-0.001 < zmin:
            plt.plot(S[i,0], S[i,1], 'o', ms = 6, c=bcol[orb[i]%ncol])
            if show_labels: 
                plt.text(S[i,0], S[i,1]-offset, f'${i+1}$', va='top', ha='center', color='b', fontsize=8)
            if show_orb_labels: 
                plt.text(S[i,0]+orb_offset, S[i,1]+ z_offset*(S[i,2]-zmin), f'${orb[i]+1}$', va='center', ha='left', color='g', fontsize=8)

    #...............................................................................................
    # plotting the neighbors

    if show_neighbors:
        for j in range(Neigh.shape[0]):
            for i in range(S.shape[0]):
                plt.plot(S[i,0]+Neigh[j,0], S[i,1]+Neigh[j,1], 'ko', ms = 6, alpha=0.5)
        for e in hop:
            if ';0' in e: continue
            if np.abs(hop[e])<0.001: continue
            s1 = Si[e][0]
            s2 = Si[e][1]
            neighbor = Si[e][2]-1
            plt.plot([S[s1,0], S[s2,0] + Neigh[neighbor][0]], [S[s1,1], S[s2,1] + Neigh[neighbor][1]], 'r:', lw=1)
        for e in anom:
            if ';0' in e: continue
            s1 = Si[e][1]
            s2 = Si[e][0]
            neighbor = Si[e][2]-1
            plt.plot([S[s1,0], S[s2,0] + Neigh[neighbor][0]], [S[s1,1], S[s2,1] + Neigh[neighbor][1]], 'r:', lw=1)

    #...............................................................................................
    plt.title(f'${op_name}$', pad=18)
    if plt_ax is None: plt.show()



####################################################################################################

def draw_cluster_operator(self, clus, op_name, show_labels=True, values=False, spin_offset = 0.15, plt_ax = None):
    """
    Draws an operator defined on a cluster model to the screen (for debugging purposes)

    :param cluster clus: instance of a cluster
    :param str op_name: name of the operator
    :param boolean show_labels: if True, shows the labels of the cluster 
    :param boolean values: if True, prints de values of the elements 
    :param float spin_offset: graphical offset between up and down spins 
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    """

    if plt_ax is not None: plt.sca(plt_ax)

    file = 'tmp_model.out'

    clus_name = clus.cluster_model.name
    nb = clus.cluster_model.n_bath
    clus_I = clus.index

    self.print_model(file)
    fin = open(file, 'r')

    #...............................................................................................
    # reading positions of sites
    L = ''
    while " sites " not in L:
        L = fin.readline()
        if not L:
            raise ValueError('draw: file ended without sites')

    L = fin.readline()
    sites = []
    orb = []
    cluster = []
    while True:
        L = fin.readline()
        if L == '\n': break
        X = re.split("[(,)\t ]+", L)
        if int(X[1]) == clus_I:
            sites.append((int(X[4]), int(X[5]), int(X[6])))

    ns = len(sites)
    no = nb + ns
    #...............................................................................................
    # reading basis
    L = ''
    while "phys:" not in L:
        L = fin.readline()
        if not L:
            raise ValueError('file ended without physical basis')

    basis = []
    for i in range(3):
        L = fin.readline()
        X = re.split("[(,) ]+", L)
        basis.append((float(X[1]), float(X[2]), float(X[3])))
    B = np.array(basis)
    B = np.linalg.inv(B)

    S = np.zeros((len(sites), 3))
    for i,s in enumerate(sites):
        S[i,:] = (np.array(s)@B).T

    #...............................................................................................
    # find the cluster description

    L = ''
    while clus_name+' ' not in L:
        L = fin.readline()
        if not L:
            raise ValueError('file ended without finding cluster {:s}'.format(clus_name))

    #...............................................................................................
    # reading operator

    L = ''
    while op_name+'\t' not in L:
        L = fin.readline()
        if not L or '----' in L:
            raise ValueError('file ended without findind operator {:s}'.format(op_name))

    elements = []
    while True:
        L = fin.readline()
        if L == '\n' or len(L) < 4 : break
        X = re.split("[,\;)\t ]+", L.rstrip())
        if len(X) == 4:
            complex = True
            X[2] = X[2][1:]
            v = float(X[2]) + float(X[3])*1j
        else:
            v = float(X[2])

        if X[0][-1] == '-': continue
        I = int(X[0])
        J = int(X[1])
        if I > J : continue
        s1 = 0
        if I > no : 
            s1 = 1
            I -= no
        s2 = 0
        if J > no : 
            s2 = 1
            J -= no
        if I > ns: elements.append((J-1,I-1,v,s2,s1))
        else: elements.append((I-1,J-1,v, s1, s2))

    fin.close()

    
    # finding the maximum value of x and y among the sites
    ymax = np.max(S[:,1])
    xmax = np.max(S[:,0])
    xmin = np.min(S[:,0])
    if xmax-xmin < 1.1:
        xmax += 0.8
        xmin -= 0.8
    bpos = np.zeros((nb,3))
    yb = 1.0
    if nb==0: deltax = 0
    else: deltax = (xmax-xmin)/(nb-1)
    for i in range(nb):
        bpos[i,:] = np.array([xmin + i*deltax, ymax + yb, 0])

    S = np.concatenate((S, np.array(bpos)))

    
    S = np.concatenate((S, S + np.array([spin_offset,spin_offset,0])))
    E = []
    for e in elements:
        E.append((e[0]+no*e[3], e[1]+no*e[4], e[2]))
    elements = E

    #...............................................................................................
    # plotting the elements
    for e in elements:
        if np.abs(S[e[0],2]-S[e[1],2]) > 0.001:
            pf = 'k--'
        else:
            pf = 'k-'
        if np.linalg.norm(S[e[0],0:2] -  S[e[1],0:2]) < 0.0001 :
            plt.plot([S[e[0],0]], [S[e[0],1]], 'o', ms = 24, c='w', mec='r', mew=2)
        else:
            plt.plot([S[e[0],0], S[e[1],0]], [S[e[0],1], S[e[1],1]], pf, mew=2)
            if values:
                plt.text(0.5*(S[e[0],0]+S[e[1],0]), 0.5*(S[e[0],1]+S[e[1],1]), f'${np.round(e[2],5)}$', va='bottom', ha='center', c='r')
    
    is_in_cluster = np.zeros(2*no,int)
    is_in_cluster[0:ns] = 1
    is_in_cluster[no:no+ns] = 1
    #...............................................................................................
    # plotting the sites

    bcol = ['k', 'r', 'b', 'g', 'c', 'm', 'y']
    ncol = len(bcol)
    plt.gca().set_aspect(1)
    plt.gca().axis('off')
    offset = 0.03*np.max(S)

    zmin = np.min(S[:,2])
    zmax = np.max(S[:,2])
    for i in range(S.shape[0]):
        if S[i,2]-0.001 < zmin:
            pass
        else:
            if is_in_cluster[i]:
                plt.plot(S[i,0], S[i,1], 's', ms = 12 + 6*(S[i,2]-zmin)/(zmax-zmin+0.1), mfc='w', c='b')
            else:
                plt.plot(S[i,0], S[i,1], 's', ms = 14, mfc='w', c='r')
            if show_labels: plt.text(S[i,0], S[i,1]+offset, f'${i+1}$', va='bottom', ha='center', color='r', fontsize=8)
    for i in range(S.shape[0]):
        if S[i,2]-0.001 < zmin:
            if is_in_cluster[i]:
                plt.plot(S[i,0], S[i,1], 's', ms = 12, mfc='w', c='b')
            else:
                plt.plot(S[i,0], S[i,1], 's', ms = 14, mfc='w', c='r')
            if show_labels: plt.text(S[i,0], S[i,1], f'${i+1}$', va='center', ha='center', color='b', fontsize=8)

    #...............................................................................................
    plt.title(f'${op_name}$', pad=18)
    if plt_ax is None: plt.show()


    
