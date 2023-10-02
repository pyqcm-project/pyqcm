import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=10, linewidth=512, suppress=True)

####################################################################################################
class superlattice:
    """
    Class used to analyze superlattices in view of the inter-cluster Hartree mean-field approximation
    """
    def __init__(self, sites, e):
        """
        This class is used to define a supercluster and the associated superlattice
        :param [[int]] sites: list of sites of the cluster (integer components), or tuple of such collections if more than one cluster in a supercluster.
        :param [[int]] e: superlattice basis vectors (integer components, the number of vectors is the dimension). Must be right oriented.
        """
        e = np.array(e, dtype='int')
        self.e = e
        self.dim = self.e.shape[0]
        assert (self.dim > 0 and self.dim <= 3), 'the dimension must be 1, 2 or 3'
        self.E = np.eye(3, dtype='int')
        self.E[0:self.dim, :] = self.e
        S = self.E
        self.vol = int(np.rint(np.linalg.det(S)))  # fixed : go to nearest integer
        assert self.vol > 0, 'Please use a lattice with a positive orientation. Aborting'

        self.minors = np.empty((3,3), dtype=int)
        self.minors[0,0] =  S[1,1]*S[2,2] - S[2,1]*S[1,2]
        self.minors[1,0] = -S[1,0]*S[2,2] + S[2,0]*S[1,2]
        self.minors[2,0] =  S[1,0]*S[2,1] - S[1,1]*S[2,0]
        self.minors[0,1] = -S[0,1]*S[2,2] + S[2,1]*S[0,2]
        self.minors[1,1] =  S[0,0]*S[2,2] - S[2,0]*S[0,2]
        self.minors[2,1] = -S[0,0]*S[2,1] + S[0,1]*S[2,0]
        self.minors[0,2] =  S[0,1]*S[1,2] - S[1,1]*S[0,2]
        self.minors[1,2] = -S[0,0]*S[1,2] + S[1,0]*S[0,2]
        self.minors[2,2] =  S[0,0]*S[1,1] - S[0,1]*S[1,0]
        # print('minors:\n', self.minors)
        # print('super unit cell volume = ', self.vol)

        # folding the sites
        if type(sites) is not tuple:
            sites = (sites,)
        sitesF = []
        sitesO = []
        clus=[]
        for i,c in enumerate(sites):
            csites = np.array(c, dtype='int')
            for x in csites:
                sitesF.append(self.fold_shifted(x)[1])
                sitesO.append(x)
                clus.append(i)
        self.sitesF = np.array(sitesF, dtype='int') # sites, folded into a conventional unit cell
        self.sites = np.array(sitesO, dtype='int') # sites, folded into a conventional unit cell
        self.clus = np.array(clus, dtype='int') # cluster associated with each site
        self.N = self.sites.shape[0]
        # print('liste of sites and folded sites:')
        # for i in range(self.N):
        #     print(i+1, '\t', self.sites[i], '\t', self.sitesF[i], '\tdiff = ', self.sitesF[i]-self.sites[i])
        # dictionary of sites (for getting the index from the position)
        self.siteF_index = {}
        for i,c in enumerate(sitesF):
            S = '[{:d},{:d},{:d}]'.format(c[0], c[1], c[2])
            self.siteF_index[S] = i



    #-----------------------------------------------------------------------------------------------
    def fold_shifted(self, r):
        """
        folds an integer vector back to the shifted unit cell of the superlattice
        This shifted unit cell is such that all of its vectors have positive or zero components in terms of superlattice vectors

        :param [int] r: integer component vector to be folded into the unit cell
        :return (int, [int], [int]): I, S, R where r = S + R and R is a superlattice vector, I is the index of the folded site

        """
        r = np.array(r, dtype=int)
        R = np.empty(3, dtype=int)
        S = np.empty(3, dtype=int)
        Q = np.empty(3, dtype=int)

        R = r@self.minors
        assert np.linalg.norm(R@self.E  - r*self.vol) < 1e-6, 'Folding error here : r = {}, R={}, check={}'.format(r,R, self.vol*r@np.linalg.inv(self.E))

        # now r = (R[0] e[0] + R[1] e[1] + R[2] e[2]) + (Q[0] e[0] + Q[1] e[1] + Q[2] e[2])/self.vol
        Q[0] = R[0] % self.vol
        R[0] //= self.vol
        if(Q[0] < 0):
            Q[0] += self.vol
            R[0] -= 1

        Q[1] = R[1] % self.vol
        R[1] //= self.vol
        if(Q[1] < 0):
            Q[1] += self.vol
            R[1] -= 1

        Q[2] = R[2] % self.vol
        R[2] //= self.vol
        if(Q[2] < 0):
            Q[2] += self.vol
            R[2] -= 1
	
        assert np.linalg.norm(R@self.E + Q@self.E//self.vol - r) < 1e-6, 'Folding error here : r = {}, R={}, Q={}'.format(r,R,Q)

        # S = self.E[0]*Q[0] + self.E[1]*Q[1] + self.E[2]*Q[2]
        S = np.dot(Q,self.E)
        S[0] //= self.vol
        S[1] //= self.vol
        S[2] //= self.vol
        Q = R
        # R = self.E[0]*Q[0] + self.E[1]*Q[1] + self.E[2]*Q[2]
        R = np.dot(Q,self.E)

  
        # then compensate between R and S to make the folding d-dimensional if d<3
        if(self.dim < 3):
            R -= self.E[2]*Q[2]
            S += self.E[2]*Q[2]
        if(self.dim  < 2):
            R -= self.E[1]*Q[1]
            S += self.E[1]*Q[1]

        try:
            Sstr = '[{:d},{:d},{:d}]'.format(S[0], S[1], S[2])
            I = self.siteF_index[Sstr]
        except:
            I = None

        assert np.linalg.norm(S+R-r)<1e-6, 'folding operation failed! r = {}, S={}, R={}'.format(r,S,R)
        return I, S, R

    #-----------------------------------------------------------------------------------------------
    def fold(self, r):
        """
        like 'fold_shifted', but this time folds back to the cluster itself, i.e., a different unit cell
        :param [int] r: integer component vector to be folded into the unit cell
        :return (int, [int], [int]): I, S, R where r = S + R and R is a superlattice vector, I is the index of the folded site
        """
        I, S, R = self.fold_shifted(r)
        if I != None:
            S += (self.sites[I] - self.sitesF[I])
            R += (self.sitesF[I] - self.sites[I])
        # print('fold : r = ', r, ' I = ', I, '  S = ', S, '  R = ', R)
        return I, S, R

    #-----------------------------------------------------------------------------------------------
    def draw(self, basis=None, plt_ax=None):
        """
        Plots the sites, the shifted sites and the superlattice vectors
        :param [[float]] basis: the real space, geometric basis

        """
        if self.dim != 2:
            print('the draw() function only works in two dimensions. Skipping draw()')
            return

        if plt_ax is None:
            ax = plt.gca()
        else:
            ax = plt_ax

        if basis is None:
            basis = np.eye(3)
        
        e = self.e
        e = e@basis
        ax.set_aspect(1)
        ax.plot([0, e[0,0]], [0, e[0,1]], 'r-', lw=0.5)
        ax.plot([0, e[1,0]], [0, e[1,1]], 'r-', lw=0.5)
        S = self.sites
        S = S@basis
        ax.plot(S[:,0], S[:,1], 'bo', ms=6)
        S = self.sitesF
        S = S@basis
        ax.plot(S[:,0], S[:,1], 'go', ms=3)
        if plt_ax is None:
            plt.show()

    #-----------------------------------------------------------------------------------------------
    def draw_cdw(self, cdw, basis=None, plt_ax=None):
        """
        Plots the amplitudes of a cdw
        :param [float] cdw: the cdw amplitudes (same order as the sites)
        :param [[float]] basis: the real space, geometric basis

        """
        if self.dim != 2:
            print('the draw() function only works in two dimensions. Skipping draw()')
            return

        if plt_ax is None:
            ax = plt.gca()
        else:
            ax = plt_ax

        if basis is None:
            basis = np.eye(3)
        
        e = self.e
        e = e@basis
        ax.set_aspect(1)
        ax.plot([0, e[0,0]], [0, e[0,1]], 'r-', lw=0.5)
        ax.plot([0, e[1,0]], [0, e[1,1]], 'r-', lw=0.5)
        S = self.sites
        S = S@basis
        fac = 8/np.max(np.abs(cdw))
        for j1 in range(-1,2,1):
            for j2 in range(-1,2,1):
                es = j1*e[0,:] + j2*e[1,:]
                for i in range(S.shape[0]):
                    ss = es + S[i,:]
                    if cdw[i] > 0: ax.plot(ss[0], ss[1], 'bo', ms=fac*cdw[i])
                    else: ax.plot(ss[0], ss[1], 'ro', ms=-fac*cdw[i])
                    ax.plot(ss[0], ss[1], 'k.', ms=1)
        if plt_ax is None:
            plt.show()


    #-----------------------------------------------------------------------------------------------
    def cdw_eigenstates(self, _V, plt_ax=None, basis=np.eye(3), file=None, silent=False):
        """
        Computes the possible CDW states of the cluster, to be used with the Hartree approximation

        :param cluster C: periodic cluster
        :param [([int], float)] V: density-density interactions
        :return ([float], [float,float], [float,float]): the array of eigenvalues, the matrix of eigenvectors, and the inter-cluster interaction matrix

        """

        V = []
        E = 0.0
        Vic = np.zeros((self.N, self.N))
        Vc = np.zeros((self.N, self.N))
        S2 = self.sites@basis
        if plt_ax != None:
            plt_ax.set_aspect(1.0)
            plt_ax.scatter(S2[:,0], S2[:,1], [24]*S2.shape[0],color='gray')

        assert type(_V) == list
        for y in _V:
            assert type(y) == tuple

        for v in _V:
            V.append((np.array(v[0], dtype=int), v[1]))

        if silent == False: print('-'*100)
        for v in V:
            for i,x in enumerate(self.sites):  # loop over site 1
                j, S, R = self.fold(self.sites[i] + v[0])
                if j != None:
                    if self.clus[i] == self.clus[j] and np.linalg.norm(R) < 1e-6:
                        Vc[i,j] += v[1]
                    else:
                        Vic[i,j] += v[1]
                    if plt_ax != None and silent == False:
                        DX = v[0]@basis
                        plt_ax.plot([S2[i,0], S2[i,0]+DX[0]], [S2[i,1], S2[i,1]+DX[1]])
                j, S, R = self.fold(self.sites[i] - v[0])
                if j != None:
                    if self.clus[i] == self.clus[j] and np.linalg.norm(R) < 1e-6:
                        Vc[i,j] += v[1]
                    else:
                        Vic[i,j] += v[1]
                    if plt_ax != None and silent == False:
                        DX = v[0]@basis
                        plt_ax.plot([S2[i,0], S2[i,0]-DX[0]], [S2[i,1], S2[i,1]-DX[1]])

        w, v = np.linalg.eigh(Vic)
        w = np.round(w,10)
        v = np.round(v,10)
        # making sure the first nonzero component of the eigenvector is positive
        for i in range(w.shape[0]):
            for j in range(w.shape[0]):
                if np.abs(v[j,i]) > 1e-6 and v[j,i] < 0:
                    v[:,i] *= -1
                    break


        if silent == False:
            print('intra-cluster V matrix:\n',Vc)
            print('inter-cluster V matrix:\n',Vic)

            fig, ax = plt.subplots((self.N+1)//2, 2, sharex=True, sharey=True)
            fig.set_size_inches(6, 3*ax.shape[0])
            if self.N > 2:
                ax = np.reshape(ax, 2*((self.N+1)//2))

            for i in range(self.N):
                print('\neigenvalue ', w[i], ' :\n', v[:, i])
                self.draw_cdw(v[:, i], basis=basis, plt_ax = ax[i])
                ax[i].set_title(r'No ${:d}, \lambda = {:f}$'.format(i, w[i]), fontsize=9)
                ax[i].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

            # plt.tight_layout()
            if file != None: plt.savefig(file)
            else: plt.show()

        return w, v, Vic, Vc


    #-----------------------------------------------------------------------------------------------
    def cdw_energy(self, U, _V, n, cluster=False, pr=False):
        """
        Computes the potential energy per site associated with a given CDW pattern

        :param [cluster] C: periodic cluster
        :param U : on-site interaction
        :param [([int], float)] V: density-density interactions
        :param [float] n: density pattern
        :param boolean cluster: If True, limits the computation to the cluster (no inter-cluster interactions)
        :param boolean pr: If True, prints progress
        :return float: potential energy

        """
        
        V = []
        E = 0.0
        n = np.array(n)
        assert type(_V) == list
        for y in _V:
            assert type(y) == tuple

        for v in _V:
            V.append((np.array(v[0], dtype=int), v[1]))

        # contribution of the on-site energy
        for i in range(self.N):
            E += 0.25*U*n[i]*n[i]

        if pr:
            print('\nComputing the CDW energy with n = ', n)
        for v in V:
            if cluster:
                for i,x in enumerate(self.sites):
                    R = x + v[0]
                    I = None
                    for j,r in enumerate(self.sites):
                        if np.linalg.norm(R-r) < 1e-6:
                            E += n[i]*n[j]*v[1]
                            break

            else:
                for i,x in enumerate(self.sitesF):
                    j, S, R = self.fold(x + v[0])
                    if j != None:
                        ener = n[i]*n[j]*v[1]
                        if np.abs(ener) > 1e-6 and pr:
                            print(v, '(', i+1, ',', j+1, ') --> ', ener)
                        E += ener
                
        return E/self.N
    
    #-----------------------------------------------------------------------------------------------
    def draw_V(self, _V, plt_ax, basis=None):
        """
        draws the links associated with extended interactions

        :param [([int], float)] _V: density-density interactions
        :param plt_ax: matplotlib axis set; triggers plotting
        :param [[float]] basis: the real space, geometric basis
        :return: None

        """
        
        V = []
        assert type(_V) == list
        for y in _V:
            assert type(y) == tuple

        for v in _V:
            V.append((np.array(v[0], dtype=int), v[1]))

        self.draw(basis, plt_ax)
        S2 = self.sites@basis
        color=['r', 'g', 'b', 'y', 'k']
        ic = 0
        for v in V:
            for i,x in enumerate(self.sitesF):
                j, S, R = self.fold_shifted(x + v[0])
                if j != None:
                    plt_ax.plot([S2[i,0],S2[j,0]], [S2[i,1],S2[j,1]], '-', c = color[ic%5])
            ic += 1
        plt.show()

    #-----------------------------------------------------------------------------------------------
    def draw_Vic(self, _V, plt_ax, basis=None):
        """
        draws the links associated with extended interactions

        :param [([int], float)] _V: density-density interactions
        :param plt_ax: matplotlib axis set; triggers plotting
        :param [[float]] basis: the real space, geometric basis
        :return: None

        """
        
        V = []
        assert type(_V) == list
        for y in _V:
            assert type(y) == tuple

        for v in _V:
            V.append((np.array(v[0], dtype=int), v[1]))

        self.draw(basis, plt_ax)
        S2 = self.sites@basis
        color=['r', 'g', 'b', 'y', 'k']
        ic = 0
        for v in V:
            DX = v[0]@basis
            for i,x in enumerate(self.sitesF):
                j, S, R = self.fold_shifted(x + v[0])
                
                if j != None:
                    plt_ax.plot([S2[i,0],S2[i,0]+DX[0]], [S2[i,1],S2[i,1]+DX[1]], '-', c = color[ic%5])
            ic += 1
        plt.show()

    #-----------------------------------------------------------------------------------------------
    def draw_mode(self, X, plt_ax, basis=None):
        """
        draws the links associated with extended interactions

        :param [float] X: amplitude (one per site)
        :param plt_ax: matplotlib axis set; triggers plotting
        :param [[float]] basis: the real space, geometric basis
        :return: None

        """
        
        S2 = self.sites@basis
        plt_ax.scatter(S2[:,0], S2[:,1], [24]*S2.shape[0],color='gray')
        X = X/np.linalg.norm(X)
        X *= 48
        col = ['b']*len(X)
        for i in range(len(X)):
            if X[i] < 0 :
                col[i] = 'r'
        X = np.abs(X)
        plt_ax.scatter(S2[:,0], S2[:,1], X, c=col)

    #-----------------------------------------------------------------------------------------------
    def sdw_energy(self, _J, sdw, cluster=False, pr=False):
        """
        Computes the energy per site associated with a given SDW pattern

        :param [([int], [[float]])] J: spin interactions
        :param [[float]] sdw: spin pattern (3 x N) : for each site, 3 angles are specifies : chi,theta,phi. The amplitude of the spin is sin(chi), its polar angle is theta and phi is the azimutal angle
        :param boolean cluster: If True, limits the computation to the cluster (no inter-cluster interactions)
        :param boolean pr: If True, prints progress
        :return float: energy

        """
        
        J = []
        E = 0.0
        assert type(_J) == list
        for y in _J:
            assert type(y) == tuple

        for v in _J:
            J.append((np.array(v[0], dtype=int), np.array(v[1])))

        sdw = np.array(sdw)
        if pr:
            print('\nComputing the SDW energy with sdw = ', sdw)
        s = np.zeros((self.N,3))
        for i in range(self.N):
            s[i,0] = np.abs(np.sin(sdw[0,i]))*np.sin(sdw[1,i])*np.cos(sdw[2,i])
            s[i,1] = np.abs(np.sin(sdw[0,i]))*np.sin(sdw[1,i])*np.sin(sdw[2,i])
            s[i,2] = np.abs(np.sin(sdw[0,i]))*np.cos(sdw[1,i])
        for v in J:
            if cluster:
                for i,x in enumerate(self.sites):
                    R = x + v[0]
                    for j,r in enumerate(self.sites):
                        if np.linalg.norm(R-r) < 1e-6:
                            E += s[i]@v[1]@s[j]
                            break

            else:
                for i,x in enumerate(self.sitesF):
                    j, S, R = self.fold(x + v[0])
                    if j != None:
                        ener = s[i]@v[1]@s[j]
                        # if np.abs(ener) > 1e-6 and pr:
                        #     print(v, '(', i+1, ',', j+1, ') --> ', ener)
                        E += ener
                
        return E/self.N

    #-----------------------------------------------------------------------------------------------
    def convert_to_spins(self, x, pr=False):
        s = np.zeros((self.N,3))
        sdw = np.reshape(x,(3,8))
        for i in range(self.N):
            s[i,0] = np.abs(np.sin(sdw[0,i]))*np.sin(sdw[1,i])*np.cos(sdw[2,i])
            s[i,1] = np.abs(np.sin(sdw[0,i]))*np.sin(sdw[1,i])*np.sin(sdw[2,i])
            s[i,2] = np.abs(np.sin(sdw[0,i]))*np.cos(sdw[1,i])

            if pr:
                print(self.sites[i], '\t', s[i,:])
        return s
