import pyqcm

class slab:
    """Helper class to define multi-layer systems, typically layers of planes.
    The fundamental unit is a 2D model that is repeated in the z direction
    """
    def  __init__(self, name, nlayer, cluster, sites, superlattice, lattice, thickness = None):
        """Constructor

        :param str name: name of the 2D model
        :param int nlayer: total number of layers in the slab
        :param str cluster: name of the cluster model assembled (limited to one cluster)
        :param [[int,int,int]] sites: list of sites in each layer
        :param [[int,int,int]] superlattice: superlattice vectors in 2D
        :param [[int,int,int]] lattice: lattice vectors in 2D
        :param int thickness: number of inequivalent layers (<= nlayer/2). If None, all layers are different

        """
        self.nlayer = nlayer
        self.name = name
        if thickness is None:
            self.thickness=(nlayer+1)//2
            for i in range(nlayer):
                pyqcm.add_cluster(cluster, [0,0,i], sites)
        else:
            self.thickness=thickness
            if thickness > (nlayer+1)//2:
                print('thickness cannot exceed ', (nlayer+1)//2)
                raise ValueError(f'thickness value cannot exceed {(nlayer+1)//2}')
            if thickness == (nlayer+1)//2 and nlayer%2 :
                l1 = nlayer-thickness+1
            else:
                l1 = nlayer-thickness
            for i in range(thickness):
                pyqcm.add_cluster(cluster, [0,0,i], sites)
            for i in range(thickness, nlayer-thickness):
                pyqcm.add_cluster(cluster, [0,0,i], sites, ref=thickness)
            for i in range(l1, nlayer):
                pyqcm.add_cluster(cluster, [0,0,i], sites, ref=nlayer-i)

        pyqcm.lattice_model('slab_{:s}_{:d}L'.format(name, nlayer), superlattice, lattice)
        self.nband = pyqcm.model_size()[1]//nlayer
    
    def hopping_operator(self, name, link, amplitude, **kwargs):
        """Defines a hopping operator repeated across all layers of the slab.
        Same arguments as the pyqcm.hopping_operator() function.

        :param str name: name of the hopping operator
        :param [int] link: bond vector (3-component); the z-component must be 0 (intra-layer) or 1 (inter-layer)
        :param float amplitude: hopping amplitude multiplier
        :param kwargs: additional keyword arguments passed to pyqcm.hopping_operator() (e.g. orb1, orb2, tau, sigma)

        """
        
        if 'orb1' in kwargs:
            b1 = kwargs['orb1']
        else:
             b1 = 1
        if 'orb2' in kwargs:
            b2 = kwargs['orb2']
        else:
             b2 = 1
        if 'tau' in kwargs:
            tau = kwargs['tau']
        else:
            tau = 1
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
        else:
            sigma = 0
        
        if link[2] == 0:
            for i in range(self.nlayer):
                pyqcm.hopping_operator(name, link, amplitude, orb1=b1+self.nband*i,  orb2=b2+self.nband*i, tau=tau, sigma=sigma)
        elif link[2] == 1:
            for i in range(self.nlayer-1):
                pyqcm.hopping_operator(name, link, amplitude, orb1=b1+self.nband*i,  orb2=b2+self.nband*(i+1), tau=tau, sigma=sigma)
        else:
            raise ValueError('hopping operators in slabs can only have a z-component of 0 or 1 for the link')
        
    def interaction_operator(self, name, **kwargs):
        """Defines an interaction operator repeated across all layers of the slab.
        Same arguments as the pyqcm.interaction_operator() function.

        :param str name: name of the interaction operator
        :param kwargs: additional keyword arguments passed to pyqcm.interaction_operator()
            (e.g. orb1, orb2, type, link, amplitude)

        """
        
        if 'orb1' in kwargs:
            b1 = kwargs['orb1']
        else:
             b1 = 1
        if 'orb2' in kwargs:
            b2 = kwargs['orb2']
        else:
             b2 = 1
        if 'type' in kwargs:
            the_type = kwargs['type']
        else:
            the_type = 'Hubbard'
        if 'link' in kwargs:
            link = kwargs['link']
        else:
            link = ( 0, 0, 0)
        if 'amplitude' in kwargs:
            amplitude = kwargs['amplitude']
        else:
            amplitude = 1.0
       
        if link[2] == 0:
            for i in range(self.nlayer):
                pyqcm.interaction_operator(name, link=link, amplitude=amplitude, orb1=b1+self.nband*i,  orb2=b2+self.nband*i, type=the_type)
        elif link[2] == 1:
            for i in range(self.nlayer-1):
                pyqcm.interaction_operator(name, link=link, amplitude=amplitude, orb1=b1+self.nband*i,  orb2=b2+self.nband*i, type=the_type)
        else:
            raise ValueError('hopping operators in slabs can only have a z-component of 0 or 1 for the link')

