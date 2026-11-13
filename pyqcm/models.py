import pyqcm

class rep:
    def __init__(self, _C, _signs, _name):
        self.C = _C
        self.signs = _signs
        self.name = _name



def system_2x2_C2v(reps):
    """
    Defines a cluster model on a 2x2 plaquette with C2v symmetry and sites numbered row-wise from 1 to 4 like so:
    3  4
    1  2

    param str reps : a str of irreps from A1, A2, B1, B2. E.g. "A1A1B1B2"
    returns : a cluster model object.
    """
    irrep = dict()
    R = [reps[i:i+2] for i in range(0, len(reps), 2)]
    ns = 4
    nb = len(R)
    no = ns + nb
    g1 = [3, 4, 1, 2]
    g2 = [2, 1, 4, 3]
    irrep['A1'] = rep((0,0), (1, 1, 1, 1), 'A1')
    irrep['A2'] = rep((2,2), (1,-1,-1, 1), 'A2')
    irrep['B1'] = rep((2,0), (1, 1,-1,-1), 'B1')
    irrep['B2'] = rep((0,2), (1,-1, 1,-1), 'B2')
    for i in range(nb):
        g1 += [irrep[R[i]].C[0]]
        g2 += [irrep[R[i]].C[1]]
    CM = pyqcm.cluster_model(ns, nb, name='CM_'+reps, generators=(g1, g2), bath_irrep=True)
    CM.nb = nb
    varia_N = []
    varia_SC = []
    
    # bath energies
    for i in range(nb):
        name = 'eb'+str(i+1)
        lab = i+4+1
        CM.new_operator(name, 'one-body', (
            (lab, lab, 1.0),
            (lab + no, lab + no, 1.0)
        ))
        varia_N.append(name)

    for i in range(nb):
        elem = []
        for j in range(ns):
            elem.append((j+1, i+ns+1, irrep[R[i]].signs[j]))
            elem.append((j+1+no, i+ns+no+1, irrep[R[i]].signs[j]))
        name = 'tb'+str(i+1)
        CM.new_operator(name, 'one-body', elem)
        varia_N.append(name)

    for i in range(nb):
        elem = []
        for j in range(ns):
            elem.append((j+1, i+ns+no+1, irrep[R[i]].signs[j]))
            elem.append((i+ns+1, j+1+no, irrep[R[i]].signs[j]))
        name = 'db'+str(i+1)
        CM.new_operator(name, 'anomalous', elem)
        varia_SC.append(name)
    return CM


def sq_lattice_operators(model, AF=False):
    """
    Defines standard lattice operators on the square-lattice Hubbard model

    param@ model : a lattice_model object, passed as reference (modified on output)
    para@ boolean AF : If True, adds the definition of the antiferromagnetic order parameter. This lowers the symmetry of the model.
    """
    model.interaction_operator('U')
    model.hopping_operator('t', (1,0,0), -1) 
    model.hopping_operator('t', (0,1,0), -1) 
    model.hopping_operator('t2', (1,-1,0), -1)
    model.hopping_operator('t2', (1,1,0), -1)
    model.hopping_operator('t3', (2,0,0), -1)
    model.hopping_operator('t3', (0,2,0), -1)
    model.anomalous_operator('D', (1,0,0), 1) 
    model.anomalous_operator('D', (0,1,0),-1)
    if AF:
        model.density_wave('M', 'Z', (1,1,0), 1)

