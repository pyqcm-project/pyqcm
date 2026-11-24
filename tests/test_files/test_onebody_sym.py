"""
Tests one_body_solve() (used when the cluster Hamiltonian has no interaction, here U=0)
with a nontrivial cluster symmetry group.

For each case, the same model is solved with and without symmetry generators, and the
ground state energy, the cluster Green function at a few complex frequencies and the
cluster averages are compared. Cases cover real irreps (C2, C2v) and complex irreps (C4),
all Hilbert space mixings (normal, anomalous, spin-flip, full, up/down), and a cluster
with bath orbitals.

Only one lattice model can be defined per process, hence each calculation is run in a
separate subprocess (this script calling itself with arguments).

Run with: python test_onebody_sym.py
"""
import os
import sys
import pickle
import subprocess
import tempfile
import numpy as np
import pyqcm

C2v = [[3, 4, 1, 2], [2, 1, 4, 3]]  # mirrors of the 2x2 plaquette
C2 = [[4, 3, 2, 1]]                 # rotation by pi
C4 = [[2, 4, 1, 3]]                 # rotation by pi/2 -> complex irreps

cases = {
    'C2v_normal':    dict(gen=C2v, sec='R0:N4:S0', par="t=1\nt2=0.3\nmu=0.7"),
    'C2_full':       dict(gen=C2,  sec='R0',       par="t=1\nt2=0.3\nmu=0.7\nhx=0.2\nD=0.1"),
    'C2v_anom':      dict(gen=C2v, sec='R0:S0',    par="t=1\nt2=0.3\nmu=0.7\nD=0.15"),
    'C2v_updown':    dict(gen=C2v, sec='R0:N4:S0', par="t=1\nt2=0.3\nmu=0.7\nhz=0.25"),
    'C4_complex':    dict(gen=C4,  sec='R0:N4:S0', par="t=1\nt2=0.3\nmu=0.7"),
    'C4_complex_hx': dict(gen=C4,  sec='R0:N4',    par="t=1\nt2=0.3\nmu=0.7\nhx=0.2"),
    'C4_complex_Ds': dict(gen=C4,  sec='R0:S0',    par="t=1\nt2=0.3\nmu=0.7\nDs=0.15"),
    'C4_cplx_full':  dict(gen=C4,  sec='R0',       par="t=1\nt2=0.3\nmu=0.7\nhx=0.2\nDs=0.1"),
    'C2v_bath':      dict(gen=[g + [x+4 for x in g] for g in C2v], sec='R0:N8:S0', bath=True,
                          par="t=1\nmu=0.4\neb_1=0.8\ntb_1=0.5"),
    'C2v_bath_anom': dict(gen=[g + [x+4 for x in g] for g in C2v], sec='R0:S0', bath=True,
                          par="t=1\nmu=0.4\neb_1=0.8\ntb_1=0.5\nsb_1=0.2"),
}

zs = [0.3 + 0.2j, -1.1 + 0.05j, 2j]
tol = 1e-8


def build(name, gen, bath=False):
    if bath:
        # 4 bath orbitals, each attached to one cluster site and permuted like the sites
        CM = pyqcm.cluster_model(4, 4, generators=gen, name='CM_'+name)
        no = 8
        CM.new_operator('eb', 'one-body', [(i, i, 1.0) for i in range(5, 9)] + [(i+no, i+no, 1.0) for i in range(5, 9)])
        CM.new_operator('tb', 'one-body', [(i, i+4, 1.0) for i in range(1, 5)] + [(i+no, i+4+no, 1.0) for i in range(1, 5)])
        CM.new_operator('sb', 'anomalous', [(i, i+4+no, 1.0) for i in range(1, 5)] + [(i+4, i+no, 1.0) for i in range(1, 5)])
    else:
        CM = pyqcm.cluster_model(4, generators=gen, name='CM_'+name)
    clus = pyqcm.cluster(CM, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)))
    model = pyqcm.lattice_model('L_'+name, clus, [(2, 0, 0), (0, 2, 0)])
    model.interaction_operator('U')
    model.hopping_operator('t', (1, 0, 0), -1)
    model.hopping_operator('t', (0, 1, 0), -1)
    if bath:
        return model
    model.hopping_operator('t2', (1, 1, 0), -1)
    model.hopping_operator('t2', (-1, 1, 0), -1)
    model.hopping_operator('hx', [0, 0, 0], -1, tau=0, sigma=1)
    model.hopping_operator('hz', [0, 0, 0], -1, tau=0, sigma=3)
    if 'C4' in name:  # d-wave is not C4-invariant, extended s-wave is
        model.anomalous_operator('Ds', (1, 0, 0), 1)
        model.anomalous_operator('Ds', (0, 1, 0), 1)
    else:
        model.anomalous_operator('D', (1, 0, 0), 1)
        model.anomalous_operator('D', (0, 1, 0), -1)
    return model


def run(name, sym):
    c = cases[name]
    model = build(name, c['gen'] if sym else None, c.get('bath', False))
    model.set_target_sectors(c['sec'])
    model.set_parameters("U=0\n" + c['par'])
    I = pyqcm.model_instance(model)
    E = I.ground_state()[0][0]
    G = [I.cluster_Green_function(z) for z in zs]
    if 'hz' in c['par']:
        G += [I.cluster_Green_function(z, spin_down=True) for z in zs]
    A = {k: v[0] for k, v in I.cluster_averages().items()}
    return dict(E=E, G=G, A=A)


if len(sys.argv) == 4:  # worker mode: <case> <sym|nosym> <output file>
    try:
        res = run(sys.argv[1], sys.argv[2] == 'sym')
    except Exception as e:
        res = dict(err=str(e))
    with open(sys.argv[3], 'wb') as f:
        pickle.dump(res, f)
    sys.exit(0)

ok = True
with tempfile.TemporaryDirectory() as tmp:
    for name in cases:
        R = {}
        for sym in ('sym', 'nosym'):
            out = os.path.join(tmp, f'{name}_{sym}.pkl')
            subprocess.run([sys.executable, os.path.abspath(__file__), name, sym, out], capture_output=True)
            with open(out, 'rb') as f:
                R[sym] = pickle.load(f)
        if 'err' in R['sym'] or 'err' in R['nosym']:
            print(f"{name:16s} ERROR  sym: {R['sym'].get('err')} | nosym: {R['nosym'].get('err')}")
            ok = False
            continue
        s, n = R['sym'], R['nosym']
        dE = abs(s['E'] - n['E'])
        dG = max(np.abs(a - b).max() for a, b in zip(s['G'], n['G']))
        dA = max(abs(s['A'][k] - n['A'][k]) for k in n['A'])
        flag = 'OK' if max(dE, dG, dA) < tol else 'FAIL'
        ok &= flag == 'OK'
        print(f"{name:16s} E={s['E']:+.8f}  |dE|={dE:.1e}  max|dG|={dG:.1e}  max|d<O>|={dA:.1e}  {flag}")

print("ALL OK" if ok else "SOME FAILURES")
sys.exit(0 if ok else 1)
