r"""Computational cost of the shell element matrices and internal forces

Times ``update_KC0``, ``update_fint`` and ``update_KG`` of the
shell elements over a flat plate mesh, with and without a material coordinate
system, for laminates with a different number of plies. It is used to assess
the cost of evaluating the transverse shear stiffness in the element coordinate
system, see ``CHANGELOG.md``.

Run it against the version of pyfe3d to be assessed, e.g.::

    PYTHONPATH=<path to pyfe3d repository> python bench_transverse_shear_cost.py

"""
import json
import sys
import time

import numpy as np

import pyfe3d
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
        Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF)
from pyfe3d.shellprop_utils import laminated_plate

CFRP = (138e9, 9.3e9, 0.3, 4.6e9, 4.6e9, 2.3e9)
STACKS = {
    '1 ply': [0],
    '8 plies': [0, 45, -45, 90, 90, -45, 45, 0],
    '32 plies': [0, 45, -45, 90]*8,
}
ELEMENTS = {
    'Quad4': (Quad4, Quad4Data, Quad4Probe),
    'Quad4R': (Quad4R, Quad4RData, Quad4RProbe),
    'Tria3R': (Tria3R, Tria3RData, Tria3RProbe),
}
NX = NY = 41
REPEAT = 5


def build(element, matdir, prop):
    cls, Data, Probe = ELEMENTS[element]
    data = Data()
    probe = Probe()
    xtmp = np.linspace(0, 1., NX)
    ytmp = np.linspace(0, 1., NY)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(),
                         np.zeros(NX*NY))).T
    x = ncoords.flatten()
    nids = np.arange(NX*NY).reshape(NX, NY)
    n1s = nids[:-1, :-1].flatten()
    n2s = nids[1:, :-1].flatten()
    n3s = nids[1:, 1:].flatten()
    n4s = nids[:-1, 1:].flatten()
    conn = []
    if element == 'Tria3R':
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            conn.append((n1, n2, n3))
            conn.append((n1, n3, n4))
    else:
        conn = list(zip(n1s, n2s, n3s, n4s))
    elems = []
    for k, nodes in enumerate(conn):
        e = cls(probe)
        e.n1, e.n2, e.n3 = nodes[:3]
        e.c1, e.c2, e.c3 = [DOF*n for n in nodes[:3]]
        if len(nodes) == 4:
            e.n4 = nodes[3]
            e.c4 = DOF*nodes[3]
        e.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        e.init_k_KG = k*data.KG_SPARSE_SIZE
        e.update_rotation_matrix(x, *matdir)
        e.update_probe_xe(x)
        elems.append(e)
    N = DOF*NX*NY
    rng = np.random.default_rng(0)
    u = rng.uniform(-1e-4, 1e-4, N)
    arrays = dict(
        KC0r=np.zeros(data.KC0_SPARSE_SIZE*len(elems), dtype=INT),
        KC0c=np.zeros(data.KC0_SPARSE_SIZE*len(elems), dtype=INT),
        KC0v=np.zeros(data.KC0_SPARSE_SIZE*len(elems), dtype=DOUBLE),
        KGr=np.zeros(data.KG_SPARSE_SIZE*len(elems), dtype=INT),
        KGc=np.zeros(data.KG_SPARSE_SIZE*len(elems), dtype=INT),
        KGv=np.zeros(data.KG_SPARSE_SIZE*len(elems), dtype=DOUBLE),
        fint=np.zeros(N, dtype=DOUBLE))
    return elems, x, u, arrays


def timeit(func):
    best = np.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def pin_process():
    # NOTE a single core and a high priority reduce the timing noise
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([p.cpu_affinity()[-1]])
        if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)
    except Exception:
        pass


def main():
    pin_process()
    results = {'pyfe3d': pyfe3d.__file__, 'version': pyfe3d.__version__,
               'num_quads': (NX - 1)*(NY - 1), 'cases': []}
    for element in ELEMENTS:
        for matdir_name, matdir in [('no material direction', ()),
                                    ('material direction at 30 deg',
                                     (np.cos(np.pi/6), np.sin(np.pi/6), 0.))]:
            for stack_name, stack in STACKS.items():
                prop = laminated_plate(stack=stack, plyt=0.125e-3,
                                       laminaprop=CFRP)
                elems, x, u, a = build(element, matdir, prop)
                for e in elems:
                    e.update_probe_ue(u)

                def KC0():
                    for e in elems:
                        e.update_probe_xe(x)
                        e.update_KC0(a['KC0r'], a['KC0c'], a['KC0v'], prop)

                def fint():
                    a['fint'][:] = 0
                    for e in elems:
                        e.update_probe_xe(x)
                        e.update_probe_ue(u)
                        e.update_fint(a['fint'], prop)

                def KG():
                    for e in elems:
                        e.update_probe_xe(x)
                        e.update_probe_ue(u)
                        e.update_KG(a['KGr'], a['KGc'], a['KGv'], prop)

                case = dict(element=element, matdir=matdir_name,
                            stack=stack_name, num_elements=len(elems))
                for name, func in [('KC0', KC0), ('fint', fint), ('KG', KG)]:
                    case[name] = timeit(func)/len(elems)*1e6
                results['cases'].append(case)
                print('%-7s %-29s %-9s KC0 %7.2f  fint %7.2f  KG %7.2f  [us/element]'
                      % (element, matdir_name, stack_name, case['KC0'],
                         case['fint'], case['KG']))
                sys.stdout.flush()
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
