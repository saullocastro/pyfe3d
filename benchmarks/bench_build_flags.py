r"""Computational cost of the element methods, to assess the build flags

Times ``update_KC0``, ``update_M``, ``update_KG``, ``update_fint`` with
``nonlinear=1`` and ``update_KCNL`` of the shell elements over a flat plate
mesh, called from a Python loop as in normal use. It is used to assess the
Cython directives and the compiler flags of ``setup.py``, see
``CHANGELOG.md``.

Run it against the build of pyfe3d to be assessed, e.g.::

    PYTHONPATH=<path to pyfe3d repository> python bench_build_flags.py

"""
import json
import sys

import numpy as np

import pyfe3d
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
        Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, Tria3DSG, Tria3DSGData,
        Tria3DSGProbe, INT, DOUBLE, DOF)
from pyfe3d.shellprop_utils import laminated_plate

from bench_transverse_shear_cost import pin_process, timeit, CFRP

ELEMENTS = {
    'Quad4': (Quad4, Quad4Data, Quad4Probe),
    'Quad4R': (Quad4R, Quad4RData, Quad4RProbe),
    'Tria3R': (Tria3R, Tria3RData, Tria3RProbe),
    'Tria3DSG': (Tria3DSG, Tria3DSGData, Tria3DSGProbe),
}
METHODS = ['KC0', 'M', 'KG', 'fint', 'KCNL']
NX = NY = 41


def build(element):
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
    if element.startswith('Tria3'):
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
        e.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
        e.init_k_M = k*data.M_SPARSE_SIZE
        e.update_rotation_matrix(x)
        e.update_probe_xe(x)
        elems.append(e)
    N = DOF*NX*NY
    rng = np.random.default_rng(0)
    u = rng.uniform(-1e-4, 1e-4, N)
    arrays = {}
    for name in ['KC0', 'KG', 'KCNL', 'M']:
        size = getattr(data, '%s_SPARSE_SIZE' % name)*len(elems)
        arrays[name] = (np.zeros(size, dtype=INT), np.zeros(size, dtype=INT),
                        np.zeros(size, dtype=DOUBLE))
    arrays['fint'] = np.zeros(N, dtype=DOUBLE)
    return elems, x, u, arrays


def main():
    pin_process()
    results = {'pyfe3d': pyfe3d.__file__, 'version': pyfe3d.__version__,
               'num_quads': (NX - 1)*(NY - 1), 'cases': []}
    prop = laminated_plate(stack=[0, 45, -45, 90, 90, -45, 45, 0],
                           plyt=0.125e-3, laminaprop=CFRP, rho=1600.)
    for element in ELEMENTS:
        elems, x, u, a = build(element)

        def KC0():
            for e in elems:
                e.update_probe_xe(x)
                e.update_KC0(*a['KC0'], prop)

        def M():
            for e in elems:
                e.update_probe_xe(x)
                e.update_M(*a['M'], prop)

        def KG():
            for e in elems:
                e.update_probe_xe(x)
                e.update_probe_ue(u)
                e.update_KG(*a['KG'], prop)

        def fint():
            a['fint'][:] = 0
            for e in elems:
                e.update_probe_xe(x)
                e.update_probe_ue(u)
                e.update_fint(a['fint'], prop, nonlinear=1)

        def KCNL():
            for e in elems:
                e.update_probe_xe(x)
                e.update_probe_ue(u)
                e.update_KCNL(*a['KCNL'], prop)

        funcs = dict(KC0=KC0, M=M, KG=KG, fint=fint, KCNL=KCNL)
        case = dict(element=element, num_elements=len(elems))
        for name in METHODS:
            case[name] = timeit(funcs[name])/len(elems)*1e6
        results['cases'].append(case)
        print('%-8s ' % element + '  '.join('%s %6.2f' % (name, case[name])
                                            for name in METHODS)
              + '  [us/element]')
        sys.stdout.flush()
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
