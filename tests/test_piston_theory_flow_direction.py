"""Piston-theory matrices of Quad4 and Quad4R on an element rotated in plane

The aerodynamic matrices are the matrices of the load along the element
normal, ``f = beta*KA_beta*u + gamma*KA_gamma*u + mu*CA*udot``, of the
pressure law ``q = -beta*w_{,X} + gamma*w - mu*wdot`` for a flow along the
global ``X`` axis. With a normal displacement that is linear on the element,
the load is the consistent load of a constant pressure, whose resultant is
the pressure times the area. A displacement varying only along ``Y`` has no
slope along the flow and must give no load in ``beta``, whatever the
orientation of the element axes.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, DOF, INT, DOUBLE)

ELEMENTS = {
    'Quad4': (Quad4, Quad4Data, Quad4Probe),
    'Quad4R': (Quad4R, Quad4RData, Quad4RProbe),
}


def matrices(name, thetadeg):
    cls, datacls, probecls = ELEMENTS[name]
    data = datacls()
    probe = probecls()
    # distorted quadrilateral, counterclockwise, normal along +Z
    xy = np.array([[0., 0.], [1.1, -0.1], [1.2, 0.9], [-0.1, 1.0]])
    t = np.deg2rad(thetadeg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = xy @ rot.T
    ncoords = np.column_stack((xy, np.zeros(4)))
    x = ncoords.flatten()
    quad = cls(probe)
    quad.c1, quad.c2, quad.c3, quad.c4 = 0, DOF, 2*DOF, 3*DOF
    quad.update_rotation_matrix(x)
    quad.update_probe_xe(x)
    out = []
    for size, method in [(data.KA_BETA_SPARSE_SIZE, quad.update_KA_beta),
                         (data.KA_GAMMA_SPARSE_SIZE, quad.update_KA_gamma),
                         (data.CA_SPARSE_SIZE, quad.update_CA)]:
        r = np.zeros(size, dtype=INT)
        c = np.zeros(size, dtype=INT)
        v = np.zeros(size, dtype=DOUBLE)
        method(r, c, v)
        out.append(coo_matrix((v, (r, c)), shape=(4*DOF, 4*DOF)).toarray())
    # area of the quadrilateral, shoelace formula
    X, Y = xy[:, 0], xy[:, 1]
    area = 0.5*abs(X @ np.roll(Y, -1) - Y @ np.roll(X, -1))
    return out, ncoords, area


@pytest.mark.parametrize('name', ['Quad4', 'Quad4R'])
@pytest.mark.parametrize('thetadeg', [0., 30., 115.])
def test_piston_theory_flow_along_X(name, thetadeg):
    (KA_beta, KA_gamma, CA), ncoords, area = matrices(name, thetadeg)
    u = np.zeros(4*DOF)
    # w = Y, no slope along the flow
    u[2::DOF] = ncoords[:, 1]
    f = KA_beta @ u
    assert np.allclose(f, 0, atol=1e-12)
    # w = X, unit slope along the flow: load -1 per unit area
    u[2::DOF] = ncoords[:, 0]
    f = KA_beta @ u
    assert np.isclose(f[2::DOF].sum(), -area, rtol=1e-12)
    assert np.allclose(f[0::DOF], 0) and np.allclose(f[1::DOF], 0)
    # w = 1: +1 per unit area in gamma, -1 per unit velocity in CA
    u[2::DOF] = 1.
    assert np.isclose((KA_gamma @ u)[2::DOF].sum(), area, rtol=1e-12)
    assert np.isclose((CA @ u)[2::DOF].sum(), -area, rtol=1e-12)


if __name__ == '__main__':
    test_piston_theory_flow_along_X('Quad4', 30.)
