"""Consistent mass matrix of Truss from the kinetic energy

The displacements of a point ``(y, z)`` of the rigid cross section are
``u_x = u``, ``u_y = v - z*r_x`` and ``u_z = w + y*r_x``, with ``u, v, w, r_x``
interpolated linearly, such that the kinetic energy couples ``v`` with ``r_x``
through ``-intrhoz`` and ``w`` with ``r_x`` through ``+intrhoy``, as in
BeamLR and BeamC. The matrix of the element is compared with an independent
Gauss quadrature of the kinetic energy, transformed to global coordinates.
"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import Truss, TrussData, TrussProbe, DOF, INT, DOUBLE


def test_truss_consistent_mass_kinetic_energy():
    prop = BeamProp()
    prop.A = 0.01
    prop.E = 1.
    prop.intrho = 3.
    prop.intrhoy = 0.2
    prop.intrhoz = -0.7
    prop.intrhoy2 = 0.05
    prop.intrhoz2 = 0.08
    # an element not aligned with the global axes
    ncoords = np.array([0.1, -0.2, 0.3, 1.3, 0.7, -0.4])
    data = TrussData()
    probe = TrussProbe()
    truss = Truss(probe)
    truss.c1, truss.c2 = 0, DOF
    truss.update_rotation_matrix(ncoords)
    truss.update_probe_xe(ncoords)
    Mr = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE, dtype=DOUBLE)
    truss.update_M(Mr, Mc, Mv, prop, mtype=0)
    M = coo_matrix((Mv, (Mr, Mc)), shape=(12, 12)).toarray()

    L = truss.length
    Me = np.zeros((12, 12))
    points, weights = np.polynomial.legendre.leggauss(3)
    for p, w in zip(points, weights):
        N1, N2 = (1 - p)/2, (1 + p)/2
        Nu = np.zeros(12); Nu[[0, 6]] = N1, N2
        Nv = np.zeros(12); Nv[[1, 7]] = N1, N2
        Nw = np.zeros(12); Nw[[2, 8]] = N1, N2
        Nrx = np.zeros(12); Nrx[[3, 9]] = N1, N2
        wL = w*L/2
        Me += wL*(prop.intrho*(np.outer(Nu, Nu) + np.outer(Nv, Nv)
                               + np.outer(Nw, Nw))
                  - prop.intrhoz*(np.outer(Nv, Nrx) + np.outer(Nrx, Nv))
                  + prop.intrhoy*(np.outer(Nw, Nrx) + np.outer(Nrx, Nw))
                  + (prop.intrhoy2 + prop.intrhoz2)*np.outer(Nrx, Nrx))
    Rn = np.array([[truss.r11, truss.r12, truss.r13],
                   [truss.r21, truss.r22, truss.r23],
                   [truss.r31, truss.r32, truss.r33]])
    R = np.zeros((12, 12))
    for i in range(4):
        R[3*i:3*i+3, 3*i:3*i+3] = Rn
    Mref = R @ Me @ R.T
    assert np.allclose(M, Mref, rtol=1e-12, atol=1e-13*abs(Mref).max())


if __name__ == '__main__':
    test_truss_consistent_mass_kinetic_energy()
