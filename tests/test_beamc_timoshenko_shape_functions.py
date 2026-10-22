"""Consistent Timoshenko shape functions of BeamC

The cubic interpolation of the deflection with the shear parameter, with
``beta = 1/(1 + 12*E*I/(G*A*L**2))``, satisfies the homogeneous equilibrium of
the Timoshenko beam, so that one element gives the exact tip deflection of a
cantilever under a tip force, ``P*L**3/(3*E*I) + P*L/(G*A)``. The geometric
stiffness and mass matrices must use the same interpolation; they are
compared here with an independent Gauss quadrature of the same integrals.
"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE

E = 1.
NU = 0.3
B = 0.3
L = 1.
RHO = 2.


def get_prop():
    prop = BeamProp()
    prop.A = B*B
    prop.E = E
    prop.G = E/2/(1 + NU)
    prop.Izz = prop.Iyy = B**4/12
    prop.J = 2*B**4/12
    prop.intrho = RHO*prop.A
    prop.intrhoy2 = RHO*prop.Izz
    prop.intrhoz2 = RHO*prop.Iyy
    return prop


def cantilever(num_elements):
    prop = get_prop()
    data = BeamCData()
    probe = BeamCProbe()
    x = np.linspace(0, L, num_elements + 1)
    ncoords = np.vstack((x, 0*x, 0*x)).T.flatten()
    N = DOF*(num_elements + 1)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for i in range(num_elements):
        beam = BeamC(probe)
        beam.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        beam.n1, beam.n2 = i + 1, i + 2
        beam.c1, beam.c2 = DOF*i, DOF*(i + 1)
        beam.update_rotation_matrix(0., 1., 0., ncoords)
        beam.update_probe_xe(ncoords)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    bu = np.ones(N, dtype=bool)
    bu[:DOF] = False
    tip = []
    for direction in [1, 2]:
        fext = np.zeros(N)
        fext[N - DOF + direction] = 1.
        u = np.zeros(N)
        u[bu] = np.linalg.solve(KC0[np.ix_(bu, bu)], fext[bu])
        tip.append(u[N - DOF + direction])
    return np.array(tip), prop


def test_one_element_exact_tip_deflection():
    tip, prop = cantilever(1)
    I = prop.Izz
    exact = L**3/(3*E*I) + L/(prop.G*prop.A)
    print('one element', tip, 'exact', exact)
    assert np.allclose(tip, exact, rtol=1e-10)
    tip, prop = cantilever(4)
    assert np.allclose(tip, exact, rtol=1e-10)


def shape_functions(xi, alpha):
    """Rows of v, r_z, w, r_y in element coordinates, `alpha = -12EI/(GAL^2)`"""
    beta = 1/(1 - alpha)
    Hv1 = beta*(2*xi**3 - 3*xi**2 + alpha*xi + 1 - alpha)
    Hv2 = beta*(-2*xi**3 + 3*xi**2 - alpha*xi)
    Hr1 = L*beta*(xi**3 + (alpha/2 - 2)*xi**2 + (1 - alpha/2)*xi)
    Hr2 = L*beta*(xi**3 - (1 + alpha/2)*xi**2 + (alpha/2)*xi)
    Gv1 = 6*beta/L*(xi**2 - xi)
    Gr1 = beta*(3*xi**2 + (alpha - 4)*xi + 1 - alpha)
    Gr2 = beta*(3*xi**2 - (alpha + 2)*xi)
    # derivatives with respect to x
    dHv1 = beta*(6*xi**2 - 6*xi + alpha)/L
    dHr1 = beta*(3*xi**2 + (alpha - 4)*xi + 1 - alpha/2)
    dHr2 = beta*(3*xi**2 - (2 + alpha)*xi + alpha/2)
    Nu = np.zeros(12); Nu[[0, 6]] = [1 - xi, xi]
    Nrx = np.zeros(12); Nrx[[3, 9]] = [1 - xi, xi]
    Nv = np.zeros(12); Nv[[1, 5, 7, 11]] = [Hv1, Hr1, Hv2, Hr2]
    Nrz = np.zeros(12); Nrz[[1, 5, 7, 11]] = [Gv1, Gr1, -Gv1, Gr2]
    Nw = np.zeros(12); Nw[[2, 4, 8, 10]] = [Hv1, -Hr1, Hv2, -Hr2]
    Nry = np.zeros(12); Nry[[2, 4, 8, 10]] = [-Gv1, Gr1, Gv1, Gr2]
    Nvx = np.zeros(12); Nvx[[1, 5, 7, 11]] = [dHv1, dHr1, -dHv1, dHr2]
    Nwx = np.zeros(12); Nwx[[2, 4, 8, 10]] = [dHv1, -dHr1, -dHv1, -dHr2]
    return Nu, Nv, Nw, Nrx, Nry, Nrz, Nvx, Nwx


def test_KG_M_same_interpolation():
    prop = get_prop()
    alpha = -12*prop.E*prop.Izz/(prop.G*prop.A*L**2)
    data = BeamCData()
    probe = BeamCProbe()
    ncoords = np.array([0., 0., 0., L, 0., 0.])
    beam = BeamC(probe)
    beam.c1, beam.c2 = 0, DOF
    beam.update_rotation_matrix(0., 1., 0., ncoords)
    beam.update_probe_xe(ncoords)
    # uniform axial force N = E*A*delta/L
    delta = 1.e-3
    ue = np.zeros(12)
    ue[6] = delta
    beam.update_probe_ue(ue)
    KGr = np.zeros(data.KG_SPARSE_SIZE, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE, dtype=DOUBLE)
    beam.update_KG(KGr, KGc, KGv, prop)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(12, 12)).toarray()
    Mr = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE, dtype=DOUBLE)
    beam.update_M(Mr, Mc, Mv, prop, mtype=0)
    M = coo_matrix((Mv, (Mr, Mc)), shape=(12, 12)).toarray()

    Nforce = prop.E*prop.A*delta/L
    KGref = np.zeros((12, 12))
    Mref = np.zeros((12, 12))
    points, weights = np.polynomial.legendre.leggauss(6)
    for p, w in zip(points, weights):
        xi = (p + 1)/2
        Nu, Nv, Nw, Nrx, Nry, Nrz, Nvx, Nwx = shape_functions(xi, alpha)
        wL = w*L/2
        KGref += wL*Nforce*(np.outer(Nvx, Nvx) + np.outer(Nwx, Nwx))
        Mref += wL*(prop.intrho*(np.outer(Nu, Nu) + np.outer(Nv, Nv)
                                 + np.outer(Nw, Nw))
                    + prop.intrhoy2*np.outer(Nrz, Nrz)
                    + prop.intrhoz2*np.outer(Nry, Nry)
                    + (prop.intrhoy2 + prop.intrhoz2)*np.outer(Nrx, Nrx))
    assert np.allclose(KG, KGref, rtol=1e-12, atol=1e-14*abs(KGref).max())
    assert np.allclose(M, Mref, rtol=1e-12, atol=1e-14*abs(Mref).max())


if __name__ == '__main__':
    test_one_element_exact_tip_deflection()
    test_KG_M_same_interpolation()
