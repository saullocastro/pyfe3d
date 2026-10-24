r"""Timoshenko beam benchmarks from the literature, for BeamC and BeamLR

1. Bathe, K. J., Finite Element Procedures, Prentice Hall, Englewood Cliffs,
   NJ, 1996 (ISBN 0-13-301458-4), Section 1.2, p. 5, Eq. (1.2) and Fig. 1.2:
   tip deflection of a cantilever with shear deformation,
   ``delta = W*L**3/(3*E*I) + W*L/(5/6*A*G) = 0.053 cm`` for ``L = 28 cm``,
   ``h = 6 cm``, ``t = 0.4 cm``, ``E = 2e7 N/cm2``, ``nu = 0.3``,
   ``W = 1000 N``.

2. Petyt, M., Introduction to Finite Element Vibration Analysis, 2nd ed.,
   Cambridge University Press, 2010, https://doi.org/10.1017/CBO9780511761195,
   Section 3.9:

   - Eq. (3.236): the shape functions that solve the static equilibrium
     equations (3.226) and (3.227) of the Timoshenko beam are divided by
     ``1 + 3*beta``, ``beta = E*I/(kappa*A*G*a**2)`` with ``a = L/2``, i.e.
     by ``1 + phi``, ``phi = 12*E*I/(kappa*G*A*L**2)``, which is the
     ``1/(1 - alpha)`` of BeamC only with ``alpha = -phi``;
   - Example 3.10 and Table 3.6: cantilever with ``r_z/L = 0.08``,
     ``kappa = 2/3``, ``E/G = 8/3``, with shear deformation and rotary
     inertia; one element gives the frequency coefficients
     ``omega*sqrt(rho*A*L**4/(E*I)) = 3.322`` and ``26.569`` (p. 99-100);
     the exact coefficients are ``3.284, 15.488, 34.301`` and, for ``r_z/L =
     0.02``, ``3.500, 21.353, 57.474``; three elements give the errors
     ``0.12, 2.47, 8.80`` and ``0.02, 0.59, 2.37`` per cent (Table 3.6).

The interdependent interpolation of BeamC solves the static equilibrium
equations, so that one element is exact for a cantilever under a tip force
(case 1 and the deep beam) and reproduces the one-element results of Petyt,
whose element has the same shape functions and the same consistent mass
matrix with rotary inertia, Eq. (3.237). BeamLR, with linear interpolation
and reduced integration, only converges to these values with mesh
refinement.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import (BeamC, BeamCData, BeamCProbe, BeamLR, BeamLRData,
                    BeamLRProbe, DOF, INT, DOUBLE)

BEAMS = {
    'BeamC': (BeamC, BeamCData, BeamCProbe),
    'BeamLR': (BeamLR, BeamLRData, BeamLRProbe),
}


def assemble(name, prop, L, num_elements, mass=False):
    cls, datacls, probecls = BEAMS[name]
    data = datacls()
    probe = probecls()
    x = np.linspace(0, L, num_elements + 1)
    ncoords = np.vstack((x, 0*x, 0*x)).T.flatten()
    N = DOF*(num_elements + 1)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for i in range(num_elements):
        beam = cls(probe)
        beam.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        beam.init_k_M = i*data.M_SPARSE_SIZE
        beam.n1, beam.n2 = i + 1, i + 2
        beam.c1, beam.c2 = DOF*i, DOF*(i + 1)
        beam.update_rotation_matrix(0., 1., 0., ncoords)
        beam.update_probe_xe(ncoords)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        if mass:
            beam.update_M(Mr, Mc, Mv, prop, mtype=0)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()
    return KC0, M


def cantilever_tip(name, prop, L, num_elements):
    """Tip deflections along y and z under unit tip forces"""
    KC0, _ = assemble(name, prop, L, num_elements)
    KC0 = KC0.toarray()
    N = KC0.shape[0]
    bu = np.ones(N, dtype=bool)
    bu[:DOF] = False
    tip = []
    for direction in [1, 2]:
        f = np.zeros(N)
        f[N - DOF + direction] = 1.
        u = np.zeros(N)
        u[bu] = np.linalg.solve(KC0[np.ix_(bu, bu)], f[bu])
        tip.append(u[N - DOF + direction])
    return np.array(tip)


def bathe_prop():
    E = 2.e7 # N/cm2
    nu = 0.3
    h = 6. # cm
    t = 0.4 # cm
    prop = BeamProp()
    prop.A = h*t
    prop.E = E
    # NOTE shear correction factor 5/6 of Eq. (1.2) of Bathe (1996)
    prop.G = 5/6*E/(2*(1 + nu))
    prop.Izz = prop.Iyy = t*h**3/12
    prop.J = prop.Izz + prop.Iyy
    return prop


@pytest.mark.parametrize('name, num_elements, rtol', [
    ('BeamC', 1, 1e-10),
    ('BeamC', 3, 1e-10),
    # NOTE BeamLR converges quadratically; error 0.02 % with 32 elements
    ('BeamLR', 100, 1e-4),
    ])
def test_bathe_cantilever_shear_deflection(name, num_elements, rtol):
    L = 28. # cm, L + r_N in Fig. 1.2(b) of Bathe (1996)
    W = 1000. # N
    prop = bathe_prop()
    delta = W*cantilever_tip(name, prop, L, num_elements)
    # Eq. (1.2) of Bathe (1996)
    delta_ref = W*L**3/(3*prop.E*prop.Izz) + W*L/(prop.A*prop.G)
    print(name, num_elements, delta, delta_ref)
    assert np.allclose(delta, delta_ref, rtol=rtol)
    # NOTE published value, given with two significant digits
    assert np.allclose(delta, 0.053, atol=0.0005)


@pytest.mark.parametrize('name, num_elements, rtol', [
    ('BeamC', 1, 1e-10),
    ('BeamLR', 200, 1e-4),
    ])
def test_deep_cantilever_shear_deflection(name, num_elements, rtol):
    """Cantilever as long as deep, where the shear deflection dominates"""
    E = 1.
    nu = 0.3
    h = 1.
    L = 1.
    prop = BeamProp()
    prop.A = h*h
    prop.E = E
    prop.G = E/(2*(1 + nu))
    prop.Izz = prop.Iyy = h**4/12
    prop.J = 2*h**4/12
    tip = cantilever_tip(name, prop, L, num_elements)
    tip_ref = L**3/(3*E*prop.Izz) + L/(prop.G*prop.A)
    print(name, num_elements, tip, tip_ref)
    assert np.allclose(tip, tip_ref, rtol=rtol)


def petyt_coefficients(name, rz_L, num_elements, num_modes=3):
    """Frequency coefficients omega*sqrt(rho*A*L**4/(E*I)), bending in xy"""
    L = 1.
    E = 1.
    rho = 1.
    A = 1.
    kappa = 2/3
    G = 3/8*E # E/G = 8/3
    I = (rz_L*L)**2*A
    prop = BeamProp()
    prop.A = A
    prop.E = E
    prop.G = kappa*G
    prop.Izz = prop.Iyy = I
    prop.J = 2*I
    prop.intrho = rho*A
    prop.intrhoy2 = rho*I
    prop.intrhoz2 = rho*I
    K, M = assemble(name, prop, L, num_elements, mass=True)
    N = K.shape[0]
    # NOTE only v and r_z of the free nodes, bending in the xy plane
    bu = np.zeros(N, dtype=bool)
    bu[DOF + 1::DOF] = True
    bu[DOF + 5::DOF] = True
    Kuu = K[bu, :][:, bu].toarray()
    Muu = M[bu, :][:, bu].toarray()
    eigvals = eigh(Kuu, Muu, eigvals_only=True)
    omega = np.sqrt(eigvals[:num_modes])
    return omega*np.sqrt(rho*A*L**4/(E*I))


def test_petyt_one_element():
    # Example 3.10 of Petyt (2010), p. 99-100
    coef = petyt_coefficients('BeamC', 0.08, 1, num_modes=2)
    print(coef)
    # NOTE published with three decimals
    assert np.allclose(coef, [3.322, 26.569], atol=0.0005, rtol=0)


@pytest.mark.parametrize('rz_L, exact, errors', [
    (0.08, [3.284, 15.488, 34.301], [0.12, 2.47, 8.80]),
    (0.02, [3.500, 21.353, 57.474], [0.02, 0.59, 2.37]),
    ])
def test_petyt_table_3_6(rz_L, exact, errors):
    exact = np.array(exact)
    # three elements, Table 3.6 of Petyt (2010)
    coef = petyt_coefficients('BeamC', rz_L, 3)
    err = 100*(coef - exact)/exact
    print('3 elements', coef, err)
    # NOTE the errors are published with two decimals and the exact values
    #      with three decimals, which bounds the difference of the errors
    tol = 0.005 + 100*0.0005/exact
    assert np.all(abs(err - errors) <= tol)
    # NOTE converged values: with 160 BeamC or 640 BeamLR elements both
    #      elements agree to 1e-5 and differ from the published exact
    #      values by at most 8e-5 (21.3547 against 21.353), in the same
    #      way for both elements
    coef = petyt_coefficients('BeamC', rz_L, 160)
    print('BeamC 160 elements', coef)
    assert np.allclose(coef, exact, rtol=1e-4)
    coef = petyt_coefficients('BeamLR', rz_L, 640)
    print('BeamLR 640 elements', coef)
    assert np.allclose(coef, exact, rtol=1e-4)


if __name__ == '__main__':
    test_bathe_cantilever_shear_deflection('BeamC', 1, 1e-10)
    test_petyt_one_element()
    test_petyt_table_3_6(0.08, [3.284, 15.488, 34.301], [0.12, 2.47, 8.80])
