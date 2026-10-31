import sys
sys.path.append('..')

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import Spring, SpringData, SpringProbe
from pyfe3d.solver import natural_frequency
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE


def euler_bernoulli_cantilever(num):
    r"""Frequency coefficients ``omega*sqrt(rho*A*L**4/(E*I)) = (beta*L)**2``
    of an Euler-Bernoulli cantilever, ``beta*L`` being the roots of
    ``cos(beta*L)*cosh(beta*L) + 1 = 0``, one in each interval
    ``((k - 1)*pi, k*pi)``"""
    f = lambda x: np.cos(x)*np.cosh(x) + 1
    return np.array([brentq(f, (k - 1)*np.pi, k*np.pi, xtol=1e-14)**2
                     for k in range(1, num + 1)])


def timoshenko_cantilever(S, r2, num):
    r"""Frequency coefficients ``omega*sqrt(rho*A*L**4/(E*I))`` of a
    Timoshenko cantilever, ``S = kappa*G*A*L**2/(E*I)``, ``r2 = I/(A*L**2)``

    Exact solution of Eqs. (3.241) of Petyt (2010) with the boundary
    conditions (3.242), written for ``y = [v/L, theta, M*L/(E*I),
    Q*L**2/(E*I)]``, ``Q = kappa*A*G*(v' - theta)``, as ``y' = C*y`` along
    ``x/L``: ``y(1) = expm(C)*y(0)`` with ``v = theta = 0`` at the clamp and
    ``M = Q = 0`` at the free end gives a 2x2 characteristic determinant,
    whose roots are bracketed by its changes of sign.
    """
    def det(Omega):
        C = np.array([[0, 1, 0, 1/S],
                      [0, 0, 1, 0],
                      [0, -Omega**2*r2, 0, -1],
                      [-Omega**2, 0, 0, 0]])
        T = expm(C)
        return T[2, 2]*T[3, 3] - T[2, 3]*T[3, 2]
    grid = np.linspace(0.1, 1.2*((num - 0.5)*np.pi)**2, 200)
    vals = [det(Omega) for Omega in grid]
    roots = [brentq(det, grid[i], grid[i + 1], xtol=1e-13)
             for i in range(len(grid) - 1) if vals[i]*vals[i + 1] < 0]
    return np.array(roots[:num])


def test_nat_freq_cantilever(refinement=1, mtypes=range(2)):
    r"""First three bending frequencies of a cantilever clamped through a
    stiff spring, in both planes of its square cross section

    The beam is clamped through a :class:`.Spring` of stiffness ``1e9`` in
    all six degrees-of-freedom, 8.5e4 times the tip stiffness
    ``3*E*I/L**3`` and 2.8e4 times the rotational stiffness ``E*I/L`` of the
    beam. The flexibility of the spring lowers the converged frequencies by
    0.0085, 0.025 and 0.055 % with respect to those of the clamped beam
    (measured with 49 and 99 beam elements and the consistent mass matrix).

    The expected frequencies are the exact ones of a Timoshenko beam, with
    shear deformation and rotary inertia, as BeamC: the frequency
    coefficients ``omega*sqrt(rho*A*L**4/(E*I))`` of
    :func:`timoshenko_cantilever`, from Eqs. (3.241) and (3.242) of Petyt,
    M., Introduction to Finite Element Vibration Analysis, 2nd ed.,
    Cambridge University Press, 2010, Sec. 3.9, Example 3.10, p. 99. The
    function reproduces the exact coefficients of Petyt's Table 3.6, p. 100,
    to 0.008 %.

    The Euler-Bernoulli frequencies, Eq. (3.31) of Petyt (2010), p. 52,
    ``omega_r = (beta_r*L)**2*sqrt(E*I/(rho*A*L**4))``, are those of the
    roots ``beta_r*L = 1.87510, 4.69409, 7.85476`` of ``cos(beta*L)*cosh(
    beta*L) = -1``, Table 5.1, No. 3, p. 201, of Abramovich, H.,
    Vibration of composite columns and plates, in: Abramovich, H. (ed.),
    Stability and Vibrations of Thin-Walled Composite Structures, Elsevier,
    2017, pp. 191-222, https://doi.org/10.1016/B978-0-08-100410-4.00005-3,
    where the third root is misprinted as 7.85340; the coefficients
    ``(beta_r*L)**2 = 3.516, 22.034, 61.697`` are the exact solution of
    Petyt's Table 10.2, p. 303. For this beam, ``L/r = 208``, shear
    deformation and rotary inertia lower the three frequencies by 0.022,
    0.154 and 0.366 % with respect to the Euler-Bernoulli ones.

    With 49 beam elements the frequencies of the consistent mass matrix are
    0.0085, 0.025 and 0.055 % below the Timoshenko reference of the clamped
    beam, with 24 elements 0.0085 to 0.052 %; those of the lumped mass
    matrix are 0.028, 0.091 and 0.163 % below with 49 elements, 0.088 to
    0.51 % below with 24 elements.
    """
    L = 3
    E = 203.e9 # Pa
    rho = 7.83e3 # kg/m3
    b = 0.05 # m
    h = 0.05 # m
    A = h*b
    Izz = b*h**3/12
    Iyy = b**3*h/12
    scf = 5/6.
    G = scf*E/2/(1+0.3)

    # NOTE exact coefficients of Table 10.2 of Petyt (2010)
    coeff_eb = euler_bernoulli_cantilever(3)
    assert np.allclose(coeff_eb, [3.516, 22.034, 61.697], atol=5e-4)
    # NOTE exact coefficients of Table 3.6 of Petyt (2010), kappa = 2/3,
    #      E/G = 8/3, r_z/L = 0.08 and 0.02
    for rL, coeffs in [(0.08, [3.284, 15.488, 34.301]),
                       (0.02, [3.500, 21.353, 57.474])]:
        assert np.allclose(timoshenko_cantilever(2/3/(8/3)/rL**2, rL**2, 3),
                           coeffs, rtol=1e-4)
    scale = np.sqrt(E*Izz/(rho*A*L**4))
    omega123_eb = coeff_eb*scale
    omega123 = timoshenko_cantilever(G*A*L**2/(E*Izz), Izz/(A*L**2), 3)*scale

    for mtype in mtypes:
        print('mtype', mtype)
        n = 50*refinement

        x = np.zeros(n+1)
        x[0] = -L/100
        x[1:] = np.linspace(0, L, n)
        y = np.ones_like(x)

        ncoords = np.vstack((x, y, np.zeros_like(x))).T
        nids = 1 + np.arange(ncoords.shape[0])
        nid_pos = dict(zip(nids, np.arange(len(nids))))

        num_elements = len(ncoords) - 1
        print('num_elements', num_elements)

        p = BeamCProbe()
        beamdata = BeamCData()

        springdata = SpringData()
        springprobe = SpringProbe()

        KC0r = np.zeros(springdata.KC0_SPARSE_SIZE*1 + beamdata.KC0_SPARSE_SIZE*(num_elements-1), dtype=INT)
        KC0c = np.zeros(springdata.KC0_SPARSE_SIZE*1 + beamdata.KC0_SPARSE_SIZE*(num_elements-1), dtype=INT)
        KC0v = np.zeros(springdata.KC0_SPARSE_SIZE*1 + beamdata.KC0_SPARSE_SIZE*(num_elements-1), dtype=DOUBLE)
        Mr = np.zeros(beamdata.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mc = np.zeros(beamdata.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mv = np.zeros(beamdata.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*(n + 1)
        print('num_DOF', N)

        prop = BeamProp()
        prop.A = A
        prop.E = E
        prop.G = G
        prop.Izz = Izz
        prop.Iyy = Iyy
        prop.J = Iyy + Izz
        prop.intrho = rho*A
        prop.intrhoy2 = rho*Izz
        prop.intrhoz2 = rho*Iyy

        ncoords_flatten = ncoords.flatten()

        init_k_KC0 = 0
        init_k_M = 0

        spring = Spring(springprobe)
        spring.init_k_KC0 = init_k_KC0
        spring.n1 = 0
        spring.n2 = 1
        spring.c1 = 0*DOF
        spring.c2 = 1*DOF
        spring.kxe = spring.kye = spring.kze = 1e9
        spring.krxe = spring.krye = spring.krze = 1e9
        spring.update_rotation_matrix(1, 0, 0, 1, 1, 0)
        spring.update_KC0(KC0r, KC0c, KC0v)
        init_k_KC0 += springdata.KC0_SPARSE_SIZE

        n1s = nids[1:-1]
        n2s = nids[2:]
        for n1, n2 in zip(n1s, n2s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            beam = BeamC(p)
            beam.init_k_KC0 = init_k_KC0
            beam.init_k_M = init_k_M
            beam.n1 = n1
            beam.n2 = n2
            beam.c1 = DOF*pos1
            beam.c2 = DOF*pos2
            beam.update_rotation_matrix(1., 1., 0, ncoords_flatten)
            beam.update_probe_xe(ncoords_flatten)
            beam.update_KC0(KC0r, KC0c, KC0v, prop)
            beam.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            init_k_KC0 += beamdata.KC0_SPARSE_SIZE
            init_k_M += beamdata.M_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

        print('sparse KC0 and M created')

        bk = np.zeros(N, dtype=bool)
        check = np.isclose(x, -L/100)
        bk[0::DOF][check] = True
        bk[1::DOF][check] = True
        bk[2::DOF][check] = True
        bk[3::DOF][check] = True
        bk[4::DOF][check] = True
        bk[5::DOF][check] = True

        bu = ~bk

        Kuu = KC0[bu, :][:, bu]
        Muu = M[bu, :][:, bu]

        num_eigenvalues = 6
        # NOTE pyfe3d.solver.natural_frequency equilibrates the diagonal
        #      before calling the eigensolver and returns the circular
        #      frequencies already sorted
        omegan, eigvecsu = natural_frequency(Kuu, Muu, num_eigvalues=num_eigenvalues, tol=1e-4)

        print('Euler-Bernoulli omega123', omega123_eb)
        print('Timoshenko omega123', omega123)
        print('Numerical omega123', omegan)
        print()
        # NOTE both bending planes, square cross section
        rtol = {0: 1e-3, 1: 3e-3}[mtype]
        assert np.allclose(np.repeat(omega123, 2), omegan, rtol=rtol)


if __name__ == '__main__':
    test_nat_freq_cantilever(refinement=1)
