import sys
sys.path.append('..')

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE
from pyfe3d.solver import natural_frequency


def roots_by_sign_change(f, num, Omega_max):
    grid = np.linspace(0.1, Omega_max, 200)
    vals = [f(Omega) for Omega in grid]
    roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-13)
             for i in range(len(grid) - 1) if vals[i]*vals[i + 1] < 0]
    return np.array(roots[:num])


def euler_bernoulli_cantilever(p, num):
    r"""Frequency coefficients ``Omega = omega*sqrt(rho*A*L**4/(E*I))`` of an
    Euler-Bernoulli cantilever under the compressive load ``P``, ``p =
    P*L**2/(E*I)``, roots of the characteristic equation of Bokaian (1988),
    Table 1, clamped-free beam, with ``U = p/2``"""
    U = p/2
    def f(Omega):
        M = np.sqrt(-U + np.sqrt(U**2 + Omega**2))
        N = np.sqrt(U + np.sqrt(U**2 + Omega**2))
        return (Omega**2 - Omega*U*np.sinh(M)*np.sin(N)
                + (2*U**2 + Omega**2)*np.cosh(M)*np.cos(N))/np.cosh(M)
    return roots_by_sign_change(f, num, 1.2*((num - 0.5)*np.pi)**2)


def timoshenko_cantilever(S, r2, p, num):
    r"""Frequency coefficients ``Omega = omega*sqrt(rho*A*L**4/(E*I))`` of a
    Timoshenko cantilever under the compressive load ``P``, ``S =
    kappa*G*A*L**2/(E*I)``, ``r2 = I/(A*L**2)``, ``p = P*L**2/(E*I)``

    Exact solution of Eqs. (5.52) and (5.53) of Abramovich (2017), the
    equations (3.241) of Petyt (2010) with the axial load, written for ``y
    = [v/L, theta, M*L/(E*I), Q*L**2/(E*I)]``, ``Q = kappa*A*G*(v' -
    theta)``, as ``y' = C*y`` along ``x/L``: ``y(1) = expm(C)*y(0)`` with
    ``v = theta = 0`` at the clamp and, at the free end, ``M = 0`` and ``Q -
    P*v' = 0`` (Table 5.3 of Abramovich, 2017) gives a 2x2 characteristic
    determinant, whose roots are bracketed by its changes of sign.
    """
    c = 1/(1 - p/S)
    def det(Omega):
        C = np.array([[0, 1, 0, 1/S],
                      [0, 0, 1, 0],
                      [0, -Omega**2*r2, 0, -1],
                      [-Omega**2*c, 0, p*c, 0]])
        T = expm(C)
        shear = (1 - p/S)*T[3] - p*T[1]
        return T[2, 2]*shear[3] - T[2, 3]*shear[2]
    return roots_by_sign_change(det, num, 1.2*((num - 0.5)*np.pi)**2)


def test_nat_freq_cantilever(refinement=1, mtypes=range(2)):
    r"""First three bending frequencies of a cantilever under a compressive
    tip load equal to half of its Euler load, in the plane of ``Izz``

    The pre-stress comes from a linear static solution under the tip load
    ``P = 0.5*pi**2*E*I/(4*L**2)``, half of the Euler load of the
    cantilever, Eq. (2-4) of Timoshenko, S. P., Gere, J. M., Theory of
    Elastic Stability, 2nd ed., McGraw-Hill, 1961, Sec. 2.1, p. 48; with
    shear deformation the buckling load is ``P_e/(1 + n*P_e/(A*G))``, Eq.
    (2-57), Sec. 2.17, p. 133, ``n = 1.2`` for a rectangular cross section
    (p. 132), i.e. 0.018 % below ``P_e`` for this beam. The buckling load
    of BeamLR is 0.069, 0.026 and 0.020 % above Eq. (2-57) with 20, 49 and
    99 elements, and converges to ``P_e``, without the reduction due to shear
    deformation (0.0021 and 0.0005 % above Eq. (2-4) with 99 and 199
    elements).

    The frequencies under the axial load are compared with the exact ones of
    a Timoshenko beam, with shear deformation and rotary inertia, as BeamLR,
    from the equations of motion (5.52) and (5.53), p. 206, and the
    boundary conditions of a free end of Table 5.3, p. 208, of Abramovich,
    H., Vibration of composite columns and plates, in: Abramovich, H. (ed.),
    Stability and Vibrations of Thin-Walled Composite Structures, Elsevier,
    2017, pp. 191-222, https://doi.org/10.1016/B978-0-08-100410-4.00005-3,
    see :func:`timoshenko_cantilever`. Without axial load the function
    reproduces the exact coefficients of Table 3.6, p. 100, of Petyt, M.,
    Introduction to Finite Element Vibration Analysis, 2nd ed., Cambridge
    University Press, 2010, to 0.008 %; without shear deformation and rotary
    inertia, the roots of the characteristic equation of the clamped-free
    beam of Table 1, p. 51, of Bokaian, A., Natural frequencies of beams
    under compressive axial loads, J. Sound Vib., 126(1), 1988, pp. 49-65,
    https://doi.org/10.1016/0022-460X(88)90397-5:
    ``Omega**2 - Omega*U*sinh(M)*sin(N) + (2*U**2 + Omega**2)*cosh(M)*cos(N)
    = 0``, ``U = P*L**2/(2*E*I)``, ``M, N = (-+U + (U**2 +
    Omega**2)**0.5)**0.5``. For this beam, ``L/r = 208``, shear deformation
    and rotary inertia lower the three frequencies by 0.028, 0.160 and 0.371
    % with respect to these Euler-Bernoulli ones.

    BeamLR converges quadratically. With 49 elements the frequencies of the
    consistent mass matrix are 0.020, 0.11 and 0.30 % above the Timoshenko
    reference, with 99 elements 0.015, 0.035 and 0.081 %, with 20 elements
    0.05, 0.64 and 1.79 %; those of the lumped mass matrix are 0.002, 0.002
    and 0.036 % above with 49 elements, 0.05 to 0.18 % off with 20
    elements. With 199 elements both mass matrices are still 0.009 to 0.027
    % above the reference, consistently with the buckling load converging
    to ``P_e`` instead of Eq. (2-57).
    """
    L = 3
    E = 203.e9 # Pa
    rho = 7.83e3 # kg/m3
    b = 0.05 # m
    h = 0.05 # m
    A = h*b
    Izz = b*h**3/12
    scf = 5/6.
    G = scf*E/2/(1+0.3)

    # NOTE Euler load, Eq. (2-4) of Timoshenko and Gere (1961), and with
    #      shear deformation, Eq. (2-57)
    Pe = np.pi**2*E*Izz/(4*L**2)
    Pcr_shear = Pe/(1 + Pe/(G*A))
    load_fraction = 0.5
    p_load = load_fraction*Pe*L**2/(E*Izz)

    # NOTE exact coefficients of Table 3.6 of Petyt (2010), kappa = 2/3,
    #      E/G = 8/3, r_z/L = 0.08 and 0.02
    for rL, coeffs in [(0.08, [3.284, 15.488, 34.301]),
                       (0.02, [3.500, 21.353, 57.474])]:
        assert np.allclose(timoshenko_cantilever(2/3/(8/3)/rL**2, rL**2, 0, 3),
                           coeffs, rtol=1e-4)
    # NOTE the Euler-Bernoulli limit is the characteristic equation of
    #      Bokaian (1988)
    coeff_eb = euler_bernoulli_cantilever(p_load, 3)
    assert np.allclose(timoshenko_cantilever(np.inf, 0, p_load, 3), coeff_eb,
                       rtol=1e-8)
    scale = np.sqrt(E*Izz/(rho*A*L**4))
    omega123_eb = coeff_eb*scale
    omega123 = timoshenko_cantilever(G*A*L**2/(E*Izz), Izz/(A*L**2), p_load, 3)*scale

    for mtype in mtypes:
        print('mtype', mtype)
        n = 50*refinement

        x = np.linspace(0, L, n)
        y = np.ones_like(x)

        ncoords = np.vstack((x, y, np.zeros_like(x))).T
        nids = 1 + np.arange(ncoords.shape[0])
        nid_pos = dict(zip(nids, np.arange(len(nids))))

        n1s = nids[0:-1]
        n2s = nids[1:]

        num_elements = len(n1s)
        print('num_elements', num_elements)

        p = BeamLRProbe()
        data = BeamLRData()

        KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*n
        print('num_DOF', N)

        prop = BeamProp()
        prop.A = A
        prop.E = E
        prop.G = G
        prop.Izz = Izz
        prop.intrho = rho*A
        prop.intrhoy2 = rho*Izz

        ncoords_flatten = ncoords.flatten()

        beams = []
        init_k_KC0 = 0
        init_k_KG = 0
        init_k_M = 0
        for n1, n2 in zip(n1s, n2s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            beam = BeamLR(p)
            beam.init_k_KC0 = init_k_KC0
            beam.init_k_KG = init_k_KG
            beam.init_k_M = init_k_M
            beam.n1 = n1
            beam.n2 = n2
            beam.c1 = DOF*pos1
            beam.c2 = DOF*pos2
            beam.update_rotation_matrix(1., 1., 0, ncoords_flatten)
            beam.update_probe_xe(ncoords_flatten)
            beam.update_KC0(KC0r, KC0c, KC0v, prop)
            beam.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            beams.append(beam)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_KG += data.KG_SPARSE_SIZE
            init_k_M += data.M_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

        print('sparse KC0 and M created')

        bk = np.zeros(N, dtype=bool)
        check = np.isclose(x, 0.)
        bk[0::DOF] = check
        bk[1::DOF] = check
        bk[2::DOF] = check
        bk[5::DOF] = check
        bk[3::DOF] = True
        bk[4::DOF] = True
        bu = ~bk

        # NOTE compressive tip load, half of the Euler load
        fext = np.zeros(N)
        load = -load_fraction*Pe
        check = np.isclose(x, L)
        fext[0::DOF][check] = load

        Kuu = KC0[bu, :][:, bu]
        Muu = M[bu, :][:, bu]
        fextu = fext[bu]

        PREC = np.linalg.norm(1/Kuu.diagonal())
        uu = spsolve(PREC*Kuu, PREC*fextu)
        u = np.zeros(N)
        u[bu] = uu

        for beam in beams:
            beam.update_probe_ue(u)
            beam.update_KG(KGr, KGc, KGv, prop)
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
        KGuu = KG[bu, :][:, bu]
        print('sparse KG created')

        eigvals, eigvecsu = eigsh(A=Kuu, k=1, which='SM', M=KGuu,
                tol=1e-9, sigma=1., mode='buckling')
        load_mult = -eigvals
        Pcr = -load_mult[0]*load
        print('linear buckling Pcr=', Pcr, 'Pe=', Pe, 'Pcr_shear=', Pcr_shear)
        assert np.isclose(Pcr, Pcr_shear, rtol=1e-3)

        num_eigenvalues = 3
        # NOTE pyfe3d.solver.natural_frequency equilibrates the diagonal
        #      before calling the eigensolver and returns the circular
        #      frequencies already sorted
        omegan, eigvecsu = natural_frequency(Kuu + KGuu, Muu, num_eigvalues=num_eigenvalues, tol=1e-3)

        print('Euler-Bernoulli omega123 with pre-stress', omega123_eb)
        print('Timoshenko omega123 with pre-stress', omega123)
        print('Numerical omega123', omegan)
        print()
        rtol = {0: 5e-3, 1: 1e-3}[mtype]
        assert np.allclose(omega123, omegan, rtol=rtol)

if __name__ == '__main__':
    test_nat_freq_cantilever(refinement=1)
