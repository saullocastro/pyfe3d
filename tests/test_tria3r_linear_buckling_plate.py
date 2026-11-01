import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling


def test_tria3r_linear_buckling_plate(plot=False, mode=0, refinement=1):
    r"""Linear buckling of a simply supported plate, ``a/b = 4``, under a
    uniform compression ``N_xx`` from a prestress analysis, ``KG`` from the
    displacements

    The reference is the exact solution of Leissa, A. W., Vibration of
    Plates, NASA SP-160, 1969, Sec. 10.2.1, p. 277, Eq. (10.46), minimised
    over the number ``m`` of half-waves along ``x``:
    ``N_xx,cr = -k_c*pi**2*D/b**2``, ``k_c = (m*b/a + a/(m*b))**2``. With
    ``b/h = 167`` the Mindlin plate buckles 0.02 % below it.

    With the default transverse shear since 0.12.0, the discrete shear gap
    field of ``shear_model = 1``, the buckling load converges to the
    reference, measured with this test (quadrilaterals split in two
    triangles, time on one thread):

        refinement   mesh      P_cr          error    time
        1            30x14     -42550.4     +5.14 %   0.2 s
        2            62x30     -40767.5     +0.73 %   1.2 s
        3            92x44     -40583.2     +0.28 %   3.8 s
        4            124x60    -40511.4     +0.10 %   8.4 s
        6            186x90    -40455.1     -0.04 %  25.7 s

    ``refinement=1`` is compared with the value pre-determined by this test,
    with ``rtol=1e-4`` for the round-off of the eigensolver, and with grade B
    of MacNeal and Harder (1985), p. 14, as the bound against the
    reference; from ``refinement=2`` grade A is asserted against the
    reference alone. Before 0.12.0, with the centroid-sampled shear and
    ``alpha_shear_locking = 0.7``, the load was 4.9 % above the reference
    with 30x14 elements and 13.4 % above with 92x44, see
    :mod:`pyfe3d.tria3r`. The prestress system is solved directly; the
    preconditioned conjugate gradient used before does not converge on it.
    """
    data = Tria3RData()
    probe = Tria3RProbe()
    nx = refinement*31
    ny = refinement*15
    if (nx % 2) == 0:
        nx += 1
    if (ny % 2) == 0:
        ny += 1

    a = 2.0
    b = 0.5

    E = 203.e9 # Pa
    nu = 0.33

    rho = 7.83e3 # kg/m3
    h = 0.003 # m

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T

    x = ncoords[:, 0]
    y = ncoords[:, 1]
    z = ncoords[:, 2]
    ncoords_flatten = ncoords.flatten()

    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    num_elements = len(n1s)*2
    print('num_elements', num_elements)

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)

    trias = []
    init_k_KC0 = 0
    init_k_KG = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        pos4 = nid_pos[n4]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        r4 = ncoords[pos4]

        # first tria
        normal = np.cross(r2 - r1, r3 - r1)[2]
        assert normal > 0
        tria = Tria3R(probe)
        tria.n1 = n1
        tria.n2 = n2
        tria.n3 = n3
        tria.c1 = DOF*nid_pos[n1]
        tria.c2 = DOF*nid_pos[n2]
        tria.c3 = DOF*nid_pos[n3]
        tria.init_k_KC0 = init_k_KC0
        tria.init_k_KG = init_k_KG
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

        # second tria
        normal = np.cross(r3 - r1, r4 - r1)[2]
        assert normal > 0
        tria = Tria3R(probe)
        tria.n1 = n1
        tria.n2 = n3
        tria.n3 = n4
        tria.c1 = DOF*nid_pos[n1]
        tria.c2 = DOF*nid_pos[n3]
        tria.c3 = DOF*nid_pos[n4]
        tria.init_k_KC0 = init_k_KC0
        tria.init_k_KG = init_k_KG
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    bk = np.zeros(N, dtype=bool)
    check = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF] = check
    # constraining u at x = a/2, y = 0,b
    check = isclose(x, a/2.) & (isclose(y, 0.) | isclose(y, b))
    bk[0::DOF] = check
    # constraining v at x = 0,a y = b/2
    check = isclose(y, b/2.) & (isclose(x, 0.) | isclose(x, a))
    bk[1::DOF] = check
    # removing drilling
    bk[5::DOF] = True

    bu = ~bk

    # applying load along u at x=a
    # nodes at vertices get 1/2 of the force
    fext = np.zeros(N)
    ftotal = -1000.
    print('ftotal', ftotal)
    # at x=0
    check = (isclose(x, 0) & ~isclose(y, 0) & ~isclose(y, b))
    fext[0::DOF][check] = -ftotal/(ny - 1)
    check = ((isclose(x, 0) & isclose(y, 0))
            |(isclose(x, 0) & isclose(y, b)))
    fext[0::DOF][check] = -ftotal/(ny - 1)/2
    assert np.isclose(fext.sum(), -ftotal)
    # at x=a
    check = (isclose(x, a) & ~isclose(y, 0) & ~isclose(y, b))
    fext[0::DOF][check] = ftotal/(ny - 1)
    check = ((isclose(x, a) & isclose(y, 0))
            |(isclose(x, a) & isclose(y, b)))
    fext[0::DOF][check] = ftotal/(ny - 1)/2
    assert np.isclose(fext.sum(), 0)

    KC0uu = KC0[bu, :][:, bu]
    fextu = fext[bu]

    uu = spsolve(KC0uu, fextu)
    u = np.zeros(N)
    u[bu] = uu

    print('u extremes', u[0::DOF].min(), u[0::DOF].max())
    print('v extremes', u[1::DOF].min(), u[1::DOF].max())
    print('w extremes', u[2::DOF].min(), u[2::DOF].max())

    if False:
        import matplotlib.pyplot as plt

        plt.gca().set_aspect('equal')
        uplot = u[0::DOF].reshape(nx, ny).T
        levels = np.linspace(uplot.min(), uplot.max(), 300)
        plt.contourf(xmesh, ymesh, uplot, levels=levels)
        plt.colorbar()
        plt.show()
        raise

    for tria in trias:
        tria.update_probe_ue(u) # NOTE update affects the Tria3RProbe class attribute ue
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KG(KGr, KGc, KGv, prop)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('sparse KG created')

    num_eig_lb = max(mode+1, 1)

    # NOTE pyfe3d.solver.linear_buckling equilibrates the diagonal,
    #      estimates the Cayley shift and verifies that no lower load
    #      multiplier was missed, which a hardcoded shift does not
    eigvals, eigvecsu = linear_buckling(KC0uu, KGuu, num_eigvalues=num_eig_lb, tol=1e-9)

    load_mult = eigvals[0]
    P_cr_calc = load_mult*ftotal
    print('linear buckling load_mult =', load_mult)
    print('linear buckling P_cr_calc =', P_cr_calc)

    u = np.zeros(N)
    u[bu] = eigvecsu[:, mode]

    # theoretical reference
    kcmin = 1e6
    mmin = 0
    for m in range(1, 21):
        kc = (m*b/a + a/(m*b))**2
        if kc <= kcmin:
            kcmin = kc
            mmin = m
    print('kcmin =', kcmin)
    print('m =', mmin)
    sigma_cr = -kcmin*np.pi**2*E/(12*(1-nu**2))*h**2/b**2
    P_cr_theory = sigma_cr*h*b
    print('Theoretical P_cr_theory', P_cr_theory)

    if plot:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()

    error = abs(P_cr_calc/P_cr_theory - 1)
    print('error', P_cr_calc/P_cr_theory - 1)
    if refinement == 1:
        # NOTE pre-determined by this test with refinement=1 on 2026-10-29
        assert isclose(P_cr_calc, -42550.38531708875, rtol=1e-4)
        assert error <= 0.10
    else:
        assert error <= 0.02


if __name__ == '__main__':
    test_tria3r_linear_buckling_plate(plot=True, mode=0, refinement=1)

