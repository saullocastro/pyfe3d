import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling


def test_tria3r_linear_buckling_plate_given_stress(mode=0, refinement=1):
    r"""Linear buckling of a simply supported plate, ``a/b = 4``, under a
    uniform compression ``N_xx`` given to ``KG``

    The reference is the exact solution of Leissa, A. W., Vibration of
    Plates, NASA SP-160, 1969, Sec. 10.2.1, p. 277, Eq. (10.46), minimised
    over the number ``m`` of half-waves along ``x``:
    ``N_xx,cr = -k_c*pi**2*D/b**2``, ``k_c = (m*b/a + a/(m*b))**2``. With
    ``b/h = 250`` the Mindlin plate buckles 0.01 % below it.

    The resultants given to ``update_KG_given_stress`` are in the element
    coordinate system, so the global ``N_xx`` is rotated to the axes of each
    triangle, whose x axis is along the diagonal for one triangle out of
    two. Before 0.12.0 it was passed unrotated, which applies a different
    load to those triangles; with the default element below, the unrotated
    load gives a buckling load 1.96 % under the reference with 248x120
    elements, and not approaching it.

    With the default transverse shear since 0.12.0, the discrete shear gap
    field of ``shear_model = 1``, the buckling load converges to the
    reference, measured with this test (quadrilaterals split in two
    triangles, time on one thread):

        refinement   mesh      P_cr          error    time
        1            30x14     -12864.9     +7.29 %   0.1 s
        2            62x30     -12087.0     +0.80 %   0.4 s
        3            92x44     -12030.0     +0.32 %   1.3 s
        4            124x60    -12009.5     +0.15 %   3.5 s
        6            186x90    -11994.8     +0.03 %   8.8 s

    ``refinement=1`` is compared with the value pre-determined by this test,
    with ``rtol=1e-4`` for the round-off of the eigensolver, and with grade B
    of MacNeal and Harder (1985), p. 14, as the bound against the
    reference; from ``refinement=2`` grade A is asserted against the
    reference alone. Before 0.12.0, with the centroid-sampled shear,
    ``alpha_shear_locking = 0.7`` and the unrotated load, the load was 4.8 %
    above the reference with 30x14 elements and 11.8 % above with 92x44.
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
    h = 0.002 # m

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

    # applying boundary conditions (leading to a constant Nxx)
    # simply supported in w
    bk = np.zeros(N, dtype=bool) #array to store known DOFs
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

    KC0uu = KC0[bu, :][:, bu]

    Nxx = -1

    for tria in trias:
        tria.update_probe_xe(ncoords_flatten)
        # NOTE the stress resultants are given in the element coordinate
        #      system, see Tria3R.update_KG_given_stress. The x axis of a
        #      triangle follows its edge 1-2, along the diagonal for the
        #      second triangle of each quadrilateral, so the global Nxx is
        #      rotated: the global x axis has the components (r11, r12) in
        #      the element axes. Before 0.12.0 it was passed unrotated
        ax = tria.r11
        ay = tria.r12
        tria.update_KG_given_stress(Nxx*ax*ax, Nxx*ay*ay, Nxx*ax*ay, KGr,
                                    KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('sparse KG created')

    num_eig_lb = max(mode+1, 1)

    # NOTE pyfe3d.solver.linear_buckling equilibrates the diagonal,
    #      estimates the Cayley shift and verifies that no lower load
    #      multiplier was missed, which a hardcoded shift does not
    eigvals, eigvecsu = linear_buckling(KC0uu, KGuu, num_eigvalues=num_eig_lb, tol=1e-9)
    
    load_mult = eigvals[0]
    P_cr_calc = load_mult*Nxx*b
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
    error = abs(P_cr_calc/P_cr_theory - 1)
    print('error', P_cr_calc/P_cr_theory - 1)
    if refinement == 1:
        # NOTE pre-determined by this test with refinement=1 on 2026-10-29
        assert isclose(P_cr_calc, -12864.930796789686, rtol=1e-4)
        assert error <= 0.10
    else:
        assert error <= 0.02


if __name__ == '__main__':
    test_tria3r_linear_buckling_plate_given_stress(mode=0, refinement=1)

