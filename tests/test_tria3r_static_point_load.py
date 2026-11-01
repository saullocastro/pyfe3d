import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF


def test_tria3r_static_point_load(plot=False, refinement=1):
    r"""Simply supported rectangular plate, ``a = 3``, ``b = 7``,
    ``h = 0.005``, under a point load at its centre

    The exact deflection at the centre of the Kirchhoff plate is
    ``6.5986e-5``, from the Navier solution of Timoshenko, S., and
    Woinowsky-Krieger, S., Theory of Plates and Shells, 2nd ed., McGraw-Hill,
    1959, Art. 29, p. 111, Eq. (133), summed in Art. 34, p. 143, Eq. (147),
    see :func:`test_quad4r_static_point_load.test_static_plate_quad_point_load`,
    where the transverse shear deflection of the Mindlin plate is shown to be
    negligible here.

    With the default transverse shear since 0.12.0, the discrete shear gap
    field of ``shear_model = 1``, the deflection converges to that
    reference from below, measured with this test (``b/h = 1400``,
    quadrilaterals split in two triangles, time on one thread):

        refinement   mesh      w_max          error    time
        1            6x10      3.8417e-05    -41.78 %   0.0 s
        2            14x22     6.0875e-05     -7.75 %   0.0 s
        3            20x32     6.4079e-05     -2.89 %   0.1 s
        4            28x44     6.5155e-05     -1.26 %   0.3 s
        6            42x66     6.5670e-05     -0.48 %   0.9 s
        8            56x88     6.5821e-05     -0.25 %   3.4 s

    The coarse mesh is stiff because the elements are 100 times longer
    than they are thick. Following the coarse-mesh policy of the tests,
    ``refinement=1`` is compared with the value pre-determined by this test,
    with ``rtol=1e-6`` for the round-off of the direct solution, and with
    grade D of MacNeal and Harder (1985), p. 14, as the bound against the
    reference; finer meshes are graded against the reference alone, B from
    ``refinement=2`` and A from ``refinement=4``. Before 0.12.0, with the
    centroid-sampled shear and ``alpha_shear_locking = 0.7``, the deflection
    was 26.2 % above the reference with 6x10 elements and settled 9.6 %
    below it with 112x176, see :mod:`pyfe3d.tria3r`.
    """
    data = Tria3RData()
    probe = Tria3RProbe()
    nx = 7*refinement
    ny = 11*refinement

    if (nx % 2) == 0:
        nx += 1
    if (ny % 2) == 0:
        ny += 1

    a = 3
    b = 7
    h = 0.005 # m

    E = 200e9
    nu = 0.3

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
    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu)

    trias = []
    init_k_KC0 = 0
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
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE

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
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    # applying boundary conditions
    # simply supported
    bk = np.zeros(N, dtype=bool)
    edges = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
    bk[2::DOF][edges] = True
    bk[0::DOF][edges] = True
    bk[1::DOF][edges] = True

    bu = ~bk

    # point load at center node
    fext = np.zeros(N)
    fmid = 1.
    middle = np.isclose(x, a/2) & np.isclose(y, b/2)
    fext[2::DOF][middle] = fmid

    KC0uu = KC0[bu, :][:, bu]
    assert fext[bu].sum() == fmid

    u = np.zeros(N)
    u[bu] = spsolve(KC0uu, fext[bu])

    w = u[2::DOF].reshape(nx, ny).T

    # NOTE Timoshenko and Woinowsky-Krieger (1959), Eq. (147), b/a = 7/3
    D = E*h**3/(12*(1 - nu**2))
    wmax_ref = 0.
    for m in range(1, 200, 2):
        am = m*np.pi*b/(2*a)
        wmax_ref += (np.tanh(am) - (am/np.cosh(am)**2 if am < 300 else 0.))/m**3
    wmax_ref *= fmid*a**2/(2*np.pi**3*D)
    print('w.max()', w.max())
    print('wmax_ref', wmax_ref, 'error', w.max()/wmax_ref - 1)

    error = abs(w.max()/wmax_ref - 1)
    if refinement == 1:
        # NOTE pre-determined by this test with refinement=1 on 2026-10-29
        assert np.isclose(w.max(), 3.8417365230339226e-05, rtol=1e-6)
        assert error <= 0.50
    elif refinement < 4:
        assert error <= 0.10
    else:
        assert error <= 0.02

    fint = np.zeros(N)
    for tria in trias:
        tria.update_probe_xe(ncoords_flatten)
        tria.update_probe_ue(u)
        tria.update_fint(fint, prop)

    # NOTE adding reaction forces to external force vector
    Kku = KC0[bk, :][:, bu]
    fext[bk] = Kku @ u[bu]
    assert np.allclose(fint, fext)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    test_tria3r_static_point_load(plot=True, refinement=1)
