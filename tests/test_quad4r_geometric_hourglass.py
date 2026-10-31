r"""Quad4R: slopes of `w` at the centroid in the geometric stiffness

Up to version 0.11, the geometric stiffness KG of :class:`pyfe3d.Quad4R`,
and the geometrically nonlinear terms that share its quadrature, took the
slopes `w_{,x}` and `w_{,y}` at the two-by-two Gauss-Legendre points. The
hourglass pattern of `w`, `w = \xi \eta` with zero rotations, then had the
geometric stiffness of the full quadrature, whereas KC0, integrated at the
centroid, resists it only with the artificial hourglass stiffness. A linear
buckling analysis could return a spurious mode of `w` alternating in sign
from node to node below the physical buckling load. Simply supported square
plate, side 0.5, uniaxial compression, lowest buckling load against the
Mindlin solution, with the slopes at the two-by-two points:

    plate                                 mesh     h/l     error
    b/h = 10, hard simple supports        12x12    1.20    -3.0 %
                                          16x16    1.60   -40.8 %
                                          48x48    4.80   -85.7 %
    b/h = 100, A44 = A55 = 1/1000 of the  8x8      0.08   -59.9 %
    isotropic value, hard simple supports

With the slopes at the centroid the pattern carries no geometric stiffness
and the same cases gave +4.7, +2.6, +0.3 and +7.0 %, converging to the
Mindlin solution; with the physical stabilisation of Belytschko and
Leviathan (1994), used since 0.12.0, +1.5, +0.8, +0.09 and +2.7 %.

The Mindlin solution of the hard simply supported plate, ``w = 0`` and the
rotation about the axis of each edge equal to zero, is the Navier solution
of a plate with transverse shear stiffness ``D_Q``, given by Seide, P., and
Stowell, E. Z., Elastic and Plastic Buckling of Simply Supported Solid-Core
Sandwich Plates in Compression, NACA Report 967, 1950, Eq. (1), p. 2,
derived in Appendix A, Eq. (A7), p. 7, with the shear stiffness ``D_Q`` of
the sandwich core taken equal to ``A55``: for the square plate, with one
half-wave in each direction, ``N_cr = 4*pi**2*D/b**2/(1 + 2*pi**2*D/(b**2*A55))``.
The same load follows from Wang, C. M., Wang, C. Y., and Reddy, J. N.,
Exact Solutions for Buckling of Structural Members, CRC Press, 2005, Sec.
4.7.2, Eq. (4.7.17), with the foundation moduli equal to zero and
``gamma = 0`` (page not verified). With ``A44 = A55`` reduced to 1/1000,
two half-waves along the load give a load only 0.4 % higher.

"""
import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling

E = 203.e9
NU = 0.33
SIDE = 0.5


def assemble(x, y, elements, prop, Nxx=-1.):
    data = Quad4RData()
    probe = Quad4RProbe()
    coords = np.vstack((x, y, 0*x)).T.flatten()
    num_elements = len(elements)
    N = DOF*len(x)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros_like(KC0r)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros_like(KGr)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for k, (n1, n2, n3, n4) in enumerate(elements):
        quad = Quad4R(probe)
        quad.n1, quad.n2, quad.n3, quad.n4 = n1, n2, n3, n4
        quad.c1, quad.c2, quad.c3, quad.c4 = DOF*n1, DOF*n2, DOF*n3, DOF*n4
        quad.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        quad.init_k_KG = k*data.KG_SPARSE_SIZE
        quad.update_rotation_matrix(coords)
        quad.update_probe_xe(coords)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quad.update_KG_given_stress(Nxx, 0., 0., KGr, KGc, KGv)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    return KC0, KG


def square_mesh(n):
    xtmp = np.linspace(0, SIDE, n)
    xmesh, ymesh = np.meshgrid(xtmp, xtmp)
    x = xmesh.T.flatten()
    y = ymesh.T.flatten()
    nids = np.arange(n*n).reshape(n, n)
    elements = list(zip(nids[:-1, :-1].flatten(), nids[1:, :-1].flatten(),
                        nids[1:, 1:].flatten(), nids[:-1, 1:].flatten()))
    return x, y, elements


def buckling_ss_plate(n, h, shear_factor=1.):
    r"""Lowest buckling load of a hard simply supported square plate under
    uniaxial compression, relative to the Mindlin solution, and the
    checkerboard content of `w` in its mode, between 0 and 1"""
    prop = isotropic_plate(E=E, nu=NU, thickness=h)
    prop.A44 *= shear_factor
    prop.A55 *= shear_factor
    x, y, elements = square_mesh(n)
    KC0, KG = assemble(x, y, elements, prop)
    bk = np.zeros(KC0.shape[0], dtype=bool)
    edge = (isclose(x, 0.) | isclose(x, SIDE) | isclose(y, 0.)
            | isclose(y, SIDE))
    for i in range(3):
        bk[i::DOF] = edge
    bk[3::DOF] |= isclose(x, 0.) | isclose(x, SIDE)
    bk[4::DOF] |= isclose(y, 0.) | isclose(y, SIDE)
    bu = ~bk
    eigvals, eigvecs = linear_buckling(KC0[bu][:, bu], KG[bu][:, bu],
                                       num_eigvalues=2, tol=1e-9)
    u = np.zeros(KC0.shape[0])
    u[bu] = eigvecs[:, 0]
    w = u[2::DOF].reshape(n, n)
    sign = (-1.)**np.add.outer(np.arange(n), np.arange(n))
    checkerboard = abs((w*sign).sum())/np.sqrt((w**2).sum()*n*n)
    D = E*h**3/(12*(1 - NU**2))
    k2 = 2*(np.pi/SIDE)**2
    N_mindlin = D*k2**2/(np.pi/SIDE)**2/(1 + D*k2/prop.A55)
    return eigvals[0]/N_mindlin - 1, checkerboard


def test_thick_elements():
    r"""``b/h = 10``: errors against the Mindlin solution, measured on
    2026-10-29 (time of one buckling analysis, ``OMP_NUM_THREADS=1``):

    =======  =========  ==========
    mesh     error      up to 0.11
    =======  =========  ==========
    12x12    +1.489 %   +4.667 %
    16x16    +0.833 %   +2.606 %
    24x24    +0.369 %   +1.152 %
    32x32    +0.207 %   +0.647 %
    48x48    +0.092 %   +0.287 %
    64x64    +0.052 %   +0.161 %
    =======  =========  ==========

    converging as the square of the element size to the Mindlin solution,
    a mesh error; a regression of the hourglass pattern gives the negative
    errors of the module documentation. Following the policy of the tests
    whose reference is reached only on fine meshes, the 12x12 and 16x16
    meshes are compared with the errors pre-determined by this same test,
    as buckling loads with ``rtol = 1e-4`` for the round-off of the
    eigenvalue solver, and with grade B of MacNeal, R. H. and Harder, R.
    L., A proposed standard set of problems to test finite element
    accuracy, Finite Elements in Analysis and Design, 1(1), 3-20, 1985,
    https://doi.org/10.1016/0168-874X(85)90003-4, p. 14 (at most 10 %), as
    a bound. Until 0.12.0 the 48x48 mesh was also run, with bounds that
    followed the measured errors. The values changed in 0.12.0 with the
    physical stabilisation of the bending and of the transverse shear of
    Belytschko and Leviathan (1994), see :mod:`pyfe3d.quad4r`; the 64x64
    mesh takes 5 s."""
    predetermined = {13: 0.014893691920608187, 17: 0.008331130410677456}
    for n in (13, 17):
        err, checkerboard = buckling_ss_plate(n, h=0.05)
        print('b/h = 10, %dx%d elements: %+.2f %%, checkerboard %.3f'
              % (n - 1, n - 1, 100*err, checkerboard))
        assert np.isclose(1 + err, 1 + predetermined[n], rtol=1e-4)
        assert 0 < err <= 0.10
        assert checkerboard < 0.05


def test_low_transverse_shear_stiffness():
    r"""``b/h = 100`` with ``A44 = A55`` reduced to 1/1000: errors against
    the Mindlin solution, measured on 2026-10-29: +2.700, +0.662, +0.165
    and +0.041 % with 8x8, 16x16, 32x32 and 64x64 elements (0.05 to 5 s),
    converging as the square of the element size, a mesh error; up to 0.11,
    before the physical stabilisation of Belytschko and Leviathan (1994),
    +7.001, +1.745, +0.436 and +0.109 %. The 8x8 and 16x16 meshes are compared with the errors
    pre-determined by this same test, as buckling loads with ``rtol =
    1e-4``, and with grade B of MacNeal and Harder (1985), see
    :func:`test_thick_elements`."""
    predetermined = {9: 0.026996360698192, 17: 0.0066243690464657945}
    for n in (9, 17):
        err, checkerboard = buckling_ss_plate(n, h=0.005, shear_factor=1e-3)
        print('b/h = 100, A44 = A55 = 1/1000, %dx%d elements: %+.2f %%, '
              'checkerboard %.3f' % (n - 1, n - 1, 100*err, checkerboard))
        assert np.isclose(1 + err, 1 + predetermined[n], rtol=1e-4)
        assert 0 < err <= 0.10
        assert checkerboard < 0.05


def test_KG_is_one_point():
    r"""KG with constant stress resultants equals the element area times the
    integrand at the centroid, also for a distorted element, and the
    hourglass pattern of `w` has no geometric stiffness"""
    xy = np.array([[0., 0.], [2., 0.], [2.3, 1.4], [0.2, 1.1]])
    x, y = xy[:, 0], xy[:, 1]
    prop = isotropic_plate(E=E, nu=NU, thickness=0.01)
    Nxx, Nyy, Nxy = -3., 1.5, 0.7
    data = Quad4RData()
    probe = Quad4RProbe()
    quad = Quad4R(probe)
    quad.n1, quad.n2, quad.n3, quad.n4 = 0, 1, 2, 3
    quad.c1, quad.c2, quad.c3, quad.c4 = 0, DOF, 2*DOF, 3*DOF
    quad.init_k_KG = 0
    coords = np.vstack((x, y, 0*x)).T.flatten()
    quad.update_rotation_matrix(coords)
    quad.update_probe_xe(coords)
    quad.update_area()
    KGr = np.zeros(data.KG_SPARSE_SIZE, dtype=INT)
    KGc = np.zeros_like(KGr)
    KGv = np.zeros(data.KG_SPARSE_SIZE, dtype=DOUBLE)
    quad.update_KG_given_stress(Nxx, Nyy, Nxy, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(4*DOF, 4*DOF)).toarray()
    Kww = KG[2::DOF, 2::DOF]
    # NOTE derivatives of the bilinear shape functions at the centroid
    xi = np.array([-1., 1., 1., -1.])
    eta = np.array([-1., -1., 1., 1.])
    J = np.array([[xi @ x, xi @ y], [eta @ x, eta @ y]])/4
    dN = np.linalg.solve(J, np.vstack((xi, eta))/4)
    # NOTE stress resultants from the element axes, the columns of r, to the
    #      global axes
    r = np.array([[quad.r11, quad.r12], [quad.r21, quad.r22]])
    Nmat = r @ np.array([[Nxx, Nxy], [Nxy, Nyy]]) @ r.T
    Kww_ref = quad.area*dN.T @ Nmat @ dN
    assert np.allclose(Kww, Kww_ref, rtol=1e-12, atol=1e-12*abs(Kww_ref).max())
    hourglass = np.array([1., -1., 1., -1.])
    assert abs(hourglass @ Kww @ hourglass) < 1e-12*abs(Kww).max()


if __name__ == '__main__':
    test_thick_elements()
    test_low_transverse_shear_stiffness()
    test_KG_is_one_point()
