r"""Tria3R with the discrete shear gap field, ``shear_model = 1``, the default

With ``shear_model = 0``, the default before 0.12.0, the transverse shear
strains of :class:`pyfe3d.tria3r.Tria3R` are sampled at the centroid, a field
that locks,
and the locking is relaxed by dividing the transverse shear stiffnesses by
`1 + \alpha \ell^2/h^2`. Lyly, Stenberg and Vihinen (1993), p. 348, state
that this stabilisation was introduced for a piecewise quadratic deflection,
and that linear or bilinear deflections need an MITC reduction operator with
it; the centroid sample is neither, and on a thin plate the error stalls at a
value set by `\alpha` instead of decreasing under refinement.

``shear_model = 1``, the default since 0.12.0, replaces that field by the
discrete shear gap (DSG) field of Bletzinger, Bischoff and Ramm (2000), the one of
:class:`pyfe3d.tria3dsg.Tria3DSG`, and keeps the Allman membrane of Tria3R.
The tests check that the shear stiffness is the one of Tria3DSG to round-off,
that the internal forces stay consistent with the stiffness matrix, and that
the element converges to the closed-form solution of a thin plate.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import (Tria3R, Tria3RData, Tria3RProbe, Tria3DSG, Tria3DSGData,
                    Tria3DSGProbe, INT, DOUBLE, DOF)

E = 200.e9
NU = 0.3
# NOTE an irregular triangle, rotated out of the xy plane, so that neither
#      symmetry nor the identity rotation can hide an error
IRREGULAR = np.array([[0.1, 0.05, 0.], [1.3, 0.2, 0.], [0.4, 1.1, 0.]])
TH = 0.7
ROT = np.array([[1., 0., 0.], [0., np.cos(TH), -np.sin(TH)],
                [0., np.sin(TH), np.cos(TH)]])


def element_K(Elem, Data, Probe, ncoords, prop, alpha, shear_model=None,
              drilling_model=0):
    flat = ncoords.flatten()
    data = Data()
    el = Elem(Probe())
    for k in range(3):
        setattr(el, 'n%d' % (k + 1), k + 1)
        setattr(el, 'c%d' % (k + 1), DOF*k)
    el.alpha_shear_locking = alpha
    el.drilling_model = drilling_model
    if shear_model is not None:
        el.shear_model = shear_model
    el.update_rotation_matrix(flat)
    el.update_probe_xe(flat)
    r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_KC0(r, c, v, prop)
    return coo_matrix((v, (r, c)), shape=(18, 18)).toarray(), el


def test_shear_stiffness_is_the_one_of_tria3dsg():
    r"""The DSG shear block of Tria3R equals the one of Tria3DSG

    The shear part of each element matrix is isolated as the difference
    between the matrix with the shear stiffness and the matrix with it
    removed by a huge ``alpha_shear_locking``. The two elements share the
    operator, so the tolerance is round-off; measured below ``5e-16``.
    """
    for ncoords in (IRREGULAR, IRREGULAR @ ROT.T):
        for h in (0.01, 0.2):
            prop = isotropic_plate(thickness=h, E=E, nu=NU)
            A, _ = element_K(Tria3R, Tria3RData, Tria3RProbe, ncoords, prop,
                             0., shear_model=1)
            A0, _ = element_K(Tria3R, Tria3RData, Tria3RProbe, ncoords, prop,
                              1.e30, shear_model=1)
            B, _ = element_K(Tria3DSG, Tria3DSGData, Tria3DSGProbe, ncoords,
                             prop, 0.)
            B0, _ = element_K(Tria3DSG, Tria3DSGData, Tria3DSGProbe, ncoords,
                              prop, 1.e30)
            ref = np.abs(B - B0).max()
            assert np.abs((A - A0) - (B - B0)).max() <= 1.e-12*ref
            assert np.abs(A - A.T).max() <= 1.e-12*np.abs(A).max()


def test_fint_is_KC0_times_u():
    r"""The internal forces are the product of KC0 with the displacements

    For both drilling models, to round-off; measured below ``3e-16``.
    """
    rng = np.random.default_rng(1)
    ncoords = IRREGULAR @ ROT.T
    prop = isotropic_plate(thickness=0.05, E=E, nu=NU)
    for drilling_model in (0, 1):
        K, el = element_K(Tria3R, Tria3RData, Tria3RProbe, ncoords, prop, 0.,
                          shear_model=1, drilling_model=drilling_model)
        u = rng.standard_normal(18)
        el.update_probe_ue(u)
        fint = np.zeros(18)
        el.update_fint(fint, prop)
        assert np.abs(fint - K @ u).max() <= 1.e-12*np.abs(fint).max()


def plate_centre_deflection(n, h, shear_model, alpha):
    r"""Simply supported square plate under a uniform pressure, ``a = 1``

    ``n`` by ``n`` squares, each cut along the diagonal 1-3. The support is
    `w = 0` with the rotation tangential to the edge fixed, the support of
    the Navier solution, and the in-plane displacements and the drilling
    rotation are fixed on the edges, the plate being flat.
    """
    a = 1.
    prop = isotropic_plate(thickness=h, E=E, nu=NU)
    xs = np.linspace(0., a, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    ncoords = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))
    flat = ncoords.flatten()
    ids = np.arange(X.size).reshape(n + 1, n + 1)
    conn = []
    for i in range(n):
        for j in range(n):
            n1, n2, n3, n4 = ids[i, j], ids[i+1, j], ids[i+1, j+1], ids[i, j+1]
            conn += [(n1, n2, n3), (n1, n3, n4)]
    data = Tria3RData()
    probe = Tria3RProbe()
    size = data.KC0_SPARSE_SIZE*len(conn)
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    N = DOF*len(ncoords)
    f = np.zeros(N)
    for e, nodes in enumerate(conn):
        el = Tria3R(probe)
        el.n1, el.n2, el.n3 = [nd + 1 for nd in nodes]
        el.c1, el.c2, el.c3 = [DOF*nd for nd in nodes]
        el.alpha_shear_locking = alpha
        el.shear_model = shear_model
        el.init_k_KC0 = e*data.KC0_SPARSE_SIZE
        el.update_rotation_matrix(flat)
        el.update_probe_xe(flat)
        el.update_KC0(r, c, v, prop)
        for nd in nodes:
            f[DOF*nd + 2] += el.area/3.
    K = coo_matrix((v, (r, c)), shape=(N, N)).tocsc()
    x, y = ncoords[:, 0], ncoords[:, 1]
    ex = np.isclose(x, 0.) | np.isclose(x, a)
    ey = np.isclose(y, 0.) | np.isclose(y, a)
    bk = np.zeros(N, dtype=bool)
    for d in (0, 1, 2, 5):
        bk[d::DOF] |= ex | ey
    bk[3::DOF] |= ex
    bk[4::DOF] |= ey
    bu = ~bk
    u = np.zeros(N)
    u[bu] = spsolve(K[bu][:, bu], f[bu])
    D = E*h**3/(12*(1 - NU**2))
    return u[DOF*ids[n//2, n//2] + 2]*D/a**4


def test_thin_plate_converges():
    r"""Thin simply supported square plate, ``a/h = 1000``, uniform load

    The reference is the Kirchhoff plate of Timoshenko, S., and
    Woinowsky-Krieger, S., Theory of Plates and Shells, 2nd ed.,
    McGraw-Hill, 1959, Art. 30, Table 8, p. 120: `w_{max} = 0.00406 q a^4/D`
    for `b/a = 1` and `\nu = 0.3`; the double series of the Navier solution
    summed to 400 terms in each direction gives 0.00406235, used here. At
    `a/h = 1000` the transverse shear deflection of the Mindlin plate adds
    0.0005 per cent to it.

    Measured error of the centre deflection, ``shear_model = 1`` with
    ``alpha_shear_locking = 0``, i.e. the field of Tria3DSG, against the
    former default ``shear_model = 0`` with ``alpha_shear_locking = 0.7``:

        n x n      shear_model = 1     shear_model = 0
        16x16          -12.33 %            -13.00 %
        32x32           -0.81 %            -14.10 %
        64x64           -0.09 %            -14.36 %
        128x128         -0.02 %            -14.32 %

    The DSG field converges at second order once the mesh resolves the
    plate; the former default stalls at -14 per cent. The 32x32 mesh is asserted
    with the tolerance of grade A of MacNeal and Harder (1985), p. 14, 2 %.
    """
    ref = 0.00406235
    got = plate_centre_deflection(32, 0.001, shear_model=1, alpha=0.)
    print('shear_model=1, 32x32: %.7f, error %+.3f %%' % (got, 100*(got/ref - 1)))
    assert abs(got/ref - 1) <= 0.02


if __name__ == '__main__':
    test_shear_stiffness_is_the_one_of_tria3dsg()
    test_fint_is_KC0_times_u()
    test_thin_plate_converges()
