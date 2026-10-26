import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def central_deflection(h, n=9, a=1., E=1.e7, nu=0.3):
    r"""Normalized central deflection `w D/(P a^2)` of a simply supported
    square plate of side ``a`` under a central point load `P`, with
    ``(n - 1)**2`` :class:`.Quad4` elements"""
    data = Quad4Data()
    probe = Quad4Probe()
    xtmp = np.linspace(0, a, n)
    xmesh, ymesh = np.meshgrid(xtmp, xtmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(),
                         np.zeros(n*n))).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    ncoords_flatten = ncoords.flatten()
    nids = np.arange(n*n).reshape(n, n)
    n1s = nids[:-1, :-1].flatten()
    n2s = nids[1:, :-1].flatten()
    n3s = nids[1:, 1:].flatten()
    n4s = nids[:-1, 1:].flatten()
    num_elements = len(n1s)
    N = DOF*n*n
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        quad = Quad4(probe)
        quad.n1, quad.n2, quad.n3, quad.n4 = n1, n2, n3, n4
        quad.c1, quad.c2, quad.c3, quad.c4 = DOF*n1, DOF*n2, DOF*n3, DOF*n4
        quad.init_k_KC0 = init_k_KC0
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    edges = (np.isclose(x, 0) | np.isclose(x, a) | np.isclose(y, 0)
             | np.isclose(y, a))
    bk[2::DOF] = edges
    bk[0::DOF] = True
    bk[1::DOF] = True
    bk[5::DOF] = True
    bu = ~bk
    fext = np.zeros(N)
    fext[2::DOF][np.isclose(x, a/2) & np.isclose(y, a/2)] = 1.
    u = np.zeros(N)
    u[bu] = spsolve(KC0[bu, :][:, bu], fext[bu])
    D = E*h**3/(12*(1 - nu**2))
    return u[2::DOF].max()*D/a**2


def test_quad4_thin_limit():
    r"""No correction for extreme thinness is needed in double precision

    With elements of side `\ell = a/8`, the normalized deflection at
    `\ell/h = 1.25 \times 10^5`, ten times the maximum aspect ratio
    `10^5/8` of Hughes et al. (1977) in single precision, is the one at
    `\ell/h = 1.25 \times 10^3`, the thin limit of the mesh.

    """
    a = 1.
    ell = a/8
    w_ref = central_deflection(h=ell/1.25e3, a=a)
    w_thin = central_deflection(h=ell/1.25e5, a=a)
    print('w D/(P a^2)', w_ref, w_thin)
    assert np.isclose(w_thin, w_ref, rtol=1e-4)


if __name__ == '__main__':
    test_quad4_thin_limit()
