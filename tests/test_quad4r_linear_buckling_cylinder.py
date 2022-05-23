import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import laminated_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF


def test_linear_buckling_cylinder(mode=0):
    r"""Test case from reference

        Geier, B., and Singh, G., 1997, “Some Simple Solutions for Buckling Loads of Thin and Moderately Thick Cylindrical Shells and Panels Made of Laminated Composite Material,” Aerosp. Sci. Technol., 1(1), pp. 47–63.

        Cylinder Z11, see Table 3 page 60

    """
    data = Quad4RData()
    probe = Quad4RProbe()

    L = 0.510 # m
    R = 0.250 # m
    b = 2*np.pi*R # m

    ntheta = 40 # circumferential
    nlength = 2*int(ntheta*L/b)

    E11 = 123.55e9
    E22 = 8.7079e9
    nu12 = 0.319
    G12 = 5.695e9
    G13 = 5.695e9
    G23 = 3.400e9
    plyt = 0.125e-3
    laminaprop = (E11, E22, nu12, G12, G13, G23)

    # NOTE cylinder Z11, table 3 of reference
    stack = [+60, -60, 0, 0, +68, -68, +52, -52, +37, -37]
    prop = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop)

    nids = 1 + np.arange(nlength*(ntheta+1))
    nids_mesh = nids.reshape(nlength, ntheta+1)
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    zlin = np.linspace(0, L, nlength)
    thetatmp = np.linspace(0, 2*np.pi, ntheta+1)
    thetalin = np.linspace(0, 2*np.pi-(thetatmp[-1] - thetatmp[-2]), ntheta)[::-1]
    zmesh, thetamesh = np.meshgrid(zlin, thetalin)
    zmesh = zmesh.T
    thetamesh = thetamesh.T
    xmesh = np.cos(thetamesh)*R
    ymesh = np.sin(thetamesh)*R

    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten(), zmesh.flatten())).T
    ncoords_flatten = ncoords.flatten()
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    z = ncoords[:, 2]

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    num_elements = len(n1s)
    print('num_elements', num_elements)

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*nlength*ntheta

    quads = []
    init_k_KC0 = 0
    init_k_KG = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        quad = Quad4R(probe)
        quad.n1 = n1
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.c1 = DOF*nid_pos[n1]
        quad.c2 = DOF*nid_pos[n2]
        quad.c3 = DOF*nid_pos[n3]
        quad.c4 = DOF*nid_pos[n4]
        quad.init_k_KC0 = init_k_KC0
        quad.init_k_KG = init_k_KG
        quad.update_rotation_matrix(ncoords_flatten, 0, 0, 1)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    bk = np.zeros(N, dtype=bool)

    checkSS = isclose(z, 0) | isclose(z, L)
    bk[0::DOF] = checkSS
    bk[1::DOF] = checkSS
    bk[2::DOF] = checkSS

    bu = ~bk

    u = np.zeros(N, dtype=DOUBLE)

    compression = -0.00007
    checkTopEdge = isclose(z, L)
    u[2::DOF] += checkTopEdge*compression
    uk = u[bk]

    KC0uu = KC0[bu, :][:, bu]
    KC0uk = KC0[bu, :][:, bk]
    KC0kk = KC0[bk, :][:, bk]

    fextu = -KC0uk*uk

    PREC = np.max(1/KC0uu.diagonal())
    uu = spsolve(PREC*KC0uu, PREC*fextu)
    print('static analysis OK')
    u[bu] = uu

    for quad in quads:
        quad.update_probe_ue(u) # NOTE update affects the Quad4RProbe class attribute ue
        quad.update_probe_xe(ncoords_flatten) # NOTE update affects the Quad4RProbe class attribute xe
        quad.update_KG(KGr, KGc, KGv, prop)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('sparse KG created')

    num_eig_lb = max(mode+1, 1)
    eigvals, eigvecsu = eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM',
            M=PREC*KC0uu, tol=1e-9, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    print('eigvals', eigvals)

    print('linear buckling analysis OK')
    fext = np.zeros(N)
    fk = KC0uk.T*uu + KC0kk*uk
    fext[bk] = fk
    Pcr = (eigvals[0]*fext[2::DOF][checkTopEdge]).sum()
    print('Pcr =', Pcr)
    assert np.isclose(Pcr, -409522.60151502624, rtol=1e-4)


if __name__ == '__main__':
    test_linear_buckling_cylinder(mode=0)
