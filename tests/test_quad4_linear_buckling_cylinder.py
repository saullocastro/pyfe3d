import sys
sys.path.append('..')

import time
import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import laminated_plate, isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def test_linear_buckling_cylinder(mode=0, plot_pyvista=False):
    r"""Test case from reference

        Geier, B., and Singh, G., 1997, “Some Simple Solutions for Buckling Loads of Thin and Moderately Thick Cylindrical Shells and Panels Made of Laminated Composite Material,” Aerosp. Sci. Technol., 1(1), pp. 47–63.

        Cylinder Z12, see Table 3 page 60

    """
    data = Quad4Data()
    probe = Quad4Probe()

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

    # NOTE cylinder Z12, table 3 of reference
    stack = [+51, -51, +45, -45, +37, -37, +19, -19, 0, 0]
    prop = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop)
    #prop = isotropic_plate(thickness=0.001, E=70e9, nu=0.33)

    nids = 1 + np.arange(nlength*(ntheta+1))
    nids_mesh = nids.reshape(nlength, ntheta+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    zlin = np.linspace(0, L, nlength)
    thetatmp = np.linspace(-np.pi, np.pi, ntheta+1)
    thetalin = np.linspace(-np.pi, np.pi-(thetatmp[-1] - thetatmp[-2]), ntheta)[::-1]
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
    t0 = time.time()
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        quad = Quad4(probe)
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
        quad.K6ROT = 1.
        quad.update_rotation_matrix(ncoords_flatten, 0, 0, 1)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

    print('elements created', time.time()-t0)

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    bk = np.zeros(N, dtype=bool)

    checkSS = isclose(z, 0) | isclose(z, L)
    bk[0::DOF] = checkSS
    bk[1::DOF] = checkSS
    bk[2::DOF] = checkSS

    bu = ~bk

    u = np.zeros(N, dtype=DOUBLE)

    compression = -0.0001
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
        quad.update_probe_xe(ncoords_flatten) # NOTE update affects the Quad4Probe class attribute xe
        quad.update_probe_ue(u) # NOTE update affects the Quad4Probe class attribute ue
        quad.update_KG(KGr, KGc, KGv, prop)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('sparse KG created')

    num_eig_lb = max(mode+1, 1)
    eigvecs = np.zeros((N, num_eig_lb))
    eigvals, eigvecsu = eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM',
            M=PREC*KC0uu, tol=1e-9, sigma=1., mode='cayley')
    eigvecs[bu] = eigvecsu
    eigvals = -1./eigvals
    print('eigvals', eigvals)

    print('linear buckling analysis OK')
    fext = np.zeros(N)
    fk = KC0uk.T*uu + KC0kk*uk
    fext[bk] = fk
    Pcr = (eigvals[0]*fext[2::DOF][checkTopEdge]).sum()
    print('Pcr =', Pcr)
    reference_value_Geier_Singh_Z12 =  -274300

    if plot_pyvista:
        import pyvista as pv

        contour_colorscale = 'coolwarm'
        background = 'gray'
        contour_label = 'Radial displacement'
        contour_vec = np.sqrt(eigvecs[0::DOF, mode]**2 + eigvecs[1::DOF, mode]**2)
        displ_vec = np.zeros_like(ncoords)
        displ_vec[:, 0] = eigvecs[0::DOF, mode]*100
        displ_vec[:, 1] = eigvecs[1::DOF, mode]*100
        displ_vec[:, 2] = eigvecs[2::DOF, mode]*100
        intensitymode = 'vertex'

        plotter = pv.Plotter(off_screen=False)
        faces_quad = []
        for q in quads:
            faces_quad.append([4, nid_pos[q.n1], nid_pos[q.n2], nid_pos[q.n3], nid_pos[q.n4]])
        faces_quad = np.array(faces_quad)
        quad_plot = pv.PolyData(ncoords, faces_quad)
        if contour_vec is not None:
            quad_plot[contour_label] = contour_vec
            plotter.add_mesh(quad_plot, scalars=contour_label,
                    cmap=contour_colorscale, edge_color='black', show_edges=True,
                    line_width=1.)
        else:
            plotter.add_mesh(quad_plot, edge_color='black', show_edges=True,
                    line_width=1.)
        if displ_vec is not None:
            quad_plot = pv.PolyData(ncoords + displ_vec, faces_quad)
            plotter.add_mesh(quad_plot, edge_color='red', show_edges=True,
                    line_width=1., opacity=0.5)
        #NOTE plotting coordinate system
        xaxis = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=R/3)
        plotter.add_mesh(xaxis, color='blue')
        yaxis = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=R/3)
        plotter.add_mesh(yaxis, color='yellow')
        zaxis = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=R/3)
        plotter.add_mesh(zaxis, color='green')

        plotter.set_background(background)
        plotter.parallel_projection = False
        plotter.show()


    #assert np.isclose(Pcr, reference_value_Geier_Singh_Z12, rtol=0.01)
    assert np.isclose(Pcr, -225968.28006101557, rtol=0.01)

if __name__ == '__main__':
    test_linear_buckling_cylinder(mode=0, plot_pyvista=True)
