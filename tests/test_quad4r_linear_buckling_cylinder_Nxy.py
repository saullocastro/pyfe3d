import sys
sys.path.append('..')

import time
import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import laminated_plate, isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling


def _cylindrical_transformation(ncoords):
    r"""Sparse matrix ``T`` such that ``u_global = T @ u_cyl``

    For a cylinder along the global z axis the nodal degrees-of-freedom of
    ``u_cyl`` are the radial, circumferential and axial translations and the
    rotations about the same three directions. A matrix ``K`` assembled in
    the global system becomes ``T.T @ K @ T``.

    """
    theta = np.arctan2(ncoords[:, 1], ncoords[:, 0])
    c = np.cos(theta)
    s = np.sin(theta)
    one = np.ones_like(c)
    zero = np.zeros_like(c)
    # NOTE columns of the 3x3 blocks are the radial, circumferential and
    #      axial unit vectors
    Q = np.array([[c, -s, zero],
                  [s, c, zero],
                  [zero, zero, one]]) # shape (3, 3, num_nodes)
    rows = []
    cols = []
    vals = []
    base = DOF*np.arange(len(theta))
    for offset in (0, 3):
        for i in range(3):
            for j in range(3):
                rows.append(base + offset + i)
                cols.append(base + offset + j)
                vals.append(Q[i, j])
    N = DOF*len(theta)
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows),
        np.concatenate(cols))), shape=(N, N)).tocsc()


def test_linear_buckling_cylinder_Nxy(mode=0, plot_pyvista=False, refinement=1):
    r"""Torsional buckling of the laminated cylinders Z11 and Z33 of

        Saullo G. P. Castro, Christian Mittelstedt, Francisco A. C. Monteiro, Mariano
        A. Arbelo, Gerhard Ziegmann, Richard Degenhardt. "Linear buckling predictions
        of unstiffened laminated composite cylinders and cones under various loading
        and boundary conditions using semi-analytical models". Composite Structures,
        118, 303-315, 2014. https://doi.org/10.1016/j.compstruct.2014.07.037

    The paper gives the geometry (Table 4, p. 307, R = 250 mm, H = 510 mm,
    ply thickness 0.125 mm), the material Geier1997 (Table 3, p. 307), the
    stacking sequences (Table 4, from inwards to outwards), the boundary
    condition CC2, ``v = w = w,x = 0`` with ``u`` free (Table 1, p. 304), the
    torsion load ``N_xtheta = -T/(2*pi*r**2)`` (Eq. (13), p. 305), and the shear
    correction factor 5/6 (Sec. 6, p. 307). It does not tabulate the
    torsional buckling load. The reference values are the intercepts with the
    zero axial load axis, that is pure torsion, of the FSDT-BC2 curves of
    Fig. 13, p. 311 ("Combined torsion-axial compression, FSDT-BC2, CC2"),
    read off the figure to about 0.1 kNm:

        Z11    T = +16.8 kNm   and   T = -15.8 kNm
        Z33    T = +9.2 kNm    and   T = -9.5 kNm

    the two signs differing because the laminates are unsymmetric.
    ``Tcr = eigval*Nxy*b*R`` below, with ``Nxy < 0``, is minus the torque T of
    the paper, so ``-Tcr`` is compared with the positive intercepts.

    Measured with this test, with ``ntheta = 80*refinement`` elements around
    the circumference and nearly square elements, ``|Tcr|`` in Nm and its
    relative difference to the intercept of Fig. 13 of the same sign, with
    the wall time of the test, both laminates, on a machine shared with
    other jobs::

        refinement ntheta elements  Z11, Nxy < 0      Z33, Nxy < 0     time
           0.5       40      480   25067.1 +49.2 %   15663.7 +70.3 %     1 s
           0.75      60     1080   20219.9 +20.4 %   11691.9 +27.1 %     4 s
           1         80     1920   18537.7 +10.3 %   10509.2 +14.2 %    11 s
           1.5      120     4560   17459.7  +3.9 %    9766.3  +6.2 %    31 s
           2        160     8000   17113.2  +1.9 %    9480.9  +3.1 %    68 s
           3        240    18240   16870.4  +0.4 %    9283.4  +0.9 %   208 s
        extrapolated               16684    -0.7 %    9136    -0.7 %

    The extrapolation fits ``a + c*ntheta**-p`` to the three finest meshes,
    giving p = 2.1. The element thus converges to within 0.7 per cent of the
    semi-analytical prediction for both laminates, which is the precision
    with which the figure can be read, and to the same values as the enriched
    Quad4, whose extrapolated values are 16684 and 9136 Nm too, see
    tests/test_quad4_linear_buckling_cylinder_Nxy.py, where the model is also
    run with ``Nxy > 0`` and reproduces the negative intercepts. The
    geometric stiffness taking the slopes of `w` at the centroid, in
    version 0.12.0, changed none of these values: under a pure shear
    prestress it coincides with that of the two-by-two quadrature on
    rectangles.

    The figure is matched within grade A, 2 per cent, of MacNeal, R. H.,
    and Harder, R. L., "A proposed standard set of problems to test finite
    element accuracy", Finite Elements in Analysis and Design, 1(1), 3-20,
    1985, p. 14, only from ``refinement=3``, too expensive for every run.
    The default ``refinement=1`` therefore compares ``-Tcr`` with the
    values of the table pre-determined by this same test, 18537.7 and
    10509.2 Nm, with a tolerance for the round-off of other platforms only,
    and with Fig. 13 within grade C, 20 per cent, as a sanity bound.
    ``refinement=2`` compares with 17113.2 and 9480.9 Nm and grade B, 10 per
    cent, and ``refinement=3`` with grade A.

    """
    # NOTE stacking sequences from Table 4 of Castro et al., from inwards to
    #      outwards. The element normal of this mesh points outwards and the
    #      element x axis is the cylinder axis, so the plies are listed from
    #      the bottom to the top of the laminate. That the sign conventions
    #      of the ply angles and of the torque agree with those of the paper
    #      is shown in tests/test_quad4_linear_buckling_cylinder_Nxy.py:
    #      Nxy < 0 reproduces the positive-torque intercepts of Fig. 13 and
    #      Nxy > 0 the negative ones, for both laminates
    stacks = {
        'Z11' : [+60, -60, 0, 0, +68, -68, +52, -52, +37, -37],
        'Z33' : [0, 0, 19, -19, 37, -37, 45, -45, 51, -51],
    }
    # NOTE intercepts of the FSDT-BC2 CC2 curves of Fig. 13 of Castro et al.
    #      with the axis of zero axial load, positive torque, in Nm
    reference_T = {
        'Z11' : 16.8e3,
        'Z33' : 9.2e3,
    }
    # NOTE -Tcr in Nm pre-determined with this same test at the same
    #      refinement, see the table of the docstring. The tolerance, rtol =
    #      1e-4, is for the round-off of other platforms and solvers only
    predetermined = {
        1: {'Z11': 18537.667, 'Z33': 10509.195},
        2: {'Z11': 17113.207, 'Z33': 9480.917},
    }
    # NOTE sanity bounds of the relative difference to Fig. 13 of Castro et
    #      al., the grades of MacNeal and Harder (1985), p. 14: C (20 %) with
    #      ntheta = 80, B (10 %) with ntheta = 160 and A (2 %) with
    #      ntheta = 240, which the table of the docstring shows to be met
    grade = {1: 0.20, 2: 0.10, 3: 0.02}

    for cyl in ['Z11', 'Z33']:
        stack = stacks[cyl]

        data = Quad4RData()
        probe = Quad4RProbe()

        L = 0.510 # m
        R = 0.250 # m
        b = 2*np.pi*R # m

        ntheta = int(round(80*refinement)) # circumferential
        nlength = int(ntheta*L/b)
        if nlength % 2 == 0:
            nlength += 1
        print('ntheta', ntheta)
        print('nlength', nlength)

        # NOTE material Geier1997 of Table 3 of Castro et al.
        E11 = 123.55e9
        E22 = 8.708e9
        nu12 = 0.319
        G12 = 5.695e9
        G13 = 5.695e9
        G23 = 3.400e9
        plyt = 0.125e-3
        laminaprop = (E11, E22, nu12, G12, G13, G23)

        # NOTE forcing 5/6 according to Castro et al., beginning of Section 6
        prop = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop,
                               shear_correction='constant')

        # NOTE a thousandth of the default hourglass stiffnesses, with which
        #      the values of the docstring were measured
        hgfactor = 0.001

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

        # NOTE in Fig. 1 and Eq. (13) of Castro et al. a positive torque (T)
        #      gives a negative shear force per unit length Nxy. The sign
        #      must be consistent with the sign of the ply angles because it
        #      changes the buckling load of an unsymmetric laminate
        Nxy = -1000/b
        print('Nxy', Nxy)

        quads = []
        init_k_KC0 = 0
        init_k_KG = 0
        t0 = time.time()
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
            quad.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
            quad.update_probe_xe(ncoords_flatten)
            quad.update_KC0(KC0r, KC0c, KC0v, prop, hgfactor_u=hgfactor,
                            hgfactor_v=hgfactor, hgfactor_w=hgfactor,
                            hgfactor_rx=hgfactor, hgfactor_ry=hgfactor)
            # NOTE the element x axis is the cylinder axis, so the membrane
            #      prestress in the element frame is (0, 0, Nxy)
            quad.update_KG_given_stress(0, 0, Nxy, KGr, KGc, KGv)
            quads.append(quad)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_KG += data.KG_SPARSE_SIZE

        print('elements created', time.time()-t0)

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

        print('sparse KC0 and KG created')

        # NOTE boundary condition CC2 of Table 1 of Castro et al.,
        #      v = w = w,x = 0 at both edges, with the axial displacement u
        #      and the rotation about the radial direction free. These are
        #      constraints on the radial and circumferential directions,
        #      which change from node to node, so KC0 and KG are transformed
        #      to nodal cylindrical degrees-of-freedom
        #      (u_r, u_theta, u_z, r_r, r_theta, r_z)
        T = _cylindrical_transformation(ncoords)
        KC0 = (T.T @ KC0 @ T).tocsc()
        KG = (T.T @ KG @ T).tocsc()

        bk = np.zeros(N, dtype=bool)
        edges = isclose(z, 0) | isclose(z, L)
        bk[0::DOF][edges] = True # w
        bk[1::DOF][edges] = True # v
        bk[4::DOF][edges] = True # w,x, rotation about the circumferential direction
        # NOTE removing the rigid axial translation with a single node
        middle = np.where(isclose(z, L/2))[0]
        bk[2::DOF][middle[0]] = True
        # NOTE earlier versions of this test held the axial displacement of the
        #      whole ring of nodes at mid-length and both rotations at the
        #      edges, which is not CC2. That ring constraint stiffens the
        #      torsional mode, whose buckles are inclined helices crossing
        #      the mid-length: for the enriched Quad4 at ntheta = 80 it raised
        #      Tcr of Z11 from -18980.9 to -22193.3 Nm, +16.9 per cent, and
        #      holding the rotation about the radial direction at the edges
        #      added another 1.1 per cent. Together they were the 16 per cent (Z11) and
        #      23 per cent (Z33) by which the converged values exceeded
        #      Fig. 13, -19197.8 and -11093.7 Nm at ntheta = 320 with
        #      this element

        bu = ~bk

        KC0uu = KC0[bu, :][:, bu]
        KGuu = KG[bu, :][:, bu]

        # NOTE the torsional buckling loads come in close pairs. With only 4
        #      load multipliers requested eigsh() misses members of these
        #      clusters for Z33 at ntheta = 80, which the verification of
        #      linear_buckling detects and reports as an error; 6 are enough
        num_eig_lb = max(mode+1, 6)

        eigvecs = np.zeros((N, num_eig_lb))

        # NOTE pyfe3d.solver.linear_buckling equilibrates the diagonal,
        #      estimates the Cayley shift and verifies that no lower load
        #      multiplier was missed, which a hardcoded shift does not
        eigvals, eigvecsu = linear_buckling(KC0uu, KGuu, num_eigvalues=num_eig_lb, tol=1e-9)

        eigvecs[bu] = eigvecsu
        eigvecs = T @ eigvecs # back to the global system
        print('eigvals', eigvals)

        print('linear buckling analysis OK')
        Tcr = eigvals[0]*Nxy*b*R
        print('Tcr =', Tcr)

        if plot_pyvista:
            import pyvista as pv

            contour_colorscale = 'coolwarm'
            background = 'gray'

            vector = eigvecs[:, mode]

            contour_label = 'Radial displacement'
            contour_vec = np.sqrt(vector[0::DOF]**2 + vector[1::DOF]**2)

            plotter = pv.Plotter(off_screen=False)
            faces_quad = []
            for q in quads:
                faces_quad.append([4, nid_pos[q.n1], nid_pos[q.n2], nid_pos[q.n3], nid_pos[q.n4]])
            faces_quad = np.array(faces_quad)
            quad_plot = pv.PolyData(ncoords, faces_quad)
            quad_plot[contour_label] = contour_vec
            plotter.add_mesh(quad_plot, scalars=contour_label,
                    cmap=contour_colorscale, edge_color='black', show_edges=True,
                    line_width=1.)
            #NOTE plotting coordinate system
            xaxis = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=R/3)
            plotter.add_mesh(xaxis, color='blue')
            yaxis = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=R/3)
            plotter.add_mesh(yaxis, color='yellow')
            zaxis = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=R/3)
            plotter.add_mesh(zaxis, color='green')

            plotter.set_background(background)
            plotter.parallel_projection = True
            plotter.show()

        # NOTE -Tcr is the positive torque T of Castro et al.
        error = -Tcr/reference_T[cyl] - 1
        print('relative difference to Fig. 13 of Castro et al.', error)
        if refinement in predetermined:
            assert isclose(-Tcr, predetermined[refinement][cyl], rtol=1e-4)
        if refinement in grade:
            assert abs(error) < grade[refinement]


def test_linear_buckling_cylinder_torsion_isotropic(refinement=1):
    r"""Torsional buckling of a moderately long isotropic cylinder with simply
    supported edges

    The critical shear stress is ``tau_cr = k_s*pi**2*D/(L**2*t)`` with the
    straight-line approximation ``k_s = 0.85*Z**(3/4)`` for simply supported
    edges, ``Z = L**2/(R*t)*sqrt(1 - nu**2)``, from

        Batdorf, S. B., Stein, M., Schildcrout, M., "Critical stress of
        thin-walled cylinders in torsion", NACA TN 1344, 1947, p. 3, with the
        symbols on p. 2 and the range of validity ``100 < Z < 10*R**2/t**2``
        on p. 4,

    and the same equation with the correlation factor set to one is
    Eq. (25), Sec. 4.2.4, p. 11, of NASA SP-8007, Buckling of thin-walled
    circular cylinders, 1965, revised 1968, valid there for
    ``50 < Z < 78*(R/t)**2*(1 - nu**2)``. The simply supported edges of
    Batdorf et al. are ``w = w,xx = v = 0`` with ``u`` unrestrained (TN 1344,
    p. 8), which is what the constraints below impose.

    Here ``R = 250 mm``, ``L = 4*R``, ``t = 4 mm`` and ``nu = 0.3``, so that
    ``Z = 954``, well inside both ranges, and the cylinder buckles with five
    circumferential waves. Measured with this test, with
    ``ntheta = 64*refinement`` elements around the circumference and nearly
    square elements, the critical ``N_xy`` in N/m and its ratio to the
    reference, with the wall time on a machine shared with other jobs::

        refinement  ntheta  elements    Nxy_cr    ratio    time
           0.5         32       640    765745.5   1.2962     1 s
           0.75        48      1440    659117.4   1.1157     2 s
           1           64      2560    623627.9   1.0556     6 s
           1.5         96      5760    599769.1   1.0152    24 s
           2          128     10240    591694.8   1.0016    55 s
           3          192     23424    585982.2   0.9919   191 s
        extrapolated                              0.984

    the extrapolation fitting ``a + c*ntheta**-p`` to the three finest meshes
    (p = 2.0). The converged element is therefore 1.6 per cent below the
    formula, which is within the accuracy of the formula itself: it is a
    straight-line fit of Batdorf's Galerkin solution of a Donnell-type
    equation, and Table 1, p. 20, of TN 1344 gives k_s = 153.0 at Z = 1000 in
    the second approximation, 1.2 per cent above the 151.2 of the straight
    line, while the third approximation is 1.6 and 2.2 per cent below the
    second at Z = 300 and 10000, the neighbouring entries where both are
    tabulated. Table 1 has no entry at the Z = 954 of this test;
    interpolated linearly in log k_s against log Z, the third
    approximation between Z = 300 and 10000, 61.47 and 851.9, gives
    k_s = 146.3, 0.3 per cent above the 145.9 of the straight line, and
    the second approximation between Z = 300 and 1000 gives 147.7, 1.3 per
    cent above it. The straight line is kept as the reference: it is the
    formula the paper gives, and the interpolation adds an error of its
    own. A cylinder with the R and L of
    test_linear_buckling_cylinder_Nxy and ``t = 1 mm``, Z = 992, nine to ten
    waves, converges with the enriched Quad4 to 0.995 of the formula, see
    tests/test_quad4_linear_buckling_cylinder_Nxy.py, but needs about twice
    the elements for the same accuracy, which is why the longer cylinder is
    used here. The extrapolated values of Quad4 and Quad4R on the cylinder
    of this test are both 0.984.

    The default ``refinement=1``, ``ntheta = 64``, compares ``N_xy`` with
    the value of the table pre-determined by this same test, with a
    tolerance for the round-off of other platforms only, and the ratio with
    the formula within grade B, 10 per cent, of MacNeal, R. H., and Harder,
    R. L., "A proposed standard set of problems to test finite element
    accuracy", Finite Elements in Analysis and Design, 1(1), 3-20, 1985,
    p. 14; from ``refinement=2`` the ratio is checked within grade A, 2 per
    cent.

    """
    data = Quad4RData()
    probe = Quad4RProbe()

    E = 70.e9 # Pa
    nu = 0.3
    R = 0.250 # m
    L = 4*R # m
    t = 0.004 # m
    b = 2*np.pi*R # m

    D = E*t**3/(12*(1 - nu**2))
    Z = L**2/(R*t)*np.sqrt(1 - nu**2)
    assert 100 < Z < 10*R**2/t**2
    k_s = 0.85*Z**0.75
    Nxy_cr_reference = k_s*np.pi**2*D/L**2
    print('Z', Z, 'k_s', k_s, 'Nxy_cr_reference', Nxy_cr_reference)

    ntheta = int(round(64*refinement)) # circumferential
    nlength = int(ntheta*L/b)
    if nlength % 2 == 0:
        nlength += 1
    print('ntheta', ntheta)
    print('nlength', nlength)

    prop = isotropic_plate(thickness=t, E=E, nu=nu)

    # NOTE the default hourglass stiffnesses, with which the values of the
    #      docstring were measured
    hgfactor = 1.

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

    Nxy = -1.
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
        quad.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop, hgfactor_u=hgfactor,
                        hgfactor_v=hgfactor, hgfactor_w=hgfactor,
                        hgfactor_rx=hgfactor, hgfactor_ry=hgfactor)
        quad.update_KG_given_stress(0, 0, Nxy, KGr, KGc, KGv)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

    # NOTE simply supported edges of Batdorf et al., w = v = 0 with u and the
    #      rotations free: for a cylinder along the global z axis the radial
    #      and circumferential translations are the global x and y ones
    bk = np.zeros(N, dtype=bool)
    edges = isclose(z, 0) | isclose(z, L)
    bk[0::DOF][edges] = True
    bk[1::DOF][edges] = True
    # NOTE removing the rigid axial translation with a single node
    middle = np.where(isclose(z, L/2))[0]
    bk[2::DOF][middle[0]] = True
    bu = ~bk

    KC0uu = KC0[bu, :][:, bu]
    KGuu = KG[bu, :][:, bu]

    # NOTE every buckling load of this isotropic cylinder is a degenerate
    #      pair, so 4 load multipliers are requested
    eigvals, eigvecsu = linear_buckling(KC0uu, KGuu, num_eigvalues=4, tol=1e-9)
    print('eigvals', eigvals)
    Nxy_cr = eigvals[0]*abs(Nxy)
    ratio = Nxy_cr/Nxy_cr_reference
    print('Nxy_cr', Nxy_cr, 'ratio', ratio)

    # NOTE Nxy_cr in N/m pre-determined with this same test at the same
    #      refinement, see the table of the docstring. The tolerance, rtol =
    #      1e-4, is for the round-off of other platforms and solvers only
    predetermined = {
        1: 623627.89,
        2: 591694.82,
    }
    if refinement in predetermined:
        assert isclose(Nxy_cr, predetermined[refinement], rtol=1e-4)
    # NOTE sanity bounds of the relative difference to the formula, the
    #      grades of MacNeal and Harder (1985), p. 14: B (10 %) with
    #      ntheta = 64 and A (2 %) from ntheta = 128, which the table of
    #      the docstring shows to be met
    grade = {1: 0.10, 2: 0.02, 3: 0.02}
    if refinement in grade:
        assert abs(ratio - 1) < grade[refinement]


if __name__ == '__main__':
    test_linear_buckling_cylinder_Nxy(mode=0, plot_pyvista=False, refinement=1)
    test_linear_buckling_cylinder_torsion_isotropic(refinement=1)
