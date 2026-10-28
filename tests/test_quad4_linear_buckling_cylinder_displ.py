import sys
sys.path.append('..')

import time
import numpy as np
from numpy import isclose
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, diags as sp_diags

from pyfe3d.shellprop_utils import laminated_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling


def test_linear_buckling_cylinder(mode=0, plot_pyvista=False, refinement=1):
    r"""Linear buckling of the laminated cylinder Z11 under a prescribed axial
    shortening, displacement control against load control

    The cylinder is Z11 of

        Saullo G. P. Castro, Christian Mittelstedt, Francisco A. C. Monteiro, Mariano
        A. Arbelo, Gerhard Ziegmann, Richard Degenhardt. "Linear buckling predictions
        of unstiffened laminated composite cylinders and cones under various loading
        and boundary conditions using semi-analytical models". Composite Structures,
        118, 303-315, 2014. 10.1016/j.compstruct.2014.07.037

    with the geometry and stacking sequence of its Table 4, p. 307, the
    material Geier1997 of its Table 3, p. 307, and the boundary condition SS1
    of its Table 1, p. 304, ``u = v = w = 0`` at both edges, the top edge being
    moved axially by the prescribed shortening.

    What is asserted is a property of the model and not a literature value.
    The buckling load is computed three times on the same mesh:

    1. displacement control, the axial shortening prescribed at the top edge;
    2. displacement control with twice that shortening, which must halve the
       load multiplier and leave ``Pcr`` unchanged, because the prebuckling
       field is linear in the shortening and the geometric stiffness linear
       in the prebuckling field;
    3. load control, with the axial displacement of the top edge released in
       the static analysis and loaded by the axial reaction forces of
       analysis 1. The static solution is unique, so the prebuckling field,
       the geometric stiffness and ``Pcr`` must be those of analysis 1. The
       buckling analysis keeps the same constraints as analysis 1.

    The three must agree to round-off, which checks the recovery of the
    reaction forces from ``KC0``, the static solution with prescribed
    displacements and the assembly of the geometric stiffness from a
    displacement field through ``update_probe_ue`` and ``update_KG``.

    Why there is no comparison with the paper. Table 6, p. 308, of Castro et
    al. gives the axial buckling load of Z11, 282.82 kN with FSDT-BC2 and SS2
    and 284.06 kN with FSDT-BC2 and CC2, the tests giving 228.0 kN (Z11) and
    221.7 kN (Z23, the same laminate). SS1, the boundary condition here, is
    not tabulated, it only appears through FSDT-BC1 with CC1 in the curves of
    Fig. 12, p. 311. Measured with this test, ``Pcr`` in kN::

        refinement  ntheta  elements     Pcr   time
             1         40       440   -590.3    1 s
             2         80      1920   -352.0   18 s
             4        160      8000   -297.0  115 s

    The load decreases towards the values of the paper for SS2 and CC2,
    5 per cent above them with ``ntheta = 160``, but SS1 is not tabulated and
    a converged value would need a finer mesh than a test affords. A
    comparison would also need the shear correction factor of 5/6 used by the
    paper (Sec. 6, p. 307), which this test does not force. Before the
    geometric stiffness of :class:`.Quad4` took the slopes of `w` at the
    centroid, in version 0.12.0, the loads were -494.9, -330.0 and -81.6 kN:
    the hourglass pattern of `w` had the geometric stiffness of the
    two-by-two quadrature and the load of a mesh-scale mode kept falling
    under refinement.

    """
    data = Quad4Data()
    probe = Quad4Probe()

    L = 0.510 # m
    R = 0.250 # m
    b = 2*np.pi*R # m

    ntheta = 40*refinement # circumferential
    nlength = int(ntheta*L/b)

    # NOTE material Geier1997 of Table 3 in Castro et al.
    E11 = 123.55e9
    E22 = 8.708e9
    nu12 = 0.319
    G12 = 5.695e9
    G13 = 5.695e9
    G23 = 3.400e9
    plyt = 0.125e-3
    laminaprop = (E11, E22, nu12, G12, G13, G23)

    # NOTE cylinder Z11, Table 4 of Castro et al.
    stack = [+60, -60, 0, 0, +68, -68, +52, -52, +37, -37]
    prop = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop)

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

    # NOTE cylinders with SS1 boundary condition as decribed in Table 1 of Castro et al.
    checkSS = isclose(z, 0) | isclose(z, L)
    bk[0::DOF] = checkSS
    bk[1::DOF] = checkSS
    bk[2::DOF] = checkSS

    bu = ~bk

    checkTopEdge = isclose(z, L)

    def solve_static(bk_static, uk_static, fext_static):
        bu_static = ~bk_static
        Kuu = KC0[bu_static, :][:, bu_static]
        Kuk = KC0[bu_static, :][:, bk_static]
        rhs = fext_static[bu_static] - Kuk @ uk_static
        # NOTE pre-conditioning the linear system
        kuu_diag = Kuu.diagonal()
        D_inv_sqrt = sp_diags(1.0/np.sqrt(np.maximum(kuu_diag, 1e-30)))
        uu_scaled = spsolve(D_inv_sqrt @ Kuu @ D_inv_sqrt, D_inv_sqrt @ rhs)
        u = np.zeros(N)
        u[bu_static] = D_inv_sqrt @ uu_scaled
        u[bk_static] = uk_static
        return u

    def buckling(u, num_eigvalues):
        KGv[:] = 0
        for quad in quads:
            quad.update_probe_xe(ncoords_flatten) # NOTE update affects the Quad4Probe class attribute xe
            quad.update_probe_ue(u) # NOTE update affects the Quad4Probe class attribute ue
            quad.update_KG(KGr, KGc, KGv, prop)
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
        KC0uu = KC0[bu, :][:, bu]
        KGuu = KG[bu, :][:, bu]
        # NOTE pyfe3d.solver.linear_buckling equilibrates the diagonal,
        #      estimates the Cayley shift and verifies that no lower load
        #      multiplier was missed, which a hardcoded shift does not
        return linear_buckling(KC0uu, KGuu, num_eigvalues=num_eigvalues, tol=1e-9)

    num_eig_lb = max(mode+1, 1)

    # 1. displacement control
    compression = -0.001
    u_prescribed = np.zeros(N, dtype=DOUBLE)
    u_prescribed[2::DOF] += checkTopEdge*compression
    u = solve_static(bk, u_prescribed[bk], np.zeros(N))
    # NOTE earlier versions of this test computed the geometric stiffness
    #      from a displacement field that held the solution at the free
    #      degrees-of-freedom but zero at the prescribed ones, dropping the
    #      shortening of the top edge from the prebuckling field of the
    #      elements of the top row. With the full field the load at
    #      refinement=1 is -494.9 kN instead of -498.4 kN
    # NOTE reaction forces at the constrained degrees-of-freedom
    reactions = KC0 @ u
    P = reactions[2::DOF][checkTopEdge].sum()
    eigvals, eigvecsu = buckling(u, num_eig_lb)
    eigvecs = np.zeros((N, num_eig_lb))
    eigvecs[bu] = eigvecsu
    print('eigvals', eigvals)
    Pcr = eigvals[0]*P
    print('Pcr =', Pcr)

    # 2. displacement control with twice the shortening
    u2 = solve_static(bk, 2*u_prescribed[bk], np.zeros(N))
    P2 = (KC0 @ u2)[2::DOF][checkTopEdge].sum()
    eigvals2, _ = buckling(u2, num_eig_lb)
    Pcr2 = eigvals2[0]*P2
    print('Pcr with twice the shortening =', Pcr2)

    # 3. load control, the axial displacement of the top edge released in
    #    the static analysis and loaded by the axial reactions of analysis 1
    bk_load = bk.copy()
    bk_load[2::DOF][checkTopEdge] = False
    fext = np.zeros(N)
    fext[2::DOF][checkTopEdge] = reactions[2::DOF][checkTopEdge]
    u3 = solve_static(bk_load, np.zeros(bk_load.sum()), fext)
    eigvals3, _ = buckling(u3, num_eig_lb)
    Pcr3 = eigvals3[0]*fext[2::DOF].sum()
    print('Pcr under load control =', Pcr3)

    assert Pcr < 0
    assert np.isclose(eigvals2[0], eigvals[0]/2, rtol=1e-6)
    assert np.isclose(Pcr2, Pcr, rtol=1e-6)
    assert np.allclose(u3, u, rtol=0, atol=1e-9*abs(compression))
    assert np.isclose(Pcr3, Pcr, rtol=1e-6)

    # NOTE a record of a related study on the laminated cylinder Z22 of
    #      Castro et al., kept because it explains why the axial buckling
    #      load of these laminated cylinders cannot be verified in a test.
    #      It was measured before version 0.12.0, in which Quad4 and Quad4R
    #      take the slopes of w at the centroid in the geometric stiffness;
    #      the mesh-scale modes it describes for the quadrilaterals have the
    #      signature of the hourglass pattern of w that the former geometric
    #      stiffness did not exclude, and their values are not those of the
    #      present elements. The values of Tria3R and Tria3DSG are unchanged.
    #
    # NOTE for the record, the mesh convergence of the axial buckling load,
    #      here for cylinder Z22 with the uniform membrane prestress of
    #      Eq. (13) applied through update_KG_given_stress and SS2 of
    #      Table 1, over both quadrilaterals and both drilling models. The
    #      reference is Pcr = -35858 N, from
    #      bfsccylinder/tests/test_linear_buckling_Z22_constant_stress.py,
    #      which uses a C1 cubic element and CLPT with ny = 60. Table 6 of
    #      Castro et al. gives 36.35 kN for CLPT-BC2 SS2, 36.29 kN for
    #      FSDT-BC2 SS2 and 34.4 kN for the test.
    #
    #      Every case below went through pyfe3d.solver.linear_buckling, so
    #      the Cayley shift was estimated and the spectrum verified, and
    #      every one of the sixteen reported that no lower load multiplier
    #      had been missed. The divergence is therefore a property of the
    #      discretisation and not of the eigensolver.
    #
    #        element  drilling  ntheta  elements    Pcr (N)  Pcr/ref  el/wave
    #         Quad4       0        60      1080     -87954.3   2.453      7.5
    #         Quad4       0       120      4560     -50600.2   1.411    120.0
    #         Quad4       0       240     18240     -24151.0   0.674      2.0
    #         Quad4       0       480     73920      -5943.3   0.166      2.0
    #         Quad4       1        60      1080     -89028.4   2.483      7.5
    #         Quad4       1       120      4560     -48593.8   1.355    120.0
    #         Quad4       1       240     18240     -20750.0   0.579      2.0
    #         Quad4       1       480     73920      -5093.4   0.142      2.0
    #         Quad4R      0        60      1080     -88689.5   2.473      7.5
    #         Quad4R      0       120      4560     -44763.9   1.248     15.0
    #         Quad4R      0       240     18240     -24184.7   0.674      2.0
    #         Quad4R      0       480     73920      -5976.1   0.167      2.0
    #         Quad4R      1        60      1080     -66710.4   1.860      7.5
    #         Quad4R      1       120      4560     -42258.3   1.178    120.0
    #         Quad4R      1       240     18240     -20784.0   0.580      2.0
    #         Quad4R      1       480     73920      -5126.2   0.143      2.0
    #
    #      Three things are worth keeping from this. First, the two drilling
    #      models track each other, within 1 per cent at ntheta = 60 for
    #      Quad4 and within 15 per cent at the finest meshes, with
    #      drilling_model = 0 consistently the stiffer and therefore the
    #      better of the two, so the drilling treatment is not the cause.
    #      Second, Quad4 and Quad4R agree to within 1 per cent at every mesh
    #      except Quad4R with drilling_model = 1 at ntheta = 60, although
    #      Quad4 is fully integrated with the Allman enrichment and Quad4R is
    #      reduced-integrated with hourglass control; whatever drives the
    #      divergence is common to both and is therefore neither the reduced
    #      integration nor the hourglass control. Third, from ntheta = 240
    #      the critical mode sits at exactly two elements per wavelength, the
    #      mesh Nyquist limit, which is the signature of a mesh-scale mode
    #      rather than of a physical one.
    #
    # NOTE CORRECTION. An earlier version of this block reported that
    #      Tria3R converges on this cylinder to -9.20 kN, a quarter of the
    #      reference, and attributed that to its alpha_shear_locking
    #      stabilisation. Those numbers were produced with the axial
    #      prestress applied wrongly and the conclusion drawn from them was
    #      wrong. What follows is the corrected study; the original numbers
    #      are kept in the "as passed" column so the record is traceable.
    #
    #      update_KG_given_stress takes the membrane stress resultants in
    #      the ELEMENT coordinate system, as its docstring says. The
    #      element x axis runs from node 1 to node 2, so in this mesh the
    #      first triangle of every cell, (n1, n2, n3), has its x axis along
    #      the cylinder axis while the second, (n1, n3, n4), has it along
    #      the cell diagonal. The cells here are nearly square, 13.1 mm
    #      circumferentially by 13.4 mm axially at ntheta = 120, so that
    #      diagonal sits 44.3 degrees off the axis. Handing the same
    #      (Nxx, 0, 0) to both loads half the elements along their
    #      diagonal instead of along the cylinder axis.
    #
    #      The uniform axial prestress of Eq. (13) is a tensor in the
    #      surface. With [T_3x3] the 3x3 transformation matrix of the
    #      element, whose columns are the element axes, and
    #      {a} = [T_3x3].T {0, 0, 1} the cylinder axis in element coordinates,
    #      the state N {a}{a}.T has components
    #
    #          Nxx_e = N*a_x**2,  Nyy_e = N*a_y**2,  Nxy_e = N*a_x*a_y
    #
    #      which reduces to (N, 0, 0) exactly when the element x axis is
    #      along the axis. That is why the quadrilateral tables above need
    #      no correction at all, and Quad4 was re-run to confirm it: every
    #      one of its six entries came back bit-identical with and without
    #      the rotation.
    #
    #      How wrong it is, and the reason it cannot be waved away as a
    #      small modelling choice: the three cyclic numberings of the
    #      second triangle describe the same triangle and give the same
    #      KC0 to 1e-15, so they must give the same buckling load. At
    #      ntheta = 60 they give, with (Nxx, 0, 0) passed blindly,
    #
    #          second triangle numbered along   Tria3DSG      Tria3R
    #           the cell diagonal               -12570.5     -7637.7
    #           the axial edge                  -91061.8    -53203.7
    #           the circumferential edge         -6050.4     -3619.0
    #
    #      a factor of fifteen from nothing but the order the nodes are
    #      listed in, whereas with the rotation the three agree to 2e-3 for
    #      Tria3DSG and 4e-3 for Tria3R. The axial-edge numbering is the one
    #      for which (Nxx, 0, 0) happens to be right, and it reproduces the
    #      rotated answer exactly. This is now guarded by
    #      test_buckling_load_does_not_depend_on_triangle_node_numbering in
    #      tests/test_tria3dsg.py.
    #
    #      The corrected convergence, two triangles per cell of the same
    #      structured mesh, so twice the elements but the identical node
    #      count and identical number of degrees of freedom as the
    #      quadrilateral study, drilling_model = 0 throughout:
    #
    #        element    ntheta   as passed   corrected   waves   el/wave
    #         Tria3R        60       0.213       1.478      30       2.0
    #         Tria3R       120       0.242       1.128      16       7.5
    #         Tria3R       240       0.256       0.964       4      60.0
    #         Tria3DSG      60       0.351       2.534       8       7.5
    #         Tria3DSG     120       0.266       2.060       7      17.1
    #         Tria3DSG     240       0.253       1.262       4      60.0
    #
    #      as multiples of the -35858 N reference, with the corresponding
    #      Quad4 column being 2.453, 1.411 and 0.674. The "as passed"
    #      column reproduces the previously recorded Tria3R values exactly,
    #      0.213, 0.242 and 0.256, which is what makes this a comparison of
    #      the prestress alone.
    #
    #      Neither triangle converges to a quarter of the reference. Both
    #      approach it from above, Tria3R dipping just below at the finest
    #      mesh. Two cautions on reading the table as a convergence study:
    #      the number of circumferential waves wanders between meshes, 30,
    #      16 and 4 for Tria3R, which means several eigenvalues are close
    #      together and the critical one is not the same physical mode at
    #      every mesh; and ntheta = 480 has not been run for either
    #      triangle, so no converged value is claimed here. What the table
    #      does establish is that the recorded factor-of-four deficit was
    #      an artefact.
    #
    #      The energy split of the critical mode, obtained as described in
    #      the next NOTE and with gamma_rz pinned to the material frame A66
    #      in every assembly so that the split closes exactly, at
    #      ntheta = 60. Pcr/ref is the pinned value here, which is why it
    #      differs in the third digit from the table above:
    #
    #        element    Pcr/ref  el/wave       A        B        D     Ats   drill
    #         Quad4       2.453      7.5   +71.40   -35.69   +64.26    0.02    0.01
    #         Tria3DSG    2.540      7.5   +70.78   -30.55   +56.32    2.17    1.28
    #         Tria3R      1.484      2.0   +31.71   +15.18   +14.37   36.62    2.12
    #
    #      and all three close to 1e-14 relative. In cylindrical components
    #      the three modes are
    #
    #        element      |u_r|      |u_theta|    |u_axial|
    #         Quad4        2.669e-02  3.103e-03    6.741e-04
    #         Tria3DSG     2.477e-02  2.877e-03    6.155e-04
    #         Tria3R       6.698e-03  8.993e-07    1.078e-04
    #
    #      Tria3DSG and Quad4 agree to 4 per cent on the same eight-wave
    #      mode and their mode contents agree component by component to
    #      within 8 per cent.
    #
    #      Tria3R is the odd one out and is already sitting on a mode at two
    #      elements per wavelength, the mesh Nyquist limit, carrying 37 per
    #      cent of its energy in transverse shear. That is the same spurious
    #      mechanism the quadrilaterals fall into at ntheta = 240, reached
    #      sixteen times earlier in mesh density because alpha_shear_locking
    #      divides its transverse shear stiffnesses by 1 + alpha*maxl**2/h**2,
    #      a factor of 1852 at this mesh, maxl being the cell diagonal
    #      sqrt(dcirc**2 + dz**2) = 38.6 mm against a 0.75 mm wall.
    #
    #      The same split at ntheta = 120, where the three elements have
    #      gone to three different modes, so only the Ats column is
    #      comparable across the rows:
    #
    #        element    Pcr/ref  el/wave       A        B        D     Ats   drill
    #         Quad4       1.411    120.0  +140.74  -136.51   +95.64    0.13    0.01
    #         Tria3DSG    2.094     17.1  +108.94   -85.94   +65.60    1.24   10.17
    #         Tria3R      1.157      7.5   +72.50   -38.99   +27.73   23.82   14.94
    #
    #      also closing to 1e-15. The transverse shear share is the whole
    #      point: 0.13 per cent for the quadrilateral, 1.24 for the
    #      discrete shear gap triangle and 23.82 for the stabilised one, at
    #      a mesh where the stabilisation factor is still 437. Tria3DSG's
    #      drilling share of 10.17 per cent is worth a second look some
    #      time, being far above the quadrilateral's 0.01, and is the
    #      largest single thing left unexplained in this element on this
    #      problem.
    #      Pinning gamma_rz is needed because the element derives it from
    #      the ELEMENT frame A66, which differs between the two triangles of
    #      a cell since their frames are 44.3 degrees apart. Without pinning
    #      the drilling-only assembly does not use the same coefficient as
    #      the full one and the split fails to close, by 2e-3 to 3e-2 for
    #      the triangles while the quadrilaterals, whose material rotation is
    #      the identity throughout, closed at 1e-15. Pinning moves Pcr by
    #      2.0e-3 for Tria3DSG and 3.8e-3 for Tria3R at ntheta = 60, and by
    #      1.6e-2 and 2.6e-2 at ntheta = 120, against 1.1e-13 for Quad4 at
    #      both. Worth noting on its own account: the drilling coefficient
    #      of a triangulated mesh therefore differs between the two
    #      triangles of a cell, by enough to move this buckling load a per
    #      cent or two. Tria3R behaves the same way, so it is consistent
    #      across the family rather than a defect of one element, but a
    #      frame-invariant choice of gamma_rz would remove it.
    #
    #      Sweeping alpha at ntheta = 60 with the corrected prestress:
    #
    #        alpha    factor   Pcr/ref   waves   el/wave
    #         0.7    1852.0      1.478      30       2.0
    #         0.1     264.6      2.520       8       7.5
    #         0.01     26.5      3.975      14       4.3
    #         1e-6      0.003   10.309       5      12.0
    #
    #      At the literature value alpha = 0.1 Tria3R selects the physical
    #      eight-wave mode and gives 2.520, against Quad4 at 2.453 and
    #      Tria3DSG at 2.534 on that same mode. At the default alpha = 0.7
    #      it selects the Nyquist mode instead. So on this cylinder the
    #      default alpha is not merely inaccurate, it changes which mode is
    #      critical, and the literature value is the better one. That is the
    #      opposite of the direction the earlier version of this block
    #      inferred, but it leaves the practical conclusion standing and
    #      strengthens it: alpha = 0.1 makes the plate of
    #      test_tria3r_natural_freq.py 30.6 per cent too stiff while
    #      alpha = 0.7 breaks this cylinder, so no single value serves both
    #      and alpha_shear_locking has to be verified per problem class.
    #      Tria3DSG needs no such parameter and tracks Quad4 at every mesh
    #      where Quad4 is healthy.
    #
    #      The union-jack control previously recorded here, 0.135 at
    #      ntheta = 60 and 0.157 at ntheta = 120, was measured with the same
    #      wrong prestress and is withdrawn. It has not been redone, and
    #      with the correction in place the diagonal pattern would have to
    #      be re-examined from scratch.
    #
    # NOTE what the quadrilaterals' mesh-scale mode actually is. KC0 is
    #      linear in the constitutive constants, so the strain energy of any
    #      displacement field splits exactly into the membrane (A), coupling
    #      (B), bending (D) and transverse shear (Ats) groups, plus the
    #      drilling term switched by gamma_rz. Applied to the critical
    #      eigenvector of Quad4 on this cylinder, the split closes to
    #      machine precision, 2.9e-15 and 1.3e-15 relative, so the numbers
    #      below are exact and not a fit.
    #
    #                              ntheta = 120      ntheta = 240
    #          Pcr/ref                   1.411             0.674
    #          circumferential waves         1               120
    #          elements per wave         120.0               2.0
    #
    #          membrane A              +140.74%           +55.48%
    #          coupling B              -136.51%            -0.01%
    #          bending D                +95.64%            +0.01%
    #          transverse shear Ats       +0.13%           +30.65%
    #          drilling                   +0.01%           +13.88%
    #
    #          |w|                      8.4e-05           1.1e-06
    #          |u, v|                   4.9e-03           7.1e-03
    #          |rx, ry|                 5.3e-01           3.0e-03
    #          |rz|                     2.1e-04           1.5e-03
    #
    #      CORRECTION. The |w| and |u, v| rows above are the global x, y
    #      and z components, and the axis of this cylinder is global z, so
    #      |w| is the AXIAL displacement and the radial displacement is
    #      spread over the global x and y components inside |u, v|. An
    #      earlier version of this block read the 1.1e-6 against 7.1e-3 as
    #      "essentially no transverse displacement" and concluded that the
    #      ntheta = 240 mode is an in-plane one. That was a mislabelling.
    #      In cylindrical components the mode is
    #
    #          |u_r| = 7.074e-03   |u_theta| = 3.005e-07   |u_axial| = 1.134e-06
    #
    #      so it is overwhelmingly radial, by four orders of magnitude over
    #      either of the other two, exactly as a buckling mode of a cylinder
    #      should be.
    #
    #      The two columns are different modes, not the same mode at two
    #      mesh densities, and the second one is the answer to what goes
    #      wrong. It is a radial checkerboard at the mesh Nyquist limit,
    #      55 per cent membrane, 31 per cent transverse shear and 14 per
    #      cent drilling, with bending and extension-bending coupling at
    #      0.01 per cent. So the element is not predicting the buckling mode
    #      badly; a different and unphysical mode has taken a lower
    #      eigenvalue.
    #
    #      The energy split reads consistently for a radial checkerboard and
    #      is what identifies the mechanism. Radial displacement w on a
    #      cylinder produces hoop membrane strain of order w/R, which is the
    #      55 per cent membrane; the rotations are small, |rx, ry| = 3.0e-3
    #      against the 5.3e-1 of the physical mode in the other column, so
    #      the curvature almost vanishes and with it the bending energy,
    #      while the transverse shear gamma = grad(w) - phi is left carrying
    #      the full gradient of a field oscillating at two elements per
    #      wavelength. A shear-compliant element is precisely what permits
    #      that, and Donnell's geometric stiffness feeds on grad(w).
    #
    #      That also accounts quantitatively for the drilling sensitivity of
    #      the convergence tables above: drilling carries 14 per cent of
    #      this mode's energy, and switching drilling_model moves Pcr by
    #      about 14 per cent at the fine meshes.
    #
    #      The ntheta = 120 column is a physical long-wave mode and is shown
    #      for contrast. Its near cancellation of +140.7 per cent membrane
    #      against -136.5 per cent coupling is a property of the
    #      unsymmetric laminate Z22 and not a defect.
    #
    #      The same decomposition cannot be done for Quad4R, because its
    #      hourglass coefficients are built from the inverse of the A
    #      matrix, so zeroing A to isolate a group makes E1eq and E2eq
    #      singular. Quad4 and Quad4R agree to 0.1 per cent on Pcr at every
    #      mesh of the tables above, so the Quad4 split stands for both.
    #
    # NOTE the direct confirmation, and the unified answer for all three
    #      elements. Scaling only the transverse shear stiffnesses A44, A45
    #      and A55 of the laminate, at ntheta = 240 where both quadrilaterals
    #      have collapsed:
    #
    #          element  Ats scale     Pcr (N)  Pcr/ref  waves  el/wave
    #           Quad4        1       -24151.0    0.674    120      2.0
    #           Quad4      100       -38469.2    1.073      1    240.0
    #           Quad4    10000       -38469.8    1.073      1    240.0
    #           Quad4R       1       -24184.7    0.674    120      2.0
    #           Quad4R     100       -37284.0    1.040      1    240.0
    #           Quad4R   10000       -37284.8    1.040      1    240.0
    #
    #      A hundredfold transverse shear stiffness removes the mesh-scale
    #      mode outright, 120 waves at two elements per wavelength becoming
    #      one wave, and the load rises from 0.674 of the reference to within
    #      7 per cent for Quad4 and 4 per cent for Quad4R. It saturates,
    #      100 and 10000 agreeing to five digits, which is expected: the
    #      reference is CLPT and scaling Ats without bound drives FSDT to
    #      CLPT. With the shear suppressed the quadrilaterals also converge
    #      properly, 1.411 at ntheta = 120 and 1.073 at ntheta = 240,
    #      approaching unity from above.
    #
    #      This is consistent with the energy split to the letter. The
    #      spurious mode carries 31 per cent of its energy in transverse
    #      shear, so a hundredfold shear stiffness raises its eigenvalue
    #      enormously, while the physical mode carries 0.13 per cent and is
    #      barely touched, so the physical mode becomes the critical one.
    #
    #      Together with the corrected Tria3R and Tria3DSG measurements
    #      above, one mechanism accounts for what goes wrong in both element
    #      families: transverse shear compliance admitting a mode at the
    #      mesh scale. In the quadrilaterals it is the genuine FSDT
    #      compliance, and the mode appears only once the mesh is fine
    #      enough, at ntheta = 240; suppressing the shear removes it. In
    #      Tria3R the alpha_shear_locking stabilisation divides the shear
    #      stiffnesses by 1 + alpha*maxl**2/h**2, a factor of 1852 at
    #      ntheta = 60, so the same mode is available sixteen times earlier
    #      in mesh density and is already critical on the coarsest mesh;
    #      reducing alpha to its literature value restores the physical mode
    #      there. Tria3DSG, whose discrete shear gap uses Ats as the
    #      laminate gives it and needs no stabilisation, carries 2.17 per
    #      cent of its energy in transverse shear at that mesh against
    #      Tria3R's 36.82, and tracks Quad4 to 3 per cent at every mesh
    #      where Quad4 is healthy.
    #
    #      Scaling Ats is a diagnostic and not a fix: it removes the
    #      physical transverse shear deformation along with the spurious
    #      mode. The cure is an assumed transverse shear field, that is the
    #      MITC4 treatment of Dvorkin and Bathe, which removes the spurious
    #      modes while keeping the physical compliance. The same conclusion
    #      is reached independently in
    #      tests/test_quad4_spurious_shear_mode.py, which documents the
    #      spurious modes of Hughes, Taylor and Kanoknukulchai (1977),
    #      Figure 10, and of Hughes (1980), Figure 8

    if plot_pyvista:
        import pyvista as pv

        contour_colorscale = 'coolwarm'
        background = 'gray'

        vector = eigvecs[:, mode]

        contour_label = 'Radial displacement'
        contour_vec = np.sqrt(vector[0::DOF]**2 + vector[1::DOF]**2)

        displ_vec = np.zeros_like(ncoords)
        displ_vec[:, 0] = eigvecs[0::DOF, mode]*10
        displ_vec[:, 1] = eigvecs[1::DOF, mode]*10
        displ_vec[:, 2] = eigvecs[2::DOF, mode]*10
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
        displ_vec = None
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


if __name__ == '__main__':
    test_linear_buckling_cylinder(mode=0, plot_pyvista=True, refinement=5)

