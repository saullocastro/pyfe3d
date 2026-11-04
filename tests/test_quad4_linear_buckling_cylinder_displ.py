r"""Linear buckling of a laminated cylinder modelled with Quad4 under a
prescribed axial shortening

The buckling load of the cylinder Z11 of Castro et al. (2014) with
displacement control, with twice the shortening and with load control must
agree to round-off; with the default ``refinement`` it is also compared with
a value pre-determined by this test.
"""
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

    The cylinder is Z11 of Castro et al. (2014), with the geometry and stacking sequence of its Table 4, p. 307, the
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
    ``plot_pyvista`` plots the mode ``mode``.

    Notes
    -----
    Why there is no assertion against the paper: SS1 at both edges is not
    in it. Table 6, p. 308, of Castro et al. gives the axial buckling load
    of Z11 with SS2 and CC2, 282.82 kN with FSDT-BC2 and SS2 and 284.06 kN
    with FSDT-BC2 and CC2, the tests giving 228.0 kN (Z11) and 221.7 kN
    (Z23, the same laminate). Its CLPT-BC2 value with SS2, 288.73 kN, is
    that of Fig. 4(a), p. 308, for which the laminate is assumed
    orthotropic; with the full constitutive matrix, as in pyfe3d, Fig. 4(b)
    gives 280.08 kN, and Fig. 5(b), p. 308, gives 281.12 kN for CLPT-BC2
    with SS1 at one edge and CC1 at the other. Under the uniform prestress
    of the paper, SS1 at both edges constrains more than SS2 and less than
    SS1-CC1, so its CLPT load lies between 280.08 and 281.12 kN. The model
    here differs from that of the paper: its prebuckling field is that of a
    static analysis, not uniform near the edges, its kinematics is FSDT,
    lower than CLPT by 2.0 per cent with SS2 in Table 6, and its shear
    correction is not forced to the 5/6 of the paper (Sec. 6, p. 307).
    Measured with this test, ``Pcr`` in kN, with the default Quad4, the
    assumed transverse shear strains of the MITC4 element and the slopes of
    `w` of the geometric stiffness at the two-by-two points (measured on
    2026-11-01), and with the slopes at the centroid, the one-point
    transverse shear of ``shear_model = 0`` except in row 1, measured with
    ``shear_model = 1`` and ``centroid_slopes = 1`` on 2026-10-30, with the
    wall time of the latter on a machine shared with other jobs::

        refinement  ntheta  elements   default   centroid    time
             1         40       440     -494.8     -590.0      1 s
             2         80      1920     -330.0     -352.0     20 s
             4        160      8000     -291.5     -297.0    145 s
             6        240     18240     -284.4     -286.7    385 s
             8        320     32640               -283.2    942 s
        extrapolated                    -278.8     -278.9

    With ``shear_model = 0`` row 1 is -590.3 kN, 0.05 per cent more than
    with ``centroid_slopes = 1``.

    The extrapolation fits ``a + c*ntheta**-p`` to the three finest
    meshes, p = 2.0 for the default and p = 2.1 with the slopes at the
    centroid. The load converges as the square of the element size,
    to 0.4 per cent below the lower end of the CLPT interval, which is where
    an FSDT value is expected, but that is no literature value to assert
    against. The default ``refinement=1`` therefore also compares ``Pcr``
    with the value of the table pre-determined by this same test, with a
    tolerance for the round-off of other platforms only. The one-point
    transverse shear of the thin elements with the slopes at the two-by-two
    points, a combination that Quad4 does not offer, gave -494.9, -330.0
    and -81.6 kN with refinements 1, 2 and 4: the hourglass pattern of `w`
    had the geometric stiffness of the two-by-two quadrature and no
    transverse shear stiffness, and the load of a mesh-scale mode kept
    falling under refinement. The MITC4 strains of the default give that
    pattern its transverse shear stiffness, and the same slopes converge
    from above.

    Record of a related study, on the laminated cylinder Z22::

        a record of a related study on the laminated cylinder Z22 of
        Castro et al., [+-49/+-36/0_2] of Geier1997 (Tables 3 and 4,
        p. 307), under the uniform membrane prestress of Eq. (13) applied
        through update_KG_given_stress, with SS2 of Table 1, over both
        quadrilaterals and both drilling models. Quad4 with
        shear_model = 0 and Quad4R take the slopes of w at the centroid in
        the geometric stiffness; the default Quad4, with the MITC4
        transverse shear, takes them at the two-by-two points.
        The reference of the ratios is Pcr = -35858 N, from
        bfsccylinder/tests/test_linear_buckling_Z22_constant_stress.py,
        which uses a C1 cubic element and CLPT with ny = 60. Table 6 of
        Castro et al. gives 36.35 kN for CLPT-BC2 SS2, 36.29 kN for
        FSDT-BC2 SS2 and 34.4 kN for the test; 36.29 kN is the FSDT-BC2
        load of an orthotropic laminate, Fig. 6(b), p. 308, while with the
        full constitutive matrix, as in pyfe3d, Fig. 6(c) gives 35.27 kN,
        0.984 of the reference.

        Every case went through pyfe3d.solver.linear_buckling, which
        estimates the Cayley shift and verifies that no lower load
        multiplier was missed; where it reported missed multipliers of the
        close clusters with 4 requested, the case was repeated with 8. n
        is the number of circumferential waves of the critical mode, n = 0
        an axisymmetric mode:

               element  drilling ntheta elements  Pcr (N)  Pcr/ref  n
                Quad4       0       60    1080   -93690.2   2.613   8
                Quad4       0      120    4560   -51097.4   1.425   0
                Quad4       0      240   18240   -38811.0   1.082   0
                Quad4       0      480   73920   -36093.6   1.007   0
                Quad4       1       60    1080   -94549.7   2.637   0
                Quad4       1      120    4560   -49012.8   1.367   0
                Quad4       1      240   18240   -38736.3   1.080   1
                Quad4       1      480   73920   -36083.7   1.006   0
                Quad4R      0       60    1080   -89209.9   2.488   0
                Quad4R      0      120    4560   -45184.7   1.260   0
                Quad4R      0      240   18240   -37600.1   1.049   0
                Quad4R      0      480   73920   -35792.7   0.998   0
                Quad4R      1       60    1080   -67285.6   1.876   0
                Quad4R      1      120    4560   -42684.2   1.190   0
                Quad4R      1      240   18240   -37157.8   1.036   0
                Quad4R      1      480   73920   -35694.9   0.995   0

        The Quad4 rows were measured with
        shear_model = 0 and the slopes of w at the centroid. With the
        default Quad4, the MITC4 transverse shear and the slopes
        of w at the two-by-two points, drilling_model = 0, measured on
        2026-11-01 with a script that reproduces the row ntheta = 60 of
        shear_model = 0 to the 0.1 N: -88117.6, -51925.4, -38811.0 and
        -36093.6 N with ntheta = 60, 120, 240 and 480, 2.457, 1.448, 1.082
        and 1.007 of the reference, and -93751.2 N with ntheta = 60 and
        centroid_slopes = 1. No mesh-scale mode appears: the MITC4
        strains give the hourglass pattern of w its transverse shear
        stiffness. From ntheta = 240 the critical modes are axisymmetric,
        whose slopes of w are constant along each element, so that the two
        rules give the same load. Fitting a + c*ntheta**-p to the three
        finest meshes gives -35383 N (p = 2.3), +0.3 per cent of the
        35.27 kN of Fig. 6(c).

        Quad4 with drilling_model = 1 at ntheta = 240 has an axisymmetric
        mode 0.01 per cent above the n = 1 pair. Every element and both
        drilling models converge from above, and from
        ntheta = 120 the critical mode is axisymmetric, as in Fig. 6 of
        Castro et al., with 15 axial half-waves at ntheta = 120, 2.5
        elements per half-wave, and 17 at ntheta = 240 and 480, 4.5 and
        9.1 elements per half-wave. Fitting a + c*ntheta**-p to the three
        finest meshes gives -35322 N (p = 2.2) for Quad4 with
        drilling_model = 0, -35161 N (p = 2.0) with drilling_model = 1
        and -35227 N (p = 2.1) for Quad4R with drilling_model = 0, within
        0.3 per cent of the 35.27 kN of Fig. 6(c). The energy split
        computed as in the paragraph on the energy split below, for the
        mode of Quad4 with
        drilling_model = 0 at ntheta = 240, closing to 5e-15, is +136.27 per cent membrane, -167.74 coupling,
        +131.24 bending, 0.23 transverse shear and 0.00 drilling, the
        large membrane and coupling shares cancelling, a property of the
        unsymmetric laminate Z22 and not a defect.

        Every Tria3R value below is that of shear_model = 0 with
        alpha_shear_locking = 0.7. The default Tria3R takes the discrete
        shear gap field of Tria3DSG, shear_model = 1, with
        alpha_shear_locking = 0.

        With (Nxx, 0, 0) handed to every triangle, Tria3R appears to
        converge on this cylinder to about a quarter of the reference,
        the "as passed" column below. That prestress is wrong:

        update_KG_given_stress takes the membrane stress resultants in
        the ELEMENT coordinate system, as its docstring says. The
        element x axis runs from node 1 to node 2, so in this mesh the
        first triangle of every cell, (n1, n2, n3), has its x axis along
        the cylinder axis while the second, (n1, n3, n4), has it along
        the cell diagonal. The cells here are nearly square, 13.1 mm
        circumferentially by 13.4 mm axially at ntheta = 120, so that
        diagonal sits 44.3 degrees off the axis. Handing the same
        (Nxx, 0, 0) to both loads half the elements along their
        diagonal instead of along the cylinder axis.

        The uniform axial prestress of Eq. (13) is a tensor in the
        surface. With [T_3x3] the 3x3 transformation matrix of the
        element, whose columns are the element axes, and
        {a} = [T_3x3].T {0, 0, 1} the cylinder axis in element coordinates,
        the state N {a}{a}.T has components

                 Nxx_e = N*a_x**2,  Nyy_e = N*a_y**2,  Nxy_e = N*a_x*a_y

        which reduces to (N, 0, 0) exactly when the element x axis is
        along the axis. That is why the quadrilateral tables above need
        no correction at all, and Quad4 was re-run to confirm it: every
        one of its six entries came back bit-identical with and without
        the rotation.

        How wrong it is, and the reason it cannot be waved away as a
        small modelling choice: the three cyclic numberings of the
        second triangle describe the same triangle and give the same
        KC0 to 1e-15, so they must give the same buckling load. At
        ntheta = 60 they give, with (Nxx, 0, 0) passed blindly,

                 second triangle numbered along   Tria3DSG      Tria3R
                  the cell diagonal               -12570.5     -7637.7
                  the axial edge                  -91061.8    -53203.7
                  the circumferential edge         -6050.4     -3619.0

        a factor of fifteen from nothing but the order the nodes are
        listed in, whereas with the rotation the three agree to 2e-3 for
        Tria3DSG and 4e-3 for Tria3R. The axial-edge numbering is the one
        for which (Nxx, 0, 0) happens to be right, and it reproduces the
        rotated answer exactly. This is guarded by
        test_buckling_load_does_not_depend_on_triangle_node_numbering in
        tests/test_tria3dsg.py.

        The corrected convergence, two triangles per cell of the same
        structured mesh, so twice the elements but the identical node
        count and identical number of degrees of freedom as the
        quadrilateral study, drilling_model = 0 throughout:

               element    ntheta   as passed   corrected   waves   el/wave
                Tria3R        60       0.213       1.478      30       2.0
                Tria3R       120       0.242       1.128      16       7.5
                Tria3R       240       0.256       0.964       4      60.0
                Tria3DSG      60       0.351       2.534       8       7.5
                Tria3DSG     120       0.266       2.060       7      17.1
                Tria3DSG     240       0.253       1.262       4      60.0

        as multiples of the -35858 N reference, with the corresponding
        Quad4 column being 2.613, 1.425 and 1.082. The "as passed" column
        hands (Nxx, 0, 0) to every triangle and the "corrected" column the
        rotated prestress, everything else equal, which makes this a
        comparison of the prestress alone.

        Neither triangle converges to a quarter of the reference. Both
        approach it from above, Tria3R dipping just below at the finest
        mesh. Two cautions on reading the table as a convergence study:
        the number of circumferential waves wanders between meshes, 30,
        16 and 4 for Tria3R, which means several eigenvalues are close
        together and the critical one is not the same physical mode at
        every mesh; and ntheta = 480 has not been run for either
        triangle, so no converged value is claimed here. What the table
        does establish is that the factor-of-four deficit of the "as
        passed" column is an artefact of the prestress.

        The energy split of the critical mode, obtained as described in
        the paragraph on the energy split below, with gamma_rz pinned to
        the material frame A66
        in every assembly so that the split closes exactly, at
        ntheta = 60. Pcr/ref is the pinned value here, which is why it
        differs in the third digit from the table above:

               element    Pcr/ref  el/wave       A        B        D     Ats   drill
                Quad4       2.613      7.5   +71.60   -36.13   +64.49    0.03    0.01
                Tria3DSG    2.540      7.5   +70.78   -30.55   +56.32    2.17    1.28
                Tria3R      1.484      2.0   +31.71   +15.18   +14.37   36.62    2.12

        and all three close to 1e-14 relative. In cylindrical components
        the three modes are

               element      |u_r|      |u_theta|    |u_axial|
                Quad4        2.669e-02  3.103e-03    6.749e-04
                Tria3DSG     2.477e-02  2.877e-03    6.155e-04
                Tria3R       6.698e-03  8.993e-07    1.078e-04

        Tria3DSG and Quad4 agree to 3 per cent on the same eight-wave
        mode and their mode contents agree component by component to
        within 9 per cent.

        Tria3R is the odd one out and is already sitting on a mode at two
        elements per wavelength, the mesh Nyquist limit, carrying 37 per
        cent of its energy in transverse shear: a transverse shear
        compliance that admits a mode at the mesh scale, available on this
        coarse mesh because alpha_shear_locking divides its transverse
        shear stiffnesses by 1 + alpha*maxl**2/h**2,
        a factor of 1852 at this mesh, maxl being the cell diagonal
        sqrt(dcirc**2 + dz**2) = 38.6 mm against a 0.75 mm wall.

        The same split at ntheta = 120, where the two triangles have
        gone to different modes, so only the Ats column is comparable
        across the rows:

               element    Pcr/ref  el/wave       A        B        D     Ats   drill
                Tria3DSG    2.094     17.1  +108.94   -85.94   +65.60    1.24   10.17
                Tria3R      1.157      7.5   +72.50   -38.99   +27.73   23.82   14.94

        also closing to 1e-15. The transverse shear share is the whole
        point: 1.24 per cent for the discrete shear gap triangle and 23.82
        for the stabilised one, at a mesh where the stabilisation factor
        is still 437. Tria3DSG's drilling share of 10.17 per cent is worth
        a second look some time, being far above the 0.01 of Quad4 at
        ntheta = 60, and is the
        largest single thing left unexplained in this element on this
        problem.
        Pinning gamma_rz is needed because the element derives it from
        the ELEMENT frame A66, which differs between the two triangles of
        a cell since their frames are 44.3 degrees apart. Without pinning
        the drilling-only assembly does not use the same coefficient as
        the full one and the split fails to close, by 2e-3 to 3e-2 for
        the triangles while the quadrilaterals, whose material rotation is
        the identity throughout, closed at 1e-15. Pinning moves Pcr by
        2.0e-3 for Tria3DSG and 3.8e-3 for Tria3R at ntheta = 60, and by
        1.6e-2 and 2.6e-2 at ntheta = 120, against 1.1e-13 for Quad4 at
        both. Worth noting on its own account: the drilling coefficient
        of a triangulated mesh therefore differs between the two
        triangles of a cell, by enough to move this buckling load a per
        cent or two. Tria3R behaves the same way, so it is consistent
        across the family rather than a defect of one element, but a
        frame-invariant choice of gamma_rz would remove it.

        Sweeping alpha at ntheta = 60 with the corrected prestress:

               alpha    factor   Pcr/ref   waves   el/wave
                0.7    1852.0      1.478      30       2.0
                0.1     264.6      2.520       8       7.5
                0.01     26.5      3.975      14       4.3
                1e-6      0.003   10.309       5      12.0

        At the literature value alpha = 0.1 Tria3R selects the physical
        eight-wave mode and gives 2.520, against Quad4 at 2.613 and
        Tria3DSG at 2.534 on that same mode. At alpha = 0.7
        it selects the Nyquist mode instead. So on this cylinder
        alpha = 0.7 is not merely inaccurate, it changes which mode is
        critical, and the literature value is the better one. Yet
        alpha = 0.1 makes the plate of
        test_tria3r_natural_freq.py 30.6 per cent too stiff while
        alpha = 0.7 breaks this cylinder, so no single value serves both
        and alpha_shear_locking has to be verified per problem class.
        Tria3DSG needs no such parameter and agrees with Quad4 on the
        eight-wave mode of ntheta = 60.

        The energy split: KC0 is linear in the constitutive constants, so
        the strain energy of any displacement field splits exactly into
        the membrane (A), coupling (B), bending (D) and transverse shear
        (Ats) groups, plus the drilling term switched by gamma_rz. The same
        decomposition cannot be done for Quad4R, because its hourglass
        coefficients are built from the inverse of the A matrix, so
        zeroing A to isolate a group makes E1eq and E2eq singular.

        One mechanism accounts for the Nyquist mode of Tria3R: transverse
        shear compliance admitting a mode at the mesh scale. The
        alpha_shear_locking stabilisation divides the shear stiffnesses by
        1 + alpha*maxl**2/h**2, a factor of 1852 at ntheta = 60, so the
        mode is already critical on the coarsest mesh; reducing alpha to
        its literature value restores the physical mode there. Tria3DSG,
        whose discrete shear gap uses Ats as the laminate gives it and
        needs no stabilisation, carries 2.17 per cent of its energy in
        transverse shear at that mesh against Tria3R's 36.82, and agrees
        with Quad4 to 3 per cent on the eight-wave mode of ntheta = 60.
        For the one-point transverse shear of Quad4 the remedy is an
        assumed transverse shear field, the MITC4 treatment of Dvorkin and
        Bathe of the default Quad4, which removes the spurious modes while
        keeping the physical compliance, see
        tests/test_quad4_spurious_shear_mode.py, which documents the
        spurious modes of Hughes, Taylor and Kanoknukulchai (1977),
        Figure 10, and of Hughes (1980), Figure 8

    References
    ----------
    Castro, S. G. P., Mittelstedt, C., Monteiro, F. A. C., Arbelo, M. A.,
    Ziegmann, G., Degenhardt, R., Linear buckling predictions of unstiffened
    laminated composite cylinders and cones under various loading and
    boundary conditions using semi-analytical models, Composite Structures,
    118, 2014, pp. 303-315, https://doi.org/10.1016/j.compstruct.2014.07.037.

    """
    data = Quad4Data()
    probe = Quad4Probe()

    L = 0.510 # m
    R = 0.250 # m
    b = 2*np.pi*R # m

    ntheta = 40*refinement # circumferential
    nlength = int(ntheta*L/b)

    # NOTE material Geier1997 of Table 3 in the reference paper
    E11 = 123.55e9
    E22 = 8.708e9
    nu12 = 0.319
    G12 = 5.695e9
    G13 = 5.695e9
    G23 = 3.400e9
    plyt = 0.125e-3
    laminaprop = (E11, E22, nu12, G12, G13, G23)

    # NOTE cylinder Z11, Table 4 of the reference paper
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

    # NOTE cylinders with SS1 boundary condition as decribed in Table 1 of the reference paper
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
    # NOTE the geometric stiffness takes the full displacement field: a
    #      field that held the solution at the free degrees-of-freedom but
    #      zero at the prescribed ones would drop the shortening of the top
    #      edge from the prebuckling field of the elements of the top row,
    #      and gave -498.4 kN instead of -494.9 kN at refinement=1 with the
    #      one-point transverse shear and the slopes at the two-by-two
    #      points
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
    # NOTE Pcr in N pre-determined with this same test at the same
    #      refinement, see the table of the docstring. The tolerance, rtol =
    #      1e-4, is for the round-off of other platforms and solvers only
    predetermined = {
        1: -494816.25,
    }
    if refinement in predetermined:
        assert np.isclose(Pcr, predetermined[refinement], rtol=1e-4)

    # NOTE a record of a related study on the cylinder Z22 is in the Notes

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

