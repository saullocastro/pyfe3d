r"""Inertia relief of the stick model of the MSC Nastran example HA144A

The forward-swept-wing (FSW) airplane with canard of Rodden and Johnson (1994),
Example HA144A, "FSW Airplane in Level Flight", pp. 181-205, is modelled with
:class:`.BeamC` elements from the bulk data of Listing 7-2 (pp. 196-197): half
of the airplane with symmetry constraints, the weights (CONM2) of the wing
attached to the elastic axis by rigid bars (RBAR), represented exactly by a
transformation of the degrees of freedom, and the reference support (SUPORT)
at GRID 90, vertical translation and pitch. The inertia relief is computed
with :func:`pyfe3d.solver.apply_inertia_relief` (Barnett et al. 1995).

The tests check, against the output of Listing 7-3 (pp. 198-205):

- the output of the grid point weight generator, p. 198: the rigid-body mass
  matrix about GRID 90, the weight 8000 lb, the centre of gravity 2.181625 ft
  aft of GRID 90 and the pitch inertia about it, 8.928944e5 lb ft^2;
- the stiffness, the geometry and the rigid bars: the nodal loads that are in
  equilibrium with the published forces of all bar elements (p. 204), applied
  to the model, give the published displacements relative to GRID 90 (pp.
  203-204) to the printed precision, for the subcases 2 and 3;
- the published normal forces of the 40 aerodynamic boxes of subcase 2 (p.
  201) with the weights balance in force and moment; transferred to the
  beam nodes by the inverse-distance weights of Gryszka and Castro (2026),
  Eq. (5), plus the moments of the offsets, they give zero rigid-body
  accelerations, `\boldsymbol{\Phi}^\top \boldsymbol{F}_{tot} =
  \boldsymbol{0}`, zero reactions at the reference support, and the forces of
  element 101, which carries only the weight of the 1500 lb mass at GRID 97:
  bending moment -1.5e4 lb ft at end B and shear 1.5e3 lb;
- the same box forces, applied as Nastran applies them, at the box centres
  with their moments about the midchord (pp. 20 and 186), and transferred
  with the linear splines SPLINE2 1501 and 1601 of the deck (emulated by
  :func:`spline2_matrix` from Rodden and Johnson 1994, pp. 33-43, and the
  Quick Reference Guide, pp. 3137-3138): the transfer conserves the
  resultant force and moment, the canard spline is the lever rule between
  GRID 98 and 99, and all published bar forces and displacements of
  subcase 2 are reproduced to the printed digits.

Rodden, W. P., Johnson, E. H., 1994, MSC/NASTRAN Aeroelastic Analysis User's
Guide, Version 68, The MacNeal-Schwendler Corporation, Los Angeles.

MSC Software, 2020, MSC Nastran 2020 Quick Reference Guide, Newport Beach,
Bulk Data entry SPLINE2, pp. 3137-3138.

Gryszka, M., Castro, S. G. P., 2026, Angled spar battery-wing system for large
all-electric aircraft, preprint, https://engrxiv.org/preprint/view/7893/version/10171

Barnett, A. R., Widrick, T. W., Ludwiczak, D. R., 1995, Closed-Form Static
Analysis with Inertia Relief and Displacement-Dependent Loads Using a
MSC/NASTRAN DMAP Alter, NASA TM-106836, https://ntrs.nasa.gov/citations/19950013233

Units: lb, ft, s; the masses are the weights times PARAM WTMASS = 0.031081
s^2/ft, in slug.
"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.spatial import cKDTree

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE
from pyfe3d.solver import apply_inertia_relief

WTMASS = 0.031081 # PARAM WTMASS, s^2/ft
G_ACC = 1/WTMASS  # ft/s^2

# Listing 7-2, p. 196: GRID (cards 42-52), coordinates in ft
GRIDS = {90: (15., 0., 0.), 97: (0., 0., 0.), 98: (10., 0., 0.),
         99: (20., 0., 0.), 100: (30., 0., 0.),
         110: (27.11325, 5., 0.), 111: (24.61325, 5., 0.),
         112: (29.61325, 5., 0.), 120: (21.33975, 15., 0.),
         121: (18.83975, 15., 0.), 122: (23.83975, 15., 0.)}
# CBAR (cards 12-17): EID: (PID, GA, GB), orientation vector (0, 0, 1)
CBARS = {100: (100, 90, 99), 101: (100, 97, 98), 102: (100, 98, 90),
         103: (100, 99, 100), 110: (101, 100, 110), 120: (101, 110, 120)}
# MAT1 1 (card 53), psf
E_MAT1, G_MAT1 = 1.44e9, 5.40e8
# PBAR (cards 59-64): A, I1 (plane 1, vertical), I2, J
PBARS = {100: dict(A=2.0, I1=0.173611, I2=0.15, J=0.5),
         101: dict(A=1.5, I1=0.173611, I2=2.0, J=0.462963)}
# CONM2 (cards 18-25), weights in lb
CONM2 = {97: 1500., 98: 1500., 99: 1500., 100: 1500., 111: 600.,
         112: 400., 121: 600., 122: 400.}
# RBAR (cards 65-68): dependent GB: independent GA (CNA = 123456)
RBARS = {111: 110, 112: 110, 121: 120, 122: 120}
# SPC1 1 (cards 71-72): 0-based dof components, see pyfe3d.DOF
SPCS = {90: (0, 1, 3, 5), 97: (1, 3, 5), 98: (1, 3, 5), 99: (1, 3, 5),
        100: (1, 3, 5)}
# SUPORT (card 77): GRID 90, components 3 and 5
SUPORT = {90: (2, 4)}

# Listing 7-3, p. 198, grid point weight generator, MO about GRID 90 (lb, ft)
MO_PUB = np.array([
    [8.000000E+03, 0., 0., 0., 0., -2.000000E+04],
    [0., 8.000000E+03, 0., 0., 0., 1.745300E+04],
    [0., 0., 8.000000E+03, 2.000000E+04, -1.745300E+04, 0.],
    [0., 0., 2.000000E+04, 2.500000E+05, -1.456625E+05, 0.],
    [0., 0., -1.745300E+04, -1.456625E+05, 9.309703E+05, 0.],
    [-2.000000E+04, 1.745300E+04, 0., 0., 0., 1.180970E+06]])
XCG_PUB, YCG_PUB = 2.181625, 2.500000 # ft, from GRID 90, p. 198
IS_PUB = np.array([[2.000000E+05, 1.020300E+05, 0.],
                   [1.020300E+05, 8.928944E+05, 0.],
                   [0., 0., 1.092894E+06]]) # I(S), lb ft^2, p. 198

# Listing 7-3, forces in bar elements, p. 204: EID: (bending moment plane 1
# end A, end B, shear plane 1, torque); plane 2 and axial forces are zero
BAR_PUB = {
    2: {100: (-1.885620E+04, -2.271239E+04, 7.712394E+02, 0.),
        101: (9.094947E-13, -1.500000E+04, 1.500000E+03, 0.),
        102: (-1.500000E+04, -1.885620E+04, 7.712394E+02, 0.),
        103: (-2.271239E+04, -4.598023E+04, 2.326784E+03, 0.),
        110: (6.164220E+04, 3.819550E+04, 4.061087E+03, 1.750427E+04),
        120: (3.274051E+04, 4.651097E+03, 2.432615E+03, 8.055939E+03)},
    3: {100: (-1.676073E+04, -1.852147E+04, 3.521470E+02, 0.),
        101: (0., -1.500000E+04, 1.500000E+03, 0.),
        102: (-1.500000E+04, -1.676073E+04, 3.521470E+02, 0.),
        103: (-1.852147E+04, -3.202391E+04, 1.350244E+03, 0.),
        110: (4.827697E+04, 2.721261E+04, 3.648454E+03, 9.105316E+03),
        120: (2.196529E+04, 9.632417E+00, 1.901416E+03, 1.668437E+01)}}
# Listing 7-3, displacements relative to GRID 90, subcase 2 p. 203, subcase
# 3 p. 204: GRID: (T3, R1, R2); T1, T2 and R3 are zero
DISP_PUB = {
    2: {97: (-6.264164E-03, 0., -6.385624E-04),
        98: (-8.785404E-04, 0., -3.385622E-04),
        99: (-1.007080E-03, 0., 4.156862E-04),
        100: (-1.125761E-02, 0., 1.789539E-03),
        110: (-2.503219E-03, 7.962558E-04, 2.716038E-03),
        111: (4.286876E-03, 7.962558E-04, 2.716038E-03),
        112: (-9.293314E-03, 7.962558E-04, 2.716038E-03),
        120: (2.737436E-02, 1.358045E-03, 3.470037E-03),
        121: (3.604945E-02, 1.358045E-03, 3.470037E-03),
        122: (1.869926E-02, 1.358045E-03, 3.470037E-03)},
    3: {97: (-5.984768E-03, 0., -6.176077E-04),
        98: (-8.086917E-04, 0., -3.176075E-04),
        99: (-8.673829E-04, 0., 3.528223E-04),
        100: (-9.000065E-03, 0., 1.363731E-03),
        110: (-2.312948E-03, 6.497573E-04, 1.981676E-03),
        111: (2.641242E-03, 6.497573E-04, 1.981676E-03),
        112: (-7.267138E-03, 6.497573E-04, 1.981676E-03),
        120: (1.953163E-02, 1.088871E-03, 2.236088E-03),
        121: (2.512185E-02, 1.088871E-03, 2.236088E-03),
        122: (1.394141E-02, 1.088871E-03, 2.236088E-03)}}
# Listing 7-3, p. 201, subcase 2 (Mach 0.9, q = 1200 psf): normal forces of
# the boxes 1000-1007 (canard) and 1100-1131 (wing), lb
BOX_FORCES_SC2 = np.array([
    1.126208E+03, 2.066928E+02, 8.708182E+01, 7.207389E+01,
    8.916525E+02, 1.486647E+02, 6.662047E+01, 6.373189E+01,
    -2.800374E+02, -2.920913E+01, -6.569765E+00, -1.519861E+00,
    3.960235E+01, -2.345699E+01, -4.447945E+00, -8.584092E+00,
    1.059489E+03, 1.148757E+02, 1.613169E+01, -1.779271E+01,
    8.106509E+02, 2.590438E+02, 7.052816E+01, 1.535013E+01,
    6.430974E+02, 2.752712E+02, 1.179050E+02, 3.788203E+01,
    5.166963E+02, 2.415806E+02, 1.338452E+02, 5.613377E+01,
    3.994048E+02, 1.907343E+02, 1.213484E+02, 6.331788E+01,
    2.643184E+02, 1.236906E+02, 8.494805E+01, 5.304620E+01])


def caero1_points(x1, y1, x12, x4, y4, x43, nspan, nchord, frac=0.25):
    r"""Points at the fraction ``frac`` of the box chord, midspan, of the
    boxes of a CAERO1 panel, numbered chordwise first, as the box
    identification numbers; ``frac=0.25`` gives the quarter-chord points,
    ``frac=0.5`` the box centres"""
    pts = []
    for j in range(nspan):
        eta = (j + 0.5)/nspan
        xle = x1 + eta*(x4 - x1)
        c = x12 + eta*(x43 - x12)
        y = y1 + eta*(y4 - y1)
        for i in range(nchord):
            pts.append((xle + (i + frac)/nchord*c, y, 0.))
    return np.array(pts)


def box_points(frac=0.25):
    r"""CAERO1 1000 (canard, 2 x 4 boxes) and 1100 (wing, 8 x 4 boxes),
    Listing 7-2, cards 8-11; the force of a doublet-lattice box acts at its
    quarter chord (p. 186), ``frac=0.25``; the aerodynamic grid point of a
    box, where Nastran applies the force and the moment about the midchord,
    is the box centre (pp. 20 and 186), ``frac=0.5``"""
    canard = caero1_points(10., 0., 10., 10., 5., 10., 2, 4, frac)
    wing = caero1_points(25., 0., 10., 13.45299, 20., 10., 8, 4, frac)
    return canard, wing


def skew(r):
    return np.array([[0., -r[2], r[1]], [r[2], 0., -r[0]], [-r[1], r[0], 0.]])


def build_model():
    r"""Stiffness and mass matrices of the HA144A stick model

    Returns a dict with the full matrices ``KC0`` and ``Mfull``, the RBAR
    transformation ``T`` from the independent degrees of freedom (``ind``)
    to all, the reduced ``K`` `= \boldsymbol{T}^\top \boldsymbol{K}_{C0}
    \boldsymbol{T}` and ``M``, and the sets ``bfix`` (SPC1) and ``bref``
    (SUPORT) over all degrees of freedom.
    """
    gids = list(GRIDS)
    idx = {g: i for i, g in enumerate(gids)}
    nn = len(gids)
    N = DOF*nn
    ncoords = np.array([GRIDS[g] for g in gids])
    x = ncoords.flatten()
    props = {}
    for pid, p in PBARS.items():
        prop = BeamProp()
        prop.A = p['A']
        prop.E = E_MAT1
        prop.G = G_MAT1
        prop.Iyy = p['I1'] # bending in the vertical plane, plane 1 of CBAR
        prop.Izz = p['I2']
        prop.J = p['J']
        # CBAR without K1 and K2: no transverse shear flexibility
        prop.G12 = prop.G13 = 1.e6*E_MAT1
        props[pid] = prop
    data = BeamCData()
    probe = BeamCProbe()
    ne = len(CBARS)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    beams = {}
    for i, (eid, (pid, ga, gb)) in enumerate(CBARS.items()):
        e = BeamC(probe)
        e.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        e.n1, e.n2 = idx[ga], idx[gb]
        e.c1, e.c2 = DOF*idx[ga], DOF*idx[gb]
        # element y axis horizontal, z axis vertical, such that the plane 1
        # of CBAR, (x, v) with v = (0, 0, 1), is the element plane xz
        vxy = np.cross([0., 0., 1.], ncoords[idx[gb]] - ncoords[idx[ga]])
        e.update_rotation_matrix(*vxy, x)
        e.update_probe_xe(x)
        e.update_KC0(KC0r, KC0c, KC0v, props[pid])
        beams[eid] = (e, props[pid])
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    Md = np.zeros(N)
    for g, w in CONM2.items():
        Md[DOF*idx[g]:DOF*idx[g] + 3] = w*WTMASS
    Mfull = coo_matrix((Md, (np.arange(N), np.arange(N))), shape=(N, N)).tocsc()
    # RBAR: u_B = u_A + theta_A x r = u_A - S(r) theta_A, theta_B = theta_A
    dep = np.zeros(N, dtype=bool)
    for gb in RBARS:
        dep[DOF*idx[gb]:DOF*idx[gb] + DOF] = True
    ind = ~dep
    col = -np.ones(N, dtype=int)
    col[ind] = np.arange(ind.sum())
    T = np.zeros((N, ind.sum()))
    T[ind, col[ind]] = 1.
    for gb, ga in RBARS.items():
        r = ncoords[idx[gb]] - ncoords[idx[ga]]
        ca = col[DOF*idx[ga]:DOF*idx[ga] + DOF]
        rb = DOF*idx[gb]
        T[rb:rb + 3, ca[:3]] = np.eye(3)
        T[rb:rb + 3, ca[3:]] = -skew(r)
        T[rb + 3:rb + 6, ca[3:]] = np.eye(3)
    T = csc_matrix(T)
    bfix = np.zeros(N, dtype=bool)
    for g, comps in SPCS.items():
        bfix[DOF*idx[g] + np.array(comps)] = True
    bref = np.zeros(N, dtype=bool)
    for g, comps in SUPORT.items():
        bref[DOF*idx[g] + np.array(comps)] = True
    return dict(gids=gids, idx=idx, N=N, ncoords=ncoords, KC0=KC0,
                Mfull=Mfull, Md=Md, T=T, ind=ind,
                K=csc_matrix(T.T @ KC0 @ T), M=csc_matrix(T.T @ Mfull @ T),
                beams=beams, props=props, bfix=bfix, bref=bref)


def rigid_body_modes(model, ref):
    r"""Translations and rotations about the point ``ref`` of all nodes"""
    N, ncoords = model['N'], model['ncoords']
    Phi = np.zeros((N, 6))
    for i, xyz in enumerate(ncoords):
        Phi[DOF*i:DOF*i + 3, :3] = np.eye(3)
        Phi[DOF*i + 3:DOF*i + 6, 3:] = np.eye(3)
        Phi[DOF*i:DOF*i + 3, 3:] = -skew(xyz - ref)
    return Phi


def weight_generator(model, ref_grid=90):
    r"""Rigid-body mass matrix about ``ref_grid``, in weight units, as the
    grid point weight generator: MO, the mass, the centre of gravity relative
    to ``ref_grid`` and the inertia matrix about it, I(S)"""
    ref = model['ncoords'][model['idx'][ref_grid]]
    Phi = rigid_body_modes(model, ref)
    MO = Phi.T @ (model['Mfull'] @ Phi)/WTMASS
    m = MO[0, 0]
    xcg = np.array([MO[1, 5], -MO[0, 5], MO[0, 4]])/m
    # inertia about the centre of gravity, I(S) of Nastran, positive products
    IS = np.zeros((3, 3))
    IS[0, 0] = MO[3, 3] - m*(xcg[1]**2 + xcg[2]**2)
    IS[1, 1] = MO[4, 4] - m*(xcg[0]**2 + xcg[2]**2)
    IS[2, 2] = MO[5, 5] - m*(xcg[0]**2 + xcg[1]**2)
    IS[0, 1] = IS[1, 0] = -(MO[3, 4] + m*xcg[0]*xcg[1])
    IS[0, 2] = IS[2, 0] = -(MO[3, 5] + m*xcg[0]*xcg[2])
    IS[1, 2] = IS[2, 1] = -(MO[4, 5] + m*xcg[1]*xcg[2])
    return MO, m, xcg, IS


def bar_forces(model, u):
    r"""Forces of the CBAR elements in the convention of Nastran: (bending
    moment of plane 1 at end A, at end B, shear of plane 1, torque)

    With the element `z` axis vertical, the internal forces ``finte`` of
    :class:`.BeamC` give `M_{1A}` = ``finte[4]``, `M_{1B}` = ``-finte[10]``,
    `V_1` = ``finte[8]`` and the torque ``finte[9]``; the plane 1 bending
    moment of Nastran is positive when the top fibre is in compression.
    """
    out = {}
    for eid, (e, prop) in model['beams'].items():
        e.update_probe_ue(u)
        e.update_probe_finte(prop)
        f = np.array(e.probe.finte)
        out[eid] = np.array([f[4], -f[10], f[8], f[9]])
    return out


def loads_from_bar_forces(model, bf):
    r"""Nodal loads in equilibrium with given bar forces (Nastran
    convention), the assembled element forces"""
    N = model['N']
    f = np.zeros(N)
    for eid, (M1a, M1b, V1, Tq) in bf.items():
        e, prop = model['beams'][eid]
        fl = np.zeros(12)
        fl[[2, 8, 4, 10, 3, 9]] = [-V1, V1, M1a, -M1b, -Tq, Tq]
        R = np.array([[e.r11, e.r12, e.r13], [e.r21, e.r22, e.r23],
                      [e.r31, e.r32, e.r33]])
        for k, n in enumerate((e.n1, e.n2)):
            f[DOF*n:DOF*n + 3] += R @ fl[6*k:6*k + 3]
            f[DOF*n + 3:DOF*n + 6] += R @ fl[6*k + 3:6*k + 6]
    return f


def transfer_loads(model, pts, forces, grids, npoints, moments=True):
    r"""Aerodynamic forces at ``pts`` to the nodes ``grids``

    The inverse-distance weights of Gryszka and Castro (2026), Eq. (5),
    `w_{ij} = d_{ij}^{-2}/\sum_k d_{kj}^{-2}` over the ``npoints`` nearest
    nodes `i` of each point `j`, distribute the forces to the translational
    degrees of freedom; ``moments=True`` adds the moments of the offsets,
    `w_{ij} (\boldsymbol{x}_j - \boldsymbol{x}_i) \times \boldsymbol{F}_j`,
    to the rotational degrees of freedom, such that the resultant force and
    moment are conserved.
    """
    idx, ncoords, N = model['idx'], model['ncoords'], model['N']
    xs = ncoords[[idx[g] for g in grids]]
    d, k = cKDTree(xs).query(pts, k=npoints)
    d = d.reshape(len(pts), npoints)
    k = k.reshape(len(pts), npoints)
    w = d**-2/(d**-2).sum(axis=1)[:, None]
    f = np.zeros(N)
    for j in range(len(pts)):
        for i in range(npoints):
            n = idx[grids[k[j, i]]]
            f[DOF*n:DOF*n + 3] += w[j, i]*forces[j]
            if moments:
                f[DOF*n + 3:DOF*n + 6] += w[j, i]*np.cross(
                    pts[j] - ncoords[n], forces[j])
    return f


def cord2r(a, b, c):
    r"""Origin and unit vectors `x`, `y`, `z` of a CORD2R entry given by the
    origin A, a point B on the `z` axis and a point C in the `xz` plane"""
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    ez = (b - a)/np.linalg.norm(b - a)
    ex = (c - a) - ((c - a) @ ez)*ez
    ex /= np.linalg.norm(ex)
    return a, ex, np.cross(ez, ex), ez


# CORD2R 1 (canard spline, through the hinge line at the quarter chord) and
# CORD2R 2 (wing spline, along the elastic axis), Listing 7-1, p. 192, and
# Listing 7-2, cards 50-57
CORD2R = {1: cord2r((12.5, 0., 0.), (12.5, 0., 10.), (20., 0., 0.)),
          2: cord2r((30., 0., 0.), (30., 0., 10.), (38.66025, 5., 0.))}
# SPLINE2 (Listing 7-2, cards 73-76): EID: (CAERO boxes, SET1 grids, DZ,
# DTOR, CID, DTHX, DTHY); SET1 1000 and 1100, cards 69-70
SPLINE2 = {1501: (slice(0, 8), [98, 99], 0., 1., 1, 1., -1.),
           1601: (slice(8, 40), [99, 100, 111, 112, 121, 122], 0., 1., 2,
                  -1., -1.)}


def spline2_matrix(model, grids, cid, pts, dz=0., dtor=1., dthx=-1.,
                   dthy=-1.):
    r"""Interpolation matrix of the linear (beam) spline SPLINE2 of Nastran

    The spline is a beam along the `y` axis of the coordinate system
    ``cid`` (a key of ``CORD2R``), with bending flexibility `1/EI = 1` and
    torsional flexibility `1/GJ` = ``dtor``, and rigid arms along its `x`
    axis: a point at the spline coordinates `(x, y)` has the deflection `w =
    w_b(y) - x \psi(y)` and the rotations `\varphi_x = w_b'(y)` (about the
    spline `x` axis) and `\psi(y)` (about the spline `y` axis), with `w_b`
    and `\psi` the bending deflection and the twist of the beam. The beam is
    loaded at the structural grids by the forces `P_j`, and, where attached
    (``dthx``, ``dthy`` `\geq 0`), by the couples `M_j` about the spline `x`
    axis and the torques `T_j` about its `y` axis; the deflection is the
    superposition of the fundamental solutions of the infinite beam plus
    the rigid-body terms `a_0 + a_1 y - a_2 x`, and the loads are self
    equilibrated, `\boldsymbol{R}^\top \boldsymbol{P} = \boldsymbol{0}`.
    Requiring the spline to pass through the structural deflections, with the
    attachment flexibilities ``dz``, ``dthx``, ``dthy`` added to the
    diagonal, gives `\boldsymbol{a}` and `\boldsymbol{P}`, and the
    deflections and rotations at the points ``pts``.

    Returns ``G`` such that `(w, \varphi_x, \psi)` at the point `k` are
    ``G[3*k:3*k + 3] @ u``, with ``u`` over all degrees of freedom of
    ``model``; the structural degrees of freedom used are `u_z` and, where
    attached, the components of the rotation along the spline `x` and `y`
    axes. Loads at the points, normal forces and moments, give the
    structural loads ``G.T @ Fk`` (Eq. 2-25 of Rodden and Johnson 1994).

    Notes
    -----
    Rodden and Johnson (1994): interpolation and force transformation, Eqs.
    2-22 to 2-25, pp. 33-35; theory of linear splines, Eqs. 2-44 to 2-54,
    pp. 38-41; attachment flexibilities, Eqs. 2-55 to 2-59, p. 41; rigid
    arms, p. 42; the matrices `\boldsymbol{R}_i` and `\boldsymbol{A}_{ij}`,
    pp. 42-43; input, Section 3.3, pp. 106-108. The entry SPLINE2 is defined
    in the MSC Nastran 2020 Quick Reference Guide, pp. 3137-3138: DZ is the
    linear attachment flexibility, DTOR `= EI/GJ`, the spline axis is the
    `y` axis of CID, and a negative DTHX or DTHY means no attachment
    (Remark 3). The deflection kernel `|\Delta y|^3/(12 EI) - x_i x_j
    |\Delta y|/(2 GJ)` is the one of p. 43; the kernels of the rotations
    follow from the same fundamental solutions (derived here) and are
    checked by the exact interpolation of the rigid-body motions.
    """
    idx, ncoords, N = model['idx'], model['ncoords'], model['N']
    org, ex, ey = CORD2R[cid][:3]
    EI, GJ = 1., 1./dtor
    rows = [] # (grid, kind, attachment flexibility); kind 0 w, 1 phi_x, 2 psi
    for g in grids:
        rows.append((g, 0, dz))
        if dthx >= 0:
            rows.append((g, 1, dthx))
        if dthy >= 0:
            rows.append((g, 2, dthy))

    def spline_xy(p):
        r = np.asarray(p) - org
        return r @ ex, r @ ey

    def rigid(x, y, kind):
        return ((1., y, -x), (0., 1., 0.), (0., 0., 1.))[kind]

    def kernel(xi, yi, ki, xj, yj, kj):
        # response of kind ki at i to a unit generalized load of kind kj at
        # j: force, couple about x or torque about y; at the beam, a force
        # at the arm xj adds the torque -xj
        d = yi - yj
        F, C, tau = ((1., 0., -xj), (0., 1., 0.), (0., 0., 1.))[kj]
        wb = F*abs(d)**3/(12*EI) - C*d*abs(d)/(4*EI)
        phi = F*d*abs(d)/(4*EI) - C*abs(d)/(2*EI)
        psi = -tau*abs(d)/(2*GJ)
        return (wb - xi*psi, phi, psi)[ki]

    n = len(rows)
    xy = [spline_xy(ncoords[idx[g]]) for g, _, _ in rows]
    A = np.zeros((n, n))
    R = np.zeros((n, 3))
    Ug = np.zeros((n, N))
    for i, (g, ki, flex) in enumerate(rows):
        R[i] = rigid(*xy[i], ki)
        for j, (_, kj, _) in enumerate(rows):
            A[i, j] = kernel(*xy[i], ki, *xy[j], kj)
        A[i, i] += flex
        c = DOF*idx[g]
        if ki == 0:
            Ug[i, c + 2] = 1.
        else:
            Ug[i, c + 3:c + 6] = (ex, ey)[ki - 1]
    S = np.block([[np.zeros((3, 3)), R.T], [R, A]])
    # (a, P) from the structural displacements
    Sinv = np.linalg.solve(S, np.vstack((np.zeros((3, n)), np.eye(n))))
    G = np.zeros((3*len(pts), N))
    for k, p in enumerate(pts):
        xk, yk = spline_xy(p)
        for kk in range(3):
            row = np.concatenate((rigid(xk, yk, kk),
                [kernel(xk, yk, kk, *xy[j], kj) for j, (_, kj, _) in
                 enumerate(rows)]))
            G[3*k + kk] = row @ Sinv @ Ug
    return G


def spline2_loads(model, pts, forces, moments=None):
    r"""Normal forces ``forces`` (along `z`) and moments ``moments`` (3
    components in the basic system, or ``None``) at the 40 points ``pts`` of
    the boxes 1000-1007 and 1100-1131, transferred to the structure with the
    splines SPLINE2 1501 and 1601 of HA144A, `\boldsymbol{F}_g =
    \boldsymbol{G}_{kg}^\top \boldsymbol{F}_k`

    The moments enter through the rotations of the rigid arms, by their
    components along the spline axes `x` and `y`. Nastran applies the
    doublet-lattice box forces at the box centres, with the moments of the
    quarter-chord forces about them (pp. 20 and 186): ``pts`` from
    ``box_points(0.5)``, ``moments`` from ``box_moments()``.
    """
    f = np.zeros(model['N'])
    for eid, (boxes, grids, dz, dtor, cid, dthx, dthy) in SPLINE2.items():
        ex, ey = CORD2R[cid][1:3]
        G = spline2_matrix(model, grids, cid, pts[boxes], dz, dtor, dthx,
                           dthy)
        nk = len(pts[boxes])
        Fk = np.zeros(3*nk)
        Fk[0::3] = forces[boxes]
        if moments is not None:
            Fk[1::3] = moments[boxes] @ ex
            Fk[2::3] = moments[boxes] @ ey
        f += G.T @ Fk
    return f


def box_moments(pts_from, pts_to, Fz):
    r"""Moments `(\boldsymbol{x}_j - \boldsymbol{x}_{0j}) \times
    \boldsymbol{F}_j` of the normal forces ``Fz`` at ``pts_from`` about
    ``pts_to``, which move the forces to ``pts_to`` without changing their
    resultant force and moment"""
    F3 = np.zeros((len(Fz), 3))
    F3[:, 2] = Fz
    return np.cross(pts_from - pts_to, F3)


def resultant(model, f, ref):
    r"""Resultant force and moment about ``ref`` of the nodal loads ``f``,
    `\boldsymbol{\Phi}_{ref}^\top \boldsymbol{f}`"""
    return rigid_body_modes(model, ref).T @ f


def weight_loads(model):
    r"""Weights as nodal loads, the mass times the acceleration `-g` along
    `z`"""
    f = np.zeros(model['N'])
    f[2::DOF] = -model['Md'][2::DOF]*G_ACC
    return f


def solve_inertia_relief(model, f):
    r"""Inertia relief of the reduced model, support SUPORT, with the
    displacements of all grids recovered through the RBAR transformation

    Returns a dict with ``u``, ``acc``, ``Phi`` and ``R`` of
    :func:`pyfe3d.solver.apply_inertia_relief`, of the reduced model, ``ftot``,
    the reduced loads plus the d'Alembert forces, ``ufull``, the
    displacements of all the grids, and ``Rref``, the reactions of the
    reference support."""
    ind, T = model['ind'], model['T']
    fr = T.T @ f
    bref = model['bref'][ind]
    u, acc, Phi, R = apply_inertia_relief(model['K'], model['M'], fr, bref,
                                          model['bfix'][ind],
                                          return_reactions=True)
    return dict(u=u, acc=acc, Phi=Phi, R=R,
                ftot=fr - model['M'] @ (Phi @ acc), ufull=T @ u,
                Rref=R[bref])


def test_mass_properties():
    model = build_model()
    MO, m, xcg, IS = weight_generator(model)
    assert np.allclose(MO, MO_PUB, rtol=0, atol=1e-6*abs(MO_PUB).max())
    assert np.isclose(m, 8000., rtol=1e-14)
    assert np.isclose(xcg[0], XCG_PUB, rtol=1e-12)
    assert np.isclose(xcg[1], YCG_PUB, rtol=1e-12)
    assert np.allclose(IS, IS_PUB, rtol=0, atol=1e-6*abs(IS_PUB).max())
    assert np.isclose(IS[1, 1], 8.928944e5, rtol=1e-6)
    # the reduced (RBAR) model has the same rigid-body mass matrix
    Phi = rigid_body_modes(model, model['ncoords'][model['idx'][90]])
    Phir = Phi[model['ind']]
    assert np.allclose(model['T'] @ Phir, Phi, atol=1e-12)
    assert np.allclose(Phir.T @ (model['M'] @ Phir)/WTMASS, MO, rtol=0,
                       atol=1e-12*abs(MO).max())


def test_displacements_from_published_bar_forces():
    model = build_model()
    idx = model['idx']
    for sc in (2, 3):
        f = loads_from_bar_forces(model, BAR_PUB[sc])
        out = solve_inertia_relief(model, f)
        # the loads are balanced: no rigid-body acceleration, no reaction
        assert abs(out['acc']).max() < 1e-6
        assert abs(out['Rref']).max() < 1e-8*abs(f).max()
        u = out['ufull']
        upub = np.array([DISP_PUB[sc][g] for g in DISP_PUB[sc]])
        ufe = np.array([u[DOF*idx[g] + np.array([2, 3, 4])]
                        for g in DISP_PUB[sc]])
        assert abs(ufe - upub).max() < 2e-6*abs(upub).max()
        bf = bar_forces(model, u)
        for eid, vals in BAR_PUB[sc].items():
            assert np.allclose(bf[eid], vals, rtol=0, atol=2e-6*6.2e4)


def test_published_box_forces_inertia_relief():
    model = build_model()
    canard, wing = box_points()
    pts = np.vstack((canard, wing))
    F = np.zeros((40, 3))
    F[:, 2] = BOX_FORCES_SC2
    MO, m, xcg, IS = weight_generator(model)
    xcg_abs = 15. + xcg[0]
    # balanced trimmed loads, 1 g level flight
    assert np.isclose(F[:, 2].sum(), 8000., rtol=1e-8)
    assert abs((F[:, 2]*(xcg_abs - pts[:, 0])).sum()) < 1e-3
    fw = weight_loads(model)
    for moments, ok in ((True, True), (False, False)):
        fa = (transfer_loads(model, canard, F[:8], [98, 99], 2, moments)
              + transfer_loads(model, wing, F[8:],
                               [99, 100, 111, 112, 121, 122], 3, moments))
        out = solve_inertia_relief(model, fa + fw)
        assert abs(out['Phi'].T @ out['ftot']).max() < 1e-9*8000
        assert abs(out['Rref']).max() < 1e-9*8000
        bf = bar_forces(model, out['ufull'])
        if ok:
            # resultant conserved: accelerations vanish to the rounding of
            # the published forces
            assert abs(out['acc']).max() < 1e-5*G_ACC
            # element 101 carries the weight of GRID 97 only, to the
            # rounding of the published box forces (7 digits)
            assert np.allclose(bf[101], [0., -1.5e4, 1.5e3, 0.], rtol=0,
                               atol=1e-6*1.5e4)
        else:
            # forces only: the pitching moment is not conserved
            assert abs(out['acc'][1]) > 0.1
            assert not np.isclose(bf[101][2], 1.5e3, rtol=0.05)


def test_spline2_transfer_resultant():
    r"""SPLINE2 1501 and 1601 conserve the resultant normal force and its
    moments: arbitrary normal forces and moments at arbitrary points of the
    two panels give nodal loads with the same resultant about the centre of
    gravity (derived here: the spline reproduces the rigid-body motions of
    the grids, Rodden and Johnson 1994, `\boldsymbol{R}_i`, p. 43; by virtual
    work, Eq. 2-25, p. 35, the resultant is conserved)"""
    model = build_model()
    rng = np.random.default_rng(1)
    canard, wing = box_points()
    pts = np.vstack((canard, wing))
    pts[:, :2] += rng.uniform(-1., 1., (40, 2))
    Fz = rng.uniform(-1e3, 1e3, 40)
    M = np.zeros((40, 3))
    M[:, :2] = rng.uniform(-1e3, 1e3, (40, 2))
    f = spline2_loads(model, pts, Fz, M)
    ref = np.array([15. + XCG_PUB, 0., 0.])
    F3 = np.zeros((40, 3))
    F3[:, 2] = Fz
    expected = np.concatenate((F3.sum(axis=0),
                               (np.cross(pts - ref, F3) + M).sum(axis=0)))
    got = resultant(model, f, ref)
    # out-of-plane force and the moments about x and y
    assert np.allclose(got[[2, 3, 4]], expected[[2, 3, 4]], rtol=1e-12,
                       atol=1e-9*abs(expected).max())
    # in-plane force and moment are not transferred, as in Nastran
    assert np.allclose(got[[0, 1, 5]], 0., atol=1e-9*abs(expected).max())


def test_published_box_forces_spline2():
    r"""The published box forces of subcase 2 (p. 201), at the box centres
    with their moments about the midchord (pp. 20 and 186), transferred with
    SPLINE2 1501 and 1601 (Listing 7-2, cards 50-57, 69-70 and 73-76):

    - the canard spline is the lever rule between GRID 98 and 99;
    - with the weights, the nodal loads of the independent degrees of
      freedom are those in equilibrium with the published bar forces (p.
      204), to the rounding of the box forces;
    - the inertia relief gives zero accelerations and reactions, and all
      the published bar forces (p. 204) and displacements (p. 203), in
      particular element 103 and the tip of the elastic axis, GRID 120, to
      the printed digits.
    """
    model = build_model()
    idx = model['idx']
    qc = np.vstack(box_points())
    cen = np.vstack(box_points(0.5))
    F = BOX_FORCES_SC2
    fa = spline2_loads(model, cen, F, box_moments(qc, cen, F))
    w99 = (qc[:8, 0] - 10.)/10.
    assert np.isclose(fa[DOF*idx[98] + 2], ((1 - w99)*F[:8]).sum(),
                      rtol=1e-12)
    fw = weight_loads(model)
    T, ind = model['T'], model['ind']
    free = ~model['bfix'][ind]
    fpub = loads_from_bar_forces(model, BAR_PUB[2])
    dr = (T.T @ (fa + fw) - T.T @ fpub)[free]
    assert abs(dr).max() < 1e-7*abs(T.T @ fpub).max()
    out = solve_inertia_relief(model, fa + fw)
    assert abs(out['acc']).max() < 1e-5*G_ACC
    assert abs(out['Rref']).max() < 1e-9*8000
    u = out['ufull']
    upub = np.array([DISP_PUB[2][g] for g in DISP_PUB[2]])
    ufe = np.array([u[DOF*idx[g] + np.array([2, 3, 4])] for g in DISP_PUB[2]])
    assert abs(ufe - upub).max() < 2e-6*abs(upub).max()
    assert np.isclose(u[DOF*idx[120] + 2], DISP_PUB[2][120][0], rtol=1e-6)
    bf = bar_forces(model, u)
    for eid, vals in BAR_PUB[2].items():
        assert np.allclose(bf[eid], vals, rtol=0, atol=2e-6*6.2e4)
    assert np.allclose(bf[103][:3], BAR_PUB[2][103][:3], rtol=1e-6)


if __name__ == '__main__':
    test_mass_properties()
    test_displacements_from_published_bar_forces()
    test_published_box_forces_inertia_relief()
    test_spline2_transfer_resultant()
    test_published_box_forces_spline2()
