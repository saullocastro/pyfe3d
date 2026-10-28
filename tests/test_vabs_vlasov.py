r"""Generalized Vlasov model of :mod:`pyfe3d.vabs`

Checks of :meth:`pyfe3d.vabs.CrossSection.calc_vlasov` and
:meth:`pyfe3d.vabs.CrossSection.translated` against Yu, Hodges, Volovoi and
Fuchs, "A generalized Vlasov theory for composite beams", Thin-Walled
Structures 43 (2005) 1493--1511, doi:10.1016/j.tws.2005.02.003: the isotropic,
orthotropic and anisotropic strips of Tables 1, 3 and 4 and the isotropic
I-beam of Table 5, with the meshes described in Sections 5.1.1 and 5.2.1. The
strips are reproduced to five digits or more with the warping constraints of Yu
et al. (2002), the default of ``calc_vlasov``. The constraints of Yu et al.
(2012) are tested as well: they differ in the fourth constraint, on the
rotation of the in-plane warping, which changes the warping rigidity of the
strips, by 9.5 per cent for the isotropic one, whose value is then close to
Vlasov's `E h^3 b^3/144`, Eq. (17) of the paper, but changes the I-beam and the
channel by about 0.1 per cent or less, and changes neither the first four rows
and columns of ``Kv``, nor the shear centre, nor the torsional warping `w_1`.
For the I-beam, the geometry is that of Fig. 6, flanges of width ``1.0`` and
outer height ``0.5``, with which the torsional rigidity is 0.8 per cent above
and the warping rigidity 1.0 to 1.1 per cent below the published VABS values,
see :func:`test_Ibeam_isotropic_Table5`; the text of Section 5.2 swaps the two
dimensions.
The "Analytical (VAM)" warping rigidity of Table 5, 2083.33, equals `E t a^3
b^2/24` with the dimensions of the text, but is not consistent with the VABS
values of the same table, and is not used.

The anisotropic I-beam of Table 6, whose lay-up the paper gives only as a
drawing, Fig. 6, is reconstructed with the description of Volovoi,
Hodges, Berdichevsky and Sutyrin, "Asymptotic theory for static behavior of
elastic anisotropic I-beams", International Journal of Solids and
Structures 36(7) (1999) 1017--1043, doi:10.1016/S0020-7683(97)00341-7,
p. 1038 and Fig. 8; the published effective torsional rigidity is
reproduced within 1.1 per cent and the warping rigidity within 4 per cent.
The composite I-beam of Pandey, Kabir and Sherbourne, "Flexural-torsional
stability of thin-walled composite I-section beams", Composites Engineering
5(3) (1995) 321--342, doi:10.1016/0961-9526(94)00101-E, with the sectional
constants of the asymptotic theory given by Hodges and Peters,
"Lateral-torsional buckling of cantilevered elastically coupled composite
strip- and I-beams", International Journal of Solids and Structures 38(9)
(2001) 1585--1603, doi:10.1016/S0020-7683(00)00111-6, p. 1601, is compared
with an assumption on the out-of-plane constants, see
:func:`test_Ibeam_Pandey_Hodges_Peters`.

For a thin-walled channel, the torsional warping about the shear centre is
compared with the principal sectorial coordinate `\omega` of Vlasov,
Thin-Walled Elastic Beams, 2nd ed., 1961, Eqs. (3.15) and (7.1), pp. 14-16
and 40-41, `\omega = \int \rho \, ds`,
normalised to zero average with the pole at the shear centre. With the
twist rate `\kappa_1` and `\rho` positive when the radius vector from the
pole turns counter-clockwise, from `x_2` towards `x_3`, the condition of no
shear strain on the mid-line, `u_{1,s} + \rho \kappa_1 = 0`, gives the
axial warping displacement `u_1 = -\omega \kappa_1`.

The thin-walled shear centres and sectorial moments of inertia
`J_\omega`, the warping rigidity being `E J_\omega`, are those of Vlasov
(1961), Chapter II, Sec. 1: for the I-section, Eq. (1.2), p. 58, the shear
centre at mid-height for equal flanges, and the unnumbered equation `J_\omega
= a_y^2 J_{1y} + (d_1 - a_y)^2 J_{2y}` of p. 59, which gives `t b^3 h^2/24`
with `J_{1y} = J_{2y} = t b^3/12` and `a_y = d_1/2 = h/2`; for the channel
("trough section"), web of depth `d`, flanges of width `d_1`, both of
thickness `t`, the unnumbered equations of p. 61, the distance of the shear
centre from the web, `a_x = d_1^2 \delta_1/(2 d_1 \delta_1 + d \delta/3)`,
i.e. `3 b^2/(6 b + h)`, and `J_\omega = (d_1 - 3 a_x) d^2 d_1^2
\delta_1/6 + a_x^2 J_x`, with `J_x` of Eq. (1.7) and `a_x` taken as a
distance (Vlasov's `a_x` is negative), which equals `t b^3 h^2 (3 b + 2
h)/[12 (6 b + h)]`. The same formulas are given by Timoshenko and Gere,
Theory of Elastic Stability, 2nd ed., McGraw-Hill, 1961 (International
Student Edition, 17th printing, 1985), Table A-3, p. 530: `C_w = t h^2
b^3/24` for the I-section, `e = 3 b^2/(6 b + h)` and `C_w = t b^3 h^2 (3 b +
2 h)/[12 (6 b + h)]` for the channel, and the torsion constant `J = t^3 (2 b
+ h)/3` of both; the warping rigidity of the I-beam, `C_1 = E I_f h^2/2`, is
that of Eq. (5-8), p. 219, stated on p. 224, Art. 5.3, where `C_w` of the
channel is derived.
"""
import sys
import os

import numpy as np
import pytest

from pyfe3d.vabs import CrossSection, isotropic, orthotropic

sys.path.append(os.path.dirname(__file__))
import vabs_meshes as vm


E, NU = 10e6, 0.3


def iso_section(x2, x3, els):
    return CrossSection(x2, x3, els, [isotropic(E, NU)], [0]*len(els))


def strip(etype=8, **kwargs):
    # NOTE Section 5.1.1: 2 elements through the thickness, 60 across the
    #      width, b = 0.953 in., h = 0.03 in.
    x2, x3, els = vm.rectangle(0.953, 0.03, 60, 2, etype)
    return x2, x3, els


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
def test_strip_isotropic_Table1(constraint):
    sec = iso_section(*strip())
    sec.calc_vlasov(constraint)
    assert np.isclose(sec.Kv[1, 1], 32.3418, rtol=5e-5)
    b, h = 0.953, 0.03
    if constraint == 'yu2002':
        # NOTE VABS value of Table 1 and asymptotic thin-walled value,
        #      Eq. (18)
        assert np.isclose(sec.Kv[4, 4], 1.78635, rtol=1e-5)
        assert np.isclose(sec.Kv[4, 4], E*h**3*b**3/(144*(1 - NU**2)),
                          rtol=2e-3)
    else:
        # NOTE 9.5 per cent lower, close to Vlasov's value, Eq. (17)
        assert np.isclose(sec.Kv[4, 4], E*h**3*b**3/144, rtol=5e-3)


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
@pytest.mark.parametrize('theta, C22, C55, coupled', [
    (0., 7.31562, 3.39251, False),    # Table 3
    (15., 7.96547, 1.82499, True),    # Table 4
    (-15., 7.96547, 1.82499, True)])
def test_strip_composite_Tables3_4(theta, C22, C55, coupled, constraint):
    r"""The warping rigidity of the paper is reproduced with the
    constraints ``'yu2002'``; with ``'yu2012'`` it is lower, by 1.9 per
    cent for the orthotropic strip and 12 per cent for the anisotropic one,
    and only the torsional rigidity is compared"""
    # NOTE Table 2; one layer, ply angle theta from the beam axis
    D = orthotropic(E1=20.59e6, E2=1.42e6, E3=1.42e6, G12=0.87e6, G13=0.87e6,
                    G23=0.696e6, nu12=0.42, nu13=0.42, nu23=0.42)
    x2, x3, els = strip()
    ne = len(els)
    sec = CrossSection(x2, x3, els, [D], [0]*ne, theta3=[theta]*ne)
    sec.calc_vlasov(constraint)
    Kv = sec.Kv
    # NOTE effective torsional rigidity with the bending-twist coupling,
    #      C22 - C23**2/C33 in the numbering of the paper
    C22bar = Kv[1, 1] - Kv[1, 2]**2/Kv[2, 2]
    assert np.isclose(C22bar, C22, rtol=2e-5)
    if constraint == 'yu2002':
        assert np.isclose(Kv[4, 4], C55, rtol=2e-5)
    else:
        assert 0.85*C55 < Kv[4, 4] < 0.99*C55
    assert (abs(Kv[1, 2]) > 1e-3*Kv[1, 1]) == coupled


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
def test_Ibeam_isotropic_Table5(constraint):
    r"""Isotropic I-beam of Yu et al. (2005), Table 5, VABS: `C_{22} =
    199.944` lb in.\ :sup:`2` and `C_{55} = 3553.43` lb in.\ :sup:`4`

    With the mesh of the paper, 2 elements through the thickness, 20 across
    each flange and 10 along the web (``nf = 9`` elements on each side of
    the junction), `C_{22}` is 0.78 per cent above the published value;
    refining the mesh by 2 and 4 (18 x 20 and 36 x 40) reduces the
    difference to 0.46 and 0.33 per cent, a discretisation error, within
    the 1 per cent of the assertion. `C_{55}` is 0.97 per cent below the
    published value with the constraints ``'yu2002'`` and 1.08 per cent
    below with ``'yu2012'``, and 0.94 and 0.93 (``'yu2002'``), 1.05 and
    1.03 per cent (``'yu2012'``) below with the refined meshes: this
    difference is not a discretisation error and its origin is not
    identified; with 4-node and 9-node elements it is 1.06 and 0.95 per cent
    (``'yu2002'``). The thin-walled value of Vlasov (1961), `E t b^3
    h^2/24` = 3526.67, is 0.76 per cent below the published value, and
    `C_{55}` tends to it as the walls become thinner, see
    :func:`test_Ibeam_thin_walled_limit`. The tolerance of 1.5 per cent on
    the published `C_{55}` is not derived from an error estimate; it covers
    this unexplained difference of about 1 per cent.
    """
    # NOTE Section 5.2.1: 2 elements through the thickness, 20 across the
    #      flange, 10 along the web
    t = 0.04
    sec = iso_section(*vm.ibeam(1.0, 0.5, t, 9, 2, 10, 8))
    sec.calc_vlasov(constraint)
    assert np.isclose(sec.Kv[1, 1], 199.944, rtol=1e-2)
    assert np.isclose(sec.Kv[4, 4], 3553.43, rtol=1.5e-2)
    # NOTE thin-walled warping rigidity, E t bf**3 h**2/24, h between the
    #      flange mid-planes, Vlasov (1961), Ch. II, Sec. 1, p. 59
    assert np.isclose(sec.Kv[4, 4], E*t*1.0**3*(0.5 - t)**2/24, rtol=3.5e-3)
    # NOTE doubly symmetric: no coupling of the bimoment
    assert np.abs(sec.Kv[4, :4]).max() < 1e-8*sec.Kv[4, 4]


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
def test_Ibeam_thin_walled_limit(constraint):
    errors = []
    for t in (0.04, 0.02, 0.01):
        sec = iso_section(*vm.ibeam(1.0, 0.5, t, 12, 2, 12, 8))
        sec.calc_vlasov(constraint)
        EG = E*t*1.0**3*(0.5 - t)**2/24
        errors.append(abs(sec.Kv[4, 4]/EG - 1))
    assert errors[2] < errors[1] < errors[0] < 3.5e-3


def channel_omega(sec, b, h, e):
    r"""Principal sectorial coordinate of a channel at the nodes of the
    mid-line, pole at the shear centre `(-e, 0)`, origin of the contour at
    the centre of the web, where `\omega = 0` by antisymmetry"""
    x2, x3 = sec.x2, sec.x3
    on_web = np.isclose(x2, 0.) & (np.abs(x3) <= h/2 + 1e-12)
    on_flange = np.isclose(np.abs(x3), h/2) & (x2 >= 0.) & (x2 <= b + 1e-12)
    omega = np.full(x2.shape, np.nan)
    # NOTE d omega = (x2 - x2P) dx3 - (x3 - x3P) dx2
    omega[on_web] = e*x3[on_web]
    up = on_flange & (x3 > 0)
    lo = on_flange & (x3 < 0)
    omega[up] = e*h/2 - h/2*x2[up]
    omega[lo] = -e*h/2 + h/2*x2[lo]
    return omega


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
@pytest.mark.parametrize('t, rtol, rtol_EG', [(0.02, 2e-3, 5e-3),
                                              (0.01, 5e-4, 1.5e-3)])
def test_channel_sectorial_coordinate(t, rtol, rtol_EG, constraint):
    b, h = 0.5, 1.0
    x2, x3, els = vm.channel(b, h, t, 20, 2, 20, 8)
    sec = iso_section(x2, x3, els)
    sec.calc_vlasov(constraint)
    # NOTE shear centre of the thin-walled channel, behind the web, Vlasov
    #      (1961), Ch. II, Sec. 1, p. 61
    e = 3*b**2/(h + 6*b)
    sc = sec.vlasov_section
    e2, e3 = sec.x2[0] - sc.x2[0], sec.x3[0] - sc.x3[0]
    assert np.isclose(e2, -e, rtol=5e-3)
    assert abs(e3) < 1e-10
    assert np.abs(sc.shear_centre).max() < 1e-10
    w1 = sc.warping([0., 1., 0., 0.])[:, 0]
    omega = channel_omega(sec, b, h, e)
    mid = ~np.isnan(omega)
    assert mid.sum() > 40
    assert np.abs(w1[mid] + omega[mid]).max() < rtol*np.abs(omega[mid]).max()
    # NOTE normalisation: zero average, orthogonal to x2 and x3
    tri = sc.triangles()
    X2, X3 = sc.x2[tri], sc.x3[tri]
    area = 0.5*((X2[:, 1] - X2[:, 0])*(X3[:, 2] - X3[:, 0])
                - (X2[:, 2] - X2[:, 0])*(X3[:, 1] - X3[:, 0]))
    W = w1[tri].mean(axis=1)
    scale = np.abs(W*area).sum()*h
    for f in (1., X2.mean(axis=1), X3.mean(axis=1)):
        assert abs((W*f*area).sum()) < 2e-2*scale
    # NOTE warping rigidity, E t b**3 h**2 (3b + 2h)/[12 (6b + h)], Vlasov
    #      (1961), Ch. II, Sec. 1, p. 61
    EG = E*t*b**3*h**2*(3*b + 2*h)/(12*(6*b + h))
    assert np.isclose(sec.Kv[4, 4], EG, rtol=rtol_EG)


def test_consistency():
    x2, x3, els = vm.channel(0.5, 1.0, 0.05, 10, 2, 10, 8)
    sec = iso_section(x2 + 0.3, x3 - 0.2, els)
    sec.calc_stiffness()
    Kc = sec.Kc.copy()
    sec.calc_vlasov()
    # NOTE the other attributes are not changed
    assert np.array_equal(sec.Kc, Kc)
    assert sec.constraint == 'yu2012'
    Kv = sec.Kv
    assert np.allclose(Kv, Kv.T, rtol=0, atol=1e-12*np.abs(Kv).max())
    assert np.all(np.linalg.eigvalsh(Kv) > 0)
    assert np.allclose(sec.Fv @ Kv, np.eye(5), atol=1e-8)
    # NOTE first 4 x 4 block: classical stiffness about the shear centre
    assert np.allclose(Kv[:4, :4], sec.vlasov_section.Kc)
    # NOTE independent of the reference axis
    ref = iso_section(x2, x3, els)
    ref.calc_vlasov()
    assert np.allclose(ref.Kv, Kv, rtol=1e-8, atol=1e-8*np.abs(Kv).max())
    # NOTE the constraints change only the warping rigidity
    ref.calc_vlasov('yu2012')
    assert np.allclose(ref.Kv[:4, :4], Kv[:4, :4], rtol=1e-8,
                       atol=1e-8*np.abs(Kv).max())


def test_translated():
    x2, x3, els = vm.rectangle(1., 0.2, 4, 2, 9)
    D = isotropic(1., 0.3)
    sec = CrossSection(x2, x3, els, [D, 2*D], [0, 1]*4,
                       theta1=np.arange(8.), theta3=-np.arange(8.),
                       densities=[1., 2.])
    tr = sec.translated(0.25, -0.5)
    assert np.allclose(tr.x2, x2 - 0.25)
    assert np.allclose(tr.x3, x3 + 0.5)
    assert np.array_equal(tr.conn, sec.conn)
    assert np.array_equal(tr.mat_ids, sec.mat_ids)
    assert np.array_equal(tr.theta1, sec.theta1)
    assert np.array_equal(tr.theta3, sec.theta3)
    assert np.array_equal(tr.densities, sec.densities)
    assert tr.Kc is None


# NOTE Table 2 of Yu et al. (2005): graphite-epoxy of the strips and of the
#      anisotropic I-beam, psi
D_TABLE2 = orthotropic(E1=20.59e6, E2=1.42e6, E3=1.42e6, G12=0.87e6,
                       G13=0.87e6, G23=0.696e6, nu12=0.42, nu13=0.42,
                       nu23=0.42)


def ibeam_table6_layup(x2, x3, els, theta, wrapped=True):
    r"""Ply angles ``theta1``, ``theta3`` of the anisotropic I-beam of Yu et
    al. (2005), Table 6, flanges of width ``1.0``, outer height ``0.5``,
    walls of 8 plies of thickness ``0.005``, one element per ply

    The lay-up is that of Volovoi et al. (1999), p. 1038, "[0/90]4 lay-up in
    the web and a [(0/90)3/(theta)2]" in the flanges, the two ``theta``
    plies outermost, with the ply-by-ply reading of the drawing, Fig. 6 of
    Yu et al. (2005) and Fig. 8 of Volovoi et al. (1999): two back-to-back
    channels of 4 plies, from the left face of the web 0/90/0/90 and
    0/90/0/90, which wrap around the junctions into the inner 4 plies of the
    flanges, covered by a 0/90 pair, the 90 ply outer, and by the two
    ``theta`` plies, all three over the full width. From the inner to the
    outer surface, the left half of each flange is 0/90/0/90/0/90/theta/
    theta and the right half 90/0/90/0/0/90/theta/theta. Each channel ply
    turns at the junction with a square corner, which is assigned to its
    flange part. The ``theta`` plies have the same sense, ``theta3 =
    theta``, with ``theta1 = 0``, in both flanges. With ``wrapped=False``,
    the inner 4 plies of the flanges are 0/90/0/90, from the inner surface,
    over the full width, including the junctions, i.e. the same order in
    both halves.
    """
    t, hp = 0.04, 0.005
    th1, th3 = [], []
    for el in els:
        xc, yc = x2[el[:4]].mean(), x3[el[:4]].mean()
        if abs(yc) < 0.25 - t:
            # NOTE web, ply c from the left face
            c = int((xc + t/2)/hp)
            th1.append(90.)
            th3.append(0. if c % 2 == 0 else 90.)
            continue
        # NOTE flange, row r from the outer surface
        r = int((0.25 - abs(yc))/hp)
        th1.append(0.)
        if r < 4:
            th3.append([theta, theta, 90., 0.][r])
            continue
        if not wrapped:
            th3.append([90., 0., 90., 0.][r - 4])
            continue
        left = xc < 0
        j = r - 3
        if abs(xc) < t/2:
            # NOTE junction: column cc from the face of the web on the side
            #      of the element; vertical arm of the channel ply 4 - cc
            cc = int((xc + t/2)/hp)
            cc = cc if left else 7 - cc
            if cc > 7 - r:
                j = 4 - cc
                th1[-1] = 90.
        # NOTE channel ply j from the outer: left 90/0/90/0, right 0/90/0/90
        th3.append([90., 0., 90., 0.][j - 1] if left
                   else [0., 90., 0., 90.][j - 1])
    return np.array(th1), np.array(th3)


def ibeam_table6(theta=15., wrapped=True, constraint='yu2002'):
    # NOTE Section 5.2.1: 8 elements through the thickness, here one per
    #      ply, 20 across each flange, 10 along the web
    x2, x3, els = vm.ibeam(1.0, 0.5, 0.04, 10, 8, 10, 8)
    th1, th3 = ibeam_table6_layup(x2, x3, els, theta, wrapped)
    sec = CrossSection(x2, x3, els, [D_TABLE2], [0]*len(els), theta1=th1,
                       theta3=th3)
    sec.calc_vlasov(constraint)
    Kv = sec.Kv
    # NOTE effective torsional rigidity C22 - C23**2/C33 of the paper
    return Kv[1, 1] - Kv[1, 2]**2/Kv[2, 2], Kv[4, 4], Kv


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
def test_Ibeam_anisotropic_Table6(constraint):
    r"""Anisotropic I-beam of Yu et al. (2005), Table 6, VABS: `\bar{C}_{22}
    = 55.8658` lb in.\ :sup:`2` and `C_{55} = 4232.17` lb in.\ :sup:`4`

    The lay-up is reconstructed, see :func:`ibeam_table6_layup`. With it,
    `\bar{C}_{22}` is 1.05 per cent and `C_{55}` 4.0 per cent above the
    published values, with both constraints; refining the mesh to 40
    elements per half flange and along the web changes them to 0.75 and
    3.95 per cent. The isotropic I-beam of Table 5, with the same geometry
    and mesh, is 0.9 per cent below VABS, so that the difference of `C_{55}`
    is attributed to the lay-up, of which the paper gives only the drawing,
    and not to the geometry or to the formulation. With the ``theta`` plies
    of opposite sense in the two flanges there is no bending-twist coupling
    and `\bar{C}_{22}` is 8.4 per cent above the published value, which
    confirms the same sense in both flanges.

    The formulation for unsymmetric laminated walls was checked against
    an independent thin-walled estimate, from the classical lamination
    theory of each half flange: the torsional warping through the
    thickness is `(p - n) x_2`, `n` outwards from the mid-plane, with the
    slope `p = d/2 + \gamma_{12}^0` given by the mid-plane shear of the
    flange in St Venant torsion, `N_{22} = N_{12} = M_{22} = 0`, and the
    warping rigidity is that of the axial strain `p x_2` and curvature
    `-x_2` per unit `\kappa_1'`, with `\epsilon_{22}`, `\gamma_{12}` and
    `\kappa_{22}` free. It gives `C_{55}` = 4433.1 for this lay-up and
    4475.2 with the same order in both half flanges, against 4400.4 and
    4442.0 here, i.e. 0.74 per cent below in both cases, as the isotropic
    I-beam, 3519.9 against 3535.6, 0.44 per cent below, and as I-beams with
    isotropic flanges of a stiff and a soft layer, within 0.7 per cent,
    whose `C_{55}` differ by 17 per cent between the stiff layer outer and
    inner. The published value is 4.5 per cent below the thin-walled value
    of the drawn lay-up and lies between those with the ``theta`` plies at
    mid-thickness, 4289.5, and on the inner surface, 3953.6, which points
    to a difference between the drawing and the model of the paper rather
    than to the formulation.
    """
    C22bar, C55, Kv = ibeam_table6(constraint=constraint)
    assert np.isclose(C22bar, 55.8658, rtol=1.5e-2)
    assert np.isclose(C55, 4232.17, rtol=4.5e-2)
    # NOTE bending-twist coupling, kappa_1 with kappa_2
    assert abs(Kv[1, 2]) > 0.1*np.sqrt(Kv[1, 1]*Kv[2, 2])


def test_Ibeam_anisotropic_Table6_sensitivity():
    r"""Sensitivity of the anisotropic I-beam of Table 6 to the sign of
    ``theta`` and to the plies at the junctions"""
    ref = ibeam_table6()
    # NOTE theta -> -theta reverses the coupling only
    neg = ibeam_table6(theta=-15.)
    assert np.isclose(neg[0], ref[0], rtol=1e-8)
    assert np.isclose(neg[1], ref[1], rtol=1e-8)
    assert np.isclose(neg[2][1, 2], -ref[2][1, 2], rtol=1e-8)
    # NOTE the same order in both half flanges changes C22bar by 0.07 and
    #      C55 by 0.95 per cent
    flat = ibeam_table6(wrapped=False)
    assert np.isclose(flat[0], ref[0], rtol=5e-3)
    assert np.isclose(flat[1], ref[1], rtol=2e-2)


PANDEY = {0.: (364., 71200., 1.14e6, 735.),
          45.: (610., 71200., 912000., 735.)}


@pytest.mark.parametrize('constraint', ['yu2002', 'yu2012'])
@pytest.mark.parametrize('theta', [0., 45.])
def test_Ibeam_Pandey_Hodges_Peters(theta, constraint):
    r"""Composite I-beam of Pandey, Kabir and Sherbourne (1995), Composites
    Engineering 5(3) 321--342, doi:10.1016/0961-9526(94)00101-E, see the
    module docstring, not available here: the data and the constants are
    those given by Hodges and Peters (2001), p. 1601: wall thickness
    0.00953 m, depth 0.2032 m, flange width 0.1016 m, `E_{11}` = 42.7 GPa,
    `E_{22}` = 8.02 GPa, `G_{12}` = 3.10 GPa, `\nu_{12}` = 0.248, flanges
    of 0 degrees, web of a single layer of ply angle ``theta``; the
    asymptotic values of `D_{11}` (torsion), `D_{22}` (weak-axis bending),
    `D_{33}` (strong-axis bending) and `D_{44}` (warping), Eq. (1), are
    those of p. 1601, printed with three digits

    The depth is taken between the mid-planes of the flanges. The out-of-plane
    constants are not given; the material is taken as transversely
    isotropic, `E_{33} = E_{22}`, `G_{13} = G_{12}`, `\nu_{13} = \nu_{12}`,
    `\nu_{23} = 0.4` and `G_{23} = E_{22}/[2 (1 + \nu_{23})]`; `\nu_{23}`
    from 0.25 to 0.5 changes the four constants by less than 0.03 per cent.

    The published `D_{33}` and `D_{44}` are the thin-walled values with the
    web counted over the full depth `d`, `E_{11} b t d^2/2 + E_x t d^3/12`,
    `E_x` the uniaxial modulus of the web ply along the beam axis, and `E_{11}
    t b^3 d^2/24` plus the warping of the two flanges as strips, `E_{11}
    t^3 b^3/144` each, Eq. (17) of Yu et al. (2005), which reproduce them
    within half a unit of the last printed digit (1.138e6, 911600 and
    734.8). The 2-D section has the web between the flanges, and its
    `D_{33}` is the exact `E I` of the meshed geometry, to rounding at 0
    degrees, the section being homogeneous, and within 4e-8 at 45 degrees;
    at 0 degrees it is 1.10e6 N m\ :sup:`2`, 3.5 per cent below the
    published value. The published `D_{22}`, 71200 N m\ :sup:`2`,
    constant with the ply angle, omits the web; the flanges alone give
    71130, 0.1 per cent below; the 2-D value includes the web, `E_x (d -
    t) t^3/12`, 0.84 per cent of `D_{22}` at 0 degrees and 0.17 per cent at
    45, and is 0.74 and 0.10 per cent above the published value, within the
    bound given by the web and the 0.1 per cent.

    `D_{11}` is 0.92 per cent below the published value at 0 degrees and
    2.08 per cent below at 45 degrees, 1.17 and 2.30 per cent with 16
    elements on each side of the junction, 8 through the thickness and 32
    along the web; `D_{44}` is 0.35 and 0.58 per cent below with
    ``'yu2002'``, 0.49 and 0.60 per cent with ``'yu2012'``, 0.32 to 0.56
    per cent with the refined mesh. These differences do not decrease with
    refinement; they are of the order of the terms of relative order `t/b`
    = 0.094 that the thin-walled theory neglects, e.g. the factor `1 -
    0.630 t/b` of the torsion constant `b t^3/3` of a narrow rectangle of
    width `b` and thickness `t`, due to its two short free edges,
    Timoshenko and Goodier, Theory of Elasticity, 2nd ed., McGraw-Hill,
    1951, Art. 95, Eq. (161), p. 278 (written there `M_t = G \theta (2a)^3
    (2b) (1 - 0.630 a/b)/3` for the thickness `2a` and the width `2b`):
    5.9 per cent for each flange, 3.0 per cent of `D_{11}` for the two at 0
    degrees. The thin-walled `D_{11}` at 0 degrees, `G_{12} t^3 (2 b +
    d)/3` = 363.5, Table A-3 of Timoshenko and Gere (1961), is 0.15 per
    cent below the published 364. The tolerances of 2.5 and 1 per cent are not
    derived from an error estimate.
    """
    t, d, b = 0.00953, 0.2032, 0.1016
    E1, E2, G12, nu12 = 42.7e9, 8.02e9, 3.10e9, 0.248
    nu23 = 0.4
    D = orthotropic(E1=E1, E2=E2, E3=E2, G12=G12, G13=G12,
                    G23=E2/(2*(1 + nu23)), nu12=nu12, nu13=nu12, nu23=nu23)
    x2, x3, els = vm.ibeam(b, d + t, t, 8, 4, 16, 8)
    th1, th3 = [], []
    for el in els:
        if abs(x3[el[:4]].mean()) < d/2 - t/2:
            th1.append(90.)
            th3.append(theta)
        else:
            th1.append(0.)
            th3.append(0.)
    sec = CrossSection(x2, x3, els, [D], [0]*len(els), theta1=th1,
                       theta3=th3)
    sec.calc_vlasov(constraint)
    Kv = sec.Kv
    D11, D22, D33, D44 = PANDEY[theta]
    # NOTE uniaxial modulus of the web ply along the beam axis
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    Ex = 1/(c**4/E1 + s**4/E2 + c**2*s**2*(1/G12 - 2*nu12/E1))
    # NOTE thin-walled values of the published reading, web over the full
    #      depth d, against half a unit of the last printed digit
    half_unit = {0.: (5e3, 0.5), 45.: (500., 0.5)}[theta]
    D33_pub = E1*b*t*d**2/2 + Ex*t*d**3/12
    D44_pub = E1*t*b**3*d**2/24 + 2*E1*t**3*b**3/144
    assert abs(D33_pub - D33) <= half_unit[0]
    assert abs(D44_pub - D44) <= half_unit[1]
    # NOTE the web is along x3: kappa_2 is the strong-axis bending, exact
    #      E I of the 2-D geometry, web between the flanges
    D33_2d = E1*(b*t**3/6 + b*t*d**2/2) + Ex*t*(d - t)**3/12
    assert np.isclose(Kv[2, 2], D33_2d, rtol=1e-6)
    # NOTE weak-axis bending: flanges and web, the published value omits
    #      the web
    D22_flanges = E1*t*b**3/6
    D22_web = Ex*(d - t)*t**3/12
    assert abs(Kv[3, 3] - D22_flanges - D22_web) < D22_web
    assert (abs(Kv[3, 3]/D22 - 1)
            <= D22_web/D22 + abs(D22_flanges/D22 - 1))
    assert np.isclose(Kv[1, 1], D11, rtol=2.5e-2)
    assert np.isclose(Kv[4, 4], D44, rtol=1e-2)
