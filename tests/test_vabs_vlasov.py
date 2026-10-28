r"""Generalized Vlasov model of :mod:`pyfe3d.vabs`

Checks of :meth:`pyfe3d.vabs.CrossSection.calc_vlasov` and
:meth:`pyfe3d.vabs.CrossSection.translated` against Yu, Hodges, Volovoi and
Fuchs, "A generalized Vlasov theory for composite beams", Thin-Walled
Structures 43 (2005) 1493--1511, doi:10.1016/j.tws.2005.02.003: the
isotropic, orthotropic and anisotropic strips of Tables 1, 3 and 4 and the
isotropic I-beam of Table 5, with the meshes described in Sections 5.1.1
and 5.2.1. The strips are reproduced to five digits or more with the
warping constraints of Yu et al. (2002). For the I-beam, the geometry is
that of Fig. 6, flanges of width ``1.0`` and outer height ``0.5``, which
gives the published torsional and warping rigidities within 1 per cent;
the text of Section 5.2 swaps the two dimensions. The "Analytical (VAM)"
warping rigidity of Table 5, 2083.33, equals `E t a^3 b^2/24` with the
dimensions of the text, but is not consistent with the VABS values of the
same table, and is not used.

For a thin-walled channel, the torsional warping about the shear centre is
compared with the principal sectorial coordinate `\omega` of Vlasov,
Thin-Walled Elastic Beams, 2nd ed., 1961, `\omega = \int \rho \, ds`,
normalised to zero average with the pole at the shear centre. With the
twist rate `\kappa_1` and `\rho` positive when the radius vector from the
pole turns counter-clockwise, from `x_2` towards `x_3`, the condition of no
shear strain on the mid-line, `u_{1,s} + \rho \kappa_1 = 0`, gives the
axial warping displacement `u_1 = -\omega \kappa_1`.
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


def test_strip_isotropic_Table1():
    sec = iso_section(*strip())
    sec.calc_vlasov()
    assert np.isclose(sec.Kv[1, 1], 32.3418, rtol=5e-5)
    assert np.isclose(sec.Kv[4, 4], 1.78635, rtol=1e-5)
    # NOTE asymptotic thin-walled value, Eq. (18)
    b, h = 0.953, 0.03
    assert np.isclose(sec.Kv[4, 4], E*h**3*b**3/(144*(1 - NU**2)), rtol=2e-3)
    # NOTE with the constraints of Yu et al. (2012), close to Vlasov's
    #      value, Eq. (17)
    sec.calc_vlasov('yu2012')
    assert np.isclose(sec.Kv[4, 4], E*h**3*b**3/144, rtol=5e-3)


@pytest.mark.parametrize('theta, C22, C55, coupled', [
    (0., 7.31562, 3.39251, False),    # Table 3
    (15., 7.96547, 1.82499, True),    # Table 4
    (-15., 7.96547, 1.82499, True)])
def test_strip_composite_Tables3_4(theta, C22, C55, coupled):
    # NOTE Table 2; one layer, ply angle theta from the beam axis
    D = orthotropic(E1=20.59e6, E2=1.42e6, E3=1.42e6, G12=0.87e6, G13=0.87e6,
                    G23=0.696e6, nu12=0.42, nu13=0.42, nu23=0.42)
    x2, x3, els = strip()
    ne = len(els)
    sec = CrossSection(x2, x3, els, [D], [0]*ne, theta3=[theta]*ne)
    sec.calc_vlasov()
    Kv = sec.Kv
    # NOTE effective torsional rigidity with the bending-twist coupling,
    #      C22 - C23**2/C33 in the numbering of the paper
    C22bar = Kv[1, 1] - Kv[1, 2]**2/Kv[2, 2]
    assert np.isclose(C22bar, C22, rtol=2e-5)
    assert np.isclose(Kv[4, 4], C55, rtol=2e-5)
    assert (abs(Kv[1, 2]) > 1e-3*Kv[1, 1]) == coupled


def test_Ibeam_isotropic_Table5():
    # NOTE Section 5.2.1: 2 elements through the thickness, 20 across the
    #      flange, 10 along the web
    t = 0.04
    sec = iso_section(*vm.ibeam(1.0, 0.5, t, 9, 2, 10, 8))
    sec.calc_vlasov()
    assert np.isclose(sec.Kv[1, 1], 199.944, rtol=1e-2)
    assert np.isclose(sec.Kv[4, 4], 3553.43, rtol=1.5e-2)
    # NOTE thin-walled warping rigidity, E t bf**3 h**2/24, h between the
    #      flange mid-planes
    assert np.isclose(sec.Kv[4, 4], E*t*1.0**3*(0.5 - t)**2/24, rtol=3e-3)
    # NOTE doubly symmetric: no coupling of the bimoment
    assert np.abs(sec.Kv[4, :4]).max() < 1e-8*sec.Kv[4, 4]


def test_Ibeam_thin_walled_limit():
    errors = []
    for t in (0.04, 0.02, 0.01):
        sec = iso_section(*vm.ibeam(1.0, 0.5, t, 12, 2, 12, 8))
        sec.calc_vlasov()
        EG = E*t*1.0**3*(0.5 - t)**2/24
        errors.append(abs(sec.Kv[4, 4]/EG - 1))
    assert errors[2] < errors[1] < errors[0] < 3e-3


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


@pytest.mark.parametrize('t, rtol, rtol_EG', [(0.02, 2e-3, 5e-3),
                                              (0.01, 5e-4, 1.5e-3)])
def test_channel_sectorial_coordinate(t, rtol, rtol_EG):
    b, h = 0.5, 1.0
    x2, x3, els = vm.channel(b, h, t, 20, 2, 20, 8)
    sec = iso_section(x2, x3, els)
    sec.calc_vlasov()
    # NOTE shear centre of the thin-walled channel, behind the web
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
    # NOTE warping rigidity, E t b**3 h**2 (3b + 2h)/[12 (6b + h)]
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
