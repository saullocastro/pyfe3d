r"""Warping displacements and triangulation of :mod:`pyfe3d.vabs`

Checks of :meth:`pyfe3d.vabs.CrossSection.warping` and
:meth:`pyfe3d.vabs.CrossSection.triangles`: the torsional warping of an
elliptical section against the Saint-Venant solution, `w_1 = -\kappa_1
\frac{a^2 - b^2}{a^2 + b^2} x_2 x_3` for the semi-axes `a` along `x_2` and
`b` along `x_3` (Timoshenko and Goodier, Theory of Elasticity, 3rd ed.,
Sec. 105, with `\psi = \kappa_1 \phi`), linearity, the conversion from the
strains of pyfe3d, and the triangles of every element type.
"""
import sys
import os

import numpy as np
import pytest

from pyfe3d.vabs import CrossSection, isotropic, _Ts

sys.path.append(os.path.dirname(__file__))
import vabs_meshes as vm


def section(x2, x3, els, nu=0.3):
    sec = CrossSection(x2, x3, els, [isotropic(1., nu)], [0]*len(els))
    sec.calc_stiffness()
    return sec


@pytest.mark.parametrize('etype, rtol', [(3, 2e-2), (4, 2e-2), (6, 3e-3),
                                         (8, 1e-3), (9, 1e-3)])
def test_ellipse_torsion(etype, rtol):
    a, b = 2., 1.
    x2, x3, els = vm.circle(1., 6, etype)
    sec = section(a*x2, b*x3, els)
    w = sec.warping([0., 1., 0., 0.])
    assert w.shape == (x2.shape[0], 3)
    w1 = -(a**2 - b**2)/(a**2 + b**2)*sec.x2*sec.x3
    assert np.abs(w[:, 0] - w1).max() < rtol*np.abs(w1).max()
    # NOTE isotropic section: no in-plane warping under torsion
    assert np.abs(w[:, 1:]).max() < 1e-12*np.abs(w1).max()


def test_linearity_and_conventions():
    x2, x3, els = vm.box(2., 1., 0.1, 4, 2, 1, 8)[:3]
    sec = section(x2, x3, els)
    rng = np.random.default_rng(0)
    e1, e2, d1 = rng.normal(size=(3, 4))
    w1 = sec.warping(e1)
    w2 = sec.warping(e2)
    assert np.allclose(sec.warping(e1 + 2*e2), w1 + 2*w2, atol=1e-14)
    assert np.allclose(sec.warping(e1, d1),
                       w1 + (sec.V1S @ d1).reshape(-1, 3), atol=1e-14)
    assert np.allclose(sec.warping(e1, np.zeros(4)), w1, atol=0)
    # NOTE {eps, kx, ky, kz} of pyfe3d: kappa2 = kz, kappa3 = -ky
    Ts, _ = _Ts(4, 1.)
    assert np.allclose(sec.warping(e1, d1, convention='pyfe3d'),
                       sec.warping(Ts @ e1, Ts @ d1), atol=1e-14)
    es = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(sec.warping(es, convention='pyfe3d'),
                       sec.warping([0.1, 0.2, 0.4, -0.3]), atol=1e-14)


def test_warping_errors():
    x2, x3, els = vm.rectangle(1., 1., 2, 2, 4)
    sec = CrossSection(x2, x3, els, [isotropic(1., 0.3)], [0]*len(els))
    with pytest.raises(RuntimeError):
        sec.warping([0., 1., 0., 0.])
    sec.calc_stiffness()
    with pytest.raises(ValueError):
        sec.warping([0., 1., 0.])
    with pytest.raises(ValueError):
        sec.warping([0., 1., 0., 0.], deps=[1., 0.])
    with pytest.raises(ValueError):
        sec.warping([0., 1., 0., 0.], convention='vabs')


@pytest.mark.parametrize('etype, ntri', [(3, 1), (4, 2), (6, 4), (8, 6),
                                         (9, 8)])
def test_triangles(etype, ntri):
    # NOTE straight-sided elements: the triangles cover the area exactly
    x2, x3, els = vm.rectangle(2., 1., 3, 2, etype, x0=0.3, y0=-0.2)
    sec = CrossSection(x2, x3, els, [isotropic(1., 0.3)], [0]*len(els))
    sec.calc_mass()
    tri = sec.triangles()
    assert tri.shape == (len(els)*ntri, 3)
    x, y = sec.x2, sec.x3
    area = 0.5*((x[tri[:, 1]] - x[tri[:, 0]])*(y[tri[:, 2]] - y[tri[:, 0]])
                - (x[tri[:, 2]] - x[tri[:, 0]])*(y[tri[:, 1]] - y[tri[:, 0]]))
    assert area.min() > 0
    assert np.isclose(area.sum(), sec.area, rtol=1e-13)
    assert np.array_equal(np.unique(tri), np.arange(x.shape[0]))


def test_triangles_mixed():
    # NOTE a 9-node quadrilateral on [0, 1]^2 and two 6-node triangles on
    #      its right, sharing complete edges; area 2
    x2 = [0., 1., 1., 0., 0.5, 1., 0.5, 0., 0.5, 2., 1.5, 1.5, 2., 2., 1.5]
    x3 = [0., 0., 1., 1., 0., 0.5, 1., 0.5, 0.5, 0., 0., 0.5, 1., 0.5, 1.]
    els = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 9, 2, 10, 11, 5],
           [9, 12, 2, 13, 14, 11]]
    sec = CrossSection(x2, x3, els, [isotropic(1., 0.3)], [0]*3)
    tri = sec.triangles()
    assert tri.shape == (8 + 4 + 4, 3)
    x, y = sec.x2, sec.x3
    area = 0.5*((x[tri[:, 1]] - x[tri[:, 0]])*(y[tri[:, 2]] - y[tri[:, 0]])
                - (x[tri[:, 2]] - x[tri[:, 0]])*(y[tri[:, 1]] - y[tri[:, 0]]))
    assert area.min() > 0
    assert np.isclose(area.sum(), 2., rtol=1e-14)
    assert np.array_equal(np.unique(tri), np.arange(15))
