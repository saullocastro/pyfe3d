r"""Cross-sectional analysis of homogeneous isotropic sections

Verification of :class:`pyfe3d.vabs.CrossSection` against elasticity
solutions and published values:

- rectangle: `EA`, `EI_2`, `EI_3` and the Saint-Venant torsion constant
  (series solution derived in the docstring of :func:`rect_torsion_J`), the
  shear correction factors of Eq. (25) of Yu and Hodges (2005), with the
  value ``0.784442`` given on p. 50 for `2a = 1`, `2b = 2`, `\nu = 0.3`,
  and their convergence with the mesh;
- square of Table 3 of Yu et al. (2002), prismatic column, p. 5113;
- circle: `EA`, `EI`, `GJ` and the shear correction factor of the
  Saint-Venant flexure solution, :func:`circle_k`; tube: `EA`, `EI`, `GJ`;
- T-section of Fig. 3 and Table 1 of Yu and Hodges (2005), p. 51;
- reference axis offset from the centroid: sign of the extension-bending
  couplings, conversion to the convention of pyfe3d and :class:`BeamProp`.

Notes
-----
The shear correction factors of the rectangle and of the T-section are
those of Eq. (25), p. 50, and Table 1 of Yu and Hodges (2005) [1]_, the
square that of Table 3 of Yu et al. (2002) [2]_; the axial strain of the
offset reference axis is Eq. (36) of Yu et al. (2012) [3]_. The
elasticity solutions are derived in the docstrings of
:func:`rect_torsion_J` and :func:`circle_k`.

References
----------
.. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
   Variational Asymptotic Beam Sectional Analysis", Journal of the
   American Helicopter Society, 50(1), pp. 46--55, 2005.
   https://doi.org/10.4050/1.3092842
.. [2] Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
   Timoshenko-like modeling of initially curved and twisted composite
   beams", International Journal of Solids and Structures, 39(19),
   pp. 5101--5121, 2002. https://doi.org/10.1016/S0020-7683(02)00399-2
.. [3] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
   beam sectional analysis -- An updated version", International
   Journal of Engineering Science, 59, pp. 40--64, 2012.
   https://doi.org/10.1016/j.ijengsci.2012.03.006

"""
import sys
import os

import numpy as np
import pytest

from pyfe3d.vabs import (CrossSection, isotropic, yu_to_pyfe3d,
                         beamprop_from_vabs, classical_from_timoshenko)

sys.path.append(os.path.dirname(__file__))
import vabs_meshes as vm


def rect_torsion_J(a, b, nterms=200):
    r"""Torsion constant of the rectangle `|y| \le a/2`, `|z| \le b/2`,
    Saint-Venant series solution

    Notes
    -----
    Derivation: with the Prandtl stress function `\phi`, `\nabla^2 \phi =
    -2` (`G \theta = 1`), `\phi = 0` on the boundary and `J = 2 \int \phi \,
    dA`. With the Fourier series `1 = \sum_{n \, odd} \frac{4}{n \pi}
    (-1)^{(n-1)/2} \cos(n \pi z/b)` on `|z| < b/2`, the series `\phi =
    \sum_{n \, odd} c_n \cos(n \pi z/b) [1 - \cosh(n \pi y/b)/\cosh(n \pi
    a/(2b))]` satisfies the boundary conditions term by term and the
    equation with `c_n = 8 b^2 (-1)^{(n-1)/2}/(n^3 \pi^3)`, and each term
    gives `J_n = 32 b^3 [n \pi a - 2 b \tanh(n \pi a/(2b))]/(\pi^5 n^5)`.
    """
    n = np.arange(1, 2*nterms, 2, dtype=float)
    return np.sum(32*b**3*(n*np.pi*a - 2*b*np.tanh(n*np.pi*a/(2*b)))
                  /(np.pi**5*n**5))


def rect_c2(a, b, nu, nterms=2000):
    r"""Shear correction factor `c_2` of a rectangle of width `2a` along
    `x_2` and height `2b` along `x_3`, Eq. (25) of Yu and Hodges (2005)

    Notes
    -----
    The series of Eq. (25) of [1]_ is summed with ``nterms`` terms.

    References
    ----------
    .. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    rho = a/b
    m = np.arange(1, nterms + 1, dtype=float)
    s = np.sum(np.tanh(m*np.pi*rho)/m**5)
    inv = 6/5 + (nu/(1 + nu))**2*(1/(5*rho**4)
                                   - 18/(rho**5*np.pi**5)*s)
    return 1/inv


def circle_k(nu):
    r"""Shear correction factor of a solid circle, from the Saint-Venant
    flexure solution

    Notes
    -----
    Derivation (scratch script of the author's run, reproduced here): the
    stresses `\sigma_{zz} = -P (l - z) x/I`, `\tau_{xz} = \frac{3 + 2\nu}{8
    (1 + \nu)} \frac{P}{I} (R^2 - x^2 - \frac{1 - 2\nu}{3 + 2\nu} y^2)`,
    `\tau_{yz} = -\frac{1 + 2\nu}{4 (1 + \nu)} \frac{P x y}{I}`, all other
    stresses zero, satisfy equilibrium, the six Beltrami-Michell
    compatibility equations and the traction-free lateral surface, and have
    the resultant `P` along `x` with no torque; the constants were solved
    for with SymPy from these conditions. The shear correction factor that
    gives the same shear energy, `\int \tau^2/(2G) dA = P^2/(2 k G A)`, is
    `k = 6 (1 + \nu)^2/(7 + 14\nu + 8\nu^2)`.
    """
    return 6*(1 + nu)**2/(7 + 14*nu + 8*nu**2)


def run(x2, x3, els, E, nu, rho=0., constraint='yu2012'):
    sec = CrossSection(x2, x3, els, [isotropic(E, nu)], [0]*len(els),
                       densities=[rho])
    sec.calc_stiffness(constraint)
    return sec


def test_rect_c2_exact():
    r"""Series of :func:`rect_c2` against the value printed by Yu and Hodges
    (2005)

    Notes
    -----
    The value ``0.784442`` is printed on p. 50 of [1]_ for `2a = 1`, `2b =
    2` and `\nu = 0.3`.

    References
    ----------
    .. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    assert np.isclose(rect_c2(0.5, 1., 0.3), 0.784442, rtol=1e-6)


@pytest.mark.parametrize('etype, n, rtol_c', [pytest.param(3, 32, 2.e-3,
                                                           marks=pytest.mark.slow),
                                              (4, 32, 1.e-3),
                                              (6, 8, 2.e-5), (8, 8, 5.e-5),
                                              (9, 8, 5.e-5)])
def test_rectangle(etype, n, rtol_c):
    r"""Rectangle, every element type: area, `EA`, `EI_2`, `EI_3`, the
    torsion constant and the shear correction factors against the
    analytical values, and the classical stiffness against the condensed
    Timoshenko stiffness

    Notes
    -----
    The reference values are `A = a b`, `E A`, `E a b^3/12` and `E b
    a^3/12`, `G J` with `J` the Saint-Venant series of
    :func:`rect_torsion_J`, and `c_2`, `c_3` of Eq. (25) of [1]_,
    :func:`rect_c2`.

    References
    ----------
    .. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    E, nu = 70.e9, 0.3
    G = E/(2*(1 + nu))
    a, b = 1., 2.
    x2, x3, els = vm.rectangle(a, b, n, 2*n, etype)
    sec = run(x2, x3, els, E, nu)
    A = a*b
    Kt = sec.Kt
    assert np.isclose(sec.area, A, rtol=1e-12)
    assert np.isclose(Kt[0, 0], E*A, rtol=1e-12)
    rtol_I = 1e-12 if etype in (6, 8, 9) else 2e-3
    assert np.isclose(Kt[4, 4], E*a*b**3/12, rtol=rtol_I)
    assert np.isclose(Kt[5, 5], E*b*a**3/12, rtol=rtol_I)
    assert np.isclose(Kt[3, 3], G*rect_torsion_J(b, a), rtol=10*rtol_c)
    # NOTE 2a = 1 along x2, 2b = 2 along x3, and the reverse for c3
    assert np.isclose(Kt[1, 1]/(G*A), rect_c2(0.5, 1., nu), rtol=rtol_c)
    assert np.isclose(Kt[2, 2]/(G*A), rect_c2(1., 0.5, nu), rtol=rtol_c)
    # NOTE no coupling for a doubly symmetric section at its centroid, up to
    #      the asymmetry of the meshes of triangles with parallel diagonals
    d = np.sqrt(np.diag(Kt))
    off = (Kt - np.diag(np.diag(Kt)))/np.outer(d, d)
    assert np.abs(off).max() < (1e-3 if etype in (3, 6) else 1e-8)
    assert np.allclose(classical_from_timoshenko(Kt), sec.Kc, rtol=1e-10,
                       atol=1e-10*np.abs(sec.Kc).max())


@pytest.mark.slow
def test_rectangle_convergence():
    r"""Error of `c_2` against Eq. (25) of Yu and Hodges (2005) decreases
    with the mesh; with 8-node elements it decreases by about `2^4` per
    halving of the element size

    Notes
    -----
    The reference is `c_2` of Eq. (25) of [1]_, :func:`rect_c2`; Fig. 2 of
    [1]_ shows the convergence of VABS.

    References
    ----------
    .. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    E, nu = 1., 0.3
    G = E/(2*(1 + nu))
    ref = rect_c2(0.5, 1., nu)
    errors = []
    for n in (2, 4, 8, 16):
        x2, x3, els = vm.rectangle(1., 2., n, 2*n, 8)
        sec = run(x2, x3, els, E, nu)
        errors.append(abs(sec.Kt[1, 1]/(G*2.)/ref - 1))
    errors = np.array(errors)
    rates = np.log2(errors[:-1]/errors[1:])
    assert np.all(rates > 3.5)
    assert errors[-1] < 5e-6


def test_square_Yu2002_Table3():
    r"""Isotropic square ``0.5 in. x 0.5 in.``, ``E = 2.6e7 psi``, ``nu =
    0.3``, meshed with 8 x 8 8-node quadrilaterals, against the prismatic
    column of Table 3 of Yu et al. (2002)

    Notes
    -----
    Data and mesh are those of p. 5113 of [1]_, the reference values those
    of the prismatic column of its Table 3, compared to their three printed
    digits.

    References
    ----------
    .. [1] Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
       Timoshenko-like modeling of initially curved and twisted composite
       beams", International Journal of Solids and Structures, 39(19),
       pp. 5101--5121, 2002. https://doi.org/10.1016/S0020-7683(02)00399-2

    """
    x2, x3, els = vm.rectangle(0.5, 0.5, 8, 8, 8)
    sec = run(x2, x3, els, 2.6e7, 0.3)
    Kt = sec.Kt
    ref = {(0, 0): 0.650e7, (1, 1): 0.207e7, (2, 2): 0.207e7,
           (3, 3): 0.879e5, (4, 4): 0.135e6, (5, 5): 0.135e6}
    for (i, j), v in ref.items():
        assert '%.2e' % Kt[i, j] == '%.2e' % v
    for (i, j) in [(0, 3), (1, 4), (2, 5)]:
        assert abs(Kt[i, j]) < 1e-8*Kt[0, 0]


@pytest.mark.parametrize('etype', [6, 8,
                                   pytest.param(9, marks=pytest.mark.slow)])
def test_circle(etype):
    r"""Solid circle: area, `EA`, `EI`, `GJ` and the shear correction
    factor against the elasticity solutions

    Notes
    -----
    The reference values are `A = \pi R^2`, `I = \pi R^4/4`, `J = \pi
    R^4/2` and the shear correction factor of the Saint-Venant flexure
    solution, :func:`circle_k`.
    """
    E, nu, R = 1., 0.3, 1.
    G = E/(2*(1 + nu))
    x2, x3, els = vm.circle(R, 8, etype)
    sec = run(x2, x3, els, E, nu)
    A = np.pi*R**2
    Kt = sec.Kt
    # NOTE geometric error of the mapped boundary of about 3e-6
    assert np.isclose(sec.area, A, rtol=1e-5)
    assert np.isclose(Kt[0, 0], E*sec.area, rtol=1e-12)
    assert np.isclose(Kt[4, 4], E*np.pi*R**4/4, rtol=1e-5)
    assert np.isclose(Kt[5, 5], E*np.pi*R**4/4, rtol=1e-5)
    assert np.isclose(Kt[3, 3], G*np.pi*R**4/2, rtol=1e-5)
    assert np.isclose(Kt[1, 1]/(G*A), circle_k(nu), rtol=1e-5)
    assert np.isclose(Kt[2, 2]/(G*A), circle_k(nu), rtol=1e-5)


def test_tube():
    r"""Tube: `EA`, `EI` and `GJ` against the elasticity solutions

    Notes
    -----
    The reference values are `A = \pi (R_o^2 - R_i^2)`, `I = \pi (R_o^4 -
    R_i^4)/4` and `J = 2 I`.
    """
    E, nu, Ri, Ro = 1., 0.3, 0.6, 1.
    G = E/(2*(1 + nu))
    x2, x3, els = vm.annulus(Ri, Ro, 4, 32, 8)
    sec = run(x2, x3, els, E, nu)
    A = np.pi*(Ro**2 - Ri**2)
    I = np.pi*(Ro**4 - Ri**4)/4
    assert np.isclose(sec.Kt[0, 0], E*A, rtol=1e-5)
    assert np.isclose(sec.Kt[4, 4], E*I, rtol=1e-5)
    assert np.isclose(sec.Kt[5, 5], E*I, rtol=1e-5)
    assert np.isclose(sec.Kt[3, 3], G*2*I, rtol=1e-5)


@pytest.mark.parametrize('nu, c2, c3', [(0., 0.7404, 0.6780),
                                        (0.25, 0.7367, 0.6764)])
def test_T_section_Yu2005_Table1(nu, c2, c3):
    r"""T-section of Fig. 3 of Yu and Hodges (2005): a `4a \times 2a`
    flange under a `2a \times 2a` web; shear correction factors against the
    VABS values of their Table 1. Their mesh is not given; with this mesh
    the difference is below 0.05 %, and it decreases slowly with refinement
    because of the re-entrant corners

    Notes
    -----
    Geometry of Fig. 3 and reference values of Table 1 of [1]_, p. 51.

    References
    ----------
    .. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    a = 1.
    parts = [vm.rectangle(4*a, 2*a, 16, 8, 8, 0., a),
             vm.rectangle(2*a, 2*a, 8, 8, 8, 0., 3*a)]
    x2, x3, els = vm.merge(parts)
    sec = run(x2, x3, els, 1., nu)
    G = 1/(2*(1 + nu))
    assert np.isclose(sec.area, 12*a**2)
    assert np.isclose(sec.Kt[1, 1]/(G*sec.area), c2, rtol=5e-4)
    assert np.isclose(sec.Kt[2, 2]/(G*sec.area), c3, rtol=5e-4)
    # NOTE symmetric about x3: tension and shear centres on the x3 axis
    assert abs(sec.shear_centre[0]) < 1e-10
    assert np.allclose(sec.tension_centre, [0, 5*a/3], atol=1e-10)


@pytest.mark.parametrize('d2, d3', [(0.3, 0.), (0., 0.4), (0.3, -0.2)])
def test_offset_reference_axis(d2, d3):
    r"""Rectangle with its centroid at `(d_2, d_3)`: extension-bending and
    shear-twist couplings, centres, conversion to the convention of pyfe3d
    and :class:`BeamProp` against the centroidal properties transported to
    the reference axis

    Notes
    -----
    In the convention of :mod:`pyfe3d.vabs`, the axial strain is
    `\gamma_{11} + x_3 \kappa_2 - x_2 \kappa_3`, Eq. (36) of [1]_, such
    that the extension-bending couplings of a homogeneous
    isotropic section are `\langle E (-x_2) \rangle = -E A d_2` with
    `\kappa_3` and `\langle E x_3 \rangle = E A d_3` with `\kappa_2`. In the
    convention of pyfe3d they become `E A d_2 = E Q_z` with `\kappa_y` and
    `E A d_3 = E Q_y` with `\kappa_z`, as in the rigid cross section of
    :class:`pyfe3d.BeamC`. The shear centre and the tension centre are at
    the centroid, and the shear-twist couplings are those of the transport
    of the centroidal matrix to the reference axis.

    References
    ----------
    .. [1] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
       beam sectional analysis -- An updated version", International
       Journal of Engineering Science, 59, pp. 40--64, 2012.
       https://doi.org/10.1016/j.ijengsci.2012.03.006

    """
    E, nu, rho = 1., 0.3, 2.
    a, b = 1., 2.
    A = a*b
    x2, x3, els = vm.rectangle(a, b, 8, 16, 8)
    s0 = run(x2, x3, els, E, nu)
    sec = run(x2 + d2, x3 + d3, els, E, nu, rho=rho)
    sec.calc_mass()
    Kc, Kt, K0 = sec.Kc, sec.Kt, s0.Kt
    atol = 1e-12*Kt[0, 0]
    assert np.isclose(Kc[0, 3], -E*A*d2, atol=atol)
    assert np.isclose(Kc[0, 2], E*A*d3, atol=atol)
    assert np.isclose(Kt[0, 5], -E*A*d2, atol=atol)
    assert np.isclose(Kt[0, 4], E*A*d3, atol=atol)
    assert np.isclose(Kt[2, 3], K0[2, 2]*d2, rtol=1e-10, atol=atol)
    assert np.isclose(Kt[1, 3], -K0[1, 1]*d3, rtol=1e-10, atol=atol)
    assert np.isclose(Kt[3, 3], K0[3, 3] + K0[2, 2]*d2**2 + K0[1, 1]*d3**2,
                      rtol=1e-10)
    assert np.allclose(sec.tension_centre, [d2, d3], atol=1e-10)
    assert np.allclose(sec.shear_centre, [d2, d3], atol=1e-10)
    assert np.allclose(sec.mass_centre, [d2, d3], atol=1e-10)
    # NOTE convention of pyfe3d, {N, Vy, Vz, Mx, Mz, My}
    Cs = yu_to_pyfe3d(Kt)
    assert np.isclose(Cs[0, 4], E*A*d2, atol=atol)
    assert np.isclose(Cs[0, 5], E*A*d3, atol=atol)
    Ccl = yu_to_pyfe3d(Kc)
    assert np.isclose(Ccl[0, 2], E*A*d2, atol=atol)
    assert np.isclose(Ccl[0, 3], E*A*d3, atol=atol)
    prop = beamprop_from_vabs(Cs, E, sec=sec)
    # NOTE the isotropic section has the pattern of the 11 attributes
    C11 = prop.C
    del prop.C
    assert np.allclose(prop.C, C11, rtol=1e-8, atol=1e-8*np.abs(C11).max())
    assert np.isclose(prop.A, A)
    assert np.isclose(prop.Ay, A*d2, atol=1e-12)
    assert np.isclose(prop.Az, A*d3, atol=1e-12)
    assert np.isclose(prop.Izz, a**3*b/12 + A*d2**2)
    assert np.isclose(prop.Iyy, a*b**3/12 + A*d3**2)
    assert np.isclose(prop.Iyz, A*d2*d3, atol=1e-12)
    assert np.isclose(prop.G12, K0[1, 1]/A)
    assert np.isclose(prop.G13, K0[2, 2]/A)
    assert np.isclose(prop.intrho, rho*A)
    assert np.isclose(prop.intrhoy, rho*A*d2, atol=1e-12)
    assert np.isclose(prop.intrhoz, rho*A*d3, atol=1e-12)
    assert np.isclose(prop.intrhoy2, rho*(a**3*b/12 + A*d2**2))
    assert np.isclose(prop.intrhoz2, rho*(a*b**3/12 + A*d3**2))
    assert np.isclose(prop.intrhoyz, rho*A*d2*d3, atol=1e-12)


def test_constraints():
    r"""The classical stiffness does not depend on the warping constraints;
    for this isotropic section neither does the Timoshenko stiffness

    Notes
    -----
    The independence of the classical stiffness from the warping
    constraints is stated in Section 5.2 of [1]_.

    References
    ----------
    .. [1] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
       beam sectional analysis -- An updated version", International
       Journal of Engineering Science, 59, pp. 40--64, 2012.
       https://doi.org/10.1016/j.ijengsci.2012.03.006

    """
    x2, x3, els = vm.rectangle(1., 2., 4, 8, 8)
    s1 = run(x2 + 0.1, x3 - 0.2, els, 1., 0.3, constraint='yu2012')
    s2 = run(x2 + 0.1, x3 - 0.2, els, 1., 0.3, constraint='yu2002')
    assert np.allclose(s1.Kc, s2.Kc, rtol=1e-10, atol=1e-12)
    assert np.allclose(s1.Kt, s2.Kt, rtol=1e-8, atol=1e-10)
    # NOTE the warping satisfies its constraints
    sec = CrossSection(x2, x3, els, [isotropic(1., 0.3)], [0]*len(els))
    for constraint in ('yu2012', 'yu2002'):
        sec.calc_stiffness(constraint)
        Dc = sec._assemble(0 if constraint == 'yu2012' else 1)[5]
        assert np.abs(Dc.T @ sec.V0).max() < 1e-12
        assert np.abs(Dc.T @ sec.V1S).max() < 1e-12


if __name__ == '__main__':
    test_rectangle(8, 8, 5.e-5)
    test_offset_reference_axis(0.3, 0.)
