r"""Verification of the hourglass control of :class:`pyfe3d.Quad4R`.

The reduced integration of :class:`pyfe3d.Quad4R` samples the strains at the
element centroid only, which leaves the element rank-deficient, and the
missing rank is restored with the hourglass control of

    Brockman, R. A., 1987, "Dynamics of the Bilinear Mindlin Plate Element",
    Int. J. Numer. Methods Eng., 24(12), pp. 2343-2356.
    https://doi.org/10.1002/nme.1620241208

whose Eq. (14) is quoted from, and is attributed here to,

    Belytschko, T., and Tsay, C. S., 1983, "A stabilization procedure for the
    quadrilateral plate element with one-point quadrature", Int. J. Numer.
    Methods Eng., 19(3), pp. 405-419.
    https://doi.org/10.1002/nme.1620190308

The tests below check the implementation against that reference term by term.
Writing `\boldsymbol{h} = [1, -1, 1, -1]^\top` for the hourglass pattern of the
bilinear element, `\boldsymbol{s} = [1, 1, 1, 1]^\top`, and `A` for the element area,
the quantities being verified are

* the hourglass operator of Brockman's Eq. (15),

  .. math::
      \tilde\gamma = 2 \frac{x_{31} y_{31} + x_{24} y_{42}}{A^2}
      \quad , \quad
      \boldsymbol{\gamma} = \left. \frac{\partial^2 \boldsymbol{S}}
                                 {\partial x \partial y}\right|_{\xi=\eta=0}
      = \frac{\tilde\gamma}{4} \boldsymbol{h}

  where `x_{ij} = x_i - x_j`. The two expressions of Eq. (15) are the same
  operator, which :func:`test_hourglass_operator_matches_brockman_eq15`
  confirms;

* the generalized hourglass stiffnesses of Brockman's Eq. (16),

  .. math::
      E^{(h)}_u = E^{(h)}_v = \frac{0.10 E t}{1 + 1/A}
      \quad , \quad
      E^{(h)}_w = E^{(h)}_{\theta_x} = E^{(h)}_{\theta_y}
                = \frac{0.10 E t^3}{1 + 1/A}

  which :class:`pyfe3d.Quad4R` generalises to laminates by replacing `E` with
  the equivalent membrane moduli `E_1^{eq} = 1/(h a_{11})` and `E_2^{eq} =
  1/(h a_{22})`, from the inverse `[a]` of the extensional stiffness matrix,
  for `u` and `v`, and with the equivalent bending moduli `E_1^{b} = 12/(h^3
  d_{11})` and `E_2^{b} = 12/(h^3 d_{22})`, from the lower-right block `[d]`
  of the inverse of the ABD matrix, for `\theta_x`, `\theta_y` and, with
  their mean, `w`, all of which reduce to Eq. (16) for an isotropic plate,
  see :func:`test_laminate_hourglass_stiffnesses_use_membrane_and_bending_moduli`;

* the transverse displacement `w` is not stabilised with Eq. (16b) since
  0.12.0: the transverse shear of :class:`pyfe3d.Quad4R` is that of
  Belytschko and Leviathan (1994), the constant part of the assumed strains
  of Dvorkin and Bathe (1984) and the physical stabilisation of their
  nonconstant part, which controls the hourglass mode of `w` and passes the
  constant-twist and the thin-plate bending patch tests, see
  :func:`test_transverse_shear_matches_belytschko_leviathan`,
  :func:`test_constant_twist_patch_test` and
  :func:`test_bending_patch_test_belytschko_leviathan`; the rotations `r_x`
  and `r_y` are not stabilised with Eq. (16b) either, but with the physical
  stabilisation of the bending of Belytschko and Leviathan (1994), Eq.
  (B.18), see :func:`test_bending_stabilisation_matches_belytschko_leviathan`;
  only the coefficients of `u` and `v` of Eq. (16a) are those verified
  below, scaled by ``hgfactor_membrane``;

* the classification of the element modes of Brockman's p. 2347, six proper
  rigid-body motions, eight uniform-strain states captured by the one-point
  rule, and six modes that the one-point rule misses, namely the five
  hourglass patterns `u = h`, `v = h`, `w = h`, `\theta_x = h`, `\theta_y =
  h`, which the stabilisation suppresses, and the twisting mode

  .. math::
      w = \frac{1}{4}(\boldsymbol{s}^\top \boldsymbol{y}) x
        + \frac{1}{4}(\boldsymbol{s}^\top \boldsymbol{x}) y
      \quad , \quad
      \theta_x = x \quad , \quad \theta_y = y

  which it does not, and which Brockman notes "exists for a single element,
  but cannot occur in a mesh of two or more elements, as shown by Hughes",
  citing there

      Hughes, T. J. R., 1980, "Recent developments in computer methods for
      structural analysis", Nucl. Eng. Des., 57(2), pp. 427-439.

  The mode is verified below to be the same one as Mode 2 of Figure 10 of
  Hughes, Taylor and Kanoknukulchai (1977), documented in
  ``test_quad4_spurious_shear_mode.py``, which is a separate reference from
  the one Brockman cites;

* the mass formulations of Brockman's Eqs. (21), (22) and (31) and of his
  p. 2349, offered by :meth:`pyfe3d.Quad4R.update_M` through ``mtype``.

Two further points concern the choice of hourglass operator and the
calibration of Eq. (16):

1. Brockman quotes the operator of Belytschko and Tsay as his Eq. (14),

   .. math::
       \boldsymbol{\gamma} = \boldsymbol{h} - (\boldsymbol{h}^\top \boldsymbol{x}) \boldsymbol{b}_1
                              - (\boldsymbol{h}^\top \boldsymbol{y}) \boldsymbol{b}_2

   with `\boldsymbol{b}_1` and `\boldsymbol{b}_2` the centroidal derivatives `S_{i,x}` and
   `S_{i,y}`, the orthogonalisation being due to Flanagan and Belytschko
   (1981), and observes that "the last two terms of equation (14) are
   important for irregular elements, if the hourglass strains are to vanish
   in the presence of rigid-body motion and uniform strain". He then adopts
   the definition of Eq. (15), which does not contain them, and later states
   on p. 2348 that "the hourglassing modes identified in the preceding
   section are orthogonal to both the rigid-body motions and the
   constant-strain states". That orthogonality holds for Eq. (15) only while
   the element is a parallelogram, so with Eq. (15) alone an irregular
   element gives a rigid-body rotation and a uniform strain state a spurious
   energy and fails the constant-strain patch test.

   Since 0.10.0 :class:`pyfe3d.Quad4R` therefore applies the two correction
   terms of Eq. (14), scaled by the normalisation of Eq. (15) so that the
   generalized stiffnesses of Eq. (16) keep their calibration. The
   correction changes nothing on a parallelogram, where `\boldsymbol{h}^\top
   \boldsymbol{x}` and `\boldsymbol{h}^\top \boldsymbol{y}` are zero, and changes nothing about
   the energy of the five hourglass patterns on any shape, because the
   centroidal sums `\boldsymbol{h}^\top \boldsymbol{b}_1` and `\boldsymbol{h}^\top \boldsymbol{b}_2`
   vanish identically. What it does change is that the hourglass strains are
   now orthogonal to the rigid-body motions and to the uniform strain states
   on any shape, see
   :func:`test_hourglass_strains_orthogonal_to_linear_fields`,
   :func:`test_rigid_body_rotation_energy` and
   :func:`test_constant_strain_patch_test_on_an_irregular_mesh`.

2. The factor `(1 + 1/A)` of Eq. (16) is not dimensionally homogeneous, `1/A`
   being an inverse area. Brockman introduces it deliberately, as "motivated
   by locking problems observed in elements with extremely small dimensions",
   and reports good behaviour "over a range of six orders of magnitude in the
   planform dimension". Adopting Eq. (14) does not remove this: it removes
   the dependence of the rigid-body and constant-strain states on the
   artificial stiffness, so those are now reproduced exactly whatever the
   unit of length, but the magnitude of the artificial stiffness itself still
   depends on that unit, and a response that legitimately contains hourglass
   components, which Brockman lists on p. 2348 as "torsional or inplane
   bending modes", still moves with it, see
   :func:`test_hourglass_stiffness_depends_on_the_length_unit`.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate, laminated_plate
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, INT, DOUBLE, DOF)

# NOTE Brockman's Example 1 properties, p. 2353
E_BROCKMAN = 1.e7
NU_BROCKMAN = 0.25
T_BROCKMAN = 0.05

H = np.array([1., -1., 1., -1.])
S = np.array([1., 1., 1., 1.])

RECTANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]])
PARALLELOGRAM = np.array([[0., 0., 0.], [2., 0., 0.], [2.4, 1., 0.],
                          [0.4, 1., 0.]])
DISTORTED = np.array([[0., 0., 0.], [2., 0., 0.], [2.3, 1.4, 0.],
                      [0.2, 1.1, 0.]])
TRAPEZOID = np.array([[0., 0., 0.], [3., 0., 0.], [2., 1.5, 0.],
                      [1., 1.5, 0.]])

HG_NAMES = {0: 'u = h', 1: 'v = h', 2: 'w = h', 3: 'rx = h', 4: 'ry = h'}


def make_element(ncoords, K6ROT=0., xmat=(0., 0., 1.)):
    r"""A single Quad4R with its rotation matrix and probe already updated

    The fictitious drilling penalty is switched off by default, because its
    operator reads the in-plane translations and would otherwise add energy
    to the in-plane hourglass patterns, which would mask the term being
    verified. ``xmat`` is the material direction of
    :meth:`pyfe3d.Quad4R.update_rotation_matrix`.

    """
    probe = Quad4RProbe()
    el = Quad4R(probe)
    ncoords_flatten = ncoords.flatten()
    for i in range(ncoords.shape[0]):
        setattr(el, 'n%d' % (i + 1), i + 1)
        setattr(el, 'c%d' % (i + 1), DOF*i)
    el.init_k_KC0 = 0
    el.init_k_M = 0
    el.drilling_model = 1
    el.K6ROT = K6ROT
    el.update_rotation_matrix(ncoords_flatten, *xmat)
    el.update_probe_xe(ncoords_flatten)
    el.update_area()
    return el, probe


def element_KC0(el, prop, hgfactor=1., hgfactor_physical=None):
    r"""Element matrix in global coordinates, ``hgfactor_membrane`` and
    ``hgfactor_physical`` equal to ``hgfactor`` unless ``hgfactor_physical``
    is given"""
    data = Quad4RData()
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    hgp = hgfactor if hgfactor_physical is None else hgfactor_physical
    el.update_KC0(KC0r, KC0c, KC0v, prop, 0, hgfactor, hgp)
    return coo_matrix((KC0v, (KC0r, KC0c)), shape=(24, 24)).toarray()


def rotation(el):
    return np.array([[el.r11, el.r12, el.r13],
                     [el.r21, el.r22, el.r23],
                     [el.r31, el.r32, el.r33]])


def element_xy(probe):
    r"""The in-plane element coordinates of the four nodes"""
    xe = np.asarray(probe.xe).reshape(4, 3)
    return xe[:, 0].copy(), xe[:, 1].copy()


def to_global(el, ue_element):
    r"""Bring a displacement vector from element to global coordinates

    The columns of the matrix of :func:`rotation` are the element axes, so
    a vector in element coordinates is brought to global ones by that
    matrix, not by its transpose, which this function used until 0.12.0;
    the tests that used it pass with both, the patterns they map being
    insensitive to that error, which
    :func:`test_transverse_shear_matches_belytschko_leviathan` revealed on
    the distorted element.

    """
    rmat = rotation(el)
    ug = np.zeros_like(ue_element)
    for i in range(4):
        ug[DOF*i:DOF*i + 3] = rmat @ ue_element[DOF*i:DOF*i + 3]
        ug[DOF*i + 3:DOF*i + 6] = rmat @ ue_element[DOF*i + 3:DOF*i + 6]
    return ug


def hourglass_mode(el, dof):
    r"""The pattern of Brockman's p. 2347 for one displacement component"""
    ue = np.zeros(24)
    ue[dof::DOF] = H
    return to_global(el, ue)


def gamma_tilde(el, probe):
    r"""Brockman's Eq. (15), in element coordinates"""
    x, y = element_xy(probe)
    return 2.*((x[2] - x[0])*(y[2] - y[0])
               + (x[1] - x[3])*(y[3] - y[1]))/el.area**2


def gamma_vector(el, probe):
    r"""The hourglass operator that Eq. (15) defines, `\gamma = \tilde\gamma
    \boldsymbol{h}/4`, equal to the centroidal `\partial^2 \boldsymbol{S}/\partial x \partial y`

    """
    return gamma_tilde(el, probe)/4.*H


def centroid_derivatives(probe):
    r"""`\boldsymbol{b}_1` and `\boldsymbol{b}_2` of Brockman's Eq. (14), the centroidal
    Cartesian derivatives of the bilinear shape functions

    """
    xe = np.asarray(probe.xe).reshape(4, 3)
    x, y = xe[:, 0], xe[:, 1]
    # NOTE the Jacobian of the bilinear map at xi = eta = 0
    J11 = 0.25*(-x[0] + x[1] + x[2] - x[3])
    J12 = 0.25*(-y[0] + y[1] + y[2] - y[3])
    J21 = 0.25*(-x[0] - x[1] + x[2] + x[3])
    J22 = 0.25*(-y[0] - y[1] + y[2] + y[3])
    detJ = J11*J22 - J12*J21
    j11, j12 = J22/detJ, -J12/detJ
    j21, j22 = -J21/detJ, J11/detJ
    S_xi = 0.25*np.array([-1., 1., 1., -1.])
    S_eta = 0.25*np.array([-1., -1., 1., 1.])
    return j11*S_xi + j12*S_eta, j21*S_xi + j22*S_eta


def brockman_stiffnesses(area, E=E_BROCKMAN, t=T_BROCKMAN):
    r"""Brockman's Eq. (16) for an isotropic plate, verbatim

    Kept for the comparison of
    :func:`test_homogeneous_stiffnesses_agree_with_brockman_for_small_areas`.
    pyfe3d does not use this form, see :func:`homogeneous_stiffnesses`.

    """
    factor = 1./(1. + 1./area)
    return {0: 0.1*E*t*factor, 1: 0.1*E*t*factor,
            2: 0.1*E*t**3*factor, 3: 0.1*E*t**3*factor,
            4: 0.1*E*t**3*factor}


def homogeneous_stiffnesses(area, E=E_BROCKMAN, t=T_BROCKMAN):
    r"""The dimensionally homogeneous stiffnesses pyfe3d uses

    Brockman's `1/(1 + 1/A)` is replaced by the area itself for the in-plane
    translations and for the rotations, which is what the dimensions
    require, see the section "Dimensional homogeneity of the hourglass
    stiffnesses" of the :mod:`pyfe3d.quad4r` documentation. The
    transverse displacement `w` has no such coefficient since 0.12.0, see
    :func:`test_transverse_shear_matches_belytschko_leviathan`.

    The rotations `r_x` and `r_y` have the physical stabilisation of the
    bending of Belytschko and Leviathan (1994) since 0.12.0, see
    :func:`test_bending_stabilisation_matches_belytschko_leviathan`.

    """
    return {0: 0.1*E*t*area, 1: 0.1*E*t*area}


def test_hourglass_operator_matches_brockman_eq15():
    r"""The hourglass energy against the closed form of Eqs. (15) and (16)

    Each of the hourglass patterns of p. 2347 of `u` and `v` produces no
    strain at the centroid, so the one-point rule gives it no
    energy and the whole of its energy comes from the stabilisation,

    .. math::
        U = \frac{1}{2} E^{(h)} (\boldsymbol{\gamma}^\top \boldsymbol{h})^2 A
          = \frac{1}{2} E^{(h)} \tilde\gamma^2 A

    since `\boldsymbol{\gamma}^\top \boldsymbol{h} = \tilde\gamma \boldsymbol{h}^\top \boldsymbol{h}/4 =
    \tilde\gamma`. The identity is checked on four shapes, including two
    irregular ones, and holds to machine precision on all of them, which
    verifies the operator of Eq. (15) and the stiffnesses of Eq. (16)
    together. The patterns of `w`, `r_x` and `r_y` are controlled by the
    physical stabilisation of Belytschko and Leviathan (1994) since 0.12.0,
    see :func:`test_transverse_shear_matches_belytschko_leviathan` and
    :func:`test_bending_stabilisation_matches_belytschko_leviathan`.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    for name, ncoords in [('rectangle', RECTANGLE),
                          ('parallelogram', PARALLELOGRAM),
                          ('distorted', DISTORTED),
                          ('trapezoid', TRAPEZOID)]:
        el, probe = make_element(ncoords)
        KC0 = element_KC0(el, prop)
        gt = gamma_tilde(el, probe)
        Eh = homogeneous_stiffnesses(el.area)
        for dof in (0, 1):
            ue = hourglass_mode(el, dof)
            U = 0.5*ue.dot(KC0.dot(ue))
            U_ref = 0.5*Eh[dof]*gt**2*el.area
            print(name, HG_NAMES[dof], U, U_ref)
            assert np.isclose(U, U_ref, rtol=1e-12), (name, HG_NAMES[dof],
                                                      U, U_ref)


def test_hourglass_operator_equals_second_derivative_of_N():
    r"""The two expressions of Eq. (15) define the same operator

    ``Quad4R`` builds the operator as the centroidal `\partial^2 \boldsymbol{S}/\partial x
    \partial y`, which the element code evaluates as ``0.25*(j11*j22 +
    j12*j21)`` times the alternating pattern. Eq. (15) also gives it in
    closed form as `\tilde\gamma \boldsymbol{h}/4`. The two agree on any shape.

    """
    for name, ncoords in [('rectangle', RECTANGLE),
                          ('parallelogram', PARALLELOGRAM),
                          ('distorted', DISTORTED),
                          ('trapezoid', TRAPEZOID)]:
        el, probe = make_element(ncoords)
        xe = np.asarray(probe.xe).reshape(4, 3)
        x, y = xe[:, 0], xe[:, 1]
        J11 = 0.25*(-x[0] + x[1] + x[2] - x[3])
        J12 = 0.25*(-y[0] + y[1] + y[2] - y[3])
        J21 = 0.25*(-x[0] - x[1] + x[2] + x[3])
        J22 = 0.25*(-y[0] - y[1] + y[2] + y[3])
        detJ = J11*J22 - J12*J21
        j11, j12 = J22/detJ, -J12/detJ
        j21, j22 = -J21/detJ, J11/detJ
        Nxy = 0.25*(j11*j22 + j12*j21)*H
        print(name, Nxy, gamma_vector(el, probe))
        assert np.allclose(Nxy, gamma_vector(el, probe), rtol=1e-12,
                           atol=0.), name


def test_generalized_stiffnesses_are_dimensionally_homogeneous():
    r"""The generalized stiffnesses actually implemented, one at a time

    pyfe3d departs from Brockman's Eq. (16) as set out in the
    section "Dimensional homogeneity of the hourglass stiffnesses" of the
    :mod:`pyfe3d.quad4r` documentation: his `1/(1 + 1/A)` is replaced by the
    area itself, which is what the dimensions of `K_{ij} = A E^{(h)}
    \gamma_i \gamma_j` require. The transverse displacement `w` has no such
    coefficient since 0.12.0, see
    :func:`test_transverse_shear_matches_belytschko_leviathan`.

    The energy of each hourglass pattern is linear in the corresponding
    ``hgfactor``, which is what makes the stiffnesses separable, and the
    slope must be the implemented value. Two elements of very different
    areas are used, so the area factor is exercised.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    for name, ncoords in [('rectangle', RECTANGLE),
                          ('small rectangle', 0.05*RECTANGLE)]:
        el, probe = make_element(ncoords)
        gt = gamma_tilde(el, probe)
        Eh = homogeneous_stiffnesses(el.area)
        K1 = element_KC0(el, prop, hgfactor=1.)
        K3 = element_KC0(el, prop, hgfactor=3.)
        for dof in (0, 1):
            ue = hourglass_mode(el, dof)
            slope = 0.5*(ue.dot(K3.dot(ue)) - ue.dot(K1.dot(ue)))/2.
            ref = 0.5*Eh[dof]*gt**2*el.area
            print(name, HG_NAMES[dof], slope, ref)
            assert np.isclose(slope, ref, rtol=1e-12), (name, HG_NAMES[dof])


def test_five_hourglass_patterns_are_spurious_without_stabilisation():
    r"""Brockman's p. 2347, the modes the one-point rule misses

    With the stabilisation switched off, the five hourglass patterns carry
    exactly no energy, which is the rank deficiency the scheme exists to
    remove. With it on, all five are stiffened.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    for name, ncoords in [('rectangle', RECTANGLE),
                          ('distorted', DISTORTED)]:
        el, probe = make_element(ncoords)
        K_off = element_KC0(el, prop, hgfactor=0.)
        K_on = element_KC0(el, prop, hgfactor=1.)
        scale = np.abs(K_on).max()
        for dof in range(5):
            ue = hourglass_mode(el, dof)
            U_off = ue.dot(K_off.dot(ue))/scale
            U_on = ue.dot(K_on.dot(ue))/scale
            print(name, HG_NAMES[dof], U_off, U_on)
            assert abs(U_off) < 1e-14, (name, HG_NAMES[dof], U_off)
            assert U_on > 1e-8, (name, HG_NAMES[dof], U_on)


def test_stabilisation_restores_the_rank_of_a_rectangular_element():
    r"""Brockman's p. 2347 mode count, on a rectangle

    Of the twenty-four degrees-of-freedom of ``Quad4R``, Brockman's plate
    accounts for twenty, the drilling rotation being absent from his
    element. With the stabilisation off, the element matrix is singular in
    the six proper rigid-body motions plus the five hourglass patterns plus
    the twisting mode. With it on, only the rigid-body motions remain, the
    drilling rotations being held by their own operator: the stabilisation
    of the transverse shear of Belytschko and Leviathan (1994), which
    replaces Brockman's control of `w` since 0.12.0, also stiffens the
    twisting mode, which Brockman's does not, see
    :func:`test_twisting_mode_is_stabilised`. Up to 0.11 the twisting mode
    remained, 7 zero eigenvalues.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    el, probe = make_element(RECTANGLE, K6ROT=100.)
    counts = {}
    for label, hgfactor in [('off', 0.), ('on', 1.)]:
        K = element_KC0(el, prop, hgfactor=hgfactor)
        w = np.linalg.eigvalsh(K)
        counts[label] = int((w < 1e-11*abs(w).max()).sum())
        print('hourglass', label, 'zero eigenvalues', counts[label], w[:12])
    # NOTE 6 rigid-body motions + 5 hourglass patterns + the twisting mode
    assert counts['off'] == 12, counts
    # NOTE the 6 rigid-body motions only
    assert counts['on'] == 6, counts


def test_twisting_mode_is_stabilised():
    r"""Brockman's Eq. (13) and his remark attributing it to Hughes

    Brockman's Eq. (13) is

    .. math::
        w = \frac{1}{4}(\boldsymbol{s}^\top \boldsymbol{y}) x
          + \frac{1}{4}(\boldsymbol{s}^\top \boldsymbol{x}) y
        \quad , \quad \theta_x = x \quad , \quad \theta_y = y

    which is not an hourglass pattern, so Brockman's stabilisation does not
    reach it. Brockman notes that it "cannot occur in a mesh of two or more
    elements, as shown by Hughes", which is checked here by counting the
    zero eigenvalues of a two-element mesh. Since 0.12.0 the transverse
    shear of :class:`pyfe3d.Quad4R` is that of Belytschko and Leviathan
    (1994), whose stabilisation of the nonconstant part of the assumed
    shear strains gives the mode energy: `r_x = x` and `r_y = y` vary the
    covariant shear strains along the element, Eqs. (3.38)-(3.39), so a
    single element is no longer singular in it. Without the stabilisation,
    ``hgfactor = 0``, it is a zero-energy mode, as in Brockman's element.

    Brockman measures the transverse shear as `\boldsymbol{\nabla} w -
    \boldsymbol{\theta}`, whereas :class:`pyfe3d.Quad4R` uses `\gamma_{xz} = w_{,x}
    + r_y` and `\gamma_{yz} = w_{,y} - r_x`, so `r_x = \theta_x`, `r_y =
    -\theta_y` and the sign of the first term of `w` changes,

    .. math::
        w = -\frac{1}{4}(\boldsymbol{s}^\top \boldsymbol{y}) x
          + \frac{1}{4}(\boldsymbol{s}^\top \boldsymbol{x}) y
        \quad , \quad r_x = x \quad , \quad r_y = y

    With the element centroid as the origin both terms of `w` vanish and the
    mode is `w = 0`, `r_x = x`, `r_y = y`. That is the same mode as Mode 2 of
    Figure 10 of Hughes, Taylor and Kanoknukulchai (1977), under the mapping
    `\theta_1 = -r_y`, `\theta_2 = r_x` established in
    ``test_quad4_spurious_shear_mode.py``.

    The mode is defined for a regular element and is a zero-energy mode of a
    rectangle and of a parallelogram. On an irregular element it is no longer
    exactly strain-free, which is why only the regular shapes are asserted.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    for gname, ncoords in [('rectangle', RECTANGLE),
                           ('parallelogram', PARALLELOGRAM)]:
        el, probe = make_element(ncoords, K6ROT=100.)
        x, y = element_xy(probe)
        ue = np.zeros(24)
        for i in range(4):
            ue[DOF*i + 2] = -0.25*S.dot(y)*x[i] + 0.25*S.dot(x)*y[i]
            ue[DOF*i + 3] = x[i]
            ue[DOF*i + 4] = y[i]
        ue = to_global(el, ue)
        for label, hgfactor in [('off', 0.), ('on', 1.)]:
            K = element_KC0(el, prop, hgfactor=hgfactor)
            U = ue.dot(K.dot(ue))/(np.abs(K).max()*ue.dot(ue))
            print(gname, 'twisting mode, hourglass', label, U)
            if label == 'off':
                assert abs(U) < 1e-14, (gname, label, U)
            else:
                assert U > 1e-6, (gname, label, U)

    # NOTE two elements side by side, which must remove it
    ncoords = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.],
                        [0., 1., 0.], [4., 0., 0.], [4., 1., 0.]])
    conn = [(1, 2, 3, 4), (2, 5, 6, 3)]
    data = Quad4RData()
    probe = Quad4RProbe()
    ncoords_flatten = ncoords.flatten()
    N = DOF*6
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*2, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*2, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*2, dtype=DOUBLE)
    init_k_KC0 = 0
    for e in conn:
        el = Quad4R(probe)
        for a in range(4):
            setattr(el, 'n%d' % (a + 1), e[a])
            setattr(el, 'c%d' % (a + 1), DOF*(e[a] - 1))
        el.init_k_KC0 = init_k_KC0
        el.drilling_model = 1
        el.K6ROT = 100.
        el.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    w = np.linalg.eigvalsh(K)
    nzeros = int((w < 1e-11*abs(w).max()).sum())
    print('two elements, zero eigenvalues', nzeros, w[:10])
    # NOTE only the six rigid-body motions survive
    assert nzeros == 6, (nzeros, w[:10])


def test_hourglass_strains_orthogonal_to_linear_fields():
    r"""Brockman's p. 2348 orthogonality claim, for both operators

    The scheme relies on the hourglass strains vanishing for the rigid-body
    motions and the uniform-strain states, which requires
    `\boldsymbol{\gamma}^\top \boldsymbol{s} = \boldsymbol{\gamma}^\top \boldsymbol{x} =
    \boldsymbol{\gamma}^\top \boldsymbol{y} = 0`. For the operator of Eq. (15) the first
    always holds and the other two hold only while the element is a
    parallelogram. The operator used by the element, Eq. (15) corrected by
    the two terms of Eq. (14), satisfies all three on any shape, which is
    what this test establishes.

    The two operators are built here from the element geometry rather than
    read out of the element matrix, so that the property is checked on the
    formulas. :func:`test_rigid_body_rotation_energy` checks the consequence
    on the assembled element matrix.

    """
    for name, ncoords, regular in [('rectangle', RECTANGLE, True),
                                   ('parallelogram', PARALLELOGRAM, True),
                                   ('distorted', DISTORTED, False),
                                   ('trapezoid', TRAPEZOID, False)]:
        el, probe = make_element(ncoords)
        x, y = element_xy(probe)
        g15 = gamma_vector(el, probe)
        b1, b2 = centroid_derivatives(probe)
        # NOTE Eq. (14) as quoted by Brockman, dimensionless
        g14 = H - H.dot(x)*b1 - H.dot(y)*b2
        # NOTE what the element uses, Eq. (14) in the normalisation of
        #      Eq. (15), so that the calibration of Eq. (16) is preserved
        gused = gamma_tilde(el, probe)/4.*g14
        scale = np.abs(g15).max()*max(np.abs(x).max(), np.abs(y).max())
        print(name, 'Eq.15', g15.dot(S), g15.dot(x), g15.dot(y),
              'used', gused.dot(S), gused.dot(x), gused.dot(y))
        # NOTE the constant field is always filtered out, by every operator
        for g in (g15, g14, gused):
            assert abs(g.dot(S)) < 1e-14*max(scale, 1.), name
        # NOTE the linear fields are filtered out on any shape by Eq. (14),
        #      in either normalisation
        assert abs(g14.dot(x)) < 1e-14*max(np.abs(x).max(), 1.), name
        assert abs(g14.dot(y)) < 1e-14*max(np.abs(y).max(), 1.), name
        assert abs(gused.dot(x)) < 1e-14*max(scale, 1.), name
        assert abs(gused.dot(y)) < 1e-14*max(scale, 1.), name
        # NOTE the hourglass energy is unchanged by the correction, because
        #      the centroidal sums of the derivatives against h vanish
        assert np.isclose(gused.dot(H), g15.dot(H), rtol=1e-12), name
        if regular:
            # NOTE on a parallelogram the correction terms vanish and the
            #      two operators are the same, which is what keeps every
            #      regular mesh bit-identical
            assert np.allclose(gused, g15, rtol=1e-12, atol=0.), name
        else:
            # NOTE Eq. (15) alone would not filter the linear fields here
            assert abs(g15.dot(x)) + abs(g15.dot(y)) > 1e-3*scale, name


def test_rigid_body_rotation_energy():
    r"""The consequence of the previous test on the rigid-body motions

    All six rigid-body motions carry exactly no energy, on any element shape,
    because the hourglass strains of Eq. (14) are orthogonal to them. With
    the operator of Eq. (15) alone the rotation about the normal would pick
    up about 2.5e-4 of the largest entry of the element matrix on the
    distorted element used here.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    for gname, ncoords, regular in [('rectangle', RECTANGLE, True),
                                    ('parallelogram', PARALLELOGRAM, True),
                                    ('distorted', DISTORTED, False)]:
        el, probe = make_element(ncoords, K6ROT=100.)
        modes = np.zeros((6, 24))
        for i in range(4):
            xi, yi, zi = ncoords[i]
            for k in range(3):
                modes[k, DOF*i + k] = 1.
            modes[3, DOF*i + 1] = -zi
            modes[3, DOF*i + 2] = yi
            modes[3, DOF*i + 3] = 1.
            modes[4, DOF*i + 0] = zi
            modes[4, DOF*i + 2] = -xi
            modes[4, DOF*i + 4] = 1.
            modes[5, DOF*i + 0] = -yi
            modes[5, DOF*i + 1] = xi
            modes[5, DOF*i + 5] = 1.
        for label, hgfactor in [('on', 1.), ('off', 0.)]:
            K = element_KC0(el, prop, hgfactor=hgfactor)
            scale = np.abs(K).max()
            energies = [m.dot(K.dot(m))/(scale*m.dot(m)) for m in modes]
            print(gname, 'hourglass', label,
                  '  '.join('%s %.2e' % (n, e)
                            for n, e in zip(names, energies)))
            assert max(abs(e) for e in energies) < 1e-14, (gname, label,
                                                           energies)


# NOTE MacNeal and Harder's patch, Figure 2: a rectangle of 0.24 by 0.12
#      split into five quadrilaterals by four arbitrarily placed interior
#      nodes. Reference:
#
#          MacNeal, R. H., and Harder, R. L., 1985, "A Proposed Standard Set
#          of Problems to Test Finite Element Accuracy", Finite Elem. Anal.
#          Des., 1(1), pp. 3-20. doi:10.1016/0168-874X(85)90003-4
PATCH_OUTER = np.array([[0., 0.], [0.24, 0.], [0.24, 0.12], [0., 0.12]])
PATCH_INNER = np.array([[0.04, 0.02], [0.18, 0.03], [0.16, 0.08],
                        [0.08, 0.08]])
PATCH_CONN = [(5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7),
              (4, 1, 5, 8)]
PATCH_EXX, PATCH_EYY, PATCH_GXY = 1.e-3, -0.5e-3, 0.75e-3


def patch_solution(cls, Data, Probe, scale=1., hgfactor=1.):
    r"""Prescribe a uniform strain state on the four corners of the patch and
    return the error of the interior nodes

    The whole geometry, the thickness and the modulus are scaled so that the
    physical problem is the same for any ``scale``, which is what makes the
    unit dependence of Eq. (16) measurable.

    """
    ncoords = np.column_stack((np.vstack((PATCH_OUTER, PATCH_INNER))*scale,
                               np.zeros(8)))
    ncoords_flatten = ncoords.flatten()
    prop = isotropic_plate(thickness=0.001*scale, E=1.e6/scale**2, nu=0.25)
    data = Data()
    probe = Probe()
    N = DOF*8
    nel = len(PATCH_CONN)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*nel, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*nel, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*nel, dtype=DOUBLE)
    init_k_KC0 = 0
    for e in PATCH_CONN:
        el = cls(probe)
        for a in range(4):
            setattr(el, 'n%d' % (a + 1), e[a])
            setattr(el, 'c%d' % (a + 1), DOF*(e[a] - 1))
        el.init_k_KC0 = init_k_KC0
        el.drilling_model = 1
        el.K6ROT = 100.
        el.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        if cls is Quad4R:
            el.update_KC0(KC0r, KC0c, KC0v, prop, 0, hgfactor, hgfactor)
        else:
            el.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    x, y = ncoords[:, 0], ncoords[:, 1]
    uexact = np.zeros(N)
    uexact[0::DOF] = PATCH_EXX*x + 0.5*PATCH_GXY*y
    uexact[1::DOF] = 0.5*PATCH_GXY*x + PATCH_EYY*y

    bk = np.zeros(N, dtype=bool)
    for n in (1, 2, 3, 4):
        bk[DOF*(n - 1):DOF*(n - 1) + 2] = True
    bk[2::DOF] = True
    bk[3::DOF] = True
    bk[4::DOF] = True
    bu = ~bk
    u = np.zeros(N)
    u[bk] = uexact[bk]
    u[bu] = spsolve(KC0[bu, :][:, bu], -KC0[bu, :][:, bk].dot(uexact[bk]))
    err = np.abs(u - uexact)
    return max(err[0::DOF].max(), err[1::DOF].max())/np.abs(
        uexact[0::DOF]).max()


def test_constant_strain_patch_test_on_an_irregular_mesh():
    r"""The constant-strain patch test

    ``Quad4R`` reproduces the uniform strain state of MacNeal and Harder's
    irregular patch exactly, whether the stabilisation is on or off, because
    the hourglass strains of Eq. (14) are orthogonal to the uniform strain
    states, see :func:`test_hourglass_strains_orthogonal_to_linear_fields`.
    ``Quad4`` is exact too, and the agreement of the three is what shows that
    both the reduced integration and its stabilisation are consistent.

    With the operator of Eq. (15) alone the interior nodes of this patch came
    out wrong by about eight per cent, which is what motivated adopting the
    two correction terms.

    """
    exact_q4 = patch_solution(Quad4, Quad4Data, Quad4Probe)
    off = patch_solution(Quad4R, Quad4RData, Quad4RProbe, hgfactor=0.)
    on = patch_solution(Quad4R, Quad4RData, Quad4RProbe, hgfactor=1.)
    strong = patch_solution(Quad4R, Quad4RData, Quad4RProbe, hgfactor=100.)
    print('Quad4 %.3e   Quad4R hourglass off %.3e   on %.3e   x100 %.3e'
          % (exact_q4, off, on, strong))
    assert exact_q4 < 1e-12
    assert off < 1e-12
    assert on < 1e-12, on
    # NOTE and it stays exact however strong the stabilisation is made,
    #      which is the property that Eq. (14) buys
    assert strong < 1e-12, strong


def cook_solution(scale=1., n=4):
    r"""Cook's tapered cantilever, in-plane only, tip displacement

        Cook, R. D., 1974, "Improved Two-Dimensional Finite Element",
        J. Struct. Div. ASCE, 100(ST9), pp. 1851-1863.

    Corners (0,0), (48,44), (48,60), (0,44), thickness 1, E = 1, nu = 1/3,
    clamped at x = 0 and a unit total shear on x = 48, for which the
    reference vertical displacement at (48, 52) is 23.96. The geometry and
    the thickness are scaled by ``scale`` and the modulus by ``1/scale**2``,
    so that the physical problem is the same one, and the returned value is
    brought back to the unscaled one. That isolates the effect of the unit of
    length on the artificial hourglass stiffness.

    """
    X = np.zeros((n + 1, n + 1))
    Y = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        t = i/n
        for j in range(n + 1):
            s = j/n
            X[i, j] = 48.*t*scale
            Y[i, j] = (44.*t + (44. + 16.*t - 44.*t)*s)*scale
    nid = np.arange((n + 1)*(n + 1)).reshape(n + 1, n + 1)
    els = [(nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
           for i in range(n) for j in range(n)]
    X = X.ravel()
    Y = Y.ravel()
    nn = len(X)
    ncoords_flatten = np.vstack((X, Y, np.zeros(nn))).T.flatten()
    prop = isotropic_plate(thickness=1.0*scale, E=1.0/scale**2, nu=1.0/3.0)
    data = Quad4RData()
    probe = Quad4RProbe()
    N = DOF*nn
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=DOUBLE)
    init_k_KC0 = 0
    for e in els:
        el = Quad4R(probe)
        for a in range(4):
            setattr(el, 'n%d' % (a + 1), e[a] + 1)
            setattr(el, 'c%d' % (a + 1), DOF*e[a])
        el.init_k_KC0 = init_k_KC0
        el.drilling_model = 1
        el.K6ROT = 100.
        el.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    f = np.zeros(N)
    right = np.where(np.isclose(X, 48.*scale))[0]
    right = right[np.argsort(Y[right])]
    ys = Y[right]
    for a in range(len(right) - 1):
        le = ys[a+1] - ys[a]
        f[DOF*right[a] + 1] += le/2.
        f[DOF*right[a+1] + 1] += le/2.
    f[DOF*right + 1] /= (ys[-1] - ys[0])
    bk = np.zeros(N, dtype=bool)
    left = np.isclose(X, 0.)
    bk[0::DOF] = left
    bk[1::DOF] = left
    bk[2::DOF] = True
    bk[3::DOF] = True
    bk[4::DOF] = True
    bk[5::DOF] = left
    bu = ~bk
    u = np.zeros(N)
    u[bu] = spsolve(KC0[bu, :][:, bu], f[bu])
    tip = np.argmin((X - 48.*scale)**2 + (Y - 52.*scale)**2)
    return u[DOF*tip + 1]/scale


def test_hourglass_stiffness_is_independent_of_the_length_unit():
    r"""The homogeneous stiffnesses remove the unit dependence of Eq. (16)

    Brockman's `1/(1 + 1/A)` makes the artificial stiffness depend on the
    unit of length the model is written in, `1/A` being an inverse area. He
    introduces the factor knowingly, as "motivated by locking problems
    observed in elements with extremely small dimensions", and reports good
    behaviour "over a range of six orders of magnitude in the planform
    dimension", but the scheme switches regime at `A = 1` in whatever unit
    is used.

    Replacing the factor by the area itself, as the dimensional argument in
    the :mod:`pyfe3d.quad4r` documentation requires, removes the dependence
    exactly rather than approximately. This test measures both halves of
    that claim over nine orders of magnitude of length unit:

    * the constant-strain patch test is exact in every unit, which is what
      the correction terms of Eq. (14) buy and which held before this change
      as well;
    * Cook's in-plane bending problem, whose response legitimately contains
      the hourglass components Brockman lists on p. 2348 as "torsional or
      inplane bending modes", now gives the *same* answer in every unit. It
      previously drifted by a fraction of a per cent at the metre and
      collapsed at the kilometre, where the element areas are so small in
      that unit that Eq. (16) left almost no stabilisation at all.

    """
    for name, scale in [('metre', 1.), ('millimetre', 1.e3),
                        ('micrometre', 1.e6), ('kilometre', 1.e-3)]:
        err = patch_solution(Quad4R, Quad4RData, Quad4RProbe, scale=scale)
        q4 = patch_solution(Quad4, Quad4Data, Quad4Probe, scale=scale)
        print('patch test %-12s Quad4R %.3e   Quad4 %.3e' % (name, err, q4))
        assert err < 1e-10, (name, err)
        assert q4 < 1e-10, (name, q4)

    cook = {}
    for name, scale in [('metre', 1.), ('millimetre', 1.e3),
                        ('micrometre', 1.e6), ('kilometre', 1.e-3)]:
        cook[name] = cook_solution(scale=scale)
        print('Cook 4 by 4 %-12s %.10f' % (name, cook[name]))
    vals = np.array(list(cook.values()))
    spread = (vals.max() - vals.min())/abs(vals.mean())
    print('relative spread over nine orders of magnitude: %.3e' % spread)
    # NOTE the only residual variation is the conditioning of the linear
    #      solve at the extreme scalings, not the formulation
    assert spread < 1e-8, spread


def test_homogeneous_stiffnesses_agree_with_brockman_for_small_areas():
    r"""The departure from Eq. (16) is confined to the regime that needed it

    `1/(1 + 1/A) \to A` as `A \to 0`, so for the in-plane translations the
    homogeneous form and Eq. (16a) converge as the element area shrinks in
    the unit used, which is the regime Brockman calibrated in. This test
    measures that convergence, so the change is documented as a
    reinterpretation of his factor rather than a different magnitude of
    stabilisation.

    The `w` term is excluded: it keeps Brockman's factor and is Eq. (16b)
    itself, see :func:`homogeneous_stiffnesses`.

    """
    for area, tol in [(1.e-1, 2.e-1), (1.e-2, 2.e-2), (1.e-4, 2.e-4)]:
        br = brockman_stiffnesses(area)
        hm = homogeneous_stiffnesses(area)
        for dof in (0, 1):
            rel = abs(hm[dof] - br[dof])/abs(br[dof])
            print('A = %-8g %-8s homogeneous %.6e  Eq. (16a) %.6e  %.2e'
                  % (area, HG_NAMES[dof], hm[dof], br[dof], rel))
            assert rel < tol, (area, dof, rel)
    # NOTE and they diverge without bound in a unit that makes the areas
    #      large, which is the unit dependence this change removes
    br = brockman_stiffnesses(1.e4)
    hm = homogeneous_stiffnesses(1.e4)
    assert hm[0]/br[0] > 1.e3


def strip_twist(hgfactor_physical=1., nx=6, ny=1):
    r"""Twist of the MacNeal and Harder (1985) 6 by 0.2 strip, reference 0.03208

        MacNeal, R. H., and Harder, R. L., 1985, "A proposed standard set of
        problems to test finite element accuracy," Finite Elements in
        Analysis and Design, 1(1), pp. 3-20.
        https://doi.org/10.1016/0168-874X(85)90003-4

    With Brockman's control of `w`, used up to 0.11, the twist of a
    one-element-wide strip is the hourglass pattern `w = xy`, and its answer
    was set by the coefficient `E^{(h)}_w`. Root fully clamped, a unit torque applied as a pair of
    opposite transverse forces on the tip edge, and the returned value is
    the tip twist per unit width.

    """
    a, b, h = 6., 0.2, 0.1
    E, nu = 1.e7, 0.3
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    xs = np.linspace(0., a, nx + 1)
    ys = np.linspace(0., b, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nid = np.arange((nx + 1)*(ny + 1)).reshape(nx + 1, ny + 1)
    els = [(nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
           for i in range(nx) for j in range(ny)]
    X = X.ravel()
    Y = Y.ravel()
    nn = len(X)
    ncoords_flatten = np.vstack((X, Y, np.zeros(nn))).T.flatten()
    data = Quad4RData()
    probe = Quad4RProbe()
    N = DOF*nn
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=DOUBLE)
    init_k_KC0 = 0
    for e in els:
        el = Quad4R(probe)
        for k in range(4):
            setattr(el, 'n%d' % (k + 1), e[k] + 1)
            setattr(el, 'c%d' % (k + 1), DOF*e[k])
        el.init_k_KC0 = init_k_KC0
        el.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop, hgfactor_physical=hgfactor_physical)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    root = np.isclose(X, 0.)
    for d in range(DOF):
        bk[d::DOF] |= root
    bu = ~bk
    tip = np.where(np.isclose(X, a))[0]
    tip = tip[np.argsort(Y[tip])]
    f = np.zeros(N)
    f[DOF*tip[0] + 2] = -1./b
    f[DOF*tip[-1] + 2] = +1./b
    u = np.zeros(N)
    u[bu] = spsolve(KC0[bu, :][:, bu], f[bu])
    return (u[DOF*tip[-1] + 2] - u[DOF*tip[0] + 2])/b


def test_one_element_wide_strip_in_torsion():
    r"""The straight cantilever beam of MacNeal and Harder (1985), Fig. 4,
    6 by 1 elements, under the twist load, a unit moment applied as the
    nodal forces of Fig. 11b, p. 12, whose tip twist is 0.03208, Table 3, p.
    10. Their results are graded on p. 14: A for an error of at most 2 %, B
    10 %, C 20 %, D 50 %, F above 50 %; their QUAD4 gives 0.941, Table 8,
    p. 15.

    Up to 0.11 Quad4R controlled the hourglass mode of `w` with Brockman's
    coefficient `E^{(h)}_w`, which also stiffened a uniform twist, see
    :func:`test_constant_twist_patch_test`: on a single element across the
    width the twist was the hourglass pattern `w = xy` and the 6x1 strip
    gave 4.78 times the reference, grade F. Since 0.12.0 the bending and
    the transverse shear have the physical stabilisation of Belytschko and
    Leviathan (1994), and the 6x1 strip gives 0.941 times the reference,
    grade B, as the QUAD4 of MacNeal and Harder. Tip twist over 0.03208,
    measured on 2026-10-29, the torque applied as the corner forces of this
    test (corner) or as normal forces proportional to the distance from the
    axis (distributed), with the default stabilisation
    (``hgfactor_physical = 1``) or 1/100 of it:

    ======  =========  ===========  ===========
    mesh    corner, 1  corner, .01  distrib., 1
    ======  =========  ===========  ===========
    6x1     0.9407     22.34        0.9407
    6x2     0.9407     1.0217       0.9407
    6x4     1.0136     1.0669       1.0135
    12x4    1.0185     1.0706       1.0182
    24x8    1.0507     1.0721       1.0499
    48x8    1.0532     1.0744       1.0518
    96x16   1.0649     1.0763       1.0622
    192x32  1.0700     1.0783       1.0655
    ======  =========  ===========  ===========

    the meshes from 24x8 on giving the values of :class:`pyfe3d.Quad4` with
    the corner forces to four digits, its thick elements having the same
    assumed shear strains. With the distributed torque the
    twist converges to about 1.067 times 0.03208, 0.0342, the value to which
    Quad4 and Tria3DSG converge too, 0.3 % above the Saint-Venant twist of
    the 0.2 x 0.1 rectangle, 0.03411, see
    ``test_torsion_of_a_one_element_wide_strip`` in
    ``test_quad4_spurious_shear_mode.py``; with the corner forces it keeps
    growing slowly, the deflection under concentrated forces being singular
    in a Mindlin plate. The 6.7 % between the converged plate and 0.03208 is
    a difference of the reference, whose derivation MacNeal and Harder do
    not give. With 1/100 of the stabilisation the 6x1 strip is again far too
    flexible: one element across the width cannot represent the twist
    without the nonconstant strains that the stabilisation resists.

    Following the policy of the tests whose reference is not reached on the
    mesh used, the 6x1 and 12x4 meshes are compared with the values
    pre-determined by :func:`strip_twist`, 0.03017887108568292 and
    0.032672815829645464, with ``rtol = 1e-6`` for the round-off of the
    linear solution, and with grade B against 0.03208 as a bound.

    """
    # NOTE MacNeal and Harder (1985), Table 3, p. 10
    ref = 0.03208
    one = strip_twist(hgfactor_physical=1.)
    refined = strip_twist(hgfactor_physical=1., nx=12, ny=4)
    low = strip_twist(hgfactor_physical=0.01)
    print('strip twist / reference: 6x1 %.4f, 12x4 %.4f, 6x1 with 1/100 of '
          'the stabilisation %.4g' % (one/ref, refined/ref, low/ref))
    # NOTE values pre-determined by strip_twist on 2026-10-29
    assert np.isclose(one, 0.03017887108568292, rtol=1e-6)
    assert np.isclose(refined, 0.032672815829645464, rtol=1e-6)
    # NOTE grade B of MacNeal and Harder, p. 14
    assert abs(one/ref - 1) <= 0.10
    assert abs(refined/ref - 1) <= 0.10
    # NOTE grade F without enough stabilisation
    assert low/ref - 1 > 0.5


def twist_patch_energy_ratio(n, scale=1., hgfactor_physical=1.):
    r"""Constant-twist patch test of an ``n x n`` mesh of Quad4R

    Square plate of side ``254*scale``, thickness ``6.35*scale``, ``E =
    3102.75/scale**2``, ``nu = 0.3``, the dimensions and material of the
    hinged roof of Sze et al. (2004), see
    ``tests/test_nonlinear_benchmarks_sze2004.py``, with ``scale = 1`` in
    the unit of length of that paper and ``scale = 1e-3`` in a unit 1000
    times larger, the force unit being the same. The boundary nodes carry
    the exact Mindlin solution of a uniform twist, ``w = k*x*y``, ``r_x =
    k*x``, ``r_y = -k*y`` (zero transverse shear with ``gamma_xz = w_x +
    r_y`` and ``gamma_yz = w_y - r_x``), ``u = v = 0``, and the interior
    nodes are free. Returns the strain energy over the exact one, ``2*D66*
    k**2*a**2``, ``D66 = E*h**3/(24*(1 + nu))``.
    """
    a, h = 254.*scale, 6.35*scale
    E, nu = 3102.75/scale**2, 0.3
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    xs = np.linspace(-a/2, a/2, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    nid = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    els = [(nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
           for i in range(n) for j in range(n)]
    X = X.ravel()
    Y = Y.ravel()
    ncoords_flatten = np.vstack((X, Y, 0*X)).T.flatten()
    data = Quad4RData()
    probe = Quad4RProbe()
    N = DOF*X.size
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=DOUBLE)
    for k, e in enumerate(els):
        el = Quad4R(probe)
        for i in range(4):
            setattr(el, 'n%d' % (i + 1), e[i])
            setattr(el, 'c%d' % (i + 1), DOF*e[i])
        el.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        el.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop, hgfactor_physical=hgfactor_physical)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    kt = 1.e-3/scale
    u = np.zeros(N)
    u[2::DOF] = kt*X*Y
    u[3::DOF] = kt*X
    u[4::DOF] = -kt*Y
    edge = np.isclose(abs(X), a/2) | np.isclose(abs(Y), a/2)
    bk = np.zeros(N, dtype=bool)
    for d in range(DOF):
        bk[d::DOF] = edge
    # NOTE drilling rotations restrained, they do not take part in the twist
    bk[5::DOF] = True
    bu = ~bk
    u[bu] = 0.
    u[bu] = spsolve(KC0[bu, :][:, bu], -KC0[bu, :][:, bk] @ u[bk])
    D66 = E*h**3/(24*(1 + nu))
    return 0.5*u @ (KC0 @ u)/(0.5*D66*(2*kt)**2*a**2)


def bl94_shear_reference(x, y, Cs):
    r"""Transverse shear of Belytschko and Leviathan (1994), written here
    independently of the element code, in element coordinates

    Returns ``KH`` and ``Kc``, 12 x 12 on ``w, r_x, r_y`` of nodes 1 to 4:
    ``KH`` the stabilisation of the nonconstant part of the assumed shear
    strains, their Eq. (4.13), and ``Kc`` the constant part of the assumed
    strains minus the one-point displacement-based shear, their Eq. (4.12).
    The covariant strains of Dvorkin and Bathe (1984), ``e_xi`` at the
    tying points ``(0, -1)``, ``(0, 1)`` and ``e_eta`` at ``(-1, 0)``,
    ``(1, 0)``, are obtained from ``gamma_xz = w_x + r_y`` and ``gamma_yz =
    w_y - r_x`` evaluated there with the Jacobian of that point, and mapped
    to Cartesian strains with the Jacobian at the centroid; the nonconstant
    parts are linear in ``eta`` (``gamma_xz``) and ``xi`` (``gamma_yz``),
    their Eqs. (3.38)-(3.39), and integrate with ``int eta**2 dA = int
    xi**2 dA = A/3``, Eq. (4.15).

    """
    XI = np.array([-1., 1., 1., -1.])
    ETA = np.array([-1., -1., 1., 1.])

    def covariant(xi, eta):
        N = 0.25*(1 + XI*xi)*(1 + ETA*eta)
        Nxi = 0.25*XI*(1 + ETA*eta)
        Neta = 0.25*ETA*(1 + XI*xi)
        J = np.array([[Nxi @ x, Nxi @ y], [Neta @ x, Neta @ y]])
        dN = np.linalg.solve(J, np.vstack((Nxi, Neta)))
        G = np.zeros((2, 12))
        G[0, 0::3] = dN[0]
        G[0, 2::3] = N
        G[1, 0::3] = dN[1]
        G[1, 1::3] = -N
        return J @ G, J

    eB, _ = covariant(0., -1.)
    eD, _ = covariant(0., 1.)
    eA, _ = covariant(-1., 0.)
    eC, _ = covariant(1., 0.)
    e0, J0 = covariant(0., 0.)
    A = 4*np.linalg.det(J0)
    J0inv = np.linalg.inv(J0)
    g0m = J0inv @ np.vstack((0.5*(eB[0] + eD[0]), 0.5*(eA[1] + eC[1])))
    g0d = J0inv @ e0
    D1 = J0inv @ np.vstack((0.5*(eD[0] - eB[0]), np.zeros(12)))
    D2 = J0inv @ np.vstack((np.zeros(12), 0.5*(eC[1] - eA[1])))
    KH = A/3*(D1.T @ Cs @ D1 + D2.T @ Cs @ D2)
    Kc = A*(g0m.T @ Cs @ g0m - g0d.T @ Cs @ g0d)
    return KH, Kc


def bl94_bending_reference(x, y, D):
    r"""Physical stabilisation of the bending of Belytschko and Leviathan
    (1994), written here independently of the element code, in element
    coordinates

    Returns the 12 x 12 matrix on ``w, r_x, r_y`` of nodes 1 to 4 of the
    nonconstant part of the bilinear curvatures, `\boldsymbol{\kappa}(\xi,
    \eta) - \boldsymbol{\kappa}(0, 0)`, with `\kappa_{xx} = r_{y,x}`,
    `\kappa_{yy} = -r_{x,y}` and `\kappa_{xy} = r_{y,y} - r_{x,x}`,
    integrated with two by two Gauss points with the bending stiffness
    ``D``: the derivatives of the shape functions are evaluated directly,
    not through the decomposition `N_{I,x} = b_{xI} + h_{,x} \gamma_I` of
    their Eq. (3.25) that the element uses, which gives the same field.

    """
    XI = np.array([-1., 1., 1., -1.])
    ETA = np.array([-1., -1., 1., 1.])

    def Bk(xi, eta):
        Nxi = 0.25*XI*(1 + ETA*eta)
        Neta = 0.25*ETA*(1 + XI*xi)
        J = np.array([[Nxi @ x, Nxi @ y], [Neta @ x, Neta @ y]])
        dN = np.linalg.solve(J, np.vstack((Nxi, Neta)))
        B = np.zeros((3, 12))
        B[0, 2::3] = dN[0]
        B[1, 1::3] = -dN[1]
        B[2, 2::3] = dN[1]
        B[2, 1::3] = -dN[0]
        return B, np.linalg.det(J)

    B0, _ = Bk(0., 0.)
    K = np.zeros((12, 12))
    g = 1/np.sqrt(3)
    for xi in (-g, g):
        for eta in (-g, g):
            B, detJ = Bk(xi, eta)
            K += detJ*(B - B0).T @ D @ (B - B0)
    return K


def test_bending_stabilisation_matches_belytschko_leviathan():
    r"""The stabilisation of the rotations of Quad4R against the closed
    form and :func:`bl94_bending_reference`

    Belytschko, T., and Leviathan, I., 1994, "Physical stabilization of the
    4-node shell element with one point quadrature", Comput. Methods Appl.
    Mech. Engrg., 113(3-4), pp. 321-350,
    https://doi.org/10.1016/0045-7825(94)90052-3: the nonconstant part of
    the curvatures, Eq. (3.28), p. 327, gives the stabilisation of Eq.
    (4.13), p. 331, and of Eqs. (B.17) and (B.18), p. 349, with the
    integrals `H_{xx}`, `H_{yy}` and `H_{xy}` of Eq. (4.14), here with the
    bending stiffness `\boldsymbol{D} - \boldsymbol{B} \boldsymbol{A}^{-1}
    \boldsymbol{B}` of the laminate in the element coordinate system, which
    is `\boldsymbol{D}` for a symmetric laminate and does not depend on the
    offset. On a rectangle of half-sides `a` and `b` the patterns `r_x = h`
    and `r_y = h`, `h = \xi\eta`, have the energies `A (D_{66}/a^2 +
    D_{22}/b^2)/6` and `A (D_{11}/a^2 + D_{66}/b^2)/6`, all of them from the
    stabilisation, checked for the plates of :func:`laminates` with the
    material direction along the element `x` axis and at 30 degrees; on
    four shapes, the stabilisation of the bending and of the transverse
    shear, the difference between ``hgfactor_physical = 1`` and ``0``, is
    the sum of :func:`bl94_bending_reference` and of the matrix ``KH`` of
    :func:`bl94_shear_reference`, for random displacements, see
    :func:`test_transverse_shear_matches_belytschko_leviathan`.

    """
    a, b = 1., 0.5
    for name, prop in laminates().items():
        for xmat in [(0., 0., 1.), (np.cos(np.pi/6), np.sin(np.pi/6), 0.)]:
            el, probe = make_element(RECTANGLE, xmat=xmat)
            thetadeg = np.rad2deg(np.arctan2(el.m21, el.m11))
            A, B, D, _ = prop.calc_constitutive_element(thetadeg)
            Ds = D - B @ np.linalg.inv(A) @ B
            K1 = element_KC0(el, prop, hgfactor_physical=1.)
            K0 = element_KC0(el, prop, hgfactor_physical=0.)
            for dof, ref in ((3, el.area*(Ds[2, 2]/a**2 + Ds[1, 1]/b**2)/6),
                             (4, el.area*(Ds[0, 0]/a**2 + Ds[2, 2]/b**2)/6)):
                ue = hourglass_mode(el, dof)
                U1 = 0.5*ue @ K1 @ ue
                U0 = 0.5*ue @ K0 @ ue
                print('%-17s theta=%4.1f %-7s %.6e %.6e' % (name, thetadeg,
                      HG_NAMES[dof], U1, ref))
                assert abs(U0) < 1e-12*np.abs(K0).max()*(ue @ ue), (name,
                                                                 thetadeg)
                assert np.isclose(U1, ref, rtol=1e-10), (name, thetadeg, dof)


def test_transverse_shear_matches_belytschko_leviathan():
    r"""The transverse shear of Quad4R against :func:`bl94_shear_reference`

    Belytschko, T., and Leviathan, I., 1994, "Physical stabilization of the
    4-node shell element with one point quadrature", Comput. Methods Appl.
    Mech. Engrg., 113(3-4), pp. 321-350,
    https://doi.org/10.1016/0045-7825(94)90052-3: the assumed transverse
    shear strains of Dvorkin and Bathe (1984) split into a constant and a
    nonconstant part, Eqs. (3.35)-(3.42), pp. 328-329; the one-point
    stiffness of the constant part, Eq. (4.12), p. 330, and the
    stabilisation of the nonconstant part, Eq. (4.13), p. 331, with no
    parameter, ``hgfactor_physical = 1`` in pyfe3d. The energy of the
    stabilisation, the difference between ``hgfactor_physical = 1`` and
    ``0``, which also contains that of the bending, see
    :func:`test_bending_stabilisation_matches_belytschko_leviathan`, and its
    linearity in ``hgfactor_physical`` are checked on four shapes with
    random displacements, and on the rectangle of sides `2a = 2` and `2b =
    1` the
    hourglass pattern `w = \xi\eta`, whose assumed shear strains are
    `\gamma_{xz} = \eta/a` and `\gamma_{yz} = \xi/b`, has the energy
    `A (A_{55}/a^2 + A_{44}/b^2)/6`, all of it from the stabilisation.

    """
    prop = isotropic_plate(thickness=T_BROCKMAN, E=E_BROCKMAN,
                           nu=NU_BROCKMAN)
    Cs = np.array([[prop.A55, prop.A45], [prop.A45, prop.A44]])
    D = prop.calc_constitutive_element(0.)[2]
    rng = np.random.default_rng(1)
    for name, ncoords in [('rectangle', RECTANGLE),
                          ('parallelogram', PARALLELOGRAM),
                          ('distorted', DISTORTED),
                          ('trapezoid', TRAPEZOID)]:
        el, probe = make_element(ncoords)
        x, y = element_xy(probe)
        KH, _ = bl94_shear_reference(x, y, Cs)
        KH = KH + bl94_bending_reference(x, y, D)
        K0 = element_KC0(el, prop, hgfactor_physical=0.)
        K1 = element_KC0(el, prop, hgfactor_physical=1.)
        K3 = element_KC0(el, prop, hgfactor_physical=3.)
        for _ in range(4):
            q = rng.standard_normal(12)
            ue = np.zeros(24)
            for i in range(4):
                ue[DOF*i + 2:DOF*i + 5] = q[3*i:3*i + 3]
            ue = to_global(el, ue)
            dU = 0.5*ue @ (K1 - K0) @ ue
            dU3 = 0.5*ue @ (K3 - K0) @ ue
            ref = 0.5*q @ KH @ q
            print(name, dU, ref)
            assert np.isclose(dU, ref, rtol=1e-10), (name, dU, ref)
            assert np.isclose(dU3, 3*ref, rtol=1e-10), (name, dU3, ref)
    el, probe = make_element(RECTANGLE)
    a, b = 1., 0.5
    ue = hourglass_mode(el, 2)
    U0 = 0.5*ue @ element_KC0(el, prop, hgfactor_physical=0.) @ ue
    U1 = 0.5*ue @ element_KC0(el, prop, hgfactor_physical=1.) @ ue
    U_ref = el.area*(prop.A55/a**2 + prop.A44/b**2)/6
    print('w = xi*eta on the rectangle:', U0, U1, U_ref)
    assert abs(U0) < 1e-12*U1
    assert np.isclose(U1, U_ref, rtol=1e-12)


def test_constant_twist_patch_test():
    r"""A uniform twist, `w = kxy`, `r_x = kx`, `r_y = -ky`, has no
    transverse shear, and Quad4R reproduces its energy, `2 D_{66} k^2`
    per unit area, exactly, for any mesh, length unit and ``hgfactor_physical``

    The assumed shear strains of Belytschko and Leviathan (1994), see
    :func:`test_transverse_shear_matches_belytschko_leviathan`, vanish for
    this state, constant and nonconstant parts, so their stabilisation adds
    no energy. Up to 0.11 the hourglass mode of `w` was controlled with
    Brockman's operator `\gamma^\top w = w_{,xy}` alone, which is `k` for
    this state: it added the fraction `0.6 (1 + \nu)/(1 + 1/A)` to the
    energy, 1.7798 times the exact one with 4x4 elements and 1.7334 with
    64x64 in the unit of length of the roof of Sze et al. (2004), but only
    1.0031 and 1.00001 in a unit 1000 times larger, a failure of the patch
    test that refinement did not remove where the element areas were large
    in the unit used. See :func:`twist_patch_energy_ratio`.

    """
    for scale in (1., 1.e-3):
        for hg in (1., 0.1, 0.):
            for n in (4, 16):
                ratio = twist_patch_energy_ratio(n, scale, hg)
                print('scale %g, hgfactor_physical %g, %dx%d: U/U_exact = %.12f'
                      % (scale, hg, n, n, ratio))
                assert abs(ratio - 1) < 1e-9, (scale, hg, n, ratio)


# NOTE Belytschko and Leviathan (1994), Fig. 3 and Table 2, p. 336: nodes 1
#      to 4 on the boundary, 5 to 8 inside
BL94_PATCH_XY = np.array([[0., 10.], [0., 0.], [10., 0.], [10., 10.],
                          [2., 2.], [8., 3.], [8., 7.], [4., 7.]])
BL94_PATCH_CONN = [(1, 2, 5, 4), (2, 3, 6, 5), (3, 0, 7, 6), (0, 1, 4, 7),
                   (4, 5, 6, 7)]


def test_bending_patch_test_belytschko_leviathan():
    r"""Thin-plate bending patch test of Belytschko and Leviathan (1994),
    Section 6.1, Fig. 3 and Table 2, p. 336, Table 3, p. 337

    The quadratic field of their Eq. (6.1), `w = 0.001 x^2 + 0.002 xy +
    0.0015 y^2 + 0.025 x + 0.03 y + 0.09`, with the Kirchhoff rotations of
    Eqs. (6.2) and (6.3), `\theta_x = w_{,y}` and `\theta_y = -w_{,x}`,
    which are `r_x` and `r_y` of pyfe3d, prescribed at the boundary nodes 1
    to 4 of the irregular mesh of Fig. 3; the interior nodes 5 to 8 must
    take the exact values, those of Table 2. The paper gives no material:
    this constant-curvature state has no transverse shear, so any thickness
    must reproduce it, and ``h = 0.01`` and ``h = 1`` are used, with ``E =
    1e6`` and ``nu = 0.25``; the in-plane displacements and the drilling
    rotations are held at zero. Table 3 lists the normalised `w` of nodes
    5 to 8: 0.92 to 1.15 for a selectively reduced element and 1.06 to 1.18
    for the YASE element, both with a constant shear strain that is not
    zero under constant curvature on a non-rectangular element, and 1.00
    for MITC4 and for their element. Quad4R, whose constant transverse shear
    is that of Dvorkin and Bathe since 0.12.0, passes for any
    ``hgfactor_physical``; up to 0.11, with the displacement-based shear at the
    centroid, it gave 0.875, 1.21, 0.98 and 1.20 times the exact `w` of
    nodes 5 to 8 with ``h = 0.01`` (measured on 2026-10-29 with
    ``hgfactor_physical = 0``).

    """
    def w(X, Y):
        return (0.001*X**2 + 0.002*X*Y + 0.0015*Y**2 + 0.025*X + 0.03*Y
                + 0.09)

    def rx(X, Y):
        return 0.002*X + 0.003*Y + 0.03

    def ry(X, Y):
        return -(0.002*X + 0.002*Y + 0.025)

    X, Y = BL94_PATCH_XY[:, 0], BL94_PATCH_XY[:, 1]
    # NOTE Table 2 of Belytschko and Leviathan (1994)
    assert np.allclose(w(X, Y), [0.54, 0.09, 0.44, 1.09, 0.2180, 0.5055,
                                 0.7495, 0.5455], atol=1e-12)
    ncoords_flatten = np.vstack((X, Y, 0*X)).T.flatten()
    N = DOF*8
    data = Quad4RData()
    probe = Quad4RProbe()
    for h in (0.01, 1.):
        prop = isotropic_plate(thickness=h, E=1.e6, nu=0.25)
        for hg in (1., 0.):
            KC0r = np.zeros(data.KC0_SPARSE_SIZE*5, dtype=INT)
            KC0c = np.zeros(data.KC0_SPARSE_SIZE*5, dtype=INT)
            KC0v = np.zeros(data.KC0_SPARSE_SIZE*5, dtype=DOUBLE)
            for k, e in enumerate(BL94_PATCH_CONN):
                el = Quad4R(probe)
                for i in range(4):
                    setattr(el, 'n%d' % (i + 1), e[i])
                    setattr(el, 'c%d' % (i + 1), DOF*e[i])
                el.init_k_KC0 = k*data.KC0_SPARSE_SIZE
                el.update_rotation_matrix(ncoords_flatten)
                el.update_probe_xe(ncoords_flatten)
                el.update_area()
                el.update_KC0(KC0r, KC0c, KC0v, prop, hgfactor_physical=hg)
            KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
            exact = np.zeros(N)
            exact[2::DOF] = w(X, Y)
            exact[3::DOF] = rx(X, Y)
            exact[4::DOF] = ry(X, Y)
            bk = np.zeros(N, dtype=bool)
            bk[:DOF*4] = True
            bk[0::DOF] = bk[1::DOF] = bk[5::DOF] = True
            bu = ~bk
            u = exact.copy()
            u[bu] = spsolve(KC0[bu, :][:, bu], -KC0[bu, :][:, bk] @ exact[bk])
            ratio = u[DOF*4 + 2::DOF]/exact[DOF*4 + 2::DOF]
            print('h = %g, hgfactor_physical = %g: w/w_exact of nodes 5-8' % (h, hg),
                  ratio)
            assert np.allclose(u, exact, rtol=1e-9, atol=1e-12), (h, hg)


# NOTE input properties of a carbon-epoxy ply and of a sandwich core, used
#      only to build laminates whose bending and membrane moduli differ
LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 2.9e9)
CORE = (100.e6, 100.e6, 0.3, 40.e6, 40.e6, 40.e6)


def laminates():
    r"""Plates whose equivalent membrane and bending moduli differ

    The isotropic plate with an offset reference surface is included because
    its `\boldsymbol{D}` is not `E t^3/(12(1 - \nu^2))` there, while its bending
    moduli must still be `E`.

    """
    t = 1.e-3
    tf = 0.125e-3
    return {
        'isotropic, offset': isotropic_plate(thickness=T_BROCKMAN,
            E=E_BROCKMAN, nu=NU_BROCKMAN, offset=0.4*T_BROCKMAN),
        '[0/90]s': laminated_plate([0, 90, 90, 0], plyt=t/4,
                                   laminaprop=LAMINAPROP),
        '[45/-45]s': laminated_plate([45, -45, -45, 45], plyt=t/4,
                                     laminaprop=LAMINAPROP),
        '[0/90]': laminated_plate([0, 90], plyt=t/2, laminaprop=LAMINAPROP),
        '[45/-45], offset': laminated_plate([45, -45], plyt=t/2,
            laminaprop=LAMINAPROP, offset=0.3*t),
        'sandwich': laminated_plate([0, 45, -45, 90, 0, 90, -45, 45, 0],
            plyts=[tf]*4 + [10.e-3] + [tf]*4,
            laminaprops=[LAMINAPROP]*4 + [CORE] + [LAMINAPROP]*4),
    }


def equivalent_moduli(prop, thetadeg):
    r"""`E_{1eq}`, `E_{2eq}`, `E_{1b}`, `E_{2b}` in the element coordinate
    system, see the Notes of :meth:`pyfe3d.Quad4R.update_KC0`

    """
    A, B, D, _ = prop.calc_constitutive_element(thetadeg)
    h = prop.h
    a = np.linalg.inv(A)
    d = np.linalg.inv(np.block([[A, B], [B, D]]))[3:, 3:]
    return (1./(h*a[0, 0]), 1./(h*a[1, 1]), 12./(h**3*d[0, 0]),
            12./(h**3*d[1, 1]))


def test_laminate_hourglass_stiffnesses_use_membrane_and_bending_moduli():
    r"""`u`, `v` with the membrane moduli, in the element coordinate
    system; `w`, `r_x` and `r_y` have the physical stabilisation of
    Belytschko and Leviathan (1994) since 0.12.0, see
    :func:`test_bending_stabilisation_matches_belytschko_leviathan`

    The generalized stiffness of each hourglass pattern is recovered from
    its energy, `E^{(h)} = 2 U/(\tilde\gamma^2 A)`, see
    :func:`test_hourglass_operator_matches_brockman_eq15`, and compared with
    the expressions of the Notes of :meth:`pyfe3d.Quad4R.update_KC0`, for
    the material direction along the element `x` axis and at 30 degrees
    from it. The isotropic plate with an offset reference surface gives
    Brockman's Eq. (16) with `E`, as without offset.

    """
    for name, prop in laminates().items():
        for xmat in [(0., 0., 1.), (np.cos(np.pi/6), np.sin(np.pi/6), 0.)]:
            el, probe = make_element(RECTANGLE, xmat=xmat)
            thetadeg = np.rad2deg(np.arctan2(el.m21, el.m11))
            E1eq, E2eq, E1b, E2b = equivalent_moduli(prop, thetadeg)
            t, area = prop.h, el.area
            ref = {0: 0.1*E1eq*t*area, 1: 0.1*E2eq*t*area}
            if name.startswith('isotropic'):
                assert np.allclose([E1eq, E2eq, E1b, E2b], E_BROCKMAN,
                                   rtol=1e-12)
            KC0 = element_KC0(el, prop)
            gt = gamma_tilde(el, probe)
            for dof in (0, 1):
                ue = hourglass_mode(el, dof)
                Eh = ue.dot(KC0.dot(ue))/(gt**2*area)
                print('%-17s theta=%4.1f %-7s %.6e %.6e' % (name, thetadeg,
                      HG_NAMES[dof], Eh, ref[dof]))
                assert np.isclose(Eh, ref[dof], rtol=1e-10), (name, thetadeg,
                                                               HG_NAMES[dof])


def test_bending_moduli_differ_from_membrane_moduli_for_laminates():
    r"""The laminates of :func:`laminates` exercise the bending moduli

    `E_{1b}/E_{1eq}` and `E_{2b}/E_{2eq}` are 1 for the isotropic plate,
    with or without offset, and differ from 1 for the laminates: about 1.7
    and 0.34 for the symmetric cross-ply, 0.41 for the unsymmetric
    cross-ply, through the coupling `\boldsymbol{B}`, and above 2.5 for the sandwich,
    whose bending stiffness comes from the faces.

    """
    ratios = {}
    for name, prop in laminates().items():
        E1eq, E2eq, E1b, E2b = equivalent_moduli(prop, 0.)
        ratios[name] = (E1b/E1eq, E2b/E2eq)
        print('%-17s E1b/E1eq %.4f E2b/E2eq %.4f' % (name, *ratios[name]))
    assert np.allclose(ratios['isotropic, offset'], 1., rtol=1e-12)
    assert ratios['[0/90]s'][0] > 1.5 and ratios['[0/90]s'][1] < 0.4
    assert ratios['[0/90]'][0] < 0.5 and ratios['[0/90]'][1] < 0.5
    assert min(ratios['sandwich']) > 2.5


def test_laminate_zero_energy_modes():
    r"""Brockman's p. 2347 mode count holds for the laminates

    As in :func:`test_stabilisation_restores_the_rank_of_a_rectangular_element`,
    for each plate of :func:`laminates` on a rectangle and on a distorted
    element: without stabilisation the element matrix has 12 zero
    eigenvalues, the six rigid-body motions, the five hourglass patterns and
    the twisting mode; with it, 6, the twisting mode being stiffened by the
    stabilisation of the transverse shear, see
    :func:`test_twisting_mode_is_stabilised`. The membrane and bending stiffnesses of these thin
    laminates differ by many orders of magnitude, so the matrix is scaled
    symmetrically by its diagonal before the eigenvalues are counted, which
    does not change the number of zero eigenvalues, and the smallest
    non-zero eigenvalue is then well separated from the zero ones.

    """
    for name, prop in laminates().items():
        for shape, ncoords in [('rectangle', RECTANGLE),
                               ('distorted', DISTORTED)]:
            el, probe = make_element(ncoords, K6ROT=100.)
            counts = {}
            for label, hgfactor in [('off', 0.), ('on', 1.)]:
                K = element_KC0(el, prop, hgfactor=hgfactor)
                scale = 1./np.sqrt(np.diag(K))
                w = np.linalg.eigvalsh(scale[:, None]*K*scale[None, :])
                tol = 1e-11*abs(w).max()
                counts[label] = int((w < tol).sum())
                print(name, shape, label, counts[label],
                      w[counts[label]]/abs(w).max())
                assert w[counts[label]] > 1e3*tol, (name, shape, label)
            assert counts == {'off': 12, 'on': 6}, (name, shape, counts)


def navier_ss1_center_deflection(prop, a, nterms=199):
    r"""Centre deflection of a simply supported square plate under a unit
    uniform pressure, FSDT Navier solution with SS-1 conditions

    For a laminate with `A_{16} = A_{26} = B_{16} = B_{26} = D_{16} = D_{26}
    = A_{45} = 0`, e.g. a cross-ply, symmetric or not, the fields

    .. math::
        u = U c_x s_y, \quad v = V s_x c_y, \quad w = W s_x s_y, \quad
        r_y = X c_x s_y, \quad -r_x = Y s_x c_y

    with `s_x = \sin(\alpha x)`, `c_x = \cos(\alpha x)`, `\alpha = m
    \pi/a`, and analogously in `y` with `\beta = n \pi/a`, satisfy `v = w =
    r_x = 0` on `x = 0, a` and `u = w = r_y = 0` on `y = 0, a`. Each strain
    of the Notes of :mod:`pyfe3d.quad4r` is then a single product of sines
    and cosines, so the strain energy of a term `(m, n)` is `a^2/8` times a
    quadratic form `Q` of `(U, V, W, X, Y)` with constant coefficients, and
    the stationarity of `a^2/8 \, Q - q_{mn} W a^2/4` gives one 5 by 5 linear
    system per term, with `q_{mn} = 16/(\pi^2 m n)` for odd `m` and `n`.
    The transverse shear stiffnesses are those of ``prop``, so the
    comparison isolates the discretisation.

    """
    ABD = prop.ABD
    for i, j in [(0, 2), (1, 2), (0, 5), (1, 5), (3, 5), (4, 5)]:
        assert abs(ABD[i, j]) < 1e-12*abs(ABD).max()
    assert abs(prop.A45) < 1e-12*max(prop.A44, prop.A55)
    A11, A12, A22, A66 = ABD[0, 0], ABD[0, 1], ABD[1, 1], ABD[2, 2]
    B11, B12, B22, B66 = ABD[0, 3], ABD[0, 4], ABD[1, 4], ABD[2, 5]
    D11, D12, D22, D66 = ABD[3, 3], ABD[3, 4], ABD[4, 4], ABD[5, 5]
    A44, A55 = prop.A44, prop.A55
    m, n = np.meshgrid(np.arange(1, nterms + 1, 2), np.arange(1, nterms + 1, 2))
    al, be = m.ravel()*np.pi/a, n.ravel()*np.pi/a
    # NOTE the Hessian of Q/2 with respect to (U, V, W, X, Y)
    K = np.zeros((al.size, 5, 5))
    K[:, 0, 0] = A11*al**2 + A66*be**2
    K[:, 0, 1] = (A12 + A66)*al*be
    K[:, 0, 3] = B11*al**2 + B66*be**2
    K[:, 0, 4] = (B12 + B66)*al*be
    K[:, 1, 1] = A22*be**2 + A66*al**2
    K[:, 1, 3] = (B12 + B66)*al*be
    K[:, 1, 4] = B22*be**2 + B66*al**2
    K[:, 2, 2] = A55*al**2 + A44*be**2
    K[:, 2, 3] = A55*al
    K[:, 2, 4] = A44*be
    K[:, 3, 3] = D11*al**2 + D66*be**2 + A55
    K[:, 3, 4] = (D12 + D66)*al*be
    K[:, 4, 4] = D22*be**2 + D66*al**2 + A44
    K = K + np.triu(K, 1).transpose(0, 2, 1)
    f = np.zeros((al.size, 5, 1))
    f[:, 2, 0] = 16./(np.pi**2*m.ravel()*n.ravel())
    W = np.linalg.solve(K, f)[:, 2, 0]
    return np.sum(W*np.sin(m.ravel()*np.pi/2)*np.sin(n.ravel()*np.pi/2))


def ss1_plate_center_deflection(prop, a, n):
    r"""Quad4R counterpart of :func:`navier_ss1_center_deflection`, on an
    ``n`` by ``n`` mesh, with the consistent nodal loads of the bilinear
    element

    """
    data = Quad4RData()
    probe = Quad4RProbe()
    xs = np.linspace(0., a, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    x, y = X.flatten(), Y.flatten()
    ncoords_flatten = np.vstack((x, y, 0*x)).T.flatten()
    nid = np.arange(len(x)).reshape(n + 1, n + 1)
    els = list(zip(nid[:-1, :-1].flatten(), nid[1:, :-1].flatten(),
                   nid[1:, 1:].flatten(), nid[:-1, 1:].flatten()))
    N = DOF*len(x)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*len(els), dtype=DOUBLE)
    for i, e in enumerate(els):
        el = Quad4R(probe)
        el.n1, el.n2, el.n3, el.n4 = e
        el.c1, el.c2, el.c3, el.c4 = [DOF*k for k in e]
        el.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        el.update_rotation_matrix(ncoords_flatten)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    xedge = np.isclose(x, 0.) | np.isclose(x, a)
    yedge = np.isclose(y, 0.) | np.isclose(y, a)
    bk[0::DOF] |= yedge
    bk[1::DOF] |= xedge
    bk[2::DOF] |= xedge | yedge
    bk[3::DOF] |= xedge
    bk[4::DOF] |= yedge
    bu = ~bk
    weights = np.ones(n + 1)
    weights[0] = weights[-1] = 0.5
    f = np.zeros(N)
    f[2::DOF] = (np.outer(weights, weights)*(a/n)**2).flatten()
    u = np.zeros(N)
    u[bu] = spsolve(KC0[bu, :][:, bu], f[bu])
    return u[2::DOF].reshape(n + 1, n + 1)[n//2, n//2]


def test_laminated_plates_against_navier():
    r"""Convergence of Quad4R to the FSDT Navier solution, and the effect of
    the bending moduli in the hourglass stiffnesses

    Simply supported square plates under uniform pressure, 8 by 8, 16 by 16
    and 32 by 32 Quad4R, against :func:`navier_ss1_center_deflection`. The
    reference is the Navier solution of the first-order shear deformation
    theory of Reddy, J. N., Mechanics of Laminated Composite Plates and
    Shells: Theory and Analysis, 2nd ed., CRC Press, 2004,
    https://doi.org/10.1201/b12409, Sec. 7.2.1, pp. 379-380: the SS-1
    conditions, Eq. (7.2.1a) and Fig. 7.2.1, the expansions, Eqs. (7.2.2) to
    (7.2.4), and the load, Eq. (7.2.6a), which for a uniform pressure gives
    ``16/(pi**2*m*n)``, and the condition ``A16 = A26 = B16 = B26 = D16 =
    D26 = A45 = 0``, p. 380, under which the coefficients follow from the 5
    by 5 system of Eqs. (7.2.7a,b), Eq. (7.2.8) in statics, p. 381. These
    are the conditions and the coefficients of
    :func:`navier_ss1_center_deflection`, with `\phi_x = r_y` and `\phi_y =
    -r_x`. The same conditions are stated as Eqs. (14) and (15), pp. 9-10,
    of Part 1 of Reddy, J. N., Chandrashekhara, K., and Chao, W. C.,
    "Geometrically nonlinear analysis of laminated elastic structures", NASA
    CR-191055, 1993. The series changes by less than 1e-8 from 199 to 399
    terms in each direction, and with the transverse shear stiffnesses of
    Reddy, ``K = 5/6`` times the thickness-weighted sum of `G_{13}` and
    `G_{23}`, it reproduces the deflections of the (0/90/90/0) square plate
    under uniform load of his Table 7.2.1, p. 385, which uses the ply of
    this test, ``100*w*E_2*h**3/(a**4*q_0)`` = 1.0250, 0.7694 and 0.6833 for
    `a/h` = 10, 20 and 100, within half a unit of the last printed digit.
    The plates below keep the transverse shear stiffnesses of
    :func:`pyfe3d.shellprop_utils.laminated_plate`, so their reference
    differs from Reddy's, by 0.21 % for the [0/90]s plate with `a/h = 100`.

    The ply has `E_1/E_2 = 25`, `G_{12} = G_{13} = 0.5 E_2`, `G_{23} = 0.2
    E_2`, `
u_{12} = 0.25`, and the sandwich has two `[0/90]` faces of 0.5
    mm on a 10 mm core of 100 MPa and 40 MPa.

    Quad4R is stiffer than the reference, so its deflection is smaller, and
    the error decreases as `h^2`, the rate of the bilinear element.
    Measured `w/w_{ref}` on 2026-10-29, with the physical stabilisation of
    the bending and of the transverse shear of Belytschko and Leviathan
    (1994) used since 0.12.0, and, up to 0.11, with Brockman's hourglass
    stiffnesses of the rotations with the bending moduli:

    ==========  ===============  ===============  ===============
    plate       8x8              16x16            32x32
    ==========  ===============  ===============  ===============
    isotropic   0.9949 (0.9210)  0.9987 (0.9794)  0.9997 (0.9948)
    [0/90]s     0.9924 (0.8521)  0.9981 (0.9590)  0.9995 (0.9895)
    [0/90]      0.9687 (0.8465)  0.9921 (0.9572)  0.9980 (0.9890)
    sandwich    0.9913 (0.8612)  0.9979 (0.9617)  0.9995 (0.9902)
    ==========  ===============  ===============  ===============

    The observed order of convergence from 16x16 to 32x32 is 2.00 to 2.01,
    and the Richardson extrapolation of these two meshes with the order 2
    agrees with the Navier solution within 1e-5, so the meshes converge to
    the reference and not to another limit. Up to 0.11 the artificial
    stiffness of the rotations, 0.1 times the bending modulus, made the 8x8
    meshes 8 to 15 % too stiff; the physical stabilisation, the
    nonconstant part of the curvatures integrated with the bending
    stiffness of the laminate, leaves 0.5 to 3 %.

    """
    E2 = 1.e9
    laminaprop = (25*E2, E2, 0.25, 0.5*E2, 0.5*E2, 0.2*E2)
    core = (100.e6, 100.e6, 0.3, 40.e6, 40.e6, 40.e6)
    tf = 0.25e-3
    plates = {
        'isotropic': (isotropic_plate(thickness=0.01, E=70.e9, nu=0.3), 1.),
        '[0/90]s': (laminated_plate([0, 90, 90, 0], plyt=0.01/4,
                                    laminaprop=laminaprop), 1.),
        '[0/90]': (laminated_plate([0, 90], plyt=0.01/2,
                                   laminaprop=laminaprop), 1.),
        'sandwich': (laminated_plate([0, 90, 0, 90, 0],
            plyts=[tf, tf, 10.e-3, tf, tf],
            laminaprops=[laminaprop, laminaprop, core, laminaprop,
                         laminaprop]), 0.5),
    }
    # NOTE Reddy (2004), Table 7.2.1, (0/90/90/0), uniform load
    for a_h, wbar in ((10, 1.0250), (20, 0.7694), (100, 0.6833)):
        h = 1./a_h
        prop = laminated_plate([0, 90, 90, 0], plyt=h/4,
                               laminaprop=laminaprop)
        prop.A44 = prop.A55 = 5/6*h*(laminaprop[4] + laminaprop[5])/2
        w = 100*navier_ss1_center_deflection(prop, 1.)*E2*h**3
        print('Reddy Table 7.2.1, a/h = %3d: %.4f, Navier %.6f'
              % (a_h, wbar, w))
        assert abs(w - wbar) <= 0.5e-4
    meshes = (8, 16, 32)
    for name, (prop, a) in plates.items():
        wref = navier_ss1_center_deflection(prop, a)
        # NOTE truncation of the Navier series
        assert abs(navier_ss1_center_deflection(prop, a, 399)/wref - 1) < 1e-8
        w = np.array([ss1_plate_center_deflection(prop, a, n)
                      for n in meshes])/wref
        e = 1 - w
        order = np.log2(e[1]/e[2])
        w_inf = w[2] + (w[2] - w[1])/3
        print('%-10s a/h = %5.1f  w/wref %s  order %.2f  Richardson %.5f'
              % (name, a/prop.h, ' '.join('%.4f' % wi for wi in w), order,
                 w_inf))
        # NOTE stiffer than the reference, monotonic convergence at the
        #      rate h**2 of the bilinear element
        assert np.all(e > 0), name
        assert e[0] > e[1] > e[2], name
        assert 1.5 < order < 2.5, name
        # NOTE the extrapolation removes at least two thirds of the error
        #      of the finest mesh: the limit is the Navier solution
        assert abs(w_inf - 1) < e[2]/3, name


def test_mass_formulations_match_brockman():
    r"""Equations (21), (22) and (31) and the lumped mass of p. 2349

    For a parallelogram the Jacobian determinant is constant, which is the
    assumption under which Brockman evaluates his Eq. (22),

    .. math::
        \boldsymbol{H} = \frac{A}{36}
        \begin{bmatrix} 4 & 2 & 1 & 2 \\ 2 & 4 & 2 & 1 \\
                        1 & 2 & 4 & 2 \\ 2 & 1 & 2 & 4 \end{bmatrix}

    used by ``mtype = 0``. The reduced mass of his Eq. (31) replaces it with
    `\boldsymbol{H} = (A/16) \boldsymbol{s} \boldsymbol{s}^\top`, every entry equal, used by
    ``mtype = 2``. The lumped mass of pp. 2348-2349, by Gauss-Lobatto
    integration at the nodes, used by ``mtype = 1``, is diagonal provided
    `R_2 = 0`, as here without offset, with `R_1 A/4` and `R_3 A/4` per
    node.

    All three conserve the total mass on any shape. The per-node closed
    forms hold where the constant-Jacobian assumption holds.

    """
    prop = isotropic_plate(thickness=0.01, E=70.e9, nu=0.3, rho=2700.)
    R1 = prop.intrho
    R3 = prop.intrhoz2
    data = Quad4RData()
    for gname, ncoords, regular in [('rectangle', RECTANGLE, True),
                                    ('parallelogram', PARALLELOGRAM, True),
                                    ('distorted', DISTORTED, False)]:
        el, probe = make_element(ncoords)
        A = el.area
        for mtype in (0, 1, 2):
            Mr = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
            Mc = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
            Mv = np.zeros(data.M_SPARSE_SIZE, dtype=DOUBLE)
            el.update_M(Mr, Mc, Mv, prop, mtype)
            M = coo_matrix((Mv, (Mr, Mc)), shape=(24, 24)).toarray()
            # NOTE total mass, from the rigid-body translation
            s = np.zeros(24)
            s[0::DOF] = 1.
            total = s.dot(M.dot(s))
            print(gname, 'mtype', mtype, 'total mass', total, 'exact', A*R1)
            assert np.isclose(total, A*R1, rtol=1e-12), (gname, mtype)
            if not regular:
                continue
            if mtype == 0:
                assert np.isclose(M[0, 0], 4.*A*R1/36., rtol=1e-12), gname
                assert np.isclose(M[0, 6], 2.*A*R1/36., rtol=1e-12), gname
                assert np.isclose(M[0, 12], A*R1/36., rtol=1e-12), gname
            elif mtype == 2:
                assert np.isclose(M[0, 0], A*R1/16., rtol=1e-12), gname
                assert np.isclose(M[0, 6], A*R1/16., rtol=1e-12), gname
                assert np.isclose(M[0, 12], A*R1/16., rtol=1e-12), gname
            else:
                assert np.isclose(M[0, 0], A*R1/4., rtol=1e-12), gname
                assert np.isclose(M[3, 3], A*R3/4., rtol=1e-12), gname
                offdiag = np.abs(M - np.diag(np.diag(M))).max()
                assert offdiag < 1e-14*np.abs(M).max(), gname


if __name__ == '__main__':
    test_hourglass_operator_matches_brockman_eq15()
    test_hourglass_operator_equals_second_derivative_of_N()
    test_generalized_stiffnesses_are_dimensionally_homogeneous()
    test_five_hourglass_patterns_are_spurious_without_stabilisation()
    test_stabilisation_restores_the_rank_of_a_rectangular_element()
    test_twisting_mode_is_stabilised()
    test_hourglass_strains_orthogonal_to_linear_fields()
    test_rigid_body_rotation_energy()
    test_constant_strain_patch_test_on_an_irregular_mesh()
    test_hourglass_stiffness_is_independent_of_the_length_unit()
    test_homogeneous_stiffnesses_agree_with_brockman_for_small_areas()
    test_one_element_wide_strip_in_torsion()
    test_transverse_shear_matches_belytschko_leviathan()
    test_bending_stabilisation_matches_belytschko_leviathan()
    test_constant_twist_patch_test()
    test_bending_patch_test_belytschko_leviathan()
    test_laminate_hourglass_stiffnesses_use_membrane_and_bending_moduli()
    test_bending_moduli_differ_from_membrane_moduli_for_laminates()
    test_laminate_zero_energy_modes()
    test_laminated_plates_against_navier()
    test_mass_formulations_match_brockman()
