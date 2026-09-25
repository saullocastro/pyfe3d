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
Writing `\pmb{h} = [1, -1, 1, -1]^\top` for the hourglass pattern of the
bilinear element, `\pmb{s} = [1, 1, 1, 1]^\top`, and `A` for the element area,
the quantities being verified are

* the hourglass operator of Brockman's Eq. (15),

  .. math::
      \tilde\gamma = 2 \frac{x_{31} y_{31} + x_{24} y_{42}}{A^2}
      \quad , \quad
      \pmb{\gamma} = \left. \frac{\partial^2 \pmb{N}}
                                 {\partial x \partial y}\right|_{\xi=\eta=0}
      = \frac{\tilde\gamma}{4} \pmb{h}

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
  the equivalent moduli `E_1^{eq} = 1/(h a_{11})` and `E_2^{eq} = 1/(h
  a_{22})` obtained from the inverse of the extensional stiffness matrix, and
  by taking `E^{(h)}_w` as the mean of `E^{(h)}_{\theta_x}` and
  `E^{(h)}_{\theta_y}`, both of which reduce to Eq. (16) for an isotropic
  plate;

* the classification of the element modes of Brockman's p. 2347, six proper
  rigid-body motions, eight uniform-strain states captured by the one-point
  rule, and six modes that the one-point rule misses, namely the five
  hourglass patterns `u = h`, `v = h`, `w = h`, `\theta_x = h`, `\theta_y =
  h`, which the stabilisation suppresses, and the twisting mode

  .. math::
      w = \frac{1}{4}(\pmb{s}^\top \pmb{y}) x
        + \frac{1}{4}(\pmb{s}^\top \pmb{x}) y
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
       \pmb{\gamma} = \pmb{h} - (\pmb{h}^\top \pmb{x}) \pmb{b}_1
                              - (\pmb{h}^\top \pmb{y}) \pmb{b}_2

   with `\pmb{b}_1` and `\pmb{b}_2` the centroidal derivatives `N_{i,x}` and
   `N_{i,y}`, the orthogonalisation being due to Flanagan and Belytschko
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
   correction changes nothing on a parallelogram, where `\pmb{h}^\top
   \pmb{x}` and `\pmb{h}^\top \pmb{y}` are zero, and changes nothing about
   the energy of the five hourglass patterns on any shape, because the
   centroidal sums `\pmb{h}^\top \pmb{b}_1` and `\pmb{h}^\top \pmb{b}_2`
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

from pyfe3d.shellprop_utils import isotropic_plate
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


def make_element(ncoords, K6ROT=0.):
    r"""A single Quad4R with its rotation matrix and probe already updated

    The fictitious drilling penalty is switched off by default, because its
    operator reads the in-plane translations and would otherwise add energy
    to the in-plane hourglass patterns, which would mask the term being
    verified.

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
    el.update_rotation_matrix(ncoords_flatten, 0., 0., 1.)
    el.update_probe_xe(ncoords_flatten)
    el.update_area()
    return el, probe


def element_KC0(el, prop, hgfactor=1.):
    data = Quad4RData()
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_KC0(KC0r, KC0c, KC0v, prop, 0, hgfactor, hgfactor, hgfactor,
                  hgfactor, hgfactor)
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
    r"""Bring a displacement vector from element to global coordinates"""
    rmat = rotation(el)
    ug = np.zeros_like(ue_element)
    for i in range(4):
        ug[DOF*i:DOF*i + 3] = rmat.T @ ue_element[DOF*i:DOF*i + 3]
        ug[DOF*i + 3:DOF*i + 6] = rmat.T @ ue_element[DOF*i + 3:DOF*i + 6]
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
    \pmb{h}/4`, equal to the centroidal `\partial^2 N/\partial x \partial y`

    """
    return gamma_tilde(el, probe)/4.*H


def centroid_derivatives(probe):
    r"""`\pmb{b}_1` and `\pmb{b}_2` of Brockman's Eq. (14), the centroidal
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
    Nxi = 0.25*np.array([-1., 1., 1., -1.])
    Neta = 0.25*np.array([-1., -1., 1., 1.])
    return j11*Nxi + j12*Neta, j21*Nxi + j22*Neta


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
    require. The `w` term keeps his factor, because there the required
    amount of stabilisation is problem dependent rather than
    element dependent and no constant reproduces it, see
    :func:`test_transverse_coefficient_has_no_unit_invariant_constant` and
    the section "Dimensional homogeneity of the hourglass stiffnesses" of
    the :mod:`pyfe3d.quad4r` documentation. For an isotropic plate
    `E_{1eq} = E_{2eq} = E`, so the mean that generalises the `w` term to
    laminates reduces to `E`.

    """
    return {0: 0.1*E*t*area, 1: 0.1*E*t*area,
            2: 0.1*E*t**3/(1. + 1./area),
            3: 0.1*E*t**3*area, 4: 0.1*E*t**3*area}


def test_hourglass_operator_matches_brockman_eq15():
    r"""The hourglass energy against the closed form of Eqs. (15) and (16)

    Each of the five hourglass patterns of p. 2347 produces no strain at the
    centroid, so the one-point rule gives it no energy and the whole of its
    energy comes from the stabilisation,

    .. math::
        U = \frac{1}{2} E^{(h)} (\pmb{\gamma}^\top \pmb{h})^2 A
          = \frac{1}{2} E^{(h)} \tilde\gamma^2 A

    since `\pmb{\gamma}^\top \pmb{h} = \tilde\gamma \pmb{h}^\top \pmb{h}/4 =
    \tilde\gamma`. The identity is checked on four shapes, including two
    irregular ones, and holds to machine precision on all of them, which
    verifies the operator of Eq. (15) and the stiffnesses of Eq. (16)
    together.

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
        for dof in range(5):
            ue = hourglass_mode(el, dof)
            U = 0.5*ue.dot(KC0.dot(ue))
            U_ref = 0.5*Eh[dof]*gt**2*el.area
            print(name, HG_NAMES[dof], U, U_ref)
            assert np.isclose(U, U_ref, rtol=1e-12), (name, HG_NAMES[dof],
                                                      U, U_ref)


def test_hourglass_operator_equals_second_derivative_of_N():
    r"""The two expressions of Eq. (15) define the same operator

    ``Quad4R`` builds the operator as the centroidal `\partial^2 N/\partial x
    \partial y`, which the element code evaluates as ``0.25*(j11*j22 +
    j12*j21)`` times the alternating pattern. Eq. (15) also gives it in
    closed form as `\tilde\gamma \pmb{h}/4`. The two agree on any shape.

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

    pyfe3d departs from Brockman's Eq. (16) in two ways, both set out in the
    section "Dimensional homogeneity of the hourglass stiffnesses" of the
    :mod:`pyfe3d.quad4r` documentation: his `1/(1 + 1/A)` is replaced by the
    area itself, which is what the dimensions of `K_{ij} = A E^{(h)}
    \gamma_i \gamma_j` require, and `w` is moved from the `E t^3` group of
    Eq. (16b) into the `E t A` group, because the mode the one-point rule
    misses in `w` is resisted by transverse shear rather than by bending.

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
        for dof in range(5):
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
    the twisting mode. With it on, only the rigid-body motions and the
    twisting mode remain, the drilling rotations being held by their own
    operator.

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
    # NOTE 6 rigid-body motions + the twisting mode of Eq. (13)
    assert counts['on'] == 7, counts


def test_twisting_mode_is_not_stabilised_and_needs_two_elements():
    r"""Brockman's Eq. (13) and his remark attributing it to Hughes

    Brockman's Eq. (13) is

    .. math::
        w = \frac{1}{4}(\pmb{s}^\top \pmb{y}) x
          + \frac{1}{4}(\pmb{s}^\top \pmb{x}) y
        \quad , \quad \theta_x = x \quad , \quad \theta_y = y

    which is not an hourglass pattern, so the stabilisation does not reach
    it, and it stays a zero-energy mode of a single element whether the
    stabilisation is on or off. Brockman notes that it "cannot occur in a
    mesh of two or more elements, as shown by Hughes", which is checked here
    by counting the zero eigenvalues of a two-element mesh.

    Brockman measures the transverse shear as `\pmb{\nabla} w -
    \pmb{\theta}`, whereas :class:`pyfe3d.Quad4R` uses `\gamma_{xz} = w_{,x}
    + r_y` and `\gamma_{yz} = w_{,y} - r_x`, so `r_x = \theta_x`, `r_y =
    -\theta_y` and the sign of the first term of `w` changes,

    .. math::
        w = -\frac{1}{4}(\pmb{s}^\top \pmb{y}) x
          + \frac{1}{4}(\pmb{s}^\top \pmb{x}) y
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
            assert abs(U) < 1e-14, (gname, label, U)

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
    `\pmb{\gamma}^\top \pmb{s} = \pmb{\gamma}^\top \pmb{x} =
    \pmb{\gamma}^\top \pmb{y} = 0`. For the operator of Eq. (15) the first
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
            el.update_KC0(KC0r, KC0c, KC0v, prop, 0, hgfactor, hgfactor,
                          hgfactor, hgfactor, hgfactor)
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

    The `w` term is excluded: it is moved from the `E t^3` group to the
    `E t A` group on physical grounds, so it does not converge to Eq. (16b)
    and is not expected to.

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


def strip_twist(hgfactor_w=1., nx=6, ny=1):
    r"""Twist of the MacNeal and Harder (1985) 6 by 0.2 strip, reference 0.03208

        MacNeal, R. H., and Harder, R. L., 1985, "A proposed standard set of
        problems to test finite element accuracy," Finite Elements in
        Analysis and Design, 1(1), pp. 3-20.
        https://doi.org/10.1016/0168-874X(85)90003-4

    A one-element-wide strip carrying torsion is the case where the twist
    *is* the hourglass pattern, so the answer is set almost entirely by
    `E^{(h)}_w`. Root fully clamped, a unit torque applied as a pair of
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
        el.update_KC0(KC0r, KC0c, KC0v, prop, hgfactor_w=hgfactor_w)
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


def test_transverse_coefficient_has_no_unit_invariant_constant():
    r"""Why `E^{(h)}_w` alone keeps Brockman's `1/(1 + 1/A)`

    The hourglass operator cannot distinguish the spurious pattern `w = xy`
    with `\theta_x = \theta_y = 0` from a legitimate twist curvature, both
    giving `\gamma^T w = \partial^2 w/\partial x \partial y`, so the
    stabilisation also stiffens real twist and the acceptable amount is a
    property of the problem rather than of the element. This test measures
    the two ends of that range on this element's own benchmarks and shows
    they are two orders of magnitude apart, which is why no dimensionless
    constant replaces the factor and why the other four coefficients could
    be made homogeneous while this one could not.

    Expressed as a multiple of `E t^3`, Brockman's factor supplies
    `0.1/(1 + 1/A)`, so what it gives is set by the element area in the unit
    used. The two cases are the plate of
    ``test_quad4r_linear_buckling_plate.py``, whose elements have
    `A = 2.381 \times 10^{-3}` in metres, and the one-element-wide torsion
    strip of MacNeal and Harder (1985), whose elements have `A = 0.2`.

    """
    A_plate = (2.0/30.)*(0.5/14.)
    A_strip = (6.0/6.)*0.2
    c_plate = 0.1/(1. + 1./A_plate)
    c_strip = 0.1/(1. + 1./A_strip)
    print('coefficient of E*t**3 that Brockman supplies:')
    print('   thin plate,  A = %.4g m2 : %.4e' % (A_plate, c_plate))
    print('   torsion strip, A = %.4g m2 : %.4e' % (A_strip, c_strip))
    print('   ratio: %.1f' % (c_strip/c_plate))
    # NOTE the spread is what a single constant would have to span
    assert c_strip/c_plate > 50.

    # and the consequence, measured: holding the coefficient down at the
    # value the thin plate wants leaves the torsion strip far too flexible
    ref = 0.03208
    brockman = strip_twist(hgfactor_w=1.)
    reduced = strip_twist(hgfactor_w=c_plate/c_strip)
    print('strip twist / reference: Brockman %.3g, at the thin-plate '
          'coefficient %.3g' % (brockman/ref, reduced/ref))
    assert 3. < brockman/ref < 8.
    assert reduced/ref > 4.*brockman/ref


def test_mass_formulations_match_brockman():
    r"""Equations (21), (22) and (31) and the lumped mass of p. 2349

    For a parallelogram the Jacobian determinant is constant, which is the
    assumption under which Brockman evaluates his Eq. (22),

    .. math::
        \pmb{H} = \frac{A}{36}
        \begin{bmatrix} 4 & 2 & 1 & 2 \\ 2 & 4 & 2 & 1 \\
                        1 & 2 & 4 & 2 \\ 2 & 1 & 2 & 4 \end{bmatrix}

    used by ``mtype = 0``. The reduced mass of his Eq. (31) replaces it with
    `\pmb{H} = (A/16) \pmb{s} \pmb{s}^\top`, every entry equal, used by
    ``mtype = 1``. The lumped mass of p. 2349, by Gauss-Lobatto integration
    at the nodes, is diagonal with `R_1 A/4` and `R_3 A/4` per node, used by
    ``mtype = 2``.

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
            elif mtype == 1:
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
    test_twisting_mode_is_not_stabilised_and_needs_two_elements()
    test_hourglass_strains_orthogonal_to_linear_fields()
    test_rigid_body_rotation_energy()
    test_constant_strain_patch_test_on_an_irregular_mesh()
    test_hourglass_stiffness_is_independent_of_the_length_unit()
    test_homogeneous_stiffnesses_agree_with_brockman_for_small_areas()
    test_transverse_coefficient_has_no_unit_invariant_constant()
    test_mass_formulations_match_brockman()
