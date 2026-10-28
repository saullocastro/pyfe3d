#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
===============================================================
Variational asymptotic sectional analysis (:mod:`pyfe3d.vabs`)
===============================================================

.. currentmodule:: pyfe3d.vabs

Cross-sectional analysis of prismatic beams with arbitrary geometry and
anisotropic, non-homogeneous materials, discretized with two-dimensional
finite elements over the cross section. The class :class:`.CrossSection`
receives the mesh, the material matrices and the orientation of each
element, and computes the classical `4 \times 4` and the generalized
Timoshenko `6 \times 6` stiffness and flexibility matrices, the tension
centre, the shear centre and the mass properties, and the `5 \times 5`
stiffness of the generalized Vlasov model, for thin-walled open sections.
The functions
:func:`.isotropic` and :func:`.orthotropic` create the material matrices,
:func:`.rotate_material` rotates them, :func:`.yu_to_pyfe3d` and
:func:`.pyfe3d_to_yu` convert the stiffness matrices between the ordering
and signs of this module and those of the beam elements of pyfe3d, and
:func:`.beamprop_from_vabs` creates a :class:`.BeamProp` from them when
possible. The methods :meth:`.CrossSection.warping` and
:meth:`.CrossSection.triangles` give the warping displacements of the nodes
and a triangulation of the mesh, to plot the warping with any plotting
library.

The coordinates of the cross section are `x_2` and `x_3`, which are the
`y` and `z` of the element coordinate system of :class:`.BeamC` and
:class:`.BeamLR`, and `x_1` is the beam axis. The six strain measures of
the stiffness matrices of this module are, in this order, the extensional
strain, the two engineering transverse shear strains, the twist rate and
the two bending curvatures, `\{\gamma_{11}, 2\gamma_{12}, 2\gamma_{13},
\kappa_1, \kappa_2, \kappa_3\}`; the classical matrices omit the two shear
strains. The axial strain at the point `(x_2, x_3)` of a cross section
without warping is `\gamma_{11} + x_3 \kappa_2 - x_2 \kappa_3`, such that
the curvatures are related to those of pyfe3d by `\kappa_1 = \kappa_x`,
`\kappa_2 = \kappa_z` and `\kappa_3 = -\kappa_y`.

Notes
-----
The module implements the cross-sectional analysis of the variational
asymptotic method of Cesnik and Hodges [1]_, with the generalized
Timoshenko refinement of Yu et al. [2]_, as updated by Yu et al. [3]_ for
the warping constraints and the energy transformation of prismatic beams,
see :meth:`.CrossSection.calc_stiffness`. The ordering of the strain
measures is that of Eq. (63) of [2]_, Eq. (18) of [4]_ and Eq. (109) of
[3]_, and the signs of the curvatures follow from the operator
`\boldsymbol{\Gamma}_\epsilon` of Eq. (16) of [2]_ and Eq. (36) of [3]_,
whose first row, `\{1, 0, x_3, -x_2\}` for `\{\gamma_{11}, \kappa_1,
\kappa_2, \kappa_3\}`, gives the axial strain above. The moments
work-conjugate to `\kappa_2` and `\kappa_3` are then `M_2 = \int
\sigma_{11} x_3 \, dA = M_y` and `M_3 = -\int \sigma_{11} x_2 \, dA = -M_z`,
with `M_y` and `M_z` those of pyfe3d, see :func:`.yu_to_pyfe3d`.

The generalized Vlasov model is that of Yu et al. [5]_, see
:meth:`.CrossSection.calc_vlasov`.

Only prismatic (straight and untwisted) beams are implemented: the
initial twist `k_1` and curvatures `k_2`, `k_3` of [2]_ and [3]_, which
enter the operator `\boldsymbol{\Gamma}_R` and the perturbation solution
of the appendix of [3]_, are not.

References
----------
.. [1] Cesnik, C. E. S., and Hodges, D. H., "VABS: A New Concept for
   Composite Rotor Blade Cross-Sectional Modeling", Journal of the
   American Helicopter Society, 42(1), pp. 27--38, 1997.
   https://doi.org/10.4050/JAHS.42.27
.. [2] Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
   Timoshenko-like modeling of initially curved and twisted composite
   beams", International Journal of Solids and Structures, 39(19),
   pp. 5101--5121, 2002. https://doi.org/10.1016/S0020-7683(02)00399-2
.. [3] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic beam
   sectional analysis -- An updated version", International Journal of
   Engineering Science, 59, pp. 40--64, 2012.
   https://doi.org/10.1016/j.ijengsci.2012.03.006
.. [4] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
   Variational Asymptotic Beam Sectional Analysis", Journal of the
   American Helicopter Society, 50(1), pp. 46--55, 2005.
   https://doi.org/10.4050/1.3092842
.. [5] Yu, W., Hodges, D. H., Volovoi, V. V., and Fuchs, E. D., "A
   generalized Vlasov theory for composite beams", Thin-Walled
   Structures, 43(9), pp. 1493--1511, 2005.
   https://doi.org/10.1016/j.tws.2005.02.003

"""
from libc.math cimport sqrt, sin, cos

import numpy as np
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import splu

from .beamprop cimport BeamProp

DOUBLE = np.float64

# NOTE maximum number of nodes per element and of warping degrees-of-freedom
#      per element, three per node
cdef enum:
    MAX_NODES = 9
    MAX_EDOF = 27
    MAX_GAUSS = 9

# NOTE position of the pairs (i, j) of the strain components in the ordering
#      {11, 12, 13, 22, 23, 33} of this module
cdef int PAIR_I[6]
cdef int PAIR_J[6]
PAIR_I[:] = [0, 0, 0, 1, 1, 2]
PAIR_J[:] = [0, 1, 2, 1, 2, 2]

# NOTE ordering {11, 22, 33, 23, 13, 12} (Voigt) to {11, 12, 13, 22, 23, 33}
VOIGT_TO_YU = np.array([0, 5, 4, 1, 3, 2])

# NOTE positions of {gamma11, kappa1, kappa2, kappa3, 2gamma12, 2gamma13} in
#      {gamma11, 2gamma12, 2gamma13, kappa1, kappa2, kappa3}
BLOCK_TO_INTERLEAVED = np.array([0, 4, 5, 1, 2, 3])

# NOTE local nodes of the linear triangles of each element, counter-clockwise,
#      see CrossSection.triangles
SUB_TRIANGLES = {
    3: ((0, 1, 2),),
    6: ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)),
    4: ((0, 1, 2), (0, 2, 3)),
    8: ((0, 4, 7), (4, 1, 5), (5, 2, 6), (6, 3, 7), (4, 5, 6), (4, 6, 7)),
    9: ((0, 4, 8), (0, 8, 7), (4, 1, 5), (4, 5, 8), (8, 5, 2), (8, 2, 6),
        (7, 8, 6), (7, 6, 3)),
}


def isotropic(double E, double nu):
    r"""Material matrix of an isotropic material

    Parameters
    ----------
    E : float
        Young modulus, force per unit area.
    nu : float
        Poisson ratio, dimensionless.

    Returns
    -------
    D : np.ndarray
        Material matrix, force per unit area, shape ``(6, 6)``, relating the
        stresses `\{\sigma_{11}, \sigma_{12}, \sigma_{13}, \sigma_{22},
        \sigma_{23}, \sigma_{33}\}` to the strains `\{\Gamma_{11},
        2\Gamma_{12}, 2\Gamma_{13}, \Gamma_{22}, 2\Gamma_{23},
        \Gamma_{33}\}`, the ordering used by :class:`.CrossSection`.

    Raises
    ------
    ValueError
        If ``E`` is not positive or ``nu`` is not between ``-1`` and
        ``0.5``, both excluded.

    Examples
    --------
    >>> from pyfe3d.vabs import isotropic
    >>> D = isotropic(E=70e9, nu=0.3)
    >>> D.shape
    (6, 6)

    """
    if E <= 0:
        raise ValueError('E must be positive')
    if not (-1 < nu < 0.5):
        raise ValueError('nu must be in the interval (-1, 0.5)')
    G = E/(2*(1 + nu))
    return orthotropic(E, E, E, G, G, G, nu, nu, nu)


def orthotropic(double E1, double E2, double E3, double G12, double G13,
                double G23, double nu12, double nu13, double nu23):
    r"""Material matrix of an orthotropic material in its material axes

    Parameters
    ----------
    E1, E2, E3 : float
        Young moduli along the material axes `1`, `2` and `3`, force per
        unit area.
    G12, G13, G23 : float
        Shear moduli of the planes `12`, `13` and `23`, force per unit area.
    nu12, nu13, nu23 : float
        Poisson ratios, dimensionless, with ``nu12`` the contraction along
        `2` for a tension along `1`, and likewise for ``nu13`` and ``nu23``.

    Returns
    -------
    D : np.ndarray
        Material matrix in the material axes, force per unit area, shape
        ``(6, 6)``, with the ordering of :func:`.isotropic`.

    Raises
    ------
    ValueError
        If the matrix is not positive definite.

    Examples
    --------
    >>> from pyfe3d.vabs import orthotropic
    >>> D = orthotropic(E1=20.59e6, E2=1.42e6, E3=1.42e6, G12=0.87e6,
    ...                 G13=0.87e6, G23=0.696e6, nu12=0.30, nu13=0.30,
    ...                 nu23=0.34)
    >>> D.shape
    (6, 6)

    """
    # NOTE compliance in the ordering {11, 22, 33, 23, 13, 12}
    S = np.zeros((6, 6), dtype=DOUBLE)
    S[0, 0] = 1/E1
    S[1, 1] = 1/E2
    S[2, 2] = 1/E3
    S[0, 1] = S[1, 0] = -nu12/E1
    S[0, 2] = S[2, 0] = -nu13/E1
    S[1, 2] = S[2, 1] = -nu23/E2
    S[3, 3] = 1/G23
    S[4, 4] = 1/G13
    S[5, 5] = 1/G12
    if np.any(np.linalg.eigvalsh(S) <= 0):
        raise ValueError('the material matrix is not positive definite')
    D = np.linalg.inv(S)
    D = D[np.ix_(VOIGT_TO_YU, VOIGT_TO_YU)]
    return 0.5*(D + D.T)


cdef void _strain_transformation(double th1, double th3, double *T) noexcept nogil:
    r"""Transformation `\boldsymbol{T}` of the strains from the beam axes to
    the material axes, written row by row in ``T`` (36 values)

    The material axis `m_1` is at the angle ``th3`` from the beam axis `x_1`
    towards the axis `y_2`, and `y_2` at the angle ``th1`` from `x_2`
    towards `x_3`; both angles are in radians. See :func:`.rotate_material`.

    Notes
    -----
    With `R_{ik} = \boldsymbol{m}_i \cdot \boldsymbol{x}_k` the direction
    cosines, the tensor components transform as `\Gamma^m_{ij} = R_{ik}
    R_{jl} \Gamma_{kl}`, which, for the engineering strains of the ordering
    of :func:`.isotropic`, gives the entries computed below, such that
    `\boldsymbol{\Gamma}^m = \boldsymbol{T} \boldsymbol{\Gamma}`.

    """
    cdef double R[3][3]
    cdef double c1, s1, c3, s3, f
    cdef int I, J, i, j, k, l
    c1 = cos(th1)
    s1 = sin(th1)
    c3 = cos(th3)
    s3 = sin(th3)
    # NOTE y1 = x1, y2 = c1 x2 + s1 x3, y3 = -s1 x2 + c1 x3
    #      m1 = c3 y1 + s3 y2, m2 = -s3 y1 + c3 y2, m3 = y3
    R[0][0] = c3
    R[0][1] = s3*c1
    R[0][2] = s3*s1
    R[1][0] = -s3
    R[1][1] = c3*c1
    R[1][2] = c3*s1
    R[2][0] = 0
    R[2][1] = -s1
    R[2][2] = c1
    for I in range(6):
        i = PAIR_I[I]
        j = PAIR_J[I]
        f = 1. if i == j else 2.
        for J in range(6):
            k = PAIR_I[J]
            l = PAIR_J[J]
            if k == l:
                T[6*I + J] = f*R[i][k]*R[j][k]
            else:
                T[6*I + J] = f*0.5*(R[i][k]*R[j][l] + R[i][l]*R[j][k])


cdef void _rotate(double *Dm, double th1, double th3, double *Dout) noexcept nogil:
    r"""Material matrix in the beam axes, ``Dout`` `= \boldsymbol{T}^\top`
    ``Dm`` `\boldsymbol{T}`, with `\boldsymbol{T}` of
    :func:`._strain_transformation`; 36 values each, row by row

    Notes
    -----
    The strain energy density is invariant, `\boldsymbol{\Gamma}^{m\top}
    \boldsymbol{D}_m \boldsymbol{\Gamma}^m = \boldsymbol{\Gamma}^\top
    \boldsymbol{T}^\top \boldsymbol{D}_m \boldsymbol{T}
    \boldsymbol{\Gamma}`.

    """
    cdef double T[36]
    cdef double DT[36]
    cdef int i, j, k
    _strain_transformation(th1, th3, T)
    for i in range(6):
        for j in range(6):
            DT[6*i + j] = 0
            for k in range(6):
                DT[6*i + j] += Dm[6*i + k]*T[6*k + j]
    for i in range(6):
        for j in range(6):
            Dout[6*i + j] = 0
            for k in range(6):
                Dout[6*i + j] += T[6*k + i]*DT[6*k + j]


def rotate_material(D, double theta1, double theta3):
    r"""Rotate a material matrix from the material axes to the beam axes

    The local axes `y_1, y_2, y_3` of an element are obtained from the beam
    axes `x_1, x_2, x_3` by the rotation ``theta1`` about `x_1`, positive
    from `x_2` towards `x_3`, such that `y_1 = x_1`. The material axes `m_1,
    m_2, m_3` are obtained from the local axes by the rotation ``theta3``
    about `y_3`, positive from `y_1` towards `y_2`, such that `m_3 = y_3`.
    For a laminated wall with `y_2` along the wall, ``theta3`` is the ply
    angle measured from the beam axis. Both angles are shown in the figure
    of :class:`.CrossSection`; this convention is used by
    :class:`.CrossSection` for the angles of each element.

    Parameters
    ----------
    D : array-like
        Material matrix in the material axes, force per unit area, shape
        ``(6, 6)``, with the ordering of :func:`.isotropic`.
    theta1 : float
        Angle of the local axes, in degrees.
    theta3 : float
        Ply angle, in degrees.

    Returns
    -------
    Drot : np.ndarray
        Material matrix in the beam axes, force per unit area, shape
        ``(6, 6)``.

    Examples
    --------
    >>> from pyfe3d.vabs import orthotropic, rotate_material
    >>> D = orthotropic(20.59e6, 1.42e6, 1.42e6, 0.87e6, 0.87e6, 0.696e6,
    ...                 0.30, 0.30, 0.34)
    >>> Drot = rotate_material(D, theta1=90., theta3=15.)

    Notes
    -----
    With this convention, the stiffness of the box beam CUS of Table 2 of
    [1]_ is reproduced with all the signs of its VABS 3.2 column, for the
    plies at ``theta3 = 15`` in every wall, `y_2` tangent to the wall
    counter-clockwise around the section and `y_3` pointing inwards. Two
    other published examples give couplings of the same magnitude but of
    the opposite sign: `S_{12}`, `S_{45}` and `S_{46}` of the box beam CAS1
    of Table 2 of [2]_, which has no sketch of its axes and whose signs are
    obtained for the section rotated by 180 degrees, and `S_{14}`, `S_{25}`
    and `S_{36}` of the pipe of Table 3 of [3]_, whose signs are obtained
    with the opposite ply angles in the half circles. These differences are
    attributed to the labelling of the axes and of the ply angles in [2]_
    and [3]_; they are checked in ``tests/test_vabs_composite.py``.

    References
    ----------
    .. [1] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
       beam sectional analysis -- An updated version", International
       Journal of Engineering Science, 59, pp. 40--64, 2012.
       https://doi.org/10.1016/j.ijengsci.2012.03.006
    .. [2] Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
       Timoshenko-like modeling of initially curved and twisted composite
       beams", International Journal of Solids and Structures, 39(19),
       pp. 5101--5121, 2002. https://doi.org/10.1016/S0020-7683(02)00399-2
    .. [3] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    cdef double [:, ::1] Dm = np.ascontiguousarray(D, dtype=DOUBLE)
    cdef double [:, ::1] Dout = np.zeros((6, 6), dtype=DOUBLE)
    if Dm.shape[0] != 6 or Dm.shape[1] != 6:
        raise ValueError('D must have shape (6, 6)')
    _rotate(&Dm[0, 0], np.deg2rad(theta1), np.deg2rad(theta3), &Dout[0, 0])
    return np.asarray(Dout)


cdef int _gauss_points(int nen, double *xi, double *eta, double *w) noexcept nogil:
    r"""Integration points of an element with ``nen`` nodes, written in
    ``xi``, ``eta`` and ``w``; returns their number

    Notes
    -----
    Triangles, in the area coordinates `\xi`, `\eta` with weights summing
    to `1/2`: for 3 nodes the 3-point rule of degree 2, at `(1/6, 1/6)`,
    `(2/3, 1/6)`, `(1/6, 2/3)`; for 6 nodes the 6-point rule of degree 4 of
    Dunavant [1]_, Appendix II, p. 1140, with the weights halved.
    Quadrilaterals, in `-1 \le \xi, \eta \le 1`: the `2 \times 2` Gauss
    rule for 4 nodes and the `3 \times 3` rule for 8 and 9 nodes. The
    degrees are checked in ``tests/test_vabs_elements.py``.

    References
    ----------
    .. [1] Dunavant, D. A., "High degree efficient symmetrical Gaussian
       quadrature rules for the triangle", International Journal for
       Numerical Methods in Engineering, 21(6), pp. 1129--1148, 1985.
       https://doi.org/10.1002/nme.1620210612

    """
    cdef double a, b, wa, wb, g, g0, w0, w1
    cdef int i, j, n
    cdef double gp[3]
    cdef double gw[3]
    if nen == 3:
        xi[0] = 1./6; eta[0] = 1./6; w[0] = 1./6
        xi[1] = 2./3; eta[1] = 1./6; w[1] = 1./6
        xi[2] = 1./6; eta[2] = 2./3; w[2] = 1./6
        return 3
    elif nen == 6:
        a = 0.445948490915965
        b = 0.091576213509771
        wa = 0.223381589678011/2
        wb = 0.109951743655322/2
        xi[0] = a; eta[0] = a; w[0] = wa
        xi[1] = 1 - 2*a; eta[1] = a; w[1] = wa
        xi[2] = a; eta[2] = 1 - 2*a; w[2] = wa
        xi[3] = b; eta[3] = b; w[3] = wb
        xi[4] = 1 - 2*b; eta[4] = b; w[4] = wb
        xi[5] = b; eta[5] = 1 - 2*b; w[5] = wb
        return 6
    elif nen == 4:
        g = 1/sqrt(3.)
        gp[0] = -g; gp[1] = g
        n = 0
        for j in range(2):
            for i in range(2):
                xi[n] = gp[i]; eta[n] = gp[j]; w[n] = 1.
                n += 1
        return 4
    else:
        g = sqrt(0.6)
        gp[0] = -g; gp[1] = 0.; gp[2] = g
        gw[0] = 5./9; gw[1] = 8./9; gw[2] = 5./9
        n = 0
        for j in range(3):
            for i in range(3):
                xi[n] = gp[i]; eta[n] = gp[j]; w[n] = gw[i]*gw[j]
                n += 1
        return 9


cdef void _shape(int nen, double xi, double eta, double *N, double *Nxi,
                 double *Neta) noexcept nogil:
    r"""Shape functions ``N`` and their derivatives ``Nxi``, ``Neta`` with
    respect to the natural coordinates, for an element with ``nen`` nodes

    Notes
    -----
    Triangles use the area coordinates `L_1 = 1 - \xi - \eta`, `L_2 = \xi`,
    `L_3 = \eta`: linear, `N_i = L_i`, for 3 nodes, and quadratic for 6
    nodes, `N_i = L_i (2 L_i - 1)` at the corners and `N_4 = 4 L_1 L_2`,
    `N_5 = 4 L_2 L_3`, `N_6 = 4 L_3 L_1` at the mid-sides. Quadrilaterals
    use the bilinear functions for 4 nodes, the serendipity functions for 8
    nodes and the biquadratic Lagrange functions for 9 nodes, with the
    corners at `(\pm 1, \pm 1)` and the mid-side nodes 5 to 8 at `(0, -1)`,
    `(1, 0)`, `(0, 1)`, `(-1, 0)`, and the node 9 at `(0, 0)`.

    """
    cdef double L1, L2, L3
    cdef double xa[4]
    cdef double ea[4]
    cdef double l0x, l1x, l2x, l0e, l1e, l2e
    cdef double d0x, d1x, d2x, d0e, d1e, d2e
    cdef double lx[3]
    cdef double le[3]
    cdef double dlx[3]
    cdef double dle[3]
    cdef int i, j, k
    cdef int ix[9]
    cdef int ie[9]
    if nen == 3:
        N[0] = 1 - xi - eta; Nxi[0] = -1; Neta[0] = -1
        N[1] = xi; Nxi[1] = 1; Neta[1] = 0
        N[2] = eta; Nxi[2] = 0; Neta[2] = 1
    elif nen == 6:
        L1 = 1 - xi - eta
        L2 = xi
        L3 = eta
        N[0] = L1*(2*L1 - 1); Nxi[0] = -(4*L1 - 1); Neta[0] = -(4*L1 - 1)
        N[1] = L2*(2*L2 - 1); Nxi[1] = 4*L2 - 1; Neta[1] = 0
        N[2] = L3*(2*L3 - 1); Nxi[2] = 0; Neta[2] = 4*L3 - 1
        N[3] = 4*L1*L2; Nxi[3] = 4*(L1 - L2); Neta[3] = -4*L2
        N[4] = 4*L2*L3; Nxi[4] = 4*L3; Neta[4] = 4*L2
        N[5] = 4*L3*L1; Nxi[5] = -4*L3; Neta[5] = 4*(L1 - L3)
    elif nen == 4 or nen == 8:
        xa[0] = -1; xa[1] = 1; xa[2] = 1; xa[3] = -1
        ea[0] = -1; ea[1] = -1; ea[2] = 1; ea[3] = 1
        if nen == 4:
            for i in range(4):
                N[i] = 0.25*(1 + xa[i]*xi)*(1 + ea[i]*eta)
                Nxi[i] = 0.25*xa[i]*(1 + ea[i]*eta)
                Neta[i] = 0.25*ea[i]*(1 + xa[i]*xi)
        else:
            for i in range(4):
                N[i] = 0.25*(1 + xa[i]*xi)*(1 + ea[i]*eta)*(xa[i]*xi + ea[i]*eta - 1)
                Nxi[i] = 0.25*xa[i]*(1 + ea[i]*eta)*(2*xa[i]*xi + ea[i]*eta)
                Neta[i] = 0.25*ea[i]*(1 + xa[i]*xi)*(xa[i]*xi + 2*ea[i]*eta)
            # NOTE mid-side nodes at (0, -1), (1, 0), (0, 1), (-1, 0)
            N[4] = 0.5*(1 - xi*xi)*(1 - eta); Nxi[4] = -xi*(1 - eta); Neta[4] = -0.5*(1 - xi*xi)
            N[5] = 0.5*(1 + xi)*(1 - eta*eta); Nxi[5] = 0.5*(1 - eta*eta); Neta[5] = -eta*(1 + xi)
            N[6] = 0.5*(1 - xi*xi)*(1 + eta); Nxi[6] = -xi*(1 + eta); Neta[6] = 0.5*(1 - xi*xi)
            N[7] = 0.5*(1 - xi)*(1 - eta*eta); Nxi[7] = -0.5*(1 - eta*eta); Neta[7] = -eta*(1 - xi)
    else:
        # NOTE 1-D quadratic Lagrange functions at -1, 0, 1
        lx[0] = 0.5*xi*(xi - 1); lx[1] = 1 - xi*xi; lx[2] = 0.5*xi*(xi + 1)
        dlx[0] = xi - 0.5; dlx[1] = -2*xi; dlx[2] = xi + 0.5
        le[0] = 0.5*eta*(eta - 1); le[1] = 1 - eta*eta; le[2] = 0.5*eta*(eta + 1)
        dle[0] = eta - 0.5; dle[1] = -2*eta; dle[2] = eta + 0.5
        # NOTE positions (index along xi, index along eta) of the 9 nodes
        ix[0] = 0; ix[1] = 2; ix[2] = 2; ix[3] = 0; ix[4] = 1
        ix[5] = 2; ix[6] = 1; ix[7] = 0; ix[8] = 1
        ie[0] = 0; ie[1] = 0; ie[2] = 2; ie[3] = 2; ie[4] = 0
        ie[5] = 1; ie[6] = 2; ie[7] = 1; ie[8] = 1
        for k in range(9):
            i = ix[k]
            j = ie[k]
            N[k] = lx[i]*le[j]
            Nxi[k] = dlx[i]*le[j]
            Neta[k] = lx[i]*dle[j]


def _quadrature(int nen):
    r"""Integration points and weights of :func:`._gauss_points` for an
    element with ``nen`` nodes, as three 1-D arrays ``xi``, ``eta``, ``w``"""
    cdef double xi[MAX_GAUSS]
    cdef double eta[MAX_GAUSS]
    cdef double w[MAX_GAUSS]
    cdef int n
    n = _gauss_points(nen, xi, eta, w)
    return (np.array([xi[i] for i in range(n)]),
            np.array([eta[i] for i in range(n)]),
            np.array([w[i] for i in range(n)]))


def _shape_functions(int nen, double xi, double eta):
    r"""Shape functions of :func:`._shape` and their derivatives with
    respect to ``xi`` and ``eta`` at one point, three 1-D arrays of size
    ``nen``"""
    cdef double N[MAX_NODES]
    cdef double Nxi[MAX_NODES]
    cdef double Neta[MAX_NODES]
    _shape(nen, xi, eta, N, Nxi, Neta)
    return (np.array([N[i] for i in range(nen)]),
            np.array([Nxi[i] for i in range(nen)]),
            np.array([Neta[i] for i in range(nen)]))


cdef int _integrate(double [::1] x2, double [::1] x3, int [:, ::1] conn,
                    int [::1] nen, double [:, :, ::1] Del, double [::1] rhoel,
                    int constraint, int [::1] rows, int [::1] cols,
                    double [::1] Dhhv, double [::1] Dhlv, double [::1] Dllv,
                    double [:, ::1] Dheps, double [:, ::1] Dleps,
                    double [:, ::1] Dc, double [:, ::1] Depseps,
                    double [::1] secint) noexcept nogil:
    r"""Integrate and assemble the matrices of the cross-sectional analysis

    The element matrices of `\boldsymbol{D}_{hh}`, `\boldsymbol{D}_{hl}`
    and `\boldsymbol{D}_{ll}` are written in COO format in ``rows``,
    ``cols`` and ``Dhhv``, ``Dhlv``, ``Dllv``, element after element, each
    element with ``(3*nen[e])**2`` consecutive entries. The matrices
    `\boldsymbol{D}_{h\epsilon}`, `\boldsymbol{D}_{l\epsilon}` (``Dheps``,
    ``Dleps``), the constraint matrix ``Dc``, all of shape ``(ndof, 4)``,
    and `\boldsymbol{D}_{\epsilon\epsilon}` (``Depseps``, ``(4, 4)``) are
    accumulated. ``secint`` receives the integrals over the cross section
    of `1, x_2, x_3, x_2^2, x_3^2, x_2 x_3` and of the same functions times
    the density. Returns ``-1``, or the index of the first element with a
    non-positive Jacobian determinant, in which case the outputs are
    incomplete.

    Notes
    -----
    With the warping `\boldsymbol{w} = \boldsymbol{S} \boldsymbol{V}`, the
    shape functions `\boldsymbol{S}` and three warping degrees-of-freedom
    `w_1, w_2, w_3` per node, and the strains `\boldsymbol{\Gamma} =
    \{\Gamma_{11}, 2\Gamma_{12}, 2\Gamma_{13}, \Gamma_{22}, 2\Gamma_{23},
    \Gamma_{33}\}`, the three-dimensional strain of a prismatic beam is
    Eq. (34) of [1]_ with `\boldsymbol{\Gamma}_R = \boldsymbol{0}` and
    `\sqrt{g} = 1`,

    .. math::
        \boldsymbol{\Gamma} = \boldsymbol{\Gamma}_h \boldsymbol{S}
        \boldsymbol{V} + \boldsymbol{\Gamma}_\epsilon
        \bar{\boldsymbol{\epsilon}} + \boldsymbol{\Gamma}_l \boldsymbol{S}
        \boldsymbol{V}'

    where `\bar{\boldsymbol{\epsilon}} = \{\gamma_{11}, \kappa_1, \kappa_2,
    \kappa_3\}` are the classical strain measures and, Eqs. (35), (36) and
    (38) of [1]_,

    .. math::
        \boldsymbol{\Gamma}_h = \begin{bmatrix}
            0 & 0 & 0 \\ \partial_2 & 0 & 0 \\ \partial_3 & 0 & 0 \\
            0 & \partial_2 & 0 \\ 0 & \partial_3 & \partial_2 \\
            0 & 0 & \partial_3 \end{bmatrix}
        \qquad
        \boldsymbol{\Gamma}_\epsilon = \begin{bmatrix}
            1 & 0 & x_3 & -x_2 \\ 0 & -x_3 & 0 & 0 \\ 0 & x_2 & 0 & 0 \\
            0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
        \qquad
        \boldsymbol{\Gamma}_l = \begin{bmatrix}
            \boldsymbol{I}_3 \\ \boldsymbol{0}_3 \end{bmatrix}

    with `\partial_\alpha = \partial/\partial x_\alpha`. The matrices are
    those of Eqs. (76) and (89) of [1]_, named here as in the book of the
    author, `\boldsymbol{D}_{hh} = \langle (\boldsymbol{\Gamma}_h
    \boldsymbol{S})^\top \boldsymbol{D} \boldsymbol{\Gamma}_h \boldsymbol{S}
    \rangle` (`E` in [1]_), `\boldsymbol{D}_{h\epsilon} = \langle
    (\boldsymbol{\Gamma}_h \boldsymbol{S})^\top \boldsymbol{D}
    \boldsymbol{\Gamma}_\epsilon \rangle`, `\boldsymbol{D}_{\epsilon\epsilon}
    = \langle \boldsymbol{\Gamma}_\epsilon^\top \boldsymbol{D}
    \boldsymbol{\Gamma}_\epsilon \rangle`, `\boldsymbol{D}_{hl} = \langle
    (\boldsymbol{\Gamma}_h \boldsymbol{S})^\top \boldsymbol{D}
    \boldsymbol{\Gamma}_l \boldsymbol{S} \rangle`, `\boldsymbol{D}_{ll} =
    \langle (\boldsymbol{\Gamma}_l \boldsymbol{S})^\top \boldsymbol{D}
    \boldsymbol{\Gamma}_l \boldsymbol{S} \rangle` and
    `\boldsymbol{D}_{l\epsilon} = \langle (\boldsymbol{\Gamma}_l
    \boldsymbol{S})^\top \boldsymbol{D} \boldsymbol{\Gamma}_\epsilon
    \rangle`, where `\langle \cdot \rangle` is the integral over the cross
    section and `\boldsymbol{D}` the material matrix in the beam axes. The
    constraint matrix is `\boldsymbol{D}_c = \langle
    \boldsymbol{C}_c \boldsymbol{S} \rangle^\top`, Eq. (78) of [1]_: with
    ``constraint=0``, the matrix `\boldsymbol{C}_c` of Eq. (19) of [1]_,
    for `\langle w_i \rangle = 0` and `\langle w_{2,3} - w_{3,2} \rangle =
    0`, Eqs. (15) and (17) of [1]_; with ``constraint=1``, the kernel
    `\boldsymbol{\Psi}^\top` of Eq. (41) of [1]_, for `\langle w_i \rangle
    = 0` and `\langle x_2 w_3 - x_3 w_2 \rangle = 0`, Eqs. (9) and (10) of
    [2]_.

    References
    ----------
    .. [1] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
       beam sectional analysis -- An updated version", International
       Journal of Engineering Science, 59, pp. 40--64, 2012.
       https://doi.org/10.1016/j.ijengsci.2012.03.006
    .. [2] Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
       Timoshenko-like modeling of initially curved and twisted composite
       beams", International Journal of Solids and Structures, 39(19),
       pp. 5101--5121, 2002. https://doi.org/10.1016/S0020-7683(02)00399-2

    """
    cdef int e, ne, n, a, b, i, j, k, m, gpi, ngp, edof, pos, row, col
    cdef double xi[MAX_GAUSS]
    cdef double eta[MAX_GAUSS]
    cdef double wg[MAX_GAUSS]
    cdef double N[MAX_NODES]
    cdef double Nxi[MAX_NODES]
    cdef double Neta[MAX_NODES]
    cdef double N2[MAX_NODES]
    cdef double N3[MAX_NODES]
    cdef double Bh[6][MAX_EDOF]
    cdef double Bl[6][MAX_EDOF]
    cdef double Ge[6][4]
    cdef double DBh[6][MAX_EDOF]
    cdef double DBl[6][MAX_EDOF]
    cdef double DGe[6][4]
    cdef double Dhhe[MAX_EDOF][MAX_EDOF]
    cdef double Dhle[MAX_EDOF][MAX_EDOF]
    cdef double Dlle[MAX_EDOF][MAX_EDOF]
    cdef double J11, J12, J21, J22, detJ, wdet, xg2, xg3, rho, tmp
    cdef int gdof[MAX_EDOF]

    for k in range(12):
        secint[k] = 0
    pos = 0
    ne = conn.shape[0]
    for e in range(ne):
        n = nen[e]
        edof = 3*n
        rho = rhoel[e]
        for a in range(n):
            for k in range(3):
                gdof[3*a + k] = 3*conn[e, a] + k
        for i in range(edof):
            for j in range(edof):
                Dhhe[i][j] = 0
                Dhle[i][j] = 0
                Dlle[i][j] = 0
        for k in range(6):
            for i in range(edof):
                Bh[k][i] = 0
                Bl[k][i] = 0
        ngp = _gauss_points(n, xi, eta, wg)
        for gpi in range(ngp):
            _shape(n, xi[gpi], eta[gpi], N, Nxi, Neta)
            J11 = 0; J12 = 0; J21 = 0; J22 = 0
            xg2 = 0; xg3 = 0
            for a in range(n):
                J11 += Nxi[a]*x2[conn[e, a]]
                J12 += Nxi[a]*x3[conn[e, a]]
                J21 += Neta[a]*x2[conn[e, a]]
                J22 += Neta[a]*x3[conn[e, a]]
                xg2 += N[a]*x2[conn[e, a]]
                xg3 += N[a]*x3[conn[e, a]]
            detJ = J11*J22 - J12*J21
            if detJ <= 0:
                return e
            wdet = wg[gpi]*detJ
            for a in range(n):
                N2[a] = (J22*Nxi[a] - J12*Neta[a])/detJ
                N3[a] = (-J21*Nxi[a] + J11*Neta[a])/detJ
            # NOTE Gamma_h S and Gamma_l S, degrees-of-freedom w1, w2, w3 per
            #      node; only the non-zero entries are written, the others
            #      remain zero from the initialization above
            for a in range(n):
                Bh[1][3*a + 0] = N2[a]
                Bh[2][3*a + 0] = N3[a]
                Bh[3][3*a + 1] = N2[a]
                Bh[4][3*a + 1] = N3[a]
                Bh[4][3*a + 2] = N2[a]
                Bh[5][3*a + 2] = N3[a]
                Bl[0][3*a + 0] = N[a]
                Bl[1][3*a + 1] = N[a]
                Bl[2][3*a + 2] = N[a]
            # NOTE Gamma_epsilon for {gamma11, kappa1, kappa2, kappa3}
            for k in range(6):
                for m in range(4):
                    Ge[k][m] = 0
            Ge[0][0] = 1
            Ge[0][2] = xg3
            Ge[0][3] = -xg2
            Ge[1][1] = -xg3
            Ge[2][1] = xg2
            for k in range(6):
                for i in range(edof):
                    DBh[k][i] = 0
                    DBl[k][i] = 0
                    for j in range(6):
                        DBh[k][i] += Del[e, k, j]*Bh[j][i]
                    for j in range(3):
                        DBl[k][i] += Del[e, k, j]*Bl[j][i]
                for m in range(4):
                    DGe[k][m] = 0
                    for j in range(6):
                        DGe[k][m] += Del[e, k, j]*Ge[j][m]
            for i in range(edof):
                for j in range(edof):
                    tmp = 0
                    for k in range(6):
                        tmp += Bh[k][i]*DBh[k][j]
                    Dhhe[i][j] += tmp*wdet
                    tmp = 0
                    for k in range(6):
                        tmp += Bh[k][i]*DBl[k][j]
                    Dhle[i][j] += tmp*wdet
                    tmp = 0
                    for k in range(3):
                        tmp += Bl[k][i]*DBl[k][j]
                    Dlle[i][j] += tmp*wdet
                for m in range(4):
                    tmp = 0
                    for k in range(6):
                        tmp += Bh[k][i]*DGe[k][m]
                    Dheps[gdof[i], m] += tmp*wdet
                    tmp = 0
                    for k in range(3):
                        tmp += Bl[k][i]*DGe[k][m]
                    Dleps[gdof[i], m] += tmp*wdet
            for m in range(4):
                for j in range(4):
                    tmp = 0
                    for k in range(6):
                        tmp += Ge[k][m]*DGe[k][j]
                    Depseps[m, j] += tmp*wdet
            for a in range(n):
                Dc[gdof[3*a + 0], 0] += N[a]*wdet
                Dc[gdof[3*a + 1], 1] += N[a]*wdet
                Dc[gdof[3*a + 2], 2] += N[a]*wdet
                if constraint == 0:
                    Dc[gdof[3*a + 1], 3] += N3[a]*wdet
                    Dc[gdof[3*a + 2], 3] -= N2[a]*wdet
                else:
                    Dc[gdof[3*a + 1], 3] -= xg3*N[a]*wdet
                    Dc[gdof[3*a + 2], 3] += xg2*N[a]*wdet
            secint[0] += wdet
            secint[1] += xg2*wdet
            secint[2] += xg3*wdet
            secint[3] += xg2*xg2*wdet
            secint[4] += xg3*xg3*wdet
            secint[5] += xg2*xg3*wdet
            secint[6] += rho*wdet
            secint[7] += rho*xg2*wdet
            secint[8] += rho*xg3*wdet
            secint[9] += rho*xg2*xg2*wdet
            secint[10] += rho*xg3*xg3*wdet
            secint[11] += rho*xg2*xg3*wdet
        for i in range(edof):
            row = gdof[i]
            for j in range(edof):
                col = gdof[j]
                rows[pos] = row
                cols[pos] = col
                Dhhv[pos] = Dhhe[i][j]
                Dhlv[pos] = Dhle[i][j]
                Dllv[pos] = Dlle[i][j]
                pos += 1
    return -1


cdef class CrossSection:
    r"""Cross section of a prismatic beam discretized with finite elements

    Stores the mesh, the materials and the element orientations. The
    analysis is done by :meth:`calc_stiffness`, which sets the stiffness
    attributes, and by :meth:`calc_mass`, which sets the mass attributes;
    both set ``area``, ``centroid`` and the integrals of the cross section.
    After :meth:`calc_stiffness`, :meth:`warping` gives the warping of the
    nodes for given beam strains, and :meth:`triangles` a triangulation of
    the mesh on which to plot it. :meth:`calc_vlasov` computes the
    stiffness of the generalized Vlasov model, in the axes of the shear
    centre obtained with :meth:`translated`. All results refer to the
    origin of the coordinates `x_2`, `x_3`, i.e. the beam reference axis
    passes through `(0, 0)`, except those of :meth:`calc_vlasov`.

    .. figure:: ../figures/vabs_axes.svg

       (a) Axes `x_1, x_2, x_3` of a cross section, here a flared channel,
       with the local axes `y_1, y_2, y_3` of an element, oriented by
       ``theta1``, and the tension and shear centres computed by
       :meth:`calc_stiffness`; (b) the material axes `m_1, m_2, m_3` of the
       element, oriented by the ply angle ``theta3``, see
       :func:`.rotate_material`.

    Elements with 3, 4, 6, 8 and 9 nodes are supported: the linear and the
    quadratic triangles and the bilinear, serendipity and biquadratic
    quadrilaterals, with the nodes numbered counter-clockwise in the
    `x_2 x_3` plane, the corners first, then the mid-side nodes, starting
    with the one between the first two corners, then the centre node of the
    9-node element. Different element types may be mixed when they share
    complete edges.

    Parameters
    ----------
    x2, x3 : array-like
        Coordinates of the nodes, dimension length, 1-D arrays of size
        ``n_nodes``.
    elements : sequence of sequences of int
        Connectivity: for each element, the indices of its nodes in ``x2``
        and ``x3``, starting at ``0``; size ``n_elements``, each with 3, 4,
        6, 8 or 9 indices.
    materials : sequence of array-like
        Material matrices in the material axes, force per unit area, each of
        shape ``(6, 6)`` with the ordering of :func:`.isotropic`.
    mat_ids : array-like
        Index of the material of each element in ``materials``, 1-D array of
        integers of size ``n_elements``.
    theta1 : array-like, optional
        Angle of the local axes of each element, in degrees, 1-D array of
        size ``n_elements``, see :func:`.rotate_material`. Default ``None``,
        meaning zero for all elements.
    theta3 : array-like, optional
        Ply angle of each element, in degrees, 1-D array of size
        ``n_elements``, see :func:`.rotate_material`. Default ``None``,
        meaning zero for all elements.
    densities : sequence of float, optional
        Density of each material, mass per unit volume, size
        ``len(materials)``. Default ``None``, meaning zero for all
        materials.
    k1, k2, k3 : float, optional
        Initial twist and initial curvatures of the beam, dimension inverse
        of length. Default ``0.``, the only value implemented.

    Raises
    ------
    NotImplementedError
        If ``k1``, ``k2`` or ``k3`` is not zero.
    ValueError
        If ``x2`` and ``x3`` have different sizes, if there is no element,
        if an element does not have 3, 4, 6, 8 or 9 nodes or refers to a
        node that does not exist, if a node is not used by any element, if
        a material matrix does not have the shape ``(6, 6)``, or if
        ``mat_ids``, ``theta1``, ``theta3`` or ``densities`` do not have one
        entry per element or per material, or refer to a material that does
        not exist.

    Attributes
    ----------
    x2, x3 : np.ndarray
        Coordinates of the nodes, dimension length.
    conn : np.ndarray
        Connectivity, shape ``(n_elements, 9)``, padded with ``-1``.
    nen : np.ndarray
        Number of nodes of each element.
    mat_ids, theta1, theta3 : np.ndarray
        As the parameters, size ``n_elements``.
    materials : list
        Material matrices, as the parameter.
    densities : np.ndarray
        Density of each material, mass per unit volume.
    Del : np.ndarray
        Material matrix of each element in the beam axes, force per unit
        area, shape ``(n_elements, 6, 6)``.
    ndof : int
        Number of warping degrees-of-freedom, ``3*n_nodes``.
    constraint : str
        Warping constraints of the last call to :meth:`calc_stiffness`.
    area : float
        Area of the cross section, dimension length squared.
    centroid : np.ndarray
        Coordinates `x_2`, `x_3` of the centroid of the area, dimension
        length, size ``2``.
    Kc : np.ndarray
        Classical stiffness matrix, shape ``(4, 4)``, of the strains
        `\{\gamma_{11}, \kappa_1, \kappa_2, \kappa_3\}`; the dimensions of
        the entries are those of ``Kt`` without the shear rows and columns.
    Fc : np.ndarray
        Classical flexibility matrix, the inverse of ``Kc``.
    Kt : np.ndarray
        Generalized Timoshenko stiffness matrix, shape ``(6, 6)``, of the
        strains `\{\gamma_{11}, 2\gamma_{12}, 2\gamma_{13}, \kappa_1,
        \kappa_2, \kappa_3\}`, i.e. of the forces `\{F_1, F_2, F_3, M_1,
        M_2, M_3\}`: dimension force in the upper-left `3 \times 3` block,
        force times length in the off-diagonal blocks and force times length
        squared in the lower-right block.
    Ft : np.ndarray
        Generalized Timoshenko flexibility matrix, the inverse of ``Kt``.
    tension_centre : np.ndarray
        Coordinates `(a_2, a_3)` of the point where an axial force `F_1`
        gives no bending curvature, `\kappa_2 = \kappa_3 = 0`, from ``Fc``,
        dimension length, size ``2``, see the figure.
    shear_centre : np.ndarray
        Coordinates `(e_2, e_3)` of the point where a transverse force gives
        no twist, `\kappa_1 = 0`, from ``Ft``, dimension length, size ``2``,
        see the figure.
    V0 : np.ndarray
        Nodal warping displacements of the classical model per unit
        classical strain, shape ``(ndof, 4)``, with the warping
        displacements `w_1, w_2, w_3` of node ``i`` in the rows ``3*i``,
        ``3*i + 1``, ``3*i + 2``; the first column, per unit extension, has
        the dimension length, the others, per unit curvature, length
        squared.
    V1S : np.ndarray
        Nodal warping displacements per unit derivative along the beam of
        the classical strains, shape ``(ndof, 4)``, rows as in ``V0``;
        dimension length squared in the first column and length cubed in
        the others.
    A, B, C : np.ndarray
        Matrices of the second-order energy, shape ``(4, 4)``, dimension of
        ``Kc`` times length to the power ``0``, ``1`` and ``2``.
    Kv : np.ndarray
        Stiffness matrix of the generalized Vlasov model, set by
        :meth:`calc_vlasov`, shape ``(5, 5)``, of the strains
        `\{\gamma_{11}, \kappa_1, \kappa_2, \kappa_3, \kappa_1'\}`
        referred to the shear centre, i.e. of the forces `\{F_1, M_1, M_2,
        M_3, M_\omega\}`, with `M_\omega` the bimoment; the last row and
        column have the dimensions of those of ``Kc`` times length, and the
        last diagonal entry, the warping rigidity, force times length to the
        fourth.
    Fv : np.ndarray
        Flexibility matrix of the generalized Vlasov model, the inverse of
        ``Kv``.
    vlasov_section : CrossSection
        The cross section with the origin at the shear centre, from
        :meth:`translated`, analysed by :meth:`calc_vlasov`; its
        :meth:`warping` gives the warping about the shear centre.
    mass_per_length : float
        Mass per unit length.
    mass_centre : np.ndarray
        Coordinates `x_2`, `x_3` of the mass centre, dimension length, size
        ``2``.
    mass_matrix : np.ndarray
        Mass matrix per unit length, shape ``(6, 6)``, of the velocities
        `\{V_1, V_2, V_3, \Omega_1, \Omega_2, \Omega_3\}` of the reference
        axis and of the angular velocities of the cross section, such that
        the kinetic energy per unit length is half its quadratic form; mass
        per unit length in the upper-left block, mass in the off-diagonal
        blocks, mass times length in the lower-right block.
    intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz : float
        Integrals of the density times `1`, `x_2`, `x_3`, `x_2^2`, `x_3^2`
        and `x_2 x_3`, with the names of the attributes of
        :class:`.BeamProp`, where `y = x_2` and `z = x_3`.

    Examples
    --------
    A square of side ``1`` meshed with four bilinear quadrilaterals:

    >>> import numpy as np
    >>> from pyfe3d.vabs import CrossSection, isotropic
    >>> x2, x3 = np.meshgrid([-0.5, 0., 0.5], [-0.5, 0., 0.5])
    >>> elements = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    >>> sec = CrossSection(x2.ravel(), x3.ravel(), elements,
    ...                    materials=[isotropic(1., 0.3)], mat_ids=[0]*4)
    >>> sec.calc_stiffness()
    >>> sec.Kt.shape
    (6, 6)

    Notes
    -----
    The cross-sectional analysis follows Cesnik and Hodges [1]_ and Yu et
    al. [2]_, see :meth:`calc_stiffness` for the equations and
    :func:`._integrate` for the discretization.

    References
    ----------
    .. [1] Cesnik, C. E. S., and Hodges, D. H., "VABS: A New Concept for
       Composite Rotor Blade Cross-Sectional Modeling", Journal of the
       American Helicopter Society, 42(1), pp. 27--38, 1997.
       https://doi.org/10.4050/JAHS.42.27
    .. [2] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
       beam sectional analysis -- An updated version", International
       Journal of Engineering Science, 59, pp. 40--64, 2012.
       https://doi.org/10.1016/j.ijengsci.2012.03.006

    """
    cdef public object x2, x3, conn, nen, mat_ids, theta1, theta3
    cdef public object materials, densities, Del
    cdef public int ndof
    cdef public object constraint
    cdef public double area
    cdef public object centroid
    cdef public object Kc, Fc, Kt, Ft, tension_centre, shear_centre
    cdef public object V0, V1S, A, B, C
    cdef public object Kv, Fv, vlasov_section
    cdef public double mass_per_length
    cdef public object mass_centre, mass_matrix
    cdef public double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
    cdef object _mats

    def __init__(CrossSection self, x2, x3, elements, materials, mat_ids,
                 theta1=None, theta3=None, densities=None, double k1=0.,
                 double k2=0., double k3=0.):
        cdef int e, ne, k
        # TODO(author): initially twisted and curved beams, operator Gamma_R
        #      and the perturbation solution of the appendix of Yu et al.
        #      (2012), are not implemented
        if k1 != 0 or k2 != 0 or k3 != 0:
            raise NotImplementedError('initially twisted or curved beams '
                                      '(k1, k2, k3 not zero) are not '
                                      'implemented')
        cdef double [:, :, ::1] Del
        cdef double [:, :, ::1] Dmat
        cdef int [::1] mids
        cdef double [::1] th1, th3
        self.x2 = np.ascontiguousarray(x2, dtype=DOUBLE).ravel()
        self.x3 = np.ascontiguousarray(x3, dtype=DOUBLE).ravel()
        if self.x2.shape[0] != self.x3.shape[0]:
            raise ValueError('x2 and x3 must have the same size')
        nnodes = self.x2.shape[0]
        ne = len(elements)
        if ne == 0:
            raise ValueError('at least one element is required')
        self.conn = -np.ones((ne, MAX_NODES), dtype=np.intc)
        self.nen = np.zeros(ne, dtype=np.intc)
        for e in range(ne):
            nodes = np.asarray(elements[e], dtype=np.intc).ravel()
            if nodes.shape[0] not in (3, 4, 6, 8, 9):
                raise ValueError('element %d has %d nodes, only 3, 4, 6, 8 '
                                 'and 9 are supported'
                                 % (e, nodes.shape[0]))
            if nodes.min() < 0 or nodes.max() >= nnodes:
                raise ValueError('element %d refers to a node that does not '
                                 'exist' % e)
            self.conn[e, :nodes.shape[0]] = nodes
            self.nen[e] = nodes.shape[0]
        used = np.zeros(nnodes, dtype=bool)
        for e in range(ne):
            used[self.conn[e, :self.nen[e]]] = True
        if not np.all(used):
            raise ValueError('node %d is not used by any element'
                             % np.argmin(used))
        self.materials = [np.ascontiguousarray(D, dtype=DOUBLE)
                          for D in materials]
        for D in self.materials:
            if D.shape != (6, 6):
                raise ValueError('each material matrix must have shape (6, 6)')
        self.mat_ids = np.ascontiguousarray(mat_ids, dtype=np.int_).ravel()
        if self.mat_ids.shape[0] != ne:
            raise ValueError('mat_ids must have one entry per element')
        if (self.mat_ids.min() < 0
                or self.mat_ids.max() >= len(self.materials)):
            raise ValueError('mat_ids refers to a material that does not '
                             'exist')
        if theta1 is None:
            self.theta1 = np.zeros(ne, dtype=DOUBLE)
        else:
            self.theta1 = np.ascontiguousarray(theta1, dtype=DOUBLE).ravel()
        if theta3 is None:
            self.theta3 = np.zeros(ne, dtype=DOUBLE)
        else:
            self.theta3 = np.ascontiguousarray(theta3, dtype=DOUBLE).ravel()
        if self.theta1.shape[0] != ne or self.theta3.shape[0] != ne:
            raise ValueError('theta1 and theta3 must have one entry per '
                             'element')
        if densities is None:
            self.densities = np.zeros(len(self.materials), dtype=DOUBLE)
        else:
            self.densities = np.ascontiguousarray(densities,
                                                  dtype=DOUBLE).ravel()
        if self.densities.shape[0] != len(self.materials):
            raise ValueError('densities must have one entry per material')
        self.ndof = 3*nnodes
        # NOTE material matrix of each element in the beam axes
        self._mats = np.ascontiguousarray(np.array(self.materials))
        Dmat = self._mats
        self.Del = np.zeros((ne, 6, 6), dtype=DOUBLE)
        Del = self.Del
        mids = self.mat_ids.astype(np.intc)
        th1 = np.deg2rad(self.theta1)
        th3 = np.deg2rad(self.theta3)
        with nogil:
            for e in range(ne):
                _rotate(&Dmat[mids[e], 0, 0], th1[e], th3[e], &Del[e, 0, 0])
        self.constraint = None
        self.area = 0
        self.centroid = None
        self.Kc = None
        self.Fc = None
        self.Kt = None
        self.Ft = None
        self.tension_centre = None
        self.shear_centre = None
        self.V0 = None
        self.V1S = None
        self.A = None
        self.B = None
        self.C = None
        self.Kv = None
        self.Fv = None
        self.vlasov_section = None
        self.mass_per_length = 0
        self.mass_centre = None
        self.mass_matrix = None
        self.intrho = 0
        self.intrhoy = 0
        self.intrhoz = 0
        self.intrhoy2 = 0
        self.intrhoz2 = 0
        self.intrhoyz = 0

    def _assemble(CrossSection self, int constraint):
        r"""Assemble the matrices of the cross-sectional analysis

        Calls :func:`._integrate` and returns `\boldsymbol{D}_{hh}`,
        `\boldsymbol{D}_{hl}`, `\boldsymbol{D}_{ll}` as sparse CSC matrices
        of shape ``(ndof, ndof)``, `\boldsymbol{D}_{h\epsilon}`,
        `\boldsymbol{D}_{l\epsilon}`, `\boldsymbol{D}_c` as arrays of shape
        ``(ndof, 4)``, `\boldsymbol{D}_{\epsilon\epsilon}` of shape
        ``(4, 4)`` and the 12 integrals of :func:`._integrate`. Sets
        ``area`` and ``centroid``.

        """
        cdef int ne, size, err
        cdef double [::1] rhoel
        ne = self.conn.shape[0]
        size = int(np.sum((3*self.nen)**2))
        rows = np.zeros(size, dtype=np.intc)
        cols = np.zeros(size, dtype=np.intc)
        Dhhv = np.zeros(size, dtype=DOUBLE)
        Dhlv = np.zeros(size, dtype=DOUBLE)
        Dllv = np.zeros(size, dtype=DOUBLE)
        Dheps = np.zeros((self.ndof, 4), dtype=DOUBLE)
        Dleps = np.zeros((self.ndof, 4), dtype=DOUBLE)
        Dc = np.zeros((self.ndof, 4), dtype=DOUBLE)
        Depseps = np.zeros((4, 4), dtype=DOUBLE)
        secint = np.zeros(12, dtype=DOUBLE)
        rhoel = np.ascontiguousarray(self.densities[self.mat_ids],
                                     dtype=DOUBLE)
        err = _integrate(self.x2, self.x3, self.conn, self.nen, self.Del,
                         rhoel, constraint, rows, cols, Dhhv, Dhlv, Dllv,
                         Dheps, Dleps, Dc, Depseps, secint)
        if err >= 0:
            raise ValueError('element %d has a non-positive Jacobian '
                             'determinant, check that its nodes are '
                             'numbered counter-clockwise' % err)
        shape = (self.ndof, self.ndof)
        Dhh = coo_matrix((Dhhv, (rows, cols)), shape=shape).tocsc()
        Dhl = coo_matrix((Dhlv, (rows, cols)), shape=shape).tocsc()
        Dll = coo_matrix((Dllv, (rows, cols)), shape=shape).tocsc()
        self.area = secint[0]
        self.centroid = np.array([secint[1], secint[2]])/secint[0]
        return Dhh, Dhl, Dll, Dheps, Dleps, Dc, Depseps, secint

    def calc_stiffness(CrossSection self, str constraint='yu2012'):
        r"""Compute the classical and generalized Timoshenko stiffnesses

        Assembles the matrices of the cross section, solves the warping
        problems of the classical model and of its refinement with one
        sparse LU factorization, and sets ``Kc``, ``Fc``, ``Kt``, ``Ft``,
        ``tension_centre``, ``shear_centre``, ``V0``, ``V1S``, ``A``, ``B``,
        ``C``, ``constraint``, ``area`` and ``centroid``. The tension centre
        is the point where an axial force gives no bending curvature, and
        the shear centre the point where a transverse force gives no twist,
        see the figure of :class:`.CrossSection`.

        Parameters
        ----------
        constraint : str, optional
            The four constraints that make the warping unique: ``'yu2012'``,
            the default, zero average warping and zero average rotation of
            the warping, or ``'yu2002'``, zero average warping and zero
            average of `x_2 w_3 - x_3 w_2`. They change ``V0`` and the
            Timoshenko matrices, not ``Kc``.

        Raises
        ------
        ValueError
            If ``constraint`` is not one of the above, or if an element has
            a non-positive Jacobian determinant, e.g. when its nodes are
            numbered clockwise.
        RuntimeError
            If the constrained warping problem is singular, e.g. for a mesh
            that is not connected.

        Notes
        -----
        **Classical model.** With the matrices of :func:`._integrate`, the
        energy per unit length of a prismatic beam without the terms in the
        derivatives of the warping is, Eq. (75) of [1]_,

        .. math::
            2 \mathcal{U}_0 = \boldsymbol{V}^\top \boldsymbol{D}_{hh}
            \boldsymbol{V} + 2 \boldsymbol{V}^\top \boldsymbol{D}_{h\epsilon}
            \bar{\boldsymbol{\epsilon}} + \bar{\boldsymbol{\epsilon}}^\top
            \boldsymbol{D}_{\epsilon\epsilon} \bar{\boldsymbol{\epsilon}}

        and its minimum subject to the constraints `\boldsymbol{D}_c^\top
        \boldsymbol{V} = \boldsymbol{0}`, Eqs. (78)--(85) of [1]_, is
        `\boldsymbol{V}_0 = \hat{\boldsymbol{V}}_0 \bar{\boldsymbol{\epsilon}}`
        (``V0``), from the system with the Lagrange multipliers
        `\boldsymbol{\Lambda}`

        .. math::
            \begin{bmatrix} \boldsymbol{D}_{hh} & \boldsymbol{D}_c \\
            \boldsymbol{D}_c^\top & \boldsymbol{0} \end{bmatrix}
            \begin{Bmatrix} \hat{\boldsymbol{V}}_0 \\ \boldsymbol{\Lambda}
            \end{Bmatrix} = \begin{Bmatrix} -\boldsymbol{D}_{h\epsilon} \\
            \boldsymbol{0} \end{Bmatrix}

        which is regular because `\boldsymbol{D}_c^\top \boldsymbol{\Psi}`
        is regular, `\boldsymbol{\Psi}` being the four rigid-body modes of
        the kernel of `\boldsymbol{D}_{hh}`, Eq. (41) of [1]_. The
        multipliers vanish for this right-hand side, which is orthogonal to
        `\boldsymbol{\Psi}`, Eq. (81) of [1]_, and the constraints do not
        change the classical stiffness, Eq. (86) of [1]_,

        .. math::
            \boldsymbol{A} = \hat{\boldsymbol{V}}_0^\top
            \boldsymbol{D}_{h\epsilon} + \boldsymbol{D}_{\epsilon\epsilon}

        stored in ``Kc`` and ``A``.

        **Refinement.** With `\boldsymbol{V} = \boldsymbol{V}_0 +
        \boldsymbol{V}_1` and the terms in `\boldsymbol{V}'`, the first-order
        warping of a prismatic beam is `\boldsymbol{V}_1 =
        \hat{\boldsymbol{V}}_{1S} \bar{\boldsymbol{\epsilon}}'` (``V1S``),
        Eqs. (96)--(101) of [1]_ with `\boldsymbol{D}_R = \boldsymbol{0}`,
        no applied loads, and

        .. math::
            \boldsymbol{D}_S = (\boldsymbol{D}_{hl} - \boldsymbol{D}_{hl}^\top)
            \hat{\boldsymbol{V}}_0 - \boldsymbol{D}_{l\epsilon}

        solved from the same system with `-\boldsymbol{D}_S` on the
        right-hand side, whose multipliers do not vanish, Eq. (100) of
        [1]_. The second-order energy is `2\mathcal{U} =
        \bar{\boldsymbol{\epsilon}}^\top \boldsymbol{A}
        \bar{\boldsymbol{\epsilon}} + 2 \bar{\boldsymbol{\epsilon}}^\top
        \boldsymbol{B} \bar{\boldsymbol{\epsilon}}' +
        \bar{\boldsymbol{\epsilon}}'^\top \boldsymbol{C}
        \bar{\boldsymbol{\epsilon}}' + 2 \bar{\boldsymbol{\epsilon}}^\top
        \boldsymbol{D} \bar{\boldsymbol{\epsilon}}''`, Eqs. (102) and (103)
        of [1]_ with the initial curvatures set to zero,

        .. math::
            \boldsymbol{B} = \hat{\boldsymbol{V}}_0^\top \boldsymbol{D}_{hl}
            \hat{\boldsymbol{V}}_0 + \boldsymbol{D}_{l\epsilon}^\top
            \hat{\boldsymbol{V}}_0 \qquad
            \boldsymbol{C} = \hat{\boldsymbol{V}}_{1S}^\top
            \bar{\boldsymbol{D}}_S + \hat{\boldsymbol{V}}_0^\top
            \boldsymbol{D}_{ll} \hat{\boldsymbol{V}}_0 \qquad
            \bar{\boldsymbol{D}}_S = (\boldsymbol{D}_{hl} +
            \boldsymbol{D}_{hl}^\top) \hat{\boldsymbol{V}}_0 +
            \boldsymbol{D}_{l\epsilon}

        where `\boldsymbol{C}` is stored symmetrized; `\boldsymbol{D}` does
        not enter the stiffness of a prismatic beam.

        **Transformation to the generalized Timoshenko form.** For a
        prismatic beam, the energy `2\mathcal{U} = \boldsymbol{\epsilon}^\top
        \boldsymbol{X} \boldsymbol{\epsilon} + 2 \boldsymbol{\epsilon}^\top
        \boldsymbol{Y} \boldsymbol{\gamma}_s + \boldsymbol{\gamma}_s^\top
        \boldsymbol{G} \boldsymbol{\gamma}_s`, with
        `\boldsymbol{\gamma}_s = \{2\gamma_{12}, 2\gamma_{13}\}`, is
        obtained in closed form, Eqs. (105)--(107) of [1]_, as

        .. math::
            \boldsymbol{G} = \left[ \boldsymbol{Q}^\top \boldsymbol{A}^{-1}
            (\boldsymbol{C} - \boldsymbol{B}^\top \boldsymbol{A}^{-1}
            \boldsymbol{B}) \boldsymbol{A}^{-1} \boldsymbol{Q} \right]^{-1}
            \qquad \boldsymbol{Y} = \boldsymbol{B}^\top \boldsymbol{A}^{-1}
            \boldsymbol{Q} \boldsymbol{G} \qquad
            \boldsymbol{X} = \boldsymbol{A} + \boldsymbol{Y}
            \boldsymbol{G}^{-1} \boldsymbol{Y}^\top

        with `\boldsymbol{Q}` of Eq. (28) of [1]_, rows `\{0, 0\}`, `\{0,
        0\}`, `\{0, -1\}`, `\{1, 0\}`, which follows from the kinematic
        relation between the classical and the Timoshenko strains, Eq. (27)
        of [1]_, and from the one-dimensional equilibrium equations. The
        matrix ``Kt`` is `[\boldsymbol{X}, \boldsymbol{Y};
        \boldsymbol{Y}^\top, \boldsymbol{G}]` reordered to interleave the
        shear strains, as in Eq. (109) of [1]_. By construction the
        classical stiffness obtained from the Timoshenko flexibility by
        deleting the shear rows and columns and inverting, as prescribed
        on p. 52 of [2]_, is ``Kc``.

        **Centres.** The shear centre is Eq. (66) of [3]_, `e_2 = -P_{34} /
        P_{44}`, `e_3 = P_{24}/P_{44}`, with `P_{ij}` the entries of ``Ft``
        numbered from 1. The tension centre `(a_2, a_3)` is the point where
        an axial force `F_1`, with the moments `M_2 = a_3 F_1` and `M_3 =
        -a_2 F_1` about the reference axis, gives `\kappa_2 = \kappa_3 = 0`
        through ``Fc``, a definition of this module.

        References
        ----------
        .. [1] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
           beam sectional analysis -- An updated version", International
           Journal of Engineering Science, 59, pp. 40--64, 2012.
           https://doi.org/10.1016/j.ijengsci.2012.03.006
        .. [2] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of
           the Variational Asymptotic Beam Sectional Analysis", Journal of
           the American Helicopter Society, 50(1), pp. 46--55, 2005.
           https://doi.org/10.4050/1.3092842
        .. [3] Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S.,
           "On Timoshenko-like modeling of initially curved and twisted
           composite beams", International Journal of Solids and
           Structures, 39(19), pp. 5101--5121, 2002.
           https://doi.org/10.1016/S0020-7683(02)00399-2

        """
        if constraint == 'yu2012':
            icons = 0
        elif constraint == 'yu2002':
            icons = 1
        else:
            raise ValueError("constraint must be 'yu2012' or 'yu2002'")
        Dhh, Dhl, Dll, Dheps, Dleps, Dc, Depseps, _ = self._assemble(icons)
        Dcs = coo_matrix(Dc).tocsc()
        Kaug = bmat([[Dhh, Dcs], [Dcs.T, None]], format='csc')
        try:
            lu = splu(Kaug)
        except RuntimeError as err:
            raise RuntimeError('the constrained warping problem is singular, '
                               'check that the mesh is connected') from err
        nc = 4
        rhs = np.zeros((self.ndof + nc, 4), dtype=DOUBLE)
        # NOTE classical warping
        rhs[:self.ndof] = -Dheps
        V0 = lu.solve(rhs)[:self.ndof]
        A = V0.T @ Dheps + Depseps
        A = 0.5*(A + A.T)
        # NOTE first-order warping of a prismatic beam
        DS = (Dhl - Dhl.T) @ V0 - Dleps
        rhs[:self.ndof] = -DS
        V1S = lu.solve(rhs)[:self.ndof]
        DSbar = (Dhl + Dhl.T) @ V0 + Dleps
        B = V0.T @ (Dhl @ V0) + Dleps.T @ V0
        C = V1S.T @ DSbar + V0.T @ (Dll @ V0)
        C = 0.5*(C + C.T)
        # NOTE transformation to the generalized Timoshenko form
        Q = np.array([[0., 0.], [0., 0.], [0., -1.], [1., 0.]])
        Ainv = np.linalg.inv(A)
        Ginv = Q.T @ Ainv @ (C - B.T @ Ainv @ B) @ Ainv @ Q
        Ginv = 0.5*(Ginv + Ginv.T)
        G = np.linalg.inv(Ginv)
        Y = B.T @ Ainv @ Q @ G
        X = A + Y @ Ginv @ Y.T
        Kblk = np.zeros((6, 6), dtype=DOUBLE)
        Kblk[:4, :4] = X
        Kblk[:4, 4:] = Y
        Kblk[4:, :4] = Y.T
        Kblk[4:, 4:] = G
        Kt = Kblk[np.ix_(BLOCK_TO_INTERLEAVED, BLOCK_TO_INTERLEAVED)]
        Kt = 0.5*(Kt + Kt.T)
        self.constraint = constraint
        self.V0 = V0
        self.V1S = V1S
        self.A = A
        self.B = B
        self.C = C
        self.Kc = A.copy()
        self.Fc = np.linalg.inv(A)
        self.Kt = Kt
        self.Ft = np.linalg.inv(Kt)
        Fc = self.Fc
        Ft = self.Ft
        # NOTE tension centre: kappa2 = kappa3 = 0 for F1 at (a2, a3)
        M = np.array([[-Fc[2, 3], Fc[2, 2]], [-Fc[3, 3], Fc[3, 2]]])
        self.tension_centre = np.linalg.solve(M, -Fc[[2, 3], 0])
        # NOTE shear centre, Yu et al. (2002), Eq. (66)
        self.shear_centre = np.array([-Ft[2, 3]/Ft[3, 3], Ft[1, 3]/Ft[3, 3]])

    def translated(CrossSection self, double e2, double e3):
        r"""Copy of the cross section with the origin moved to a given point

        Creates a new :class:`.CrossSection` with the same mesh, materials,
        orientations and densities, whose coordinates are `x_2 - e_2` and
        `x_3 - e_3`, i.e. whose reference axis passes through the point
        `(e_2, e_3)` of this cross section. Nothing is computed.

        Parameters
        ----------
        e2, e3 : float
            Coordinates of the new origin in the axes of this cross section,
            dimension length, e.g. ``*shear_centre``.

        Returns
        -------
        sec : CrossSection
            The translated cross section.

        Examples
        --------
        >>> import numpy as np
        >>> from pyfe3d.vabs import CrossSection, isotropic
        >>> x2, x3 = np.meshgrid([-0.5, 0., 0.5], [-0.5, 0., 0.5])
        >>> elements = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        >>> sec = CrossSection(x2.ravel(), x3.ravel(), elements,
        ...                    materials=[isotropic(1., 0.3)], mat_ids=[0]*4)
        >>> float(sec.translated(0.5, 0.).x2.min())
        -1.0

        """
        elements = [self.conn[e, :self.nen[e]]
                    for e in range(self.conn.shape[0])]
        return CrossSection(self.x2 - e2, self.x3 - e3, elements,
                            self.materials, self.mat_ids, theta1=self.theta1,
                            theta3=self.theta3, densities=self.densities)

    def calc_vlasov(CrossSection self, str constraint='yu2002'):
        r"""Compute the stiffness of the generalized Vlasov model

        Locates the shear centre, analyses with :meth:`calc_stiffness` the
        cross section translated to it, stored in ``vlasov_section``, and
        sets ``Kv`` and ``Fv``, the `5 \times 5` stiffness and flexibility
        of the strains `\{\gamma_{11}, \kappa_1, \kappa_2, \kappa_3,
        \kappa_1'\}` referred to the shear centre, where `\kappa_1'` is the
        derivative of the twist rate along the beam and the bimoment
        `M_\omega` its work-conjugate. The model is intended for thin-walled
        open sections, whose restrained warping is not represented by the
        classical and Timoshenko models. The other attributes are not
        changed. No beam element of pyfe3d uses
        ``Kv``: :class:`.BeamC` and :class:`.BeamLR` have no warping
        degree-of-freedom, and ``Kv`` is meant for one-dimensional Vlasov
        beam analyses outside pyfe3d.

        Parameters
        ----------
        constraint : str, optional
            The warping constraints of :meth:`calc_stiffness`. Default
            ``'yu2002'``, with which the published values of the generalized
            Vlasov model are reproduced, see the Notes. ``'yu2012'``, the
            default of :meth:`calc_stiffness`, gives the same first four
            rows and columns and, in the tested cases, the same shear
            centre, but a different warping rigidity ``Kv[4, 4]``: lower by
            up to about 12 per cent for a strip, and by about 0.1 per cent
            for I- and channel sections.

        Raises
        ------
        ValueError
            As :meth:`calc_stiffness`.
        RuntimeError
            As :meth:`calc_stiffness`.

        Examples
        --------
        Warping rigidity of a thin-walled I-section, flanges of width ``1``
        and walls of thickness ``0.04``, outer height ``0.5``, close to
        `E t b_f^3 h^2/24` with `h = 0.46` between the flange mid-planes:

        >>> import numpy as np
        >>> from pyfe3d.vabs import CrossSection, isotropic
        >>> t, H, w = 0.04, 0.5, 0.48
        >>> def rect(a, b, nx, ny, x0, y0):
        ...     x2, x3 = np.meshgrid(x0 + np.linspace(-a/2, a/2, nx + 1),
        ...                          y0 + np.linspace(-b/2, b/2, ny + 1))
        ...     els = [[j*(nx + 1) + i, j*(nx + 1) + i + 1,
        ...             (j + 1)*(nx + 1) + i + 1, (j + 1)*(nx + 1) + i]
        ...            for j in range(ny) for i in range(nx)]
        ...     return x2.ravel(), x3.ravel(), els
        >>> parts = [rect(w, t, 12, 2, -t/2 - w/2, y) for y in (-0.23, 0.23)]
        >>> parts += [rect(w, t, 12, 2, t/2 + w/2, y) for y in (-0.23, 0.23)]
        >>> parts += [rect(t, t, 2, 2, 0., y) for y in (-0.23, 0.23)]
        >>> parts += [rect(t, H - 2*t, 2, 10, 0., 0.)]
        >>> xy = np.round(np.concatenate([np.column_stack(p[:2])
        ...                               for p in parts]), 9)
        >>> uniq, inv = np.unique(xy, axis=0, return_inverse=True)
        >>> inv = inv.ravel()
        >>> offsets = np.cumsum([0] + [len(p[0]) for p in parts])
        >>> elements = [[int(inv[o + n]) for n in el]
        ...             for p, o in zip(parts, offsets) for el in p[2]]
        >>> sec = CrossSection(uniq[:, 0], uniq[:, 1], elements,
        ...                    materials=[isotropic(10e6, 0.3)],
        ...                    mat_ids=[0]*len(elements))
        >>> sec.calc_vlasov()
        >>> EG = 10e6*t*1.**3*0.46**2/24
        >>> bool(abs(sec.Kv[4, 4]/EG - 1) < 0.01)
        True

        Notes
        -----
        The second-order energy of a prismatic beam, `2\mathcal{U} =
        \bar{\boldsymbol{\epsilon}}^\top \boldsymbol{A}
        \bar{\boldsymbol{\epsilon}} + 2 \bar{\boldsymbol{\epsilon}}^\top
        \boldsymbol{B} \bar{\boldsymbol{\epsilon}}' +
        \bar{\boldsymbol{\epsilon}}'^\top \boldsymbol{C}
        \bar{\boldsymbol{\epsilon}}' + 2 \bar{\boldsymbol{\epsilon}}^\top
        \boldsymbol{D} \bar{\boldsymbol{\epsilon}}''`, Eq. (1) of [1]_, see
        :meth:`calc_stiffness`, depends on the reference axis. Following
        Section 2 of [1]_, the shear centre is obtained from the generalized
        Timoshenko flexibility, Eq. (3) of [1]_, ``shear_centre``, the
        origin is moved to it, and only the derivative `\kappa_1'` of the
        twist rate is kept, `\gamma_{11}' = \kappa_2' = \kappa_3' = 0`,
        which gives the generalized Vlasov model, Eq. (4) of [1]_,

        .. math::
            \boldsymbol{K}_v = \begin{bmatrix} \boldsymbol{A} &
            \boldsymbol{b} \\ \boldsymbol{b}^\top & C_{22} \end{bmatrix}
            \qquad \boldsymbol{b} = \{B_{12}, B_{22}, B_{32}, B_{42}\}

        with `\boldsymbol{A}`, `\boldsymbol{B}` and `\boldsymbol{C}` those
        of the translated section and the entries numbered from 1. The first
        four rows and columns are the classical stiffness in the axes of the
        shear centre. The term in `\boldsymbol{D}` contains
        `\kappa_1''` and has no place in Eq. (4) of [1]_; it is omitted, as
        in [1]_, whose values are reproduced without it.

        The two sets of warping constraints differ in the fourth one, on
        the rotation of the in-plane warping. This changes `\boldsymbol{C}`,
        hence `K_{v,55}`, but not `\boldsymbol{A}` and `\boldsymbol{B}`.
        With the constraints of Yu et al. (2002), ``constraint='yu2002'``,
        `K_{v,55}` reproduces the values computed with VABS in Tables 1, 3
        and 4 of [1]_ for strips to five digits or more, and for the
        isotropic strip it agrees with the asymptotic thin-walled value `E
        h^3 b^3/[144 (1 - \nu^2)]`, Eq. (18) of [1]_. With
        ``constraint='yu2012'`` it is lower for the strips, by `9.5` per
        cent for the isotropic one, close to the value `E h^3 b^3/144` of
        Vlasov's theory, Eq. (17) of [1]_, by `1.9` per cent for the
        orthotropic one and by `12` per cent for the anisotropic one of
        Table 4 of [1]_. For I- and channel sections, the two are within
        about `0.1` per cent of each other and within `0.4` per cent of the
        thin-walled warping rigidity. Since only ``'yu2002'`` reproduces
        [1]_, it is the default.

        For a thin-walled open section, the torsional warping per unit twist
        rate about the shear centre, the first column of
        ``vlasov_section.warping([0, 1, 0, 0])``, approaches `-\omega`,
        with `\omega` the principal sectorial coordinate of Vlasov [2]_,
        normalised to zero average with the pole at the shear centre, and
        `K_{v,55}` approaches the warping rigidity `E \Gamma`.

        References
        ----------
        .. [1] Yu, W., Hodges, D. H., Volovoi, V. V., and Fuchs, E. D., "A
           generalized Vlasov theory for composite beams", Thin-Walled
           Structures, 43(9), pp. 1493--1511, 2005.
           https://doi.org/10.1016/j.tws.2005.02.003
        .. [2] Vlasov, V. Z., Thin-Walled Elastic Beams, 2nd ed., Israel
           Program for Scientific Translations, Jerusalem, 1961.

        """
        if (self.shear_centre is not None and self.constraint == constraint):
            e = self.shear_centre
        else:
            tmp = self.translated(0., 0.)
            tmp.calc_stiffness(constraint)
            e = tmp.shear_centre
        sec = self.translated(e[0], e[1])
        sec.calc_stiffness(constraint)
        Kv = np.zeros((5, 5), dtype=DOUBLE)
        Kv[:4, :4] = sec.A
        Kv[:4, 4] = sec.B[:, 1]
        Kv[4, :4] = sec.B[:, 1]
        Kv[4, 4] = sec.C[1, 1]
        self.Kv = Kv
        self.Fv = np.linalg.inv(Kv)
        self.vlasov_section = sec

    def calc_mass(CrossSection self):
        r"""Compute the mass properties of the cross section

        Integrates the densities of the materials over the cross section and
        sets ``mass_per_length``, ``mass_centre``, ``mass_matrix``,
        ``intrho``, ``intrhoy``, ``intrhoz``, ``intrhoy2``, ``intrhoz2``,
        ``intrhoyz``, ``area`` and ``centroid``. The mass centre is set to
        ``nan`` when the mass is zero.

        Notes
        -----
        The kinetic energy per unit length is Eq. (51) of [1]_,
        `\mathcal{K} = \frac{1}{2} (\mu \boldsymbol{V}^\top \boldsymbol{V} +
        2 \boldsymbol{\Omega}^\top \widetilde{\mu \bar{\xi}} \boldsymbol{V}
        + \boldsymbol{\Omega}^\top \boldsymbol{i} \boldsymbol{\Omega})`,
        with, Eq. (52) of [1]_, `\mu = \langle \rho \rangle`, `\mu
        \bar{\xi} = \langle \rho \xi \rangle`, `\boldsymbol{i} = \langle
        \rho (\xi^\top \xi \boldsymbol{I} - \xi \xi^\top) \rangle` and `\xi
        = \{0, x_2, x_3\}`, where the tilde is the skew-symmetric matrix of
        the cross product. The mass matrix is then

        .. math::
            \boldsymbol{M} = \begin{bmatrix} \mu \boldsymbol{I} &
            \widetilde{\mu \bar{\xi}}^\top \\ \widetilde{\mu \bar{\xi}} &
            \boldsymbol{i} \end{bmatrix}

        References
        ----------
        .. [1] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
           beam sectional analysis -- An updated version", International
           Journal of Engineering Science, 59, pp. 40--64, 2012.
           https://doi.org/10.1016/j.ijengsci.2012.03.006

        """
        cdef int ne, size, err
        cdef double [::1] rhoel
        ne = self.conn.shape[0]
        rhoel = np.ascontiguousarray(self.densities[self.mat_ids],
                                     dtype=DOUBLE)
        # NOTE the integrals are computed by the same kernel as the
        #      stiffness, whose matrices are discarded
        size = int(np.sum((3*self.nen)**2))
        rows = np.zeros(size, dtype=np.intc)
        cols = np.zeros(size, dtype=np.intc)
        v = np.zeros(size, dtype=DOUBLE)
        d = np.zeros((self.ndof, 4), dtype=DOUBLE)
        secint = np.zeros(12, dtype=DOUBLE)
        err = _integrate(self.x2, self.x3, self.conn, self.nen, self.Del,
                         rhoel, 0, rows, cols, v, v, v, d, d, d,
                         np.zeros((4, 4), dtype=DOUBLE), secint)
        if err >= 0:
            raise ValueError('element %d has a non-positive Jacobian '
                             'determinant, check that its nodes are '
                             'numbered counter-clockwise' % err)
        self.area = secint[0]
        self.centroid = np.array([secint[1], secint[2]])/secint[0]
        mu, m2, m3, m22, m33, m23 = secint[6:]
        self.intrho = mu
        self.intrhoy = m2
        self.intrhoz = m3
        self.intrhoy2 = m22
        self.intrhoz2 = m33
        self.intrhoyz = m23
        self.mass_per_length = mu
        if mu > 0:
            self.mass_centre = np.array([m2/mu, m3/mu])
        else:
            self.mass_centre = np.array([np.nan, np.nan])
        # NOTE skew-symmetric matrix of mu*xi_bar = {0, m2, m3}
        S = np.array([[0., -m3, m2],
                      [m3, 0., 0.],
                      [-m2, 0., 0.]])
        i = np.array([[m22 + m33, 0., 0.],
                      [0., m33, -m23],
                      [0., -m23, m22]])
        M = np.zeros((6, 6), dtype=DOUBLE)
        M[:3, :3] = mu*np.eye(3)
        M[3:, :3] = S
        M[:3, 3:] = S.T
        M[3:, 3:] = i
        self.mass_matrix = M

    def warping(CrossSection self, eps, deps=None, str convention='yu'):
        r"""Nodal warping displacements for given beam strains

        Combines the warping per unit strain stored by
        :meth:`calc_stiffness`, ``V0`` for the classical strains and ``V1S``
        for their derivatives along the beam, into the warping displacements
        `w_1, w_2, w_3` of every node, e.g. to plot the warping of a
        cross section with :meth:`triangles`.

        Parameters
        ----------
        eps : array-like
            Classical strains, size ``4``: `\{\gamma_{11}, \kappa_1,
            \kappa_2, \kappa_3\}` for ``convention='yu'``, or
            `\{\bar{\varepsilon}, \kappa_x, \kappa_y, \kappa_z\}` for
            ``convention='pyfe3d'``, see :func:`.yu_to_pyfe3d`; dimension
            inverse of length for the curvatures.
        deps : array-like, optional
            Derivatives of the classical strains along the beam axis `x_1`,
            size ``4``, same convention as ``eps``. Default ``None``,
            meaning zero, i.e. without the warping of the transverse shear
            of the generalized Timoshenko model.
        convention : str, optional
            ``'yu'``, the default, or ``'pyfe3d'``.

        Returns
        -------
        w : np.ndarray
            Warping displacements `w_1, w_2, w_3` of each node, shape
            ``(n_nodes, 3)``, dimension length; `w_1` is along the beam axis
            and `w_2`, `w_3` in the plane of the cross section.

        Raises
        ------
        RuntimeError
            If :meth:`calc_stiffness` has not been called.
        ValueError
            If ``eps`` or ``deps`` does not have size ``4``, or if
            ``convention`` is not ``'yu'`` or ``'pyfe3d'``.

        Examples
        --------
        Torsional warping per unit twist rate `\kappa_1 = 1` of a
        rectangle, which is antisymmetric about both axes:

        >>> import numpy as np
        >>> from pyfe3d.vabs import CrossSection, isotropic
        >>> x2, x3 = np.meshgrid([-1., 0., 1.], [-0.25, 0., 0.25])
        >>> elements = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        >>> sec = CrossSection(x2.ravel(), x3.ravel(), elements,
        ...                    materials=[isotropic(1., 0.3)], mat_ids=[0]*4)
        >>> sec.calc_stiffness()
        >>> w = sec.warping([0., 1., 0., 0.])
        >>> w.shape
        (9, 3)
        >>> bool(np.isclose(w[0, 0], -w[2, 0]))
        True

        """
        if self.V0 is None:
            raise RuntimeError('calc_stiffness must be called first')
        if convention == 'yu':
            T = np.eye(4)
        elif convention == 'pyfe3d':
            T, _ = _Ts(4, 1.)
        else:
            raise ValueError("convention must be 'yu' or 'pyfe3d'")
        eps = np.asarray(eps, dtype=DOUBLE).ravel()
        if eps.shape[0] != 4:
            raise ValueError('eps must have size 4')
        V = self.V0 @ (T @ eps)
        if deps is not None:
            deps = np.asarray(deps, dtype=DOUBLE).ravel()
            if deps.shape[0] != 4:
                raise ValueError('deps must have size 4')
            V = V + self.V1S @ (T @ deps)
        return V.reshape(-1, 3)

    def triangles(CrossSection self):
        r"""Linear triangles over all the nodes of the mesh, for plotting

        Splits every element into 3-node triangles whose vertices are its
        nodes, counter-clockwise, such that a field given at the nodes, e.g.
        a column of :meth:`warping`, can be drawn with any plotting library
        that accepts a triangulation. The bilinear quadrilateral gives two
        triangles; the 6-node triangle four, through its mid-side nodes; the
        9-node quadrilateral eight, two for each quarter around the centre
        node; and the 8-node quadrilateral six, one at each corner, cut
        through the two adjacent mid-side nodes, and two for the
        quadrilateral of the four mid-side nodes.

        Returns
        -------
        tri : np.ndarray
            Node indices of the triangles, shape ``(n_triangles, 3)``.

        Examples
        --------
        >>> import numpy as np
        >>> from pyfe3d.vabs import CrossSection, isotropic
        >>> x2, x3 = np.meshgrid([-0.5, 0., 0.5], [-0.5, 0., 0.5])
        >>> elements = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        >>> sec = CrossSection(x2.ravel(), x3.ravel(), elements,
        ...                    materials=[isotropic(1., 0.3)], mat_ids=[0]*4)
        >>> sec.triangles().shape
        (8, 3)

        """
        tri = []
        for e in range(self.conn.shape[0]):
            n = self.conn[e]
            for t in SUB_TRIANGLES[int(self.nen[e])]:
                tri.append([n[t[0]], n[t[1]], n[t[2]]])
        return np.array(tri, dtype=np.intc)


def _Ts(int n, double h):
    r"""Matrices `\boldsymbol{T}_s` and `\boldsymbol{H}` of
    :func:`.yu_to_pyfe3d`, of size ``n``, ``4`` or ``6``"""
    if n == 6:
        Ts = np.zeros((6, 6), dtype=DOUBLE)
        Ts[0, 0] = Ts[1, 1] = Ts[2, 2] = Ts[3, 3] = 1
        Ts[4, 5] = 1
        Ts[5, 4] = -1
        H = np.diag([1., 1., 1., h, h, h])
    elif n == 4:
        Ts = np.zeros((4, 4), dtype=DOUBLE)
        Ts[0, 0] = Ts[1, 1] = 1
        Ts[2, 3] = 1
        Ts[3, 2] = -1
        H = np.diag([1., h, h, h])
    else:
        raise ValueError('the matrix must have shape (4, 4) or (6, 6)')
    return Ts, H


def yu_to_pyfe3d(C, double h=1.):
    r"""Convert a sectional matrix from the convention of this module to
    that of pyfe3d

    The stiffness matrices of :class:`.CrossSection` refer to the strains
    `\{\gamma_{11}, 2\gamma_{12}, 2\gamma_{13}, \kappa_1, \kappa_2,
    \kappa_3\}`; those of the beam elements of pyfe3d to
    `\{\bar{\varepsilon}, \bar{\gamma}_y, \bar{\gamma}_z, \kappa_x,
    \kappa_y, \kappa_z\}`, work-conjugate to `\{N, V_y, V_z, M_x, M_z,
    M_y\}`. The fifth and sixth rows and columns are exchanged and the sign
    of every entry that couples `\kappa_3` with another strain is reversed.
    A classical matrix, of `\{\gamma_{11}, \kappa_1, \kappa_2, \kappa_3\}`,
    is converted to the order `\{\bar{\varepsilon}, \kappa_x, \kappa_y,
    \kappa_z\}` in the same way.

    Parameters
    ----------
    C : array-like
        Stiffness matrix in the convention of this module, shape ``(6, 6)``
        or ``(4, 4)``.
    h : float, optional
        Length that multiplies the curvatures in the strains to which ``C``
        refers, `\{\ldots, h\kappa_1, h\kappa_2, h\kappa_3\}`, dimension
        length. Default ``1.``, for the matrices of :class:`.CrossSection`,
        which refer to the curvatures themselves.

    Returns
    -------
    Cs : np.ndarray
        Stiffness matrix in the convention of pyfe3d, same shape as ``C``.

    Raises
    ------
    ValueError
        If ``C`` does not have the shape ``(6, 6)`` or ``(4, 4)``.

    See Also
    --------
    pyfe3d_to_yu : the inverse conversion.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfe3d.vabs import yu_to_pyfe3d
    >>> C = np.diag([1., 2., 3., 4., 5., 6.])
    >>> np.diag(yu_to_pyfe3d(C))
    array([1., 2., 3., 4., 6., 5.])

    Notes
    -----
    Without warping, the strains of the two conventions are related by
    `\boldsymbol{\epsilon} = \boldsymbol{H} \boldsymbol{T}_s
    \boldsymbol{\epsilon}_s`, with `\boldsymbol{H} = \mathrm{diag}(1, 1, 1,
    h, h, h)` and

    .. math::
        \boldsymbol{T}_s = \begin{bmatrix}
            \boldsymbol{I}_3 & \boldsymbol{0} \\
            \boldsymbol{0} & \begin{matrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\
            0 & -1 & 0 \end{matrix} \end{bmatrix}

    i.e. `\kappa_1 = \kappa_x`, `\kappa_2 = \kappa_z` and `\kappa_3 =
    -\kappa_y`, which follows from comparing the first row of
    `\boldsymbol{\Gamma}_\epsilon`, see :func:`._integrate`, with the axial
    strain `\bar{\varepsilon} + y \kappa_y + z \kappa_z` of :class:`.BeamC`,
    `y = x_2`, `z = x_3`. The energy is invariant, hence `\boldsymbol{C}_s
    = \boldsymbol{T}_s^\top \boldsymbol{H} \boldsymbol{C} \boldsymbol{H}
    \boldsymbol{T}_s`. For the classical matrices, the rows and columns of
    the shear strains are removed from `\boldsymbol{T}_s` and
    `\boldsymbol{H}`.

    """
    C = np.asarray(C, dtype=DOUBLE)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('the matrix must have shape (4, 4) or (6, 6)')
    Ts, H = _Ts(C.shape[0], h)
    return Ts.T @ H @ C @ H @ Ts


def pyfe3d_to_yu(Cs, double h=1.):
    r"""Convert a sectional matrix from the convention of pyfe3d to that of
    this module

    Inverse of :func:`.yu_to_pyfe3d`.

    Parameters
    ----------
    Cs : array-like
        Stiffness matrix in the convention of pyfe3d, shape ``(6, 6)`` or
        ``(4, 4)``.
    h : float, optional
        As in :func:`.yu_to_pyfe3d`, dimension length. Default ``1.``.

    Returns
    -------
    C : np.ndarray
        Stiffness matrix in the convention of this module, same shape as
        ``Cs``.

    Raises
    ------
    ValueError
        If ``Cs`` does not have the shape ``(6, 6)`` or ``(4, 4)``.

    See Also
    --------
    yu_to_pyfe3d : the direct conversion.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfe3d.vabs import pyfe3d_to_yu, yu_to_pyfe3d
    >>> C = np.diag([1., 2., 3., 4.])
    >>> np.allclose(pyfe3d_to_yu(yu_to_pyfe3d(C)), C)
    True

    Notes
    -----
    `\boldsymbol{C} = \boldsymbol{H}^{-1} \boldsymbol{T}_s \boldsymbol{C}_s
    \boldsymbol{T}_s^\top \boldsymbol{H}^{-1}`, with `\boldsymbol{T}_s`
    orthogonal, see :func:`.yu_to_pyfe3d`.

    """
    Cs = np.asarray(Cs, dtype=DOUBLE)
    if Cs.ndim != 2 or Cs.shape[0] != Cs.shape[1]:
        raise ValueError('the matrix must have shape (4, 4) or (6, 6)')
    Ts, H = _Ts(Cs.shape[0], h)
    Hinv = np.linalg.inv(H)
    return Hinv @ Ts @ Cs @ Ts.T @ Hinv


def classical_from_timoshenko(Kt):
    r"""Classical stiffness matrix from a generalized Timoshenko one

    Inverts the `6 \times 6` stiffness, deletes the rows and columns of the
    two transverse shear strains from the flexibility, and inverts the
    remaining `4 \times 4` matrix.

    Parameters
    ----------
    Kt : array-like
        Generalized Timoshenko stiffness matrix in the convention of this
        module, shape ``(6, 6)``.

    Returns
    -------
    Kc : np.ndarray
        Classical stiffness matrix of `\{\gamma_{11}, \kappa_1, \kappa_2,
        \kappa_3\}`, shape ``(4, 4)``.

    Raises
    ------
    ValueError
        If ``Kt`` does not have the shape ``(6, 6)``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfe3d.vabs import classical_from_timoshenko
    >>> Kc = classical_from_timoshenko(np.diag([1., 2., 3., 4., 5., 6.]))
    >>> np.diag(Kc)
    array([1., 4., 5., 6.])

    Notes
    -----
    This is the reduction prescribed on p. 52 of Yu and Hodges [1]_, a
    static condensation of the shear strains under zero shear forces. For
    the prismatic beams of :class:`.CrossSection` it returns ``Kc``, see
    :meth:`.CrossSection.calc_stiffness`.

    References
    ----------
    .. [1] Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
       Variational Asymptotic Beam Sectional Analysis", Journal of the
       American Helicopter Society, 50(1), pp. 46--55, 2005.
       https://doi.org/10.4050/1.3092842

    """
    Kt = np.asarray(Kt, dtype=DOUBLE)
    if Kt.shape != (6, 6):
        raise ValueError('Kt must have shape (6, 6)')
    F = np.linalg.inv(Kt)
    idx = [0, 3, 4, 5]
    return np.linalg.inv(F[np.ix_(idx, idx)])


def beamprop_from_vabs(Cs, double E, double rtol=1.e-6, CrossSection sec=None):
    r"""Create a :class:`.BeamProp` from a sectional stiffness matrix

    The beam elements of pyfe3d read 11 cross-section parameters of a
    :class:`.BeamProp`, which give a sectional stiffness matrix with a
    fixed pattern, see the Notes. This function sets them from the entries
    of ``Cs`` and checks that the matrix of the property reproduces ``Cs``.
    The integrals of the cross section are modulus-weighted: ``A`` is
    ``Cs[0, 0]/E``, ``Iyy`` is ``Cs[5, 5]/E``, etc., with ``G12``, ``G13``
    and ``GJ`` read from the shear and torsion entries.

    Parameters
    ----------
    Cs : array-like
        Generalized Timoshenko stiffness matrix in the convention of pyfe3d,
        e.g. ``yu_to_pyfe3d(sec.Kt)``, shape ``(6, 6)``.
    E : float
        Reference Young modulus stored in ``E`` of the property, force per
        unit area.
    rtol : float, optional
        Largest admissible value, for every pair ``i``, ``j``, of the
        difference between ``Cs`` and the matrix of the property divided by
        ``sqrt(Cs[i, i]*Cs[j, j])``, dimensionless. Default ``1.e-6``.
    sec : :class:`.CrossSection`, optional
        When given, the mass integrals ``intrho``, ``intrhoy``, ``intrhoz``,
        ``intrhoy2``, ``intrhoz2`` and ``intrhoyz`` are copied from it, so
        :meth:`.CrossSection.calc_mass` must have been called. Default
        ``None``, leaving them equal to zero.

    Returns
    -------
    prop : :class:`.BeamProp`
        The property, with ``A``, ``E``, ``G12``, ``G13``, ``GJ``, ``Ay``,
        ``Az``, ``Iyy``, ``Izz``, ``Iyz`` and, optionally, the mass
        integrals set.

    Raises
    ------
    ValueError
        If ``Cs`` does not have the shape ``(6, 6)``, if ``E`` or
        ``Cs[0, 0]`` is not positive, or if ``Cs`` has couplings that a
        :class:`.BeamProp` cannot represent, i.e. if the difference above
        exceeds ``rtol``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfe3d.vabs import beamprop_from_vabs
    >>> Cs = np.diag([7e9, 2e9, 2e9, 1e6, 6e6, 6e6])
    >>> prop = beamprop_from_vabs(Cs, E=70e9)
    >>> prop.A
    0.1

    Notes
    -----
    The strain energy of :class:`.BeamC` corresponds to the sectional
    stiffness matrix, in the order `\{\bar{\varepsilon}, \bar{\gamma}_y,
    \bar{\gamma}_z, \kappa_x, \kappa_y, \kappa_z\}`,

    .. math::
        \begin{bmatrix}
            E A & 0 & 0 & 0 & E A_y & E A_z \\
            0 & G_{12} A & 0 & -G_{12} A_z & 0 & 0 \\
            0 & 0 & G_{13} A & G_{13} A_y & 0 & 0 \\
            0 & -G_{12} A_z & G_{13} A_y & G J & 0 & 0 \\
            E A_y & 0 & 0 & 0 & E I_{zz} & E I_{yz} \\
            E A_z & 0 & 0 & 0 & E I_{yz} & E I_{yy}
        \end{bmatrix}

    with 11 parameters, against the 21 independent entries of a general
    `6 \times 6` matrix: the extension-shear, extension-twist, bending-shear
    and bending-twist couplings cannot be represented, and the shear-twist
    couplings are tied to the extension-bending ones, i.e. the shear centre
    of the property is its tension centre. This function does not change
    the element code; a section whose matrix does not have this pattern
    raises a ``ValueError``.

    """
    cdef BeamProp prop
    Cs = np.asarray(Cs, dtype=DOUBLE)
    if Cs.shape != (6, 6):
        raise ValueError('Cs must have shape (6, 6)')
    if E <= 0 or Cs[0, 0] <= 0:
        raise ValueError('E and Cs[0, 0] must be positive')
    prop = BeamProp()
    prop.E = E
    prop.A = Cs[0, 0]/E
    prop.Ay = Cs[0, 4]/E
    prop.Az = Cs[0, 5]/E
    prop.Izz = Cs[4, 4]/E
    prop.Iyz = Cs[4, 5]/E
    prop.Iyy = Cs[5, 5]/E
    prop.G12 = Cs[1, 1]/prop.A
    prop.G13 = Cs[2, 2]/prop.A
    prop.GJ = Cs[3, 3]
    Cp = np.zeros((6, 6), dtype=DOUBLE)
    Cp[0, 0] = E*prop.A
    Cp[0, 4] = Cp[4, 0] = E*prop.Ay
    Cp[0, 5] = Cp[5, 0] = E*prop.Az
    Cp[1, 1] = prop.G12*prop.A
    Cp[2, 2] = prop.G13*prop.A
    Cp[1, 3] = Cp[3, 1] = -prop.G12*prop.Az
    Cp[2, 3] = Cp[3, 2] = prop.G13*prop.Ay
    Cp[3, 3] = prop.GJ
    Cp[4, 4] = E*prop.Izz
    Cp[4, 5] = Cp[5, 4] = E*prop.Iyz
    Cp[5, 5] = E*prop.Iyy
    d = np.sqrt(np.abs(np.diag(Cs)))
    err = np.abs(Cs - Cp)/np.outer(d, d)
    if err.max() > rtol:
        i, j = np.unravel_index(np.argmax(err), err.shape)
        raise ValueError('the entry (%d, %d) of Cs cannot be represented by '
                         'a BeamProp, relative difference %g > rtol = %g'
                         % (i, j, err.max(), rtol))
    if sec is not None:
        prop.intrho = sec.intrho
        prop.intrhoy = sec.intrhoy
        prop.intrhoz = sec.intrhoz
        prop.intrhoy2 = sec.intrhoy2
        prop.intrhoz2 = sec.intrhoz2
        prop.intrhoyz = sec.intrhoyz
    return prop
