#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
================================================================
Co-rotational kinematics, EICR (:mod:`pyfe3d.corotational`)
================================================================

.. currentmodule:: pyfe3d.corotational

The element-independent co-rotational (EICR) formulation turns the linear
elements :class:`.Tria3R`, :class:`.Tria3DSG`, :class:`.Quad4`,
:class:`.Quad4R`, :class:`.BeamC`, :class:`.BeamLR`, :class:`.Truss` and
:class:`.Spring` into elements for large displacements and large rotations
with small strains, without changing them: the local linear stiffness
matrix of each element is taken once in its base configuration, and the
rigid motion of the element is removed from the nodal displacements and
rotations at every evaluation. With ``kinematics='corotational_von_karman'``
the local response is that of the von Kármán strains of the element
instead, evaluated by its own routines at the deformational displacements,
see :class:`CorotationalShell`; this changes the results of
:class:`.BeamC` only, since the von Kármán terms of the other elements
vanish in their co-rotational frame.

- :class:`CorotationalModel` assembles a model whose elements are added with
  ``kinematics='corotational'``, ``'corotational_von_karman'``,
  ``'von_karman'`` or ``'linear'``, and provides the callables
  ``calc_fext``, ``calc_fint``, ``calc_kC`` and ``calc_kG`` of
  :class:`pyfe3d.solver.Analysis`, which is used unchanged.
- :class:`CorotationalShell`, :class:`CorotationalBeam`,
  :class:`CorotationalTruss` and :class:`CorotationalSpring` wrap one
  element each.
- The finite-rotation utilities :func:`spin`, :func:`axial`,
  :func:`expso3`, :func:`logso3`, :func:`Lambda`, :func:`dLambdaT_m`,
  :func:`Hmat` and :func:`Lmat` are used by the wrappers and by the
  applied-moment helpers of :class:`CorotationalModel`.

The module is a Cython extension, as the elements: the kinematics, the
projector, and the internal force and tangent stiffness matrix of each
wrapper are computed by typed C kernels, of which the finite-rotation
utilities are the Python interface. The classes are Python classes, whose
attributes can be read and changed as usual.

Module constants:

.. data:: SERIES_ANGLE
    :value: 0.2

    Below this angle, in radians, the coefficients of the rotation matrices
    are evaluated with their Taylor series.

.. data:: SERIES_ANGLE_ETA_MU
    :value: 0.7

    Below this angle, in radians, the coefficients of :func:`Hmat` and
    :func:`Lmat` are evaluated with their Taylor series.

.. data:: WARP_RATIO_WARNING
    :value: 0.01

    Warp ratio of a quadrilateral above which :class:`CorotationalShell`
    warns with ``warping=None``, see its attribute ``warp_ratio``.

.. data:: WARP_RATIO_WARNING_OFFSETS
    :value: 0.07

    Warp ratio of a quadrilateral above which :class:`CorotationalShell`
    warns with ``warping='offsets'``.

The rotational degrees-of-freedom of the nodes are total rotation vectors,
`\boldsymbol{\psi}_a`, whose nodal rotation matrix is `\boldsymbol{R}_a =
\exp(\mathrm{Spin}(\boldsymbol{\psi}_a))`, and they are added by the solvers
like the translations. No state is kept between the calls. The map is
singular at `|\boldsymbol{\psi}_a| = 2 k \pi`, `k \ge 1`, which limits a
node to less than one full revolution.

Notes
-----
Notation: `\boldsymbol{T}_0` and `\boldsymbol{T}_R` are the base and the
current co-rotational (CR) frames of an element, as 3x3 matrices whose
columns are the element axes, the convention of the ``r11..r33`` of the
elements, such that global components are `\boldsymbol{T}` times local
components. Felippa and Haugen [1]_ write the transposes of these matrices;
Nour-Omid and Rankin [2]_ write `\boldsymbol{E}_0` and `\boldsymbol{E}`.

The spatial spin `\delta \boldsymbol{\omega}` of a node, `\delta
\boldsymbol{R} \boldsymbol{R}^T = \mathrm{Spin}(\delta \boldsymbol{\omega})`,
is related to the variation of the rotation vector by `\delta
\boldsymbol{\omega} = \boldsymbol{\Lambda}(\boldsymbol{\psi}) \delta
\boldsymbol{\psi}`, see :func:`Lambda`. This `\boldsymbol{\Lambda}` is the
`\boldsymbol{H}(\boldsymbol{\theta})` of Crisfield [3]_, Eq. (16.90), and
the `\boldsymbol{H}^{-1}` of [1]_, Eq. (100); here `\boldsymbol{H}` is
always the matrix of [1]_, Eq. (101), see :func:`Hmat`.

The internal force and the consistent tangent stiffness matrix of an
element are those of the consistent symmetrizable equilibrated (CSE)
formulation of [1]_, Section 6.3, Eq. (53), with the local linear stiffness
matrix of the element, written in terms of the spins, and transformed to
the additive rotation vectors with `\boldsymbol{\Lambda}` as in [3]_,
Section 17.4.2, Eqs. (17.173)-(17.177), see :class:`CorotationalShell`.

References
----------
.. [1] Felippa, C. A. and Haugen, B., A unified formulation of small-strain
   corotational finite elements: I. Theory, Computer Methods in Applied
   Mechanics and Engineering, 194(21-24), 2285-2335, 2005,
   https://doi.org/10.1016/j.cma.2004.07.035.
.. [2] Nour-Omid, B. and Rankin, C. C., Finite rotation analysis and
   consistent linearization using projectors, Computer Methods in Applied
   Mechanics and Engineering, 93(3), 353-384, 1991,
   https://doi.org/10.1016/0045-7825(91)90248-5.
.. [3] Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
   Structures, Volume 2: Advanced Topics, John Wiley & Sons, Chichester,
   1997.

"""
import ctypes
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from libc.math cimport sqrt, sin, cos, tan, atan2, fabs

from .tria3r import Tria3R, Tria3RData
from .quad4 import Quad4, Quad4Data
from .quad4r import Quad4R, Quad4RData
from .beamc import BeamC, BeamCData
from .tria3dsg import Tria3DSG, Tria3DSGData
from .beamlr import BeamLR, BeamLRData
from .spring import Spring, SpringData
from .truss import Truss, TrussData

# NOTE the same definitions as in pyfe3d/__init__.py, repeated here such that
#      this module does not depend on the import order of the package
DOF = 6
INT = np.int64 if ctypes.sizeof(ctypes.c_long) == 8 else np.int32

# NOTE the four constants below are documented in the module docstring,
#      since autodoc cannot read #: comments in an extension module
SERIES_ANGLE = 0.2
SERIES_ANGLE_ETA_MU = 0.7
WARP_RATIO_WARNING = 0.01
WARP_RATIO_WARNING_OFFSETS = 0.07

_I3 = np.eye(3)

# NOTE C copies of the module constants, read by the kernels below; the
#      Python constants above document them and are not read again
cdef int _DOF = 6
cdef double _SERIES_ANGLE = SERIES_ANGLE
cdef double _SERIES_ANGLE_ETA_MU = SERIES_ANGLE_ETA_MU
cdef double _PI = np.pi

# NOTE sizes of the largest element, 4 nodes and 24 degrees-of-freedom, for
#      the work arrays of the kernels
cdef enum:
    MAX_NN = 4
    MAX_N6 = 24
    MAX_NN6 = 576

# NOTE frame rules of the wrappers, see _CorotationalElement._frame_kind
cdef enum:
    FRAME_TRIA = 0          # Tria3R, Tria3DSG
    FRAME_QUAD = 1          # Quad4, Quad4R
    FRAME_MEAN_TRIAD = 2    # BeamC, BeamLR, Truss 'mean_triad', Spring 'axis'
    FRAME_SPRING_TRIAD = 3  # Spring 'triad'
    FRAME_CHORD = 4         # Truss 'chord'

# NOTE projection geometric stiffness: F&H Eq. (47), or the exact ones of
#      CorotationalBeam and of CorotationalSpring with frame='triad'
cdef enum:
    KGP_FH = 0
    KGP_BEAM = 1
    KGP_SPRING = 2


# ---------------------------------------------------------------------------
# C kernels of the 3x3 and small-matrix algebra
# ---------------------------------------------------------------------------
# NOTE all the matrices are stored row-major in C arrays: a 3x3 matrix M is
#      M[3*i + j], an n x m matrix A is A[m*i + j]. The kernels follow the
#      Python functions of the same name below and the steps of
#      _CorotationalElement.evaluate, operation by operation, such that the
#      results agree with the former pure-Python implementation to round-off
cdef inline double _dot3(const double *a, const double *b) noexcept nogil:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


cdef inline void _cross3(const double *a, const double *b,
                         double *c) noexcept nogil:
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]


cdef inline void _spin3(const double *v, double *S) noexcept nogil:
    S[0] = 0.
    S[1] = -v[2]
    S[2] = v[1]
    S[3] = v[2]
    S[4] = 0.
    S[5] = -v[0]
    S[6] = -v[1]
    S[7] = v[0]
    S[8] = 0.


cdef void _mm(const double *A, const double *B, double *C, int n, int k,
              int m) noexcept nogil:
    r"""``C = A @ B``, ``A`` of shape ``(n, k)``, ``B`` of shape ``(k, m)``"""
    cdef int i, j, l
    cdef double s
    for i in range(n):
        for j in range(m):
            s = 0.
            for l in range(k):
                s += A[k*i + l]*B[m*l + j]
            C[m*i + j] = s


cdef void _mtm(const double *A, const double *B, double *C, int n, int k,
               int m) noexcept nogil:
    r"""``C = A.T @ B``, ``A`` of shape ``(k, n)``, ``B`` of shape
    ``(k, m)``"""
    cdef int i, j, l
    cdef double s
    for i in range(n):
        for j in range(m):
            s = 0.
            for l in range(k):
                s += A[n*l + i]*B[m*l + j]
            C[m*i + j] = s


cdef inline void _mv3(const double *M, const double *v,
                      double *w) noexcept nogil:
    r"""``w = M @ v``, 3x3"""
    w[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2]
    w[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2]
    w[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2]


cdef inline void _mtv3(const double *M, const double *v,
                       double *w) noexcept nogil:
    r"""``w = M.T @ v``, 3x3"""
    w[0] = M[0]*v[0] + M[3]*v[1] + M[6]*v[2]
    w[1] = M[1]*v[0] + M[4]*v[1] + M[7]*v[2]
    w[2] = M[2]*v[0] + M[5]*v[1] + M[8]*v[2]


cdef void _coefficients_c(double t, double *c) noexcept nogil:
    r"""``sinc, a, b, a'/t, b'/t`` of :func:`_coefficients` in ``c[0:5]``"""
    cdef double t2, s, co
    if t < _SERIES_ANGLE:
        t2 = t*t
        c[0] = 1. + t2*(-1./6 + t2*(1./120 + t2*(-1./5040 + t2/362880.)))
        c[1] = 0.5 + t2*(-1./24 + t2*(1./720 + t2*(-1./40320
                                                   + t2/3628800.)))
        c[2] = 1./6 + t2*(-1./120 + t2*(1./5040 + t2*(-1./362880
                                                      + t2/39916800.)))
        c[3] = -1./12 + t2*(1./180 + t2*(-1./6720 + t2*(1./453600
                                                        - t2/47900160.)))
        c[4] = -1./60 + t2*(1./1260 + t2*(-1./60480 + t2*(1./4989600
                                                          - t2/622702080.)))
        return
    s = sin(t)
    co = cos(t)
    c[0] = s/t
    c[1] = (1. - co)/t**2
    c[2] = (t - s)/t**3
    c[3] = (t*s - 2.*(1. - co))/t**4
    c[4] = (3.*s - 2.*t - t*co)/t**5


cdef void _eta_mu_c(double t, double *eta, double *mu) noexcept nogil:
    r"""`\eta` and `\mu` of :func:`_eta_mu`"""
    cdef double t2
    if t < _SERIES_ANGLE_ETA_MU:
        t2 = t*t
        eta[0] = 1./12 + t2*(1./720 + t2*(1./30240 + t2*(1./1209600 + t2*(
            1./47900160 + t2*(691./1307674368000. + t2*(1./74724249600.
            + t2*3617./10670622842880000.))))))
        mu[0] = 1./360 + t2*(1./7560 + t2*(1./201600 + t2*(1./5987520 + t2*(
            691./130767436800. + t2*(1./6227020800. + t2*(
                3617./762187345920000.
                + t2*43867./319318388573184000.))))))
        return
    eta[0] = (1. - 0.5*t/tan(0.5*t))/t**2
    mu[0] = ((t**2 + 4.*cos(t) + t*sin(t) - 4.)
             /(4.*t**4*sin(0.5*t)**2))


cdef void _expso3_c(const double *psi, double *R) noexcept nogil:
    r"""Rotation matrix of :func:`expso3`"""
    cdef double c[5]
    cdef double S[9]
    cdef double SS[9]
    cdef int i
    _coefficients_c(sqrt(_dot3(psi, psi)), c)
    _spin3(psi, S)
    _mm(S, S, SS, 3, 3, 3)
    for i in range(9):
        R[i] = c[0]*S[i] + c[1]*SS[i]
    R[0] += 1.
    R[4] += 1.
    R[8] += 1.


cdef void _logso3_c(const double *R, double *theta) noexcept nogil:
    r"""Rotation vector of :func:`logso3`, Spurrier's algorithm"""
    cdef double q[4]
    cdef double tr, f, qi, s, Rii, fac
    cdef int i, j, k
    tr = R[0] + R[4] + R[8]
    # NOTE the first of the largest diagonal entries, as numpy.argmax
    i = 0
    if R[4] > R[0]:
        i = 1
    if R[8] > R[3*i + i]:
        i = 2
    Rii = R[3*i + i]
    if tr >= Rii:
        q[0] = 0.5*sqrt(1. + tr)
        f = 0.25/q[0]
        q[1] = (R[7] - R[5])*f
        q[2] = (R[2] - R[6])*f
        q[3] = (R[3] - R[1])*f
    else:
        j = (i + 1) % 3
        k = (i + 2) % 3
        qi = sqrt(0.5*Rii + 0.25*(1. - tr))
        f = 0.25/qi
        q[i + 1] = qi
        q[0] = (R[3*k + j] - R[3*j + k])*f
        q[j + 1] = (R[3*j + i] + R[3*i + j])*f
        q[k + 1] = (R[3*k + i] + R[3*i + k])*f
    if q[0] < 0:
        for i in range(4):
            q[i] = -q[i]
    s = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if s < 1e-14:
        for i in range(3):
            theta[i] = 2.*q[i + 1]/q[0]
        return
    fac = 2.*atan2(s, q[0])/s
    for i in range(3):
        theta[i] = fac*q[i + 1]


cdef void _Lambda_c(const double *psi, double *L) noexcept nogil:
    r"""`\boldsymbol{\Lambda}` of :func:`Lambda`"""
    cdef double c[5]
    cdef double S[9]
    cdef double SS[9]
    cdef int i
    _coefficients_c(sqrt(_dot3(psi, psi)), c)
    _spin3(psi, S)
    _mm(S, S, SS, 3, 3, 3)
    for i in range(9):
        L[i] = c[1]*S[i] + c[2]*SS[i]
    L[0] += 1.
    L[4] += 1.
    L[8] += 1.


cdef void _dLambdaT_m_c(const double *psi, const double *m,
                        double *K) noexcept nogil:
    r"""Derivative of :func:`dLambdaT_m`"""
    cdef double c[5]
    cdef double Sm[9]
    cdef double pxm[3]
    cdef double ppxm[3]
    cdef double pm
    cdef int i, j
    _coefficients_c(sqrt(_dot3(psi, psi)), c)
    _cross3(psi, m, pxm)
    _cross3(psi, pxm, ppxm)
    _spin3(m, Sm)
    pm = _dot3(psi, m)
    for i in range(3):
        for j in range(3):
            K[3*i + j] = (c[1]*Sm[3*i + j] - c[3]*pxm[i]*psi[j]
                          + c[2]*(psi[i]*m[j] + (pm if i == j else 0.)
                                  - 2.*m[i]*psi[j])
                          + c[4]*ppxm[i]*psi[j])


cdef void _Hmat_c(const double *theta, double *H) noexcept nogil:
    r"""`\boldsymbol{H}` of :func:`Hmat`"""
    cdef double eta, mu
    cdef double S[9]
    cdef double SS[9]
    cdef int i
    _eta_mu_c(sqrt(_dot3(theta, theta)), &eta, &mu)
    _spin3(theta, S)
    _mm(S, S, SS, 3, 3, 3)
    for i in range(9):
        H[i] = -0.5*S[i] + eta*SS[i]
    H[0] += 1.
    H[4] += 1.
    H[8] += 1.


cdef void _Lmat_c(const double *theta, const double *m,
                  double *L) noexcept nogil:
    r"""`\boldsymbol{L}` of :func:`Lmat`"""
    cdef double eta, mu, tm
    cdef double S[9]
    cdef double SS[9]
    cdef double Sm[9]
    cdef double H[9]
    cdef double dHtm[9]
    cdef double SSm[3]
    cdef int i, j
    _eta_mu_c(sqrt(_dot3(theta, theta)), &eta, &mu)
    _spin3(theta, S)
    _mm(S, S, SS, 3, 3, 3)
    _spin3(m, Sm)
    _mv3(SS, m, SSm)
    tm = _dot3(theta, m)
    for i in range(3):
        for j in range(3):
            dHtm[3*i + j] = (eta*((tm if i == j else 0.) + theta[i]*m[j]
                                  - 2.*m[i]*theta[j])
                             + mu*SSm[i]*theta[j] - 0.5*Sm[3*i + j])
    for i in range(9):
        H[i] = -0.5*S[i] + eta*SS[i]
    H[0] += 1.
    H[4] += 1.
    H[8] += 1.
    _mm(dHtm, H, L, 3, 3, 3)


cdef inline void _columns_frame(const double *e1, const double *e3,
                                double *TR) noexcept nogil:
    r"""``TR = [e1, e3 x e1, e3]``, columns = element axes"""
    cdef double e2[3]
    cdef int i
    _cross3(e3, e1, e2)
    for i in range(3):
        TR[3*i] = e1[i]
        TR[3*i + 1] = e2[i]
        TR[3*i + 2] = e3[i]


cdef void _minimal_rotation_c(const double *a, const double *b,
                              double *R) noexcept nogil:
    r"""Rotation of the smallest angle that maps the unit vector ``a`` onto
    the unit vector ``b``"""
    cdef double axis[3]
    cdef double n[3]
    cdef double ek[3]
    cdef double v[3]
    cdef double s, c, nn, ang
    cdef int i, k
    _cross3(a, b, axis)
    s = sqrt(_dot3(axis, axis))
    c = _dot3(a, b)
    if s < 1e-14:
        if c > 0:
            for i in range(9):
                R[i] = 0.
            R[0] = R[4] = R[8] = 1.
            return
        # NOTE about an axis normal to a, as numpy.argmin(abs(a))
        k = 0
        if fabs(a[1]) < fabs(a[0]):
            k = 1
        if fabs(a[2]) < fabs(a[k]):
            k = 2
        ek[0] = ek[1] = ek[2] = 0.
        ek[k] = 1.
        _cross3(a, ek, n)
        nn = sqrt(_dot3(n, n))
        for i in range(3):
            v[i] = _PI*n[i]/nn
        _expso3_c(v, R)
        return
    ang = atan2(s, c)
    for i in range(3):
        v[i] = axis[i]/s*ang
    _expso3_c(v, R)


cdef int _frame_c(int kind, const double *XD, const double *Rs,
                  const double *T0, const double *q0, const double *p0,
                  double q2_min, double *TR, double *aux,
                  double *val) noexcept nogil:
    r"""CR frame ``TR`` of the frame rule ``kind`` at the positions ``XD``
    (rows) and the nodal rotations ``Rs`` (3x3 each), see the ``_frame`` of
    the wrappers

    ``aux``: `\bar{\boldsymbol{q}}_i` (FRAME_MEAN_TRIAD) or
    `\bar{\boldsymbol{p}}_i, \bar{\boldsymbol{q}}_i` (FRAME_SPRING_TRIAD),
    rows. Returns 1 when the frame is undefined, with the values of the
    warning in ``val``, 0 otherwise."""
    cdef double d1[3]
    cdef double d2[3]
    cdef double n[3]
    cdef double e1[3]
    cdef double e3[3]
    cdef double c[3]
    cdef double qs[6]
    cdef double ps[6]
    cdef double q[3]
    cdef double w[3]
    cdef double t1[3]
    cdef double Rm[9]
    cdef double nd, A, B2
    cdef int i, a
    if kind == FRAME_TRIA:
        # NOTE side 1-2 alignment, as Tria3R.update_rotation_matrix
        for i in range(3):
            d1[i] = XD[3 + i] - XD[i]
            d2[i] = XD[6 + i] - XD[i]
        nd = sqrt(_dot3(d1, d1))
        for i in range(3):
            e1[i] = d1[i]/nd
        _cross3(d1, d2, n)
        nd = sqrt(_dot3(n, n))
        for i in range(3):
            e3[i] = n[i]/nd
        _columns_frame(e1, e3, TR)
        return 0
    if kind == FRAME_QUAD:
        # NOTE z along v42 x v13, x along (v13 + v42)/2, as
        #      Quad4R.update_rotation_matrix
        for i in range(3):
            d1[i] = XD[6 + i] - XD[i]
            d2[i] = XD[3 + i] - XD[9 + i]
        _cross3(d2, d1, n)
        nd = sqrt(_dot3(n, n))
        for i in range(3):
            e3[i] = n[i]/nd
            c[i] = 0.5*(d1[i] + d2[i])
        nd = sqrt(_dot3(c, c))
        for i in range(3):
            e1[i] = c[i]/nd
        _columns_frame(e1, e3, TR)
        return 0
    if kind == FRAME_MEAN_TRIAD:
        # NOTE mean nodal triad, see CorotationalBeam
        for i in range(3):
            d1[i] = XD[3 + i] - XD[i]
        nd = sqrt(_dot3(d1, d1))
        for i in range(3):
            e1[i] = d1[i]/nd
        for a in range(2):
            _mv3(&Rs[9*a], q0, &qs[3*a])
        for i in range(3):
            q[i] = 0.5*(qs[i] + qs[3 + i])
        _cross3(e1, q, w)
        nd = sqrt(_dot3(w, w))
        if not nd >= q2_min:
            val[0] = nd
            return 1
        for i in range(3):
            e3[i] = w[i]/nd
        _columns_frame(e1, e3, TR)
        for a in range(2):
            _mtv3(TR, &qs[3*a], &aux[3*a])
        return 0
    if kind == FRAME_SPRING_TRIAD:
        # NOTE mean of the nodal x and y axes, see CorotationalSpring
        for a in range(2):
            _mv3(&Rs[9*a], p0, &ps[3*a])
            _mv3(&Rs[9*a], q0, &qs[3*a])
        for i in range(3):
            c[i] = 0.5*(ps[i] + ps[3 + i])
            q[i] = 0.5*(qs[i] + qs[3 + i])
        A = sqrt(_dot3(c, c))
        for i in range(3):
            e1[i] = c[i]/A if A > 0. else c[i]
        _cross3(e1, q, w)
        B2 = sqrt(_dot3(w, w))
        if not (A >= q2_min and B2 >= q2_min):
            val[0] = A
            val[1] = B2
            return 1
        for i in range(3):
            e3[i] = w[i]/B2
        _columns_frame(e1, e3, TR)
        for a in range(2):
            _mtv3(TR, &ps[3*a], &aux[3*a])
            _mtv3(TR, &qs[3*a], &aux[6 + 3*a])
        return 0
    # NOTE FRAME_CHORD, for the post-processing only: the base frame turned
    #      by the smallest rotation that aligns x with the axis
    for i in range(3):
        d1[i] = XD[3 + i] - XD[i]
        t1[i] = T0[3*i]
    nd = sqrt(_dot3(d1, d1))
    for i in range(3):
        e1[i] = d1[i]/nd
    _minimal_rotation_c(t1, e1, Rm)
    _mm(Rm, T0, TR, 3, 3, 3)
    return 0


cdef int _kinematics_c(int kind, int nn, const double *X0c,
                       const double *ue, const double *psie,
                       const double *e, const double *xt, const double *T0,
                       const double *q0, const double *p0, double q2_min,
                       double *Rs, double *TR, double *xl, double *d,
                       double *aux, double *val) noexcept nogil:
    r"""Steps of :meth:`_CorotationalElement._kinematics`: nodal rotations
    ``Rs``, CR frame ``TR``, current local coordinates ``xl`` and
    deformational displacements ``d``; ``e`` is NULL without rigid
    offsets. Returns 1 when the frame is undefined."""
    cdef double XD[3*MAX_NN]
    cdef double mean[3]
    cdef double r[3]
    cdef double M1[9]
    cdef double M2[9]
    cdef int a, i, j
    # NOTE nodal positions relative to the base centroid, exact since
    #      the frame and xl use differences only: with X0 + ue the
    #      round-off of eps*|X0| depends on the origin of the coordinates
    for a in range(nn):
        for i in range(3):
            XD[3*a + i] = X0c[3*a + i] + ue[3*a + i]
        _expso3_c(&psie[3*a], &Rs[9*a])
    if e != NULL:
        # NOTE nodes of the flat element, at the offsets R_a e_a that
        #      rotate with the nodal triads, see CorotationalShell
        for a in range(nn):
            _mv3(&Rs[9*a], &e[3*a], r)
            for i in range(3):
                XD[3*a + i] = XD[3*a + i] - r[i]
    if _frame_c(kind, XD, Rs, T0, q0, p0, q2_min, TR, aux, val):
        return 1
    for i in range(3):
        mean[i] = 0.
        for a in range(nn):
            mean[i] += XD[3*a + i]
        mean[i] = mean[i]/nn
    for a in range(nn):
        for i in range(3):
            r[i] = XD[3*a + i] - mean[i]
        _mtv3(TR, r, &xl[3*a])
        for i in range(3):
            d[6*a + i] = xl[3*a + i] - xt[3*a + i]
        # NOTE F&H Eq. (26) written with column frames; N-O&R Eq. (14)
        _mtm(TR, &Rs[9*a], M1, 3, 3, 3)
        _mm(M1, T0, M2, 3, 3, 3)
        _logso3_c(M2, &d[6*a + 3])
    if kind == FRAME_CHORD:
        # NOTE the chord does not define the nodal rotations about it
        for a in range(nn):
            for i in range(3):
                d[6*a + 3 + i] = 0.
    return 0


cdef void _G_c(int kind, int nn, const double *xl, const double *aux,
               double *G) noexcept nogil:
    r"""Spin-fitter matrix ``G``, shape ``(3, 6*nn)``, see the Notes of the
    wrappers"""
    cdef int n6 = 6*nn
    cdef int i, j
    cdef double a1, a2, b1, b2, n, c, l, q1, q2, A, B1, B2
    cdef double s1p[3]
    cdef double s2p[3]
    cdef double s2q[3]
    cdef const double *pb
    cdef const double *qb
    for i in range(3*n6):
        G[i] = 0.
    if kind == FRAME_TRIA:
        a1 = xl[3] - xl[0]
        a2 = xl[6] - xl[0]
        b2 = xl[7] - xl[1]
        n = a1*b2
        # NOTE w_a at 6*a + 2, v_a at 6*a + 1
        G[2] = (a2 - a1)/n
        G[8] = -a2/n
        G[14] = a1/n
        G[n6 + 2] = 1./a1
        G[n6 + 8] = -1./a1
        G[2*n6 + 1] = -1./a1
        G[2*n6 + 7] = 1./a1
    elif kind == FRAME_QUAD:
        a1 = xl[6] - xl[0]
        b1 = xl[7] - xl[1]
        a2 = xl[3] - xl[9]
        b2 = xl[4] - xl[10]
        n = a2*b1 - a1*b2
        c = 0.5*sqrt((a1 + a2)**2 + (b1 + b2)**2)
        # NOTE w_a at 2, 8, 14, 20, v_a at 1, 7, 13, 19
        G[8] = -a1/n
        G[20] = a1/n
        G[14] = a2/n
        G[2] = -a2/n
        G[n6 + 14] = b2/n
        G[n6 + 2] = -b2/n
        G[n6 + 8] = -b1/n
        G[n6 + 20] = b1/n
        G[2*n6 + 13] = 0.5/c
        G[2*n6 + 1] = -0.5/c
        G[2*n6 + 7] = 0.5/c
        G[2*n6 + 19] = -0.5/c
    elif kind == FRAME_MEAN_TRIAD:
        qb = aux
        l = xl[3] - xl[0]
        q1 = 0.5*(qb[0] + qb[3])
        q2 = 0.5*(qb[1] + qb[4])
        for i in range(2):
            G[6*i + 3] = 0.5*qb[3*i + 1]/q2
            G[6*i + 4] = -0.5*qb[3*i]/q2
        G[2] = q1/(q2*l)
        G[8] = -q1/(q2*l)
        G[n6 + 2] = 1./l
        G[n6 + 8] = -1./l
        G[2*n6 + 1] = -1./l
        G[2*n6 + 7] = 1./l
    elif kind == FRAME_SPRING_TRIAD:
        pb = aux
        qb = &aux[6]
        A = 0.5*(pb[0] + pb[3])
        B1 = 0.5*(qb[0] + qb[3])
        B2 = 0.5*(qb[1] + qb[4])
        for i in range(2):
            # NOTE rows 2 and 3 of Spin(pb_i) and row 3 of Spin(qb_i)
            s1p[0] = pb[3*i + 2]
            s1p[1] = 0.
            s1p[2] = -pb[3*i]
            s2p[0] = -pb[3*i + 1]
            s2p[1] = pb[3*i]
            s2p[2] = 0.
            s2q[0] = -qb[3*i + 1]
            s2q[1] = qb[3*i]
            s2q[2] = 0.
            for j in range(3):
                G[2*n6 + 6*i + 3 + j] = -0.5*s1p[j]/A
                G[n6 + 6*i + 3 + j] = 0.5*s2p[j]/A
                G[6*i + 3 + j] = (B1*G[n6 + 6*i + 3 + j] - 0.5*s2q[j])/B2


cdef void _KGP_FH_c(int nn, const double *G, const double *P,
                    const double *pv, double *KGP) noexcept nogil:
    r"""``-G.T @ (Fn.T @ P)``, with `\boldsymbol{F}_n` of the forces of
    ``pv``, F&H Eq. (47)"""
    cdef int n6 = 6*nn
    cdef double Fn[3*MAX_N6]
    cdef double FnTP[3*MAX_N6]
    cdef int a, i
    for i in range(3*n6):
        Fn[i] = 0.
    for a in range(nn):
        _spin3(&pv[6*a], &Fn[18*a])
    _mtm(Fn, P, FnTP, 3, n6, n6)
    _mtm(G, FnTP, KGP, n6, 3, n6)
    for i in range(n6*n6):
        KGP[i] = -KGP[i]


cdef void _KGP_beam_c(const double *G, const double *P, const double *p,
                      const double *z, const double *xl, const double *aux,
                      double *KGP) noexcept nogil:
    r"""Exact projection geometric stiffness of :class:`CorotationalBeam`
    (derived here)"""
    cdef const double *qb = aux
    cdef double GE[36]
    cdef double Sq[9]
    cdef double Q[72]
    cdef double Qm[36]
    cdef double dl[12]
    cdef double dr[12]
    cdef double D[144]
    cdef double l, q1, q2, r
    cdef int i, k, c
    l = xl[3] - xl[0]
    q1 = 0.5*(qb[0] + qb[3])
    q2 = 0.5*(qb[1] + qb[4])
    r = q1/q2
    # NOTE variations of the arguments of G (derived here)
    for i in range(2):
        for k in range(36):
            GE[k] = G[k]
        for k in range(3):
            GE[12*k + 6*i + 3 + k] -= 1.
        _spin3(&qb[3*i], Sq)
        _mm(Sq, GE, &Q[36*i], 3, 3, 12)
    for k in range(36):
        Qm[k] = 0.5*(Q[k] + Q[36 + k])
    for c in range(12):
        dl[c] = 0.
    dl[0] = -1.
    dl[6] = 1.
    for c in range(12):
        dr[c] = (Qm[c]*q2 - q1*Qm[12 + c])/q2**2
    for k in range(144):
        D[k] = 0.
    for c in range(12):
        D[24 + c] = z[0]*(dr[c]/l - r*dl[c]/l**2) - z[1]*dl[c]/l**2
        D[96 + c] = -D[24 + c]
        D[12 + c] = z[2]*dl[c]/l**2
        D[84 + c] = -D[12 + c]
        for i in range(2):
            D[12*(6*i + 3) + c] = (0.5*z[0]*(Q[36*i + 12 + c]*q2
                                             - qb[3*i + 1]*Qm[12 + c])/q2**2)
            D[12*(6*i + 4) + c] = (-0.5*z[0]*(Q[36*i + c]*q2
                                              - qb[3*i]*Qm[12 + c])/q2**2)
    _KGP_FH_c(2, G, P, p, KGP)
    for k in range(144):
        KGP[k] = -D[k] + KGP[k]


cdef void _KGP_spring_c(const double *G, const double *P, const double *p,
                        const double *z, const double *aux,
                        double *KGP) noexcept nogil:
    r"""Exact projection geometric stiffness of :class:`CorotationalSpring`
    with ``frame='triad'`` (derived here)"""
    cdef const double *pb = aux
    cdef const double *qb = &aux[6]
    cdef double GE[36]
    cdef double Sx[9]
    cdef double Qp[72]
    cdef double Qq[72]
    cdef double D[144]
    cdef double Di[36]
    cdef double T36[36]
    cdef double dcp[9]
    cdef double dcq[9]
    cdef double s1p[3]
    cdef double s2p[3]
    cdef double s2q[3]
    cdef double ui[3]
    cdef double dalpha[3]
    # NOTE derivatives of the rows 2 and 3 of Spin(p) with respect to p
    cdef double S1[9]
    cdef double S2[9]
    cdef double A, B1, B2, alpha
    cdef int i, j, k, m
    S1[:] = [0., 0., 1., 0., 0., 0., -1., 0., 0.]
    S2[:] = [0., -1., 0., 1., 0., 0., 0., 0., 0.]
    A = 0.5*(pb[0] + pb[3])
    B1 = 0.5*(qb[0] + qb[3])
    B2 = 0.5*(qb[1] + qb[4])
    # NOTE exact variation of G^T z (derived here), see the Notes
    alpha = z[1] + z[0]*B1/B2
    dalpha[0] = z[0]*(0.5*1./B2 - 0.5*B1*0./B2**2)
    dalpha[1] = z[0]*(0.5*0./B2 - 0.5*B1*1./B2**2)
    dalpha[2] = z[0]*(0.5*0./B2 - 0.5*B1*0./B2**2)
    for j in range(2):
        for k in range(36):
            GE[k] = G[k]
        for k in range(3):
            GE[12*k + 6*j + 3 + k] -= 1.
        _spin3(&pb[3*j], Sx)
        _mm(Sx, GE, &Qp[36*j], 3, 3, 12)
        _spin3(&qb[3*j], Sx)
        _mm(Sx, GE, &Qq[36*j], 3, 3, 12)
    for k in range(144):
        D[k] = 0.
    for i in range(2):
        s1p[0] = pb[3*i + 2]
        s1p[1] = 0.
        s1p[2] = -pb[3*i]
        s2p[0] = -pb[3*i + 1]
        s2p[1] = pb[3*i]
        s2p[2] = 0.
        s2q[0] = -qb[3*i + 1]
        s2q[1] = qb[3*i]
        s2q[2] = 0.
        for k in range(3):
            ui[k] = 0.5/A*(-z[2]*s1p[k] + alpha*s2p[k])
        for k in range(36):
            Di[k] = 0.
        for j in range(2):
            for k in range(3):
                for m in range(3):
                    dcp[3*k + m] = -(ui[k]*(0.5 if m == 0 else 0.))/A
                    dcq[3*k + m] = (0.5/A*s2p[k]*dalpha[m]
                                    + z[0]*s2q[k]/(2*B2**2)
                                    *(0.5 if m == 1 else 0.))
            if i == j:
                for k in range(9):
                    dcp[k] = dcp[k] + 0.5/A*(-z[2]*S1[k] + alpha*S2[k])
                    dcq[k] = dcq[k] - z[0]/(2*B2)*S2[k]
            _mm(dcp, &Qp[36*j], T36, 3, 3, 12)
            for k in range(36):
                Di[k] += T36[k]
            _mm(dcq, &Qq[36*j], T36, 3, 3, 12)
            for k in range(36):
                Di[k] += T36[k]
        for k in range(3):
            for m in range(12):
                D[12*(6*i + 3 + k) + m] = Di[12*k + m]
    _KGP_FH_c(2, G, P, p, KGP)
    for k in range(144):
        KGP[k] = -D[k] + KGP[k]


cdef void _btkb_c(int nn, const double *TR, const double *BR,
                  const double *K, double *out) noexcept nogil:
    r"""``out = B.T @ K @ B`` with ``B = blockdiag(TR.T, BR_a)`` of 3x3
    blocks, ``BR_a = TR.T @ Lambda_a``"""
    cdef int n6 = 6*nn
    cdef int nb = 2*nn
    cdef int kb, lb, i, j, m
    cdef double TRT[9]
    cdef double Kb[9]
    cdef double T1[9]
    cdef double T2[9]
    cdef const double *Bk
    cdef const double *Bl
    for i in range(3):
        for j in range(3):
            TRT[3*i + j] = TR[3*j + i]
    for kb in range(nb):
        if kb % 2 == 0:
            Bk = TRT
        else:
            Bk = &BR[9*(kb//2)]
        for lb in range(nb):
            if lb % 2 == 0:
                Bl = TRT
            else:
                Bl = &BR[9*(lb//2)]
            for i in range(3):
                for j in range(3):
                    Kb[3*i + j] = K[n6*(3*kb + i) + 3*lb + j]
            _mtm(Bk, Kb, T1, 3, 3, 3)
            _mm(T1, Bl, T2, 3, 3, 3)
            for i in range(3):
                for j in range(3):
                    out[n6*(3*kb + i) + 3*lb + j] = T2[3*i + j]


cdef void _force_tangent_c(int nn, int kgp, int tangent,
                           const double *fbar, const double *KCl,
                           const double *KGl, const double *Pu,
                           const double *TR, const double *xl,
                           const double *d, const double *aux,
                           const double *psie, const double *Rs,
                           const double *e, const double *G,
                           double *f, double *KMpsi,
                           double *KGpsi) noexcept nogil:
    r"""Steps of :meth:`_CorotationalElement.evaluate` after the local
    internal force ``fbar`` and the local tangents ``KCl``, ``KGl`` (NULL
    when absent): CSE force and tangent of F&H Section 6.3, transformed to
    the additive rotation vectors and, with ``e`` not NULL, by the rigid
    offsets"""
    cdef int n6 = 6*nn
    cdef double Hs[9*MAX_NN]
    cdef double Lams[9*MAX_NN]
    cdef double BR[9*MAX_NN]
    cdef double Jr[9*MAX_NN]
    cdef double rs[3*MAX_NN]
    cdef double p[MAX_N6]
    cdef double pbar[MAX_N6]
    cdef double f_flat[MAX_N6]
    cdef double S[3*MAX_N6]
    cdef double SG[MAX_NN6]
    cdef double P[MAX_NN6]
    cdef double HP[MAX_NN6]
    cdef double T[MAX_NN6]
    cdef double KM[MAX_NN6]
    cdef double KG[MAX_NN6]
    cdef double J[MAX_NN6]
    cdef double Fnm[3*MAX_N6]
    cdef double T3n[3*MAX_N6]
    cdef double z[3]
    cdef double v3[3]
    cdef double w3[3]
    cdef double Lm[9]
    cdef double K3[9]
    cdef double M1[9]
    cdef double M2[9]
    cdef double Sf[9]
    cdef double Sr[9]
    cdef double TRT[9]
    cdef int a, i, j, k, r0
    for a in range(nn):
        _Hmat_c(&d[6*a + 3], &Hs[9*a])
    for i in range(n6):
        p[i] = fbar[i]
    for a in range(nn):
        _mtv3(&Hs[9*a], &fbar[6*a + 3], &p[6*a + 3])
    # NOTE rigid-rotation matrix S, F&H Eq. (12)
    for i in range(3*n6):
        S[i] = 0.
    for a in range(nn):
        _spin3(&xl[3*a], M1)
        for i in range(3):
            for j in range(3):
                S[3*(6*a + i) + j] = -M1[3*i + j]
            S[3*(6*a + 3 + i) + i] = 1.
    _mm(S, G, SG, n6, 3, n6)
    for i in range(n6*n6):
        P[i] = Pu[i] - SG[i]
    # NOTE projected force, F&H Eqs. (31)-(33)
    _mtm(S, p, z, 3, n6, 1)
    _mtm(Pu, p, pbar, n6, n6, 1)
    _mtm(G, z, T3n, n6, 3, 1)
    for i in range(n6):
        pbar[i] = pbar[i] - T3n[i]
    # NOTE B = blockdiag(T_R^T, T_R^T Lambda_a): f_psi = B^T pbar and
    #      K_psi = B^T K_R B, Cr2 Eqs. (17.176)-(17.177)
    for a in range(nn):
        _Lambda_c(&psie[3*a], &Lams[9*a])
        _mtm(TR, &Lams[9*a], &BR[9*a], 3, 3, 3)
        _mv3(TR, &pbar[6*a], &f[6*a])
        _mtv3(&BR[9*a], &pbar[6*a + 3], &f[6*a + 3])
    if e != NULL:
        # NOTE rigid offsets: f = J^T f_flat, see CorotationalShell
        for a in range(nn):
            _mv3(&Rs[9*a], &e[3*a], &rs[3*a])
            _spin3(&rs[3*a], M1)
            _mm(M1, &Lams[9*a], &Jr[9*a], 3, 3, 3)
        for i in range(n6):
            f_flat[i] = f[i]
        for a in range(nn):
            _mtv3(&Jr[9*a], &f_flat[6*a], v3)
            for i in range(3):
                f[6*a + 3 + i] = f_flat[6*a + 3 + i] + v3[i]
    if not tangent:
        return
    # NOTE material stiffness, F&H Eqs. (41)-(43), with the local
    #      tangent K_e + KCNL(d) of the von Kármán strains
    for i in range(n6*n6):
        HP[i] = P[i]
    for a in range(nn):
        _mm(&Hs[9*a], &P[n6*(6*a + 3)], &HP[n6*(6*a + 3)], 3, 3, n6)
    _mm(KCl, HP, T, n6, n6, n6)
    _mtm(HP, T, KM, n6, n6, n6)
    for i in range(n6*n6):
        KG[i] = 0.
    if KGl != NULL:
        # NOTE local geometric stiffness KG(d) of the von Kármán strains,
        #      transformed as the material stiffness (derived here)
        _mm(KGl, HP, T, n6, n6, n6)
        _mtm(HP, T, SG, n6, n6, n6)
        for i in range(n6*n6):
            KG[i] += SG[i]
    # NOTE moment-correction geometric stiffness, F&H Eqs. (46), (102)
    for a in range(nn):
        r0 = n6*(6*a + 3)
        _Lmat_c(&d[6*a + 3], &fbar[6*a + 3], Lm)
        _mtm(&P[r0], Lm, T3n, n6, 3, 3)
        _mm(T3n, &P[r0], SG, n6, 3, n6)
        for i in range(n6*n6):
            KG[i] += SG[i]
    # NOTE rotational geometric stiffness, F&H Eqs. (44)-(45)
    for a in range(nn):
        _spin3(&pbar[6*a], &Fnm[18*a])
        _spin3(&pbar[6*a + 3], &Fnm[18*a + 9])
    _mm(Fnm, G, SG, n6, 3, n6)
    for i in range(n6*n6):
        KG[i] -= SG[i]
    # NOTE equilibrium-projection geometric stiffness
    if kgp == KGP_BEAM:
        _KGP_beam_c(G, P, p, z, xl, aux, SG)
    elif kgp == KGP_SPRING:
        _KGP_spring_c(G, P, p, z, aux, SG)
    else:
        _KGP_FH_c(nn, G, P, pbar, SG)
    for i in range(n6*n6):
        KG[i] += SG[i]
    _btkb_c(nn, TR, BR, KM, KMpsi)
    _btkb_c(nn, TR, BR, KG, KGpsi)
    # NOTE extra term of the additive rotation vectors, Cr2 Eq. (17.177)
    for a in range(nn):
        _mv3(TR, &pbar[6*a + 3], v3)
        _dLambdaT_m_c(&psie[3*a], v3, K3)
        for i in range(3):
            for j in range(3):
                KGpsi[n6*(6*a + 3 + i) + 6*a + 3 + j] += K3[3*i + j]
    if e != NULL:
        # NOTE rigid offsets: congruent transformation and eccentricity
        #      geometric stiffness, see CorotationalShell
        for i in range(n6*n6):
            J[i] = 0.
        for i in range(n6):
            J[n6*i + i] = 1.
        for a in range(nn):
            for i in range(3):
                for j in range(3):
                    J[n6*(6*a + i) + 6*a + 3 + j] = Jr[9*a + 3*i + j]
        for k in range(2):
            if k == 0:
                for i in range(n6*n6):
                    KM[i] = KMpsi[i]
                _mm(KM, J, T, n6, n6, n6)
                _mtm(J, T, KMpsi, n6, n6, n6)
            else:
                for i in range(n6*n6):
                    KG[i] = KGpsi[i]
                _mm(KG, J, T, n6, n6, n6)
                _mtm(J, T, KGpsi, n6, n6, n6)
        for a in range(nn):
            _cross3(&f_flat[6*a], &rs[3*a], v3)
            _dLambdaT_m_c(&psie[3*a], v3, K3)
            _spin3(&f_flat[6*a], Sf)
            _spin3(&rs[3*a], Sr)
            _mtm(&Lams[9*a], Sf, M1, 3, 3, 3)
            _mm(M1, Sr, M2, 3, 3, 3)
            _mm(M2, &Lams[9*a], M1, 3, 3, 3)
            for i in range(3):
                for j in range(3):
                    KGpsi[n6*(6*a + 3 + i) + 6*a + 3 + j] += (
                        K3[3*i + j] - M1[3*i + j])


cdef double[::1] _vector3(v, name):
    r"""``v`` as a contiguous vector of 3 doubles"""
    cdef double[::1] out = np.ascontiguousarray(v, dtype=np.float64).ravel()
    if out.shape[0] != 3:
        raise ValueError('%s must have size 3' % name)
    return out


cdef object _aux_py(int kind, const double *aux):
    r"""Auxiliary vectors of the frame as returned by ``_frame``: the list
    `[\bar{\boldsymbol{q}}_1, \bar{\boldsymbol{q}}_2]` (FRAME_MEAN_TRIAD),
    the tuple of the lists `[\bar{\boldsymbol{p}}_i]` and
    `[\bar{\boldsymbol{q}}_i]` (FRAME_SPRING_TRIAD), or ``None``"""
    if kind == FRAME_MEAN_TRIAD:
        return [np.array([aux[0], aux[1], aux[2]]),
                np.array([aux[3], aux[4], aux[5]])]
    if kind == FRAME_SPRING_TRIAD:
        return ([np.array([aux[0], aux[1], aux[2]]),
                 np.array([aux[3], aux[4], aux[5]])],
                [np.array([aux[6], aux[7], aux[8]]),
                 np.array([aux[9], aux[10], aux[11]])])
    return None


cdef int _aux_c(int kind, object aux, double *out) except -1:
    r"""Inverse of :func:`_aux_py`"""
    cdef int i, j
    for i in range(12):
        out[i] = 0.
    if kind == FRAME_MEAN_TRIAD:
        for i in range(2):
            v = np.asarray(aux[i], dtype=np.float64)
            for j in range(3):
                out[3*i + j] = v[j]
    elif kind == FRAME_SPRING_TRIAD:
        for i in range(2):
            v = np.asarray(aux[0][i], dtype=np.float64)
            w = np.asarray(aux[1][i], dtype=np.float64)
            for j in range(3):
                out[3*i + j] = v[j]
                out[6 + 3*i + j] = w[j]
    return 0


cdef int _gather(object w, object u, double *ue,
                 double *psie) except -1:
    r"""Nodal translations and rotation vectors of the wrapper ``w``, rows of
    3, from the vector ``u`` of all the degrees-of-freedom"""
    cdef const double[:] uv = np.asarray(u, dtype=np.float64)
    cdef Py_ssize_t[::1] dofs = w._dofs_c
    cdef int n = w.nn
    cdef int a, i
    if uv.shape[0] <= w._max_dof:
        raise IndexError('u has size %d, the element needs at least %d'
                         % (uv.shape[0], w._max_dof + 1))
    for a in range(n):
        for i in range(3):
            ue[3*a + i] = uv[dofs[6*a + i]]
            psie[3*a + i] = uv[dofs[6*a + 3 + i]]
    return 0


cdef int _element_kinematics(object w, int kind, const double *ue,
                             const double *psie, double *Rs, double *TR,
                             double *xl, double *d, double *aux,
                             double *val) except -1:
    r"""Calls :func:`_kinematics_c` with the base configuration of the
    wrapper ``w``; returns 1 when the frame is undefined"""
    cdef double[:, ::1] X0c = w._X0c
    cdef double[:, ::1] xt = w._xt
    cdef double[:, ::1] T0 = w.T0
    cdef double[:, ::1] q0p0 = w._q0p0
    cdef double[:, ::1] e
    cdef const double *ep = NULL
    cdef double q2_min = 0.
    if w._e is not None:
        e = w._e
        ep = &e[0, 0]
    if kind == FRAME_MEAN_TRIAD or kind == FRAME_SPRING_TRIAD:
        q2_min = w.q2_min
    return _kinematics_c(kind, w.nn, &X0c[0, 0], ue, psie, ep, &xt[0, 0],
                         &T0[0, 0], &q0p0[0, 0], &q0p0[1, 0], q2_min, Rs, TR,
                         xl, d, aux, val)



# ---------------------------------------------------------------------------
# finite-rotation utilities
# ---------------------------------------------------------------------------
def _cross(a, b):
    r"""Cross product of two 3-vectors, faster than :func:`numpy.cross`"""
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])


def spin(v):
    r"""Skew-symmetric (spin) matrix of a 3-vector

    Parameters
    ----------
    v : array_like
        Vector of size 3.

    Returns
    -------
    S : np.ndarray
        Shape ``(3, 3)``, such that ``S @ w == np.cross(v, w)``.

    """
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def axial(W):
    r"""Axial vector of a skew-symmetric 3x3 matrix, the inverse of
    :func:`spin`

    Parameters
    ----------
    W : np.ndarray
        Shape ``(3, 3)``; only its skew-symmetric part is used.

    Returns
    -------
    v : np.ndarray
        Vector of size 3, ``(W[2, 1] - W[1, 2])/2`` etc.

    """
    return 0.5*np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0],
                         W[1, 0] - W[0, 1]])


def _coefficients(t):
    r"""Coefficients of the rotation matrices of the angle ``t >= 0``

    Returns ``sin t/t``, ``a = (1 - cos t)/t**2``, ``b = (t - sin t)/t**3``,
    ``a'/t`` and ``b'/t``, with their Taylor series up to ``t**8`` below
    :data:`SERIES_ANGLE` (series derived here with sympy)."""
    cdef double c[5]
    _coefficients_c(t, c)
    return c[0], c[1], c[2], c[3], c[4]


def expso3(psi):
    r"""Rotation matrix of a rotation vector, the exponential map

    Parameters
    ----------
    psi : array_like
        Rotation vector `\boldsymbol{\psi}`, size 3, in radians.

    Returns
    -------
    R : np.ndarray
        Shape ``(3, 3)``, the proper orthogonal matrix of the rotation by
        the angle `|\boldsymbol{\psi}|` about the axis of
        `\boldsymbol{\psi}`.

    Notes
    -----
    Rodrigues' formula, Crisfield [1]_, Eqs. (16.22) and (16.31); Felippa
    and Haugen [2]_, Eq. (89):

    .. math::

        \boldsymbol{R} = \boldsymbol{I} + \frac{\sin \psi}{\psi}
        \mathrm{Spin}(\boldsymbol{\psi}) + \frac{1 - \cos \psi}{\psi^2}
        \mathrm{Spin}(\boldsymbol{\psi})^2, \qquad \psi =
        |\boldsymbol{\psi}|

    with the Taylor series of the coefficients for small angles.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.
    .. [2] Felippa, C. A. and Haugen, B., CMAME 194, 2285-2335, 2005,
       https://doi.org/10.1016/j.cma.2004.07.035.

    """
    cdef double[::1] p = _vector3(psi, 'psi')
    R = np.empty((3, 3))
    cdef double[:, ::1] Rv = R
    _expso3_c(&p[0], &Rv[0, 0])
    return R


def logso3(R):
    r"""Rotation vector of a rotation matrix, the logarithmic map

    Parameters
    ----------
    R : np.ndarray
        Proper orthogonal matrix, shape ``(3, 3)``.

    Returns
    -------
    theta : np.ndarray
        Rotation vector of size 3 with `|\boldsymbol{\theta}| \le \pi`,
        such that ``expso3(theta)`` is ``R``.

    Notes
    -----
    The quaternion of ``R`` is extracted with Spurrier's algorithm, Felippa
    and Haugen [1]_, Eqs. (96)-(97), and Crisfield [2]_, Section 16.10,
    Eqs. (16.74)-(16.78), which picks the largest of the trace and the
    diagonal entries and is accurate for every angle, including 0 and
    `\pi`; the closed form `\sin \theta = |\mathrm{axial}(\boldsymbol{R})|`
    of [1]_, Eq. (98), is not used since it is inaccurate near 0 and
    `\pi`. The quaternion is taken with `q_0 \ge 0`, and the rotation vector
    is `\boldsymbol{\theta} = 2 \arctan(|\boldsymbol{q}|/q_0)
    \boldsymbol{q}/|\boldsymbol{q}|`.

    References
    ----------
    .. [1] Felippa, C. A. and Haugen, B., CMAME 194, 2285-2335, 2005,
       https://doi.org/10.1016/j.cma.2004.07.035.
    .. [2] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.

    """
    cdef double[:, ::1] Rv = np.ascontiguousarray(R, dtype=np.float64)
    if Rv.shape[0] != 3 or Rv.shape[1] != 3:
        raise ValueError('R must have shape (3, 3)')
    theta = np.empty(3)
    cdef double[::1] tv = theta
    _logso3_c(&Rv[0, 0], &tv[0])
    return theta


def Lambda(psi):
    r"""Matrix that maps the variation of a rotation vector to the spatial
    spin

    Parameters
    ----------
    psi : array_like
        Rotation vector `\boldsymbol{\psi}`, size 3, in radians.

    Returns
    -------
    Lam : np.ndarray
        Shape ``(3, 3)``, `\boldsymbol{\Lambda}(\boldsymbol{\psi})` with
        `\delta \boldsymbol{\omega} = \boldsymbol{\Lambda} \delta
        \boldsymbol{\psi}`, where `\delta \boldsymbol{R} \boldsymbol{R}^T =
        \mathrm{Spin}(\delta \boldsymbol{\omega})` for `\boldsymbol{R} =`
        ``expso3(psi)``.

    Notes
    -----
    Crisfield [1]_, Eq. (16.90), where it is called
    `\boldsymbol{H}(\boldsymbol{\theta})`; Felippa and Haugen [2]_,
    Eq. (100), last form, where it is `\boldsymbol{H}^{-1}`:

    .. math::

        \boldsymbol{\Lambda} = \boldsymbol{I} + \frac{1 - \cos \psi}{\psi^2}
        \mathrm{Spin}(\boldsymbol{\psi}) + \frac{\psi - \sin \psi}{\psi^3}
        \mathrm{Spin}(\boldsymbol{\psi})^2

    It is singular for `\psi = 2 k \pi`, `k \ge 1`, and
    `\boldsymbol{\Lambda}(\boldsymbol{\psi})^T =
    \boldsymbol{\Lambda}(-\boldsymbol{\psi})`.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.
    .. [2] Felippa, C. A. and Haugen, B., CMAME 194, 2285-2335, 2005,
       https://doi.org/10.1016/j.cma.2004.07.035.

    """
    cdef double[::1] p = _vector3(psi, 'psi')
    Lam = np.empty((3, 3))
    cdef double[:, ::1] Lv = Lam
    _Lambda_c(&p[0], &Lv[0, 0])
    return Lam


def dLambdaT_m(psi, m):
    r"""Derivative of `\boldsymbol{\Lambda}(\boldsymbol{\psi})^T
    \boldsymbol{m}` with respect to the rotation vector, for a fixed
    vector `\boldsymbol{m}`

    Parameters
    ----------
    psi : array_like
        Rotation vector `\boldsymbol{\psi}`, size 3, in radians.
    m : array_like
        Vector of size 3, e.g. a moment.

    Returns
    -------
    K : np.ndarray
        Shape ``(3, 3)``, `\partial (\boldsymbol{\Lambda}^T
        \boldsymbol{m})/\partial \boldsymbol{\psi}`.

    Notes
    -----
    Derived here from :func:`Lambda` and checked by finite differences in
    ``tests/test_corotational_core.py``. With `a = (1 - \cos \psi)/\psi^2`,
    `b = (\psi - \sin \psi)/\psi^3` and `\boldsymbol{\Lambda}^T
    \boldsymbol{m} = \boldsymbol{m} - a \boldsymbol{\psi} \times
    \boldsymbol{m} + b \boldsymbol{\psi} \times (\boldsymbol{\psi} \times
    \boldsymbol{m})`:

    .. math::

        \frac{\partial (\boldsymbol{\Lambda}^T \boldsymbol{m})}{\partial
        \boldsymbol{\psi}} = a \mathrm{Spin}(\boldsymbol{m}) -
        \frac{a'}{\psi} (\boldsymbol{\psi} \times \boldsymbol{m})
        \boldsymbol{\psi}^T + b \left[ \boldsymbol{\psi} \boldsymbol{m}^T
        + (\boldsymbol{\psi} \cdot \boldsymbol{m}) \boldsymbol{I} - 2
        \boldsymbol{m} \boldsymbol{\psi}^T \right] + \frac{b'}{\psi}
        (\boldsymbol{\psi} \times (\boldsymbol{\psi} \times \boldsymbol{m}))
        \boldsymbol{\psi}^T

    with the Taylor series of `a'/\psi` and `b'/\psi` for small angles. At
    `\boldsymbol{\psi} = \boldsymbol{0}` it is `\mathrm{Spin}(\boldsymbol{m})/2`,
    in agreement with Crisfield [1]_, Eq. (16.96). This is the extra term
    of the tangent stiffness matrix in additive rotation vectors, [1]_,
    Eq. (17.177).

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.

    """
    cdef double[::1] p = _vector3(psi, 'psi')
    cdef double[::1] mv = _vector3(m, 'm')
    K = np.empty((3, 3))
    cdef double[:, ::1] Kv = K
    _dLambdaT_m_c(&p[0], &mv[0], &Kv[0, 0])
    return K


def _eta_mu(t):
    r"""Coefficients `\eta` and `\mu` of [F&H] Eqs. (101) and (103), with
    their Taylor series up to ``t**14`` below :data:`SERIES_ANGLE_ETA_MU`
    (series derived here with sympy, in agreement with the terms up to
    ``t**6`` of F&H); the numerator of `\mu` cancels as ``t**6``"""
    cdef double eta, mu
    _eta_mu_c(t, &eta, &mu)
    return eta, mu


def Hmat(theta):
    r"""Jacobian of the rotation vector with respect to the spin

    Parameters
    ----------
    theta : array_like
        Rotation vector `\boldsymbol{\theta}`, size 3, in radians, with
        `|\boldsymbol{\theta}| < 2 \pi`; in the EICR a deformational
        rotation, small.

    Returns
    -------
    H : np.ndarray
        Shape ``(3, 3)``, `\boldsymbol{H}(\boldsymbol{\theta})` with
        `\delta \boldsymbol{\theta} = \boldsymbol{H} \delta
        \boldsymbol{\omega}`, the inverse of :func:`Lambda`.

    Notes
    -----
    Felippa and Haugen [1]_, Eq. (101); Nour-Omid and Rankin [2]_,
    Eqs. (A.14)-(A.15); Crisfield [3]_, Eq. (16.94), where it is
    `\boldsymbol{H}^{-1}`:

    .. math::

        \boldsymbol{H} = \boldsymbol{I} - \frac{1}{2}
        \mathrm{Spin}(\boldsymbol{\theta}) + \eta \,
        \mathrm{Spin}(\boldsymbol{\theta})^2, \qquad \eta = \frac{1 -
        (\theta/2) \cot(\theta/2)}{\theta^2}

    with the Taylor series of `\eta` for small angles.

    References
    ----------
    .. [1] Felippa, C. A. and Haugen, B., CMAME 194, 2285-2335, 2005,
       https://doi.org/10.1016/j.cma.2004.07.035.
    .. [2] Nour-Omid, B. and Rankin, C. C., CMAME 93, 353-384, 1991,
       https://doi.org/10.1016/0045-7825(91)90248-5.
    .. [3] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.

    """
    cdef double[::1] th = _vector3(theta, 'theta')
    H = np.empty((3, 3))
    cdef double[:, ::1] Hv = H
    _Hmat_c(&th[0], &Hv[0, 0])
    return H


def Lmat(theta, m):
    r"""Derivative of `\boldsymbol{H}(\boldsymbol{\theta})^T \boldsymbol{m}`
    with respect to the spin, for a fixed vector `\boldsymbol{m}`

    Parameters
    ----------
    theta : array_like
        Rotation vector `\boldsymbol{\theta}`, size 3, in radians.
    m : array_like
        Vector of size 3, the moment of a node in the local frame.

    Returns
    -------
    L : np.ndarray
        Shape ``(3, 3)``, `\boldsymbol{L} = \partial (\boldsymbol{H}^T
        \boldsymbol{m})/\partial \boldsymbol{\theta} \, \boldsymbol{H}`.

    Notes
    -----
    Felippa and Haugen [1]_, Eqs. (102)-(103); Nour-Omid and Rankin [2]_,
    Eqs. (A.16)-(A.18):

    .. math::

        \frac{\partial (\boldsymbol{H}^T \boldsymbol{m})}{\partial
        \boldsymbol{\theta}} = \eta \left[ (\boldsymbol{\theta} \cdot
        \boldsymbol{m}) \boldsymbol{I} + \boldsymbol{\theta}
        \boldsymbol{m}^T - 2 \boldsymbol{m} \boldsymbol{\theta}^T \right] +
        \mu \, \mathrm{Spin}(\boldsymbol{\theta})^2 \boldsymbol{m}
        \boldsymbol{\theta}^T - \frac{1}{2} \mathrm{Spin}(\boldsymbol{m}),
        \qquad \mu = \frac{\theta^2 + 4 \cos \theta + \theta \sin \theta -
        4}{4 \theta^4 \sin^2(\theta/2)}

    with the Taylor series of `\eta` and `\mu` for small angles.

    References
    ----------
    .. [1] Felippa, C. A. and Haugen, B., CMAME 194, 2285-2335, 2005,
       https://doi.org/10.1016/j.cma.2004.07.035.
    .. [2] Nour-Omid, B. and Rankin, C. C., CMAME 93, 353-384, 1991,
       https://doi.org/10.1016/0045-7825(91)90248-5.

    """
    cdef double[::1] th = _vector3(theta, 'theta')
    cdef double[::1] mv = _vector3(m, 'm')
    L = np.empty((3, 3))
    cdef double[:, ::1] Lv = L
    _Lmat_c(&th[0], &mv[0], &Lv[0, 0])
    return L


# ---------------------------------------------------------------------------
# element wrappers
# ---------------------------------------------------------------------------
_SHELLS = {Tria3R: Tria3RData, Quad4: Quad4Data, Quad4R: Quad4RData,
           Tria3DSG: Tria3DSGData}
_BEAMS = {BeamC: BeamCData, BeamLR: BeamLRData}
_TRUSSES = {Truss: TrussData}
_SPRINGS = {Spring: SpringData}
_ELEMENTS = {**_SHELLS, **_BEAMS, **_TRUSSES, **_SPRINGS}
_MODEL_ELEMENTS = _ELEMENTS
_NUM_NODES = {Tria3R: 3, Quad4: 4, Quad4R: 4, BeamC: 2, Tria3DSG: 3,
              BeamLR: 2, Spring: 2, Truss: 2}
_WARPING = ('offsets', None)
#: local strains of the wrappers: linear (pure EICR) or von Kármán
_LOCAL = ('linear', 'von_karman')
_R_NAMES = ('r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33')


def _element_rotation_matrix(elem):
    r"""``r11..r33`` of an element as a 3x3 matrix, columns = element axes"""
    return np.array([getattr(elem, n) for n in _R_NAMES]).reshape(3, 3)


def _num_nodes(elem):
    return _NUM_NODES[type(elem)]


def _element_dofs(elem):
    r"""Positions ``c1, c2, ...`` of the nodes of an element"""
    return np.array([getattr(elem, 'c%d' % (i + 1))
                     for i in range(_num_nodes(elem))], dtype=INT)


class _CorotationalElement(object):
    r"""Common part of :class:`CorotationalShell` and
    :class:`CorotationalBeam`, see their documentation"""
    def __init__(self, elem, prop, x, local='linear', **elem_kwargs):
        if type(elem) not in _ELEMENTS:
            raise TypeError('%s is not supported, only %s' % (
                type(elem).__name__,
                ', '.join(sorted(c.__name__ for c in _ELEMENTS))))
        if local not in _LOCAL:
            raise ValueError('local must be one of %s, got %r'
                             % (', '.join(repr(k) for k in _LOCAL), local))
        if local == 'von_karman' and type(elem) not in {**_SHELLS, **_BEAMS}:
            raise ValueError("%s has no von Kármán strains: local='von_karman'"
                             " is not supported" % type(elem).__name__)
        self.local = local
        self.elem = elem
        self.prop = prop
        self.elem_kwargs = dict(elem_kwargs)
        self.nn = _num_nodes(elem)
        self.cs = _element_dofs(elem)
        self.nodes = self.cs//DOF
        x = np.asarray(x, dtype=float)
        self.X0 = np.array([x[c//2:c//2 + 3] for c in self.cs])
        self.T0 = self._base_frame()
        self.xC0 = self.X0.mean(axis=0)
        self.xtilde = (self.X0 - self.xC0) @ self.T0
        # NOTE base data of the C kernels: positions relative to the base
        #      centroid, see _kinematics, and the base y and x axes of the
        #      beams and springs
        self._X0c = np.ascontiguousarray(self.X0 - self.xC0)
        self._q0p0 = np.array([getattr(self, 'q0', np.zeros(3)),
                               getattr(self, 'p0', np.zeros(3))])
        # NOTE rigid offsets (CorotationalShell with warping='offsets'): the
        #      nodal offsets e_a in global components at the base
        #      configuration, and the base local coordinates of the nodes of
        #      the flat element, from which d is measured
        self._e = None
        self._xt = self.xtilde
        self.Ke = self._local_stiffness()
        self.dofs = (self.cs[:, None] + np.arange(DOF)).ravel()
        self._dofs_c = np.ascontiguousarray(self.dofs, dtype=np.intp)
        self._max_dof = int(self.dofs.max())
        n = self.nn
        Pu = np.eye(DOF*n)
        for a in range(n):
            for b in range(n):
                Pu[DOF*a:DOF*a + 3, DOF*b:DOF*b + 3] -= _I3/n
        self._Pu = Pu

    # NOTE to be defined by the subclasses
    def _base_frame(self):
        raise NotImplementedError

    def _frame_kind(self):
        r"""Frame rule of the C kernels, ``FRAME_TRIA``, ``FRAME_QUAD``,
        ``FRAME_MEAN_TRIAD``, ``FRAME_SPRING_TRIAD`` or ``FRAME_CHORD``"""
        raise NotImplementedError

    def _kgp_kind(self):
        r"""Projection geometric stiffness of the C kernels: ``KGP_FH``, F&H
        Eq. (47), or the exact ones ``KGP_BEAM`` and ``KGP_SPRING``"""
        return KGP_FH

    def _warn_undefined(self, kind, v0, v1):
        r"""Warning, once per wrapper, that the CR frame is undefined"""
        if self._warned:
            return
        if kind == FRAME_SPRING_TRIAD:
            msg = ('Spring %d-%d: the co-rotational frame is undefined, |a| '
                   '= %.3g, b2 = %.3g, q2_min = %.3g'
                   % (self.nodes[0], self.nodes[1], v0, v1, self.q2_min))
        else:
            msg = ('%s %d-%d: the co-rotational frame is undefined, q2 = '
                   '%.3g < q2_min' % (type(self.elem).__name__,
                                      self.nodes[0], self.nodes[1], v0))
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        self._warned = True

    def _frame(self, XD, Rs):
        r"""CR frame `\boldsymbol{T}_R` and auxiliary vectors at the nodal
        positions ``XD``, shape ``(nn, 3)``, and the nodal rotation matrices
        ``Rs`` (``None`` for the identities), by the frame rule of the
        wrapper; ``(None, None)`` when the frame is undefined"""
        cdef int kind = self._frame_kind()
        cdef int n = self.nn
        cdef int a, i
        cdef double[:, ::1] X = np.ascontiguousarray(XD, dtype=np.float64)
        cdef double[:, ::1] T0 = np.ascontiguousarray(
            getattr(self, 'T0', _I3), dtype=np.float64)
        cdef double[:, ::1] q0p0 = np.array([
            getattr(self, 'q0', np.zeros(3)), getattr(self, 'p0',
                                                      np.zeros(3))])
        cdef double[:, ::1] Ra
        cdef double Rv[9*MAX_NN]
        cdef double aux[12]
        cdef double val[2]
        cdef double q2_min = 0.
        if X.shape[0] != n or X.shape[1] != 3:
            raise ValueError('XD must have shape (%d, 3)' % n)
        for a in range(n):
            if Rs is None:
                for i in range(9):
                    Rv[9*a + i] = 1. if i % 4 == 0 else 0.
            else:
                Ra = np.ascontiguousarray(Rs[a], dtype=np.float64)
                for i in range(9):
                    Rv[9*a + i] = Ra[i//3, i % 3]
        if kind == FRAME_MEAN_TRIAD or kind == FRAME_SPRING_TRIAD:
            q2_min = self.q2_min
        TR = np.empty((3, 3))
        cdef double[:, ::1] TRv = TR
        if _frame_c(kind, &X[0, 0], Rv, &T0[0, 0], &q0p0[0, 0], &q0p0[1, 0],
                    q2_min, &TRv[0, 0], aux, val):
            self._warn_undefined(kind, val[0], val[1])
            return None, None
        return TR, _aux_py(kind, aux)

    def _G(self, xl, aux):
        r"""Spin-fitter matrix `\boldsymbol{G}`, shape ``(3, 6*nn)``, at the
        current local coordinates ``xl`` and the auxiliary vectors ``aux`` of
        :meth:`_frame`, see the Notes of the wrappers"""
        cdef int kind = self._frame_kind()
        cdef double[:, ::1] x = np.ascontiguousarray(xl, dtype=np.float64)
        cdef double a[12]
        if x.shape[0] != self.nn or x.shape[1] != 3:
            raise ValueError('xl must have shape (%d, 3)' % self.nn)
        _aux_c(kind, aux, a)
        G = np.empty((3, DOF*self.nn))
        cdef double[:, ::1] Gv = G
        _G_c(kind, self.nn, &x[0, 0], a, &Gv[0, 0])
        return G

    def _in_local_frame(self, func, ue=None):
        r"""Call ``func(elem, probe)`` with the element in its local frame:
        rotation matrix set to the identity, node positions set to ``0, 6,
        12, 18``, ``init_k_*`` and the ``warp_offsets`` of Quad4 and Quad4R
        set to zero, probe coordinates set to the base local coordinates
        and, with ``ue``, probe displacements set to ``ue``; the attributes
        changed on the element and its probe are restored, also when
        ``func`` raises"""
        e = self.elem
        p = e.probe
        # NOTE the Spring has no coordinates, no size and no property
        is_spring = type(e) in _SPRINGS
        size_name = None if is_spring else (
            'area' if type(e) in _SHELLS else 'length')
        # NOTE the rigid offsets of a warped Quad4 or Quad4R are those of
        #      CorotationalShell, see its Notes: the element's own linear
        #      ones (warp_offsets) are switched off
        init_k = [n for n in ('init_k_KC0', 'init_k_KCNL', 'init_k_KG',
                              'warp_offsets')
                  if hasattr(e, n)]
        saved_c = [getattr(e, 'c%d' % (i + 1)) for i in range(self.nn)]
        saved_r = [getattr(e, n) for n in _R_NAMES]
        saved_k = [getattr(e, n) for n in init_k]
        saved_xe = np.array(p.xe, copy=True)
        saved_ue = np.array(p.ue, copy=True)
        saved_size = None if is_spring else getattr(e, size_name)
        try:
            for i in range(self.nn):
                setattr(e, 'c%d' % (i + 1), DOF*i)
            for n in init_k:
                setattr(e, n, 0)
            for n, v in zip(_R_NAMES, _I3.ravel()):
                setattr(e, n, v)
            if not is_spring:
                p.xe[:] = self.xtilde.ravel()
                getattr(e, 'update_' + size_name)()
            if ue is not None:
                p.ue[:] = ue
            return func(e, p)
        finally:
            for i in range(self.nn):
                setattr(e, 'c%d' % (i + 1), saved_c[i])
            for n, val in zip(_R_NAMES, saved_r):
                setattr(e, n, val)
            for n, val in zip(init_k, saved_k):
                setattr(e, n, val)
            p.xe[:] = saved_xe
            p.ue[:] = saved_ue
            if not is_spring:
                setattr(e, size_name, saved_size)

    def _dense(self, what, func, e):
        r"""Dense local matrix from the sparse entries written by
        ``func(r, c, v)``, of size ``<what>_SPARSE_SIZE`` of the element"""
        size = getattr(_ELEMENTS[type(e)](), what + '_SPARSE_SIZE')
        r = np.zeros(size, dtype=INT)
        c = np.zeros(size, dtype=INT)
        v = np.zeros(size)
        func(r, c, v)
        n = DOF*self.nn
        K = np.zeros((n, n))
        np.add.at(K, (r, c), v)
        return K

    def _local_stiffness(self):
        r"""Local linear stiffness matrix `\boldsymbol{K}_e` by the identity
        trick: ``update_KC0`` with the rotation matrix set to the identity,
        the probe coordinates set to the base local coordinates and the
        node positions set to ``0, 6, 12, 18``; the attributes changed on
        the element and its probe are restored"""
        def func(e, p):
            if type(e) in _SPRINGS:
                return self._dense('KC0', lambda r, c, v: e.update_KC0(
                    r, c, v, **self.elem_kwargs), e)
            return self._dense('KC0', lambda r, c, v: e.update_KC0(
                r, c, v, self.prop, **self.elem_kwargs), e)
        return self._in_local_frame(func)

    def _local_von_karman(self, d, tangent=True):
        r"""Local internal force and tangent of the von Kármán strains of
        the element at the deformational displacements ``d``, by the
        identity trick of :meth:`_local_stiffness`

        Returns ``fbar`` and, with ``tangent``, ``KC`` and ``KG``: the
        ``update_probe_finte(prop, nonlinear=1)`` of the element, its
        ``K_e + KCNL`` and its ``KG``, with ``ue = d``, in the element
        axes."""
        def func(e, p):
            e.update_probe_finte(self.prop, nonlinear=1, **self.elem_kwargs)
            fbar = np.array(p.finte, copy=True)
            if not tangent:
                return fbar
            KCNL = self._dense('KCNL', lambda r, c, v: e.update_KCNL(
                r, c, v, self.prop), e)
            KG = self._dense('KG', lambda r, c, v: e.update_KG(
                r, c, v, self.prop), e)
            return fbar, self.Ke + KCNL, KG
        return self._in_local_frame(func, ue=d)

    def _S(self, xl):
        r"""Rigid-rotation matrix `\boldsymbol{S}`, node blocks
        `[-\mathrm{Spin}(\bar{\boldsymbol{x}}_a); \boldsymbol{I}]`, F&H
        Eq. (12)"""
        S = np.zeros((DOF*self.nn, 3))
        for a in range(self.nn):
            S[DOF*a:DOF*a + 3] = -spin(xl[a])
            S[DOF*a + 3:DOF*a + 6] = _I3
        return S

    def _split(self, u):
        r"""Nodal translations and rotation vectors of the element, shapes
        ``(nn, 3)``, from the vector of all the degrees-of-freedom"""
        ue = np.asarray(u)[self.dofs].reshape(self.nn, DOF)
        return ue[:, :3], ue[:, 3:]

    def kinematics(self, u):
        r"""Co-rotational frame and deformational displacements at a
        configuration

        Parameters
        ----------
        u : np.ndarray
            Vector of all the degrees-of-freedom of the model, size ``N``:
            nodal translations and total rotation vectors.

        Returns
        -------
        TR : np.ndarray or None
            Current CR frame `\boldsymbol{T}_R`, shape ``(3, 3)``, columns
            = element axes; ``None`` when the frame is undefined, see
            :class:`CorotationalBeam`.
        d : np.ndarray or None
            Deformational displacements and rotations in the CR frame,
            `[\bar{\boldsymbol{u}}_{d,a}, \bar{\boldsymbol{\theta}}_{d,a}]`
            per node, size ``6*nn``, in the order of the degrees-of-freedom
            of the element.

        """
        out = self._kinematics(*self._split(u))
        if out is None:
            return None, None
        return out[0], out[2]

    def _kinematics(self, ue, psie):
        r"""F&H Section 3.8, Eq. (22) and Table 3, steps 2-3; N-O&R Eqs.
        (11), (14), (19); returns ``(TR, xl, d, aux)``, ``aux`` as in
        :meth:`_frame`, or ``None`` when the frame is undefined"""
        cdef int n = self.nn
        cdef int kind = self._frame_kind()
        cdef int a, i
        cdef double[:, ::1] uev = np.ascontiguousarray(ue, dtype=np.float64)
        cdef double[:, ::1] psv = np.ascontiguousarray(psie,
                                                       dtype=np.float64)
        cdef double Rs[9*MAX_NN]
        cdef double aux[12]
        cdef double val[2]
        if (uev.shape[0] != n or uev.shape[1] != 3 or psv.shape[0] != n
                or psv.shape[1] != 3):
            raise ValueError('ue and psie must have shape (%d, 3)' % n)
        TR = np.empty((3, 3))
        xl = np.empty((n, 3))
        d = np.empty(DOF*n)
        cdef double[:, ::1] TRv = TR
        cdef double[:, ::1] xlv = xl
        cdef double[::1] dv = d
        if _element_kinematics(self, kind, &uev[0, 0], &psv[0, 0], Rs,
                               &TRv[0, 0], &xlv[0, 0], &dv[0], aux, val):
            self._warn_undefined(kind, val[0], val[1])
            return None
        return TR, xl, d, _aux_py(kind, aux)

    def evaluate(self, u, tangent=True):
        r"""Internal force vector and tangent stiffness matrix of the
        element in the additive degrees-of-freedom

        Parameters
        ----------
        u : np.ndarray
            Vector of all the degrees-of-freedom of the model, size ``N``.
        tangent : bool, optional
            Whether the tangent stiffness matrix is computed. Default
            ``True``.

        Returns
        -------
        f : np.ndarray
            Internal force vector of the element, size ``6*nn``, at the
            degrees-of-freedom ``dofs``: forces, and moments conjugate to
            the rotation vectors.
        KM, KG : np.ndarray
            Only with ``tangent=True``. Material and geometric parts of the
            tangent stiffness matrix, shape ``(6*nn, 6*nn)``, whose sum is
            the Jacobian of ``f`` with respect to ``u[dofs]``.

        All the arrays are filled with NaN when the CR frame is undefined,
        see :class:`CorotationalBeam`.

        """
        cdef int n = self.nn
        cdef int n6 = _DOF*n
        cdef int kind = self._frame_kind()
        cdef int kgp = self._kgp_kind()
        cdef int i
        cdef double ue[3*MAX_NN]
        cdef double psie[3*MAX_NN]
        cdef double Rs[9*MAX_NN]
        cdef double TR[9]
        cdef double xl[3*MAX_NN]
        cdef double d[MAX_N6]
        cdef double aux[12]
        cdef double val[2]
        cdef double G[3*MAX_N6]
        cdef double fbar[MAX_N6]
        cdef double[:, ::1] Ke = self.Ke
        cdef double[:, ::1] Pu = self._Pu
        cdef double[:, ::1] KCl = Ke
        cdef double[:, ::1] KGl
        cdef double[:, ::1] e
        cdef double[::1] fv
        cdef double[:, ::1] KMv
        cdef double[:, ::1] KGv
        cdef double[::1] fbv
        cdef const double *KGlp = NULL
        cdef const double *ep = NULL
        cdef double *KMp = NULL
        cdef double *KGp = NULL
        _gather(self, u, ue, psie)
        if _element_kinematics(self, kind, ue, psie, Rs, TR, xl, d, aux,
                               val):
            self._warn_undefined(kind, val[0], val[1])
            nan = np.full(n6, np.nan)
            if not tangent:
                return nan
            return nan, np.full((n6, n6), np.nan), np.full((n6, n6), np.nan)
        if self.local == 'von_karman':
            # NOTE local internal force and tangent of the von Kármán strains
            #      of the element, Cr2 Section 17.1.5, step 8, see
            #      CorotationalShell
            out = self._local_von_karman(np.array(<double[:n6]> &d[0]), tangent)
            fbar_vk, KC_vk, KG_vk = out if tangent else (out, None, None)
            fbv = np.ascontiguousarray(fbar_vk, dtype=np.float64)
            for i in range(n6):
                fbar[i] = fbv[i]
            if tangent:
                KCl = np.ascontiguousarray(KC_vk, dtype=np.float64)
                KGl = np.ascontiguousarray(KG_vk, dtype=np.float64)
                KGlp = &KGl[0, 0]
        else:
            # NOTE local internal force of the linear element, F&H Eq. (53)
            _mm(&Ke[0, 0], d, fbar, n6, n6, 1)
        if self._e is not None:
            e = self._e
            ep = &e[0, 0]
        _G_c(kind, n, xl, aux, G)
        f = np.empty(n6)
        fv = f
        if tangent:
            KM = np.empty((n6, n6))
            KG = np.empty((n6, n6))
            KMv = KM
            KGv = KG
            KMp = &KMv[0, 0]
            KGp = &KGv[0, 0]
        _force_tangent_c(n, kgp, 1 if tangent else 0, fbar, &KCl[0, 0], KGlp,
                         &Pu[0, 0], TR, xl, d, aux, psie, Rs, ep, G, &fv[0],
                         KMp, KGp)
        if not tangent:
            return f
        return f, KM, KG

    def update_probe(self, u):
        r"""Write the deformational displacements into the probe of the
        element, for stress recovery with its own routines

        ``probe.xe`` receives the base local coordinates
        `\tilde{\boldsymbol{x}}_a` and ``probe.ue`` the deformational
        displacements ``d`` of :meth:`kinematics`, both in element
        coordinates, such that e.g. ``elem.update_probe_finte(prop)`` gives
        the local forces `\boldsymbol{K}_e \boldsymbol{d}`.

        Parameters
        ----------
        u : np.ndarray
            Vector of all the degrees-of-freedom of the model, size ``N``.

        Raises
        ------
        ValueError
            When the CR frame is undefined at ``u``.

        """
        TR, d = self.kinematics(u)
        if TR is None:
            raise ValueError('the co-rotational frame is undefined')
        p = self.elem.probe
        p.xe[:] = self.xtilde.ravel()
        p.ue[:] = d


class CorotationalShell(_CorotationalElement):
    r"""Co-rotational (EICR) wrapper of a :class:`.Tria3R`,
    :class:`.Tria3DSG`, :class:`.Quad4` or :class:`.Quad4R` element

    The local linear stiffness matrix `\boldsymbol{K}_e` of the element is
    computed once, in the base configuration, by its own ``update_KC0``
    with the rotation matrix set to the identity; the element is not
    modified, and the attributes changed during that call are restored.
    :meth:`evaluate` gives the internal force vector and the consistent
    tangent stiffness matrix at any configuration, in the additive
    degrees-of-freedom of the model: translations and total rotation
    vectors.

    Usually created by :meth:`CorotationalModel.add`.

    Parameters
    ----------
    elem : Tria3R, Tria3DSG, Quad4 or Quad4R
        The element, with its node positions ``c1, c2, ...`` set and
        ``update_rotation_matrix(x, ...)`` called with the same ``x``, which
        also defines its material direction.
    prop : ShellProp
        Property of the element.
    x : array_like
        Initial nodal coordinates of the model, size ``3*num_nodes``, as
        used by the elements.
    warping : str or None, optional
        Treatment of a warped quadrilateral, see the Notes: ``'offsets'``,
        rigid offsets between the nodes and the flat element of
        ``K_e``; ``None``, the flat element at the warped nodes, with the
        projector alone. Ignored for triangles and flat quadrilaterals.
        Default ``'offsets'``.
    local : str, optional
        Strains of the element in its CR frame: ``'linear'``, the local
        linear stiffness matrix `\boldsymbol{K}_e` (pure EICR), or
        ``'von_karman'``, the von Kármán strains of the element, see the
        Notes. Default ``'linear'``.
    **elem_kwargs
        Passed to ``update_KC0`` of the element, e.g. the ``hgfactor_*`` of
        :class:`.Quad4R`, and with ``local='von_karman'`` to its
        ``update_probe_finte``.

    Attributes
    ----------
    nn : int
        Number of nodes, 3 or 4.
    dofs : np.ndarray
        Positions of the ``6*nn`` degrees-of-freedom of the element in the
        vectors of the model.
    T0 : np.ndarray
        Base frame `\boldsymbol{T}_0`, shape ``(3, 3)``, columns = element
        axes, equal to ``r11..r33`` of the element.
    xtilde : np.ndarray
        Base local coordinates `\tilde{\boldsymbol{x}}_a = \boldsymbol{T}_0^T
        (\boldsymbol{x}_a - \boldsymbol{x}_{C0})`, shape ``(nn, 3)``, with
        `\boldsymbol{x}_{C0}` the mean of the nodal coordinates.
    Ke : np.ndarray
        Local linear stiffness matrix, shape ``(6*nn, 6*nn)``.
    warping : str or None
        The ``warping`` argument.
    local : str
        The ``local`` argument.
    warp_ratio : float
        ``max|xtilde[:, 2]|/sqrt(A)``, with `A` the area of the flat
        projection; zero for triangles. A warning is issued above
        :data:`WARP_RATIO_WARNING` with ``warping=None`` and above
        :data:`WARP_RATIO_WARNING_OFFSETS` with ``warping='offsets'``.

    Raises
    ------
    TypeError
        When ``elem`` is not a Tria3R, Tria3DSG, Quad4 or Quad4R.
    ValueError
        When ``r11..r33`` of the element do not match the frame of ``x``,
        i.e. ``update_rotation_matrix`` was not called with ``x``.
    ValueError
        When ``warping`` is not ``'offsets'`` or ``None``, or ``local`` is
        not ``'linear'`` or ``'von_karman'``.

    Notes
    -----
    Base configuration, Felippa and Haugen [1]_, Table 3, step 1. At a
    configuration with the deformed nodal positions `\boldsymbol{x}^D_a` and
    the nodal rotation matrices `\boldsymbol{R}_a =
    \exp(\mathrm{Spin}(\boldsymbol{\psi}_a))`, the deformational
    displacements in the CR frame `\boldsymbol{T}_R` are ([1]_, Section 3.8,
    Eq. (22), Table 3, steps 2-3; Nour-Omid and Rankin [2]_, Eqs. (11),
    (14), (19))

    .. math::

        \bar{\boldsymbol{u}}_{d,a} = \boldsymbol{T}_R^T (\boldsymbol{x}^D_a -
        \boldsymbol{x}_C) - \tilde{\boldsymbol{x}}_a, \qquad
        \bar{\boldsymbol{\theta}}_{d,a} = \mathrm{axial}(\log(
        \boldsymbol{T}_R^T \boldsymbol{R}_a \boldsymbol{T}_0))

    with `\boldsymbol{x}_C` the mean of `\boldsymbol{x}^D_a`. The positions
    `\boldsymbol{x}^D_a` are computed relative to the base centroid of the
    element, `(\boldsymbol{X}_a - \boldsymbol{X}_C) + \boldsymbol{u}_a`,
    which leaves `\bar{\boldsymbol{u}}_d` and `\boldsymbol{T}_R` unchanged
    and makes their round-off independent of the origin of the coordinates:
    from `\boldsymbol{X}_a + \boldsymbol{u}_a` it is of the order of
    `\varepsilon |\boldsymbol{X}_a|`, with `\varepsilon` the machine
    precision, and the Newton-Raphson iterations stagnate above the
    round-off floor of :mod:`pyfe3d.solver.convergence` on fine meshes far
    from the origin (derived here, measured with
    ``test_cantilever_end_shear_translated`` of
    ``tests/test_corotational_benchmarks.py``). The CR frame follows the frame
    rule of the element, applied to `\boldsymbol{x}^D_a`:

    - Tria3R and Tria3DSG, side 1-2 alignment ([1]_, Appendix B.2.1, Eq.
      (115); Battini and Pacoste [3]_, Eqs. (2)-(5)). The frame, and thus the
      solution, depends on the numbering of the nodes ([1]_, B.2.1; [3]_,
      Section 1).
    - Quad4 and Quad4R, `z` along `\boldsymbol{v}_{42} \times
      \boldsymbol{v}_{13}` and `x` along `(\boldsymbol{v}_{13} +
      \boldsymbol{v}_{42})/2`, invariant to a cyclic renumbering of the
      nodes.

    The spin-fitter matrix `\boldsymbol{G}` of these frames, `\delta
    \boldsymbol{\omega}_R = \boldsymbol{G} \delta \bar{\boldsymbol{v}}`, was
    derived here; with `\bar{\boldsymbol{x}}_a` the current local
    coordinates and `w, v` the local translations:

    - triangle, `\bar{\boldsymbol{x}}_2 - \bar{\boldsymbol{x}}_1 = (a_1, 0,
      0)`, `\bar{\boldsymbol{x}}_3 - \bar{\boldsymbol{x}}_1 = (a_2, b_2,
      0)`, `n = a_1 b_2`: `\omega_1 = -(a_2 (w_2 - w_1) - a_1 (w_3 -
      w_1))/n`, `\omega_2 = -(w_2 - w_1)/a_1`, `\omega_3 = (v_2 - v_1)/a_1`,
      the last one as in [3]_, Eq. (5);
    - quadrilateral, `\boldsymbol{d}_1 = \bar{\boldsymbol{x}}_3 -
      \bar{\boldsymbol{x}}_1 = (a_1, b_1, 0)`, `\boldsymbol{d}_2 =
      \bar{\boldsymbol{x}}_2 - \bar{\boldsymbol{x}}_4 = (a_2, b_2, 0)`,
      `n_z = a_2 b_1 - a_1 b_2`, `c = |\boldsymbol{d}_1 +
      \boldsymbol{d}_2|/2`: `\omega_1 = -(a_1 (w_2 - w_4) - a_2 (w_3 -
      w_1))/n_z`, `\omega_2 = (b_2 (w_3 - w_1) - b_1 (w_2 - w_4))/n_z`,
      `\omega_3 = ((v_3 - v_1) + (v_2 - v_4))/(2 c)`.

    Both satisfy `\boldsymbol{G} \boldsymbol{S} = \boldsymbol{I}` ([1]_,
    Eq. (17)) and the condition (37) of [1]_, such that the projection
    geometric stiffness of [1]_, Eq. (47), is exact, which is verified by
    finite differences in ``tests/test_corotational_core.py``.

    The internal force and the tangent stiffness matrix are those of the
    CSE formulation of [1]_, Section 6.3, Eq. (53), with the projector
    `\boldsymbol{P} = \boldsymbol{P}_u - \boldsymbol{S} \boldsymbol{G}`
    ([1]_, Eqs. (11), (12), (31)-(34)):

    .. math::

        \bar{\boldsymbol{p}} = \boldsymbol{P}^T \bar{\boldsymbol{H}}^T
        \boldsymbol{K}_e \boldsymbol{d}, \qquad \boldsymbol{K}_R =
        \boldsymbol{P}^T \bar{\boldsymbol{H}}^T \boldsymbol{K}_e
        \bar{\boldsymbol{H}} \boldsymbol{P} + \boldsymbol{P}^T
        \bar{\boldsymbol{L}} \boldsymbol{P} - \boldsymbol{F}_{nm}
        \boldsymbol{G} - \boldsymbol{G}^T \boldsymbol{F}_n^T \boldsymbol{P}

    with `\bar{\boldsymbol{H}} = \mathrm{diag}(\boldsymbol{I},
    \boldsymbol{H}(\bar{\boldsymbol{\theta}}_{d,a}))` ([1]_, Eq. (20)),
    `\bar{\boldsymbol{L}} = \mathrm{diag}(\boldsymbol{0},
    \boldsymbol{L}(\bar{\boldsymbol{\theta}}_{d,a}, \bar{\boldsymbol{m}}_a))`
    with the moments of `\boldsymbol{K}_e \boldsymbol{d}` ([1]_, Eqs. (21),
    (46), (102)), and `\boldsymbol{F}_{nm}`, `\boldsymbol{F}_n` built with
    `\bar{\boldsymbol{p}}` ([1]_, Eq. (44)). The first term is the material
    stiffness ([1]_, Eq. (43)), the others the geometric stiffness ([1]_,
    Eqs. (45)-(47)). With `\boldsymbol{\mathcal{T}} =
    \mathrm{diag}(\boldsymbol{T}_R)` and `\hat{\boldsymbol{\Lambda}} =
    \mathrm{diag}(\boldsymbol{I}, \boldsymbol{\Lambda}(\boldsymbol{\psi}_a))`,
    the force and the stiffness in the additive degrees-of-freedom are
    (Crisfield [4]_, Section 17.4.2, Eqs. (17.176)-(17.177))

    .. math::

        \boldsymbol{f} = \hat{\boldsymbol{\Lambda}}^T \boldsymbol{\mathcal{T}}
        \bar{\boldsymbol{p}}, \qquad \boldsymbol{K} =
        \hat{\boldsymbol{\Lambda}}^T \boldsymbol{\mathcal{T}} \boldsymbol{K}_R
        \boldsymbol{\mathcal{T}}^T \hat{\boldsymbol{\Lambda}} +
        \mathrm{diag}\left(\boldsymbol{0}, \frac{\partial
        (\boldsymbol{\Lambda}^T \boldsymbol{m}_a)}{\partial
        \boldsymbol{\psi}_a}\right)

    with `\boldsymbol{m}_a` the moments of `\boldsymbol{\mathcal{T}}
    \bar{\boldsymbol{p}}`, see :func:`dLambdaT_m`. ``evaluate`` returns the
    material part `\hat{\boldsymbol{\Lambda}}^T \boldsymbol{\mathcal{T}}
    \boldsymbol{K}_M \boldsymbol{\mathcal{T}}^T \hat{\boldsymbol{\Lambda}}`
    and the geometric part, the rest. The matrix is unsymmetric away from
    equilibrium ([1]_, Section 6; [4]_, Section 17.4.2).

    Warped quadrilaterals. `\boldsymbol{K}_e` and ``KC0`` use the flat
    projection of the element (``update_KC0`` ignores the local `z`), whose
    nodes are at `\tilde{z}_a` from the warped nodes along the normal of
    the mean plane, with `\tilde{z}_1 = \tilde{z}_3 = -\tilde{z}_2 =
    -\tilde{z}_4` for the frame of the diagonals. They do not annihilate
    the rigid rotations of the warped element, which makes the linear
    element too stiff in warped meshes, the "warping sensitivity" of [2]_,
    Section 1. With ``warping=None`` the flat element is evaluated at the
    warped nodes, and the projector built on the warped geometry restores
    the invariance to rigid motions ([2]_, Section 1; [1]_, Sections 4.2
    and 6.5): the tangent at the base configuration is `\boldsymbol{P}^T
    \boldsymbol{K}_e \boldsymbol{P}`. With ``warping='offsets'``, the
    default, the flat element is linked to the nodes by rigid offsets that
    rotate with the nodal triads, the eccentric nodes of [1]_, Sections 5.1
    and 6, whose formulation (their reference [37]) is not given there. The
    following was derived here. With `\boldsymbol{e}_a = \tilde{z}_a
    \boldsymbol{T}_0 \boldsymbol{e}_3` and `\boldsymbol{r}_a =
    \boldsymbol{R}_a \boldsymbol{e}_a`, the flat element has the nodes
    `\boldsymbol{x}^D_a - \boldsymbol{r}_a`, the rotations
    `\boldsymbol{R}_a` and the base local coordinates `(\tilde{x}_a,
    \tilde{y}_a, 0)`, from which `\boldsymbol{d}` is measured; its
    degrees-of-freedom vary as

    .. math::

        \delta \boldsymbol{u}^f_a = \delta \boldsymbol{u}_a +
        \mathrm{Spin}(\boldsymbol{r}_a) \boldsymbol{\Lambda}_a \delta
        \boldsymbol{\psi}_a, \qquad \delta \boldsymbol{\psi}^f_a = \delta
        \boldsymbol{\psi}_a

    i.e. `\delta \boldsymbol{v}^f = \boldsymbol{J} \delta \boldsymbol{v}`.
    With `\boldsymbol{f}^f` and `\boldsymbol{K}^f` the force and the
    tangent of the flat element given above, and `\boldsymbol{n}_a` the
    nodal forces of `\boldsymbol{f}^f`,

    .. math::

        \boldsymbol{f} = \boldsymbol{J}^T \boldsymbol{f}^f, \qquad
        \boldsymbol{K} = \boldsymbol{J}^T \boldsymbol{K}^f \boldsymbol{J} +
        \mathrm{diag}\left(\boldsymbol{0}, \frac{\partial
        (\boldsymbol{\Lambda}_a^T (\boldsymbol{n}_a \times
        \boldsymbol{r}_a))}{\partial \boldsymbol{\psi}_a}\right)

    The last term, the eccentricity geometric stiffness of [1]_, Section
    5.1, is `\partial(\boldsymbol{\Lambda}_a^T \boldsymbol{m})/\partial
    \boldsymbol{\psi}_a` at `\boldsymbol{m} = \boldsymbol{n}_a \times
    \boldsymbol{r}_a` (:func:`dLambdaT_m`) minus `\boldsymbol{\Lambda}_a^T
    \mathrm{Spin}(\boldsymbol{n}_a) \mathrm{Spin}(\boldsymbol{r}_a)
    \boldsymbol{\Lambda}_a`, from `\delta \boldsymbol{r}_a =
    -\mathrm{Spin}(\boldsymbol{r}_a) \boldsymbol{\Lambda}_a \delta
    \boldsymbol{\psi}_a`. The material part is `\boldsymbol{J}^T
    \boldsymbol{K}^f_M \boldsymbol{J}`. At the base configuration
    `\boldsymbol{K}` is ``KC0`` transformed by `\boldsymbol{J}`, the linear
    flat element with rigid links, which annihilates the rigid motions of
    the warped element; it differs from ``KC0`` by the order of the warp
    ratio. Both treatments keep the non-zero eigenvalues of ``KC0`` at the
    base configuration. The wrapper sets the attribute ``warp_offsets`` of a
    Quad4 or Quad4R to ``0``, which switches off the linear rigid offsets of
    the element itself: those of ``'offsets'`` are the exact ones above,
    and ``None`` uses the flat element without offsets. The stress recovery
    after :meth:`update_probe` is therefore that of the flat element.
    ``KC0`` of a :class:`CorotationalModel` is computed with the element's
    own offsets, ``warp_offsets = 1``, for ``'offsets'``, giving
    `\boldsymbol{J}^T \boldsymbol{K}_{C0} \boldsymbol{J}`, the tangent of
    ``'offsets'`` at the base configuration, and with ``warp_offsets = 0``
    for ``None``. In the linear analysis of a cantilever twisted by
    90 degrees (length 12, width 1.1, thickness 0.32, tip loads), with 6x1
    Quad4 elements (warp ratio 0.049), the tip displacements with
    ``'offsets'`` are within 0.7 % of those of a 96x16 mesh, 8-11 % too
    small with ``None`` and 74-91 % too small with the ``KC0`` of the flat
    element, ``warp_offsets = 0``; a coarser case
    is checked in ``tests/test_corotational_core.py``.

    Von Kármán local strains. With ``local='von_karman'`` the local element
    computation is step 8 of the procedure of [4]_, Section 17.1.5: the
    existing element routines are entered with the local displacements and
    give the local internal force vector and the local tangent stiffness
    matrix, which [4]_ transforms as `\boldsymbol{K}_t = \boldsymbol{F}^T
    \boldsymbol{K}_{tl} \boldsymbol{F} + \boldsymbol{K}_{t\sigma}` (step
    11). Here the element's own ``update_probe_finte(prop, nonlinear=1)``,
    ``update_KCNL`` and ``update_KG`` are evaluated with ``probe.ue = d``,
    by the identity trick of `\boldsymbol{K}_e`, giving
    `\bar{\boldsymbol{f}}(\boldsymbol{d})` and the local tangent
    `\boldsymbol{K}_e + \boldsymbol{K}_{CNL}(\boldsymbol{d}) +
    \boldsymbol{K}_G(\boldsymbol{d})`, the exact Jacobian of
    `\bar{\boldsymbol{f}}` (``update_fint`` of the elements).
    `\bar{\boldsymbol{f}}` replaces `\boldsymbol{K}_e \boldsymbol{d}` in
    `\bar{\boldsymbol{p}}`, `\bar{\boldsymbol{L}}`, `\boldsymbol{F}_n` and
    `\boldsymbol{F}_{nm}` above, and the local tangent replaces
    `\boldsymbol{K}_e` in the material stiffness. [1]_ obtain the material
    stiffness from the variation of the local force, their Eq. (41), with
    `\boldsymbol{K}_e = \partial \boldsymbol{p}_e/\partial \boldsymbol{v}_d`
    (their Eq. (42)), and note that their Eq. (43) holds only if
    `\boldsymbol{K}_e` does not depend on the deformational freedoms; the
    chain rule in Eq. (41) then gives `\boldsymbol{P}^T
    \bar{\boldsymbol{H}}^T (\partial \bar{\boldsymbol{f}}/\partial
    \boldsymbol{d}) \bar{\boldsymbol{H}} \boldsymbol{P}` (derived here),
    whereas the geometric terms of their Eqs. (44)-(47), which depend on
    the force and not on its derivative, keep their form. The material part
    returned by :meth:`evaluate` takes `\boldsymbol{K}_e +
    \boldsymbol{K}_{CNL}` and the geometric part takes `\boldsymbol{K}_G`,
    both transformed as the material stiffness: the split of
    ``kinematics='von_karman'`` in :class:`CorotationalModel`. ``KCNL`` of
    the elements includes the geometric term of the axial force of the
    non-linear strains, see their ``update_KCNL``.

    In the CR frames of the shells the transverse translations of
    `\boldsymbol{d}` are zero (triangles, whose frame contains the three
    nodes) or the hourglass pattern `\pm \bar{z}` (quadrilaterals, whose
    `x y` plane is parallel to both diagonals and contains the mean of the
    nodes), and the von Kármán terms of Tria3R, Tria3DSG, Quad4 and Quad4R
    take the slopes of `w` from the nodal translations only, at the
    centroid for the quadrilaterals, where the hourglass pattern has no
    slope. These terms therefore vanish, and ``local='von_karman'`` gives
    the internal force and the tangent of ``'linear'`` to round-off
    (derived here, checked in ``tests/test_corotational_core.py``); the
    same holds for BeamLR, see :class:`CorotationalBeam`.

    References
    ----------
    .. [1] Felippa, C. A. and Haugen, B., A unified formulation of
       small-strain corotational finite elements: I. Theory, CMAME 194,
       2285-2335, 2005, https://doi.org/10.1016/j.cma.2004.07.035.
    .. [2] Nour-Omid, B. and Rankin, C. C., Finite rotation analysis and
       consistent linearization using projectors, CMAME 93, 353-384, 1991,
       https://doi.org/10.1016/0045-7825(91)90248-5.
    .. [3] Battini, J.-M. and Pacoste, C., On the choice of local element
       frame for corotational triangular shell elements, Communications in
       Numerical Methods in Engineering, 20, 819-825, 2004,
       https://doi.org/10.1002/cnm.708.
    .. [4] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.

    """
    def __init__(self, elem, prop, x, warping='offsets', local='linear',
                 **elem_kwargs):
        if type(elem) not in _SHELLS:
            raise TypeError('%s is not a Tria3R, Tria3DSG, Quad4 or Quad4R'
                            % type(elem).__name__)
        if warping not in _WARPING:
            raise ValueError('warping must be one of %s, got %r'
                             % (', '.join(repr(w) for w in _WARPING),
                                warping))
        self.warping = warping
        if hasattr(elem, 'warp_offsets'):
            # NOTE the wrapper treats the warping itself, see the Notes
            elem.warp_offsets = 0
        super().__init__(elem, prop, x, local=local, **elem_kwargs)
        if self.nn == 4:
            d1 = self.xtilde[2, :2] - self.xtilde[0, :2]
            d2 = self.xtilde[1, :2] - self.xtilde[3, :2]
            area = 0.5*abs(d1[0]*d2[1] - d1[1]*d2[0])
            self.warp_ratio = np.abs(self.xtilde[:, 2]).max()/np.sqrt(area)
        else:
            self.warp_ratio = 0.
        if warping == 'offsets' and self.nn == 4 and self.warp_ratio > 0.:
            z = np.zeros_like(self.xtilde)
            z[:, 2] = self.xtilde[:, 2]
            self._e = z @ self.T0.T
            self._xt = self.xtilde - z
        if warping is None and self.warp_ratio > WARP_RATIO_WARNING:
            warnings.warn('warped %s, warp ratio %.3g > %g with warping=None:'
                          ' the element is too stiff in warped meshes, use '
                          "warping='offsets'"
                          % (type(elem).__name__, self.warp_ratio,
                             WARP_RATIO_WARNING),
                          RuntimeWarning, stacklevel=2)
        elif self.warp_ratio > WARP_RATIO_WARNING_OFFSETS:
            warnings.warn('warped %s, warp ratio %.3g > %g: the rigid offsets'
                          ' were verified up to this warp ratio only, refine '
                          'the mesh' % (type(elem).__name__, self.warp_ratio,
                                        WARP_RATIO_WARNING_OFFSETS),
                          RuntimeWarning, stacklevel=2)

    def _frame_kind(self):
        return FRAME_TRIA if self.nn == 3 else FRAME_QUAD

    def _base_frame(self):
        T0 = self._frame(self.X0, None)[0]
        if not np.allclose(T0, _element_rotation_matrix(self.elem), rtol=0,
                           atol=1e-10):
            raise ValueError('r11..r33 of the element do not match the '
                             'frame of x: call elem.update_rotation_matrix(x)'
                             ' with the same x before')
        return T0


class CorotationalBeam(_CorotationalElement):
    r"""Co-rotational (EICR) wrapper of a :class:`.BeamC` or
    :class:`.BeamLR` element

    As :class:`CorotationalShell`, with the CR frame of the mean nodal
    triad.

    Usually created by :meth:`CorotationalModel.add`.

    Parameters
    ----------
    elem : BeamC or BeamLR
        The element, with ``c1, c2`` set and ``update_rotation_matrix(vxyi,
        vxyj, vxyk, x)`` called with the same ``x``.
    prop : BeamProp
        Property of the element.
    x : array_like
        Initial nodal coordinates of the model, size ``3*num_nodes``.
    q2_min : float, optional
        Smallest value of `\bar{q}_2`, see the Notes, below which the CR
        frame is taken as undefined: the forces and the stiffness are then
        NaN, such that the non-linear solvers reduce the increment, and a
        warning is issued. Default ``1.e-2``.
    exact_KGP : bool, optional
        Whether the projection geometric stiffness includes the variation
        of `\boldsymbol{G}`, which depends on the nodal rotations, see the
        Notes. ``False`` gives Felippa and Haugen's Eq. (47) alone. Default
        ``True``.
    local : str, optional
        ``'linear'`` or ``'von_karman'``, see :class:`CorotationalShell`
        and the Notes. Default ``'linear'``.
    **elem_kwargs
        Passed to ``update_KC0`` of the element.

    Attributes
    ----------
    q0 : np.ndarray
        Base `y` axis of the element, `\boldsymbol{q}_0 = \boldsymbol{T}_0
        \boldsymbol{e}_2`.

    The other attributes are those of :class:`CorotationalShell`, without
    ``warp_ratio``.

    Raises
    ------
    TypeError
        When ``elem`` is not a BeamC or a BeamLR.
    ValueError
        When ``r11..r33`` of the element are not orthonormal or their `x`
        axis is not along the line from node 1 to node 2 of ``x``, i.e.
        ``update_rotation_matrix`` was not called with ``x``, or when
        ``local`` is not ``'linear'`` or ``'von_karman'``.

    Notes
    -----
    The CR frame is the mean nodal triad, Crisfield [1]_, Eqs.
    (17.11)-(17.14), the "midpoint interpolation" of Nour-Omid and Rankin
    [2]_, Section 5:

    .. math::

        \boldsymbol{e}_1 = \frac{\boldsymbol{x}^D_2 -
        \boldsymbol{x}^D_1}{l}, \qquad \boldsymbol{q} = \frac{1}{2}
        (\boldsymbol{R}_1 + \boldsymbol{R}_2) \boldsymbol{q}_0, \qquad
        \boldsymbol{e}_3 = \frac{\boldsymbol{e}_1 \times
        \boldsymbol{q}}{|\boldsymbol{e}_1 \times \boldsymbol{q}|}, \qquad
        \boldsymbol{e}_2 = \boldsymbol{e}_3 \times \boldsymbol{e}_1

    With `\bar{\boldsymbol{q}} = \boldsymbol{T}_R^T \boldsymbol{q} = (q_1,
    q_2, 0)` and `\bar{\boldsymbol{q}}_i = \boldsymbol{T}_R^T
    \boldsymbol{R}_i \boldsymbol{q}_0`, its spin-fitter matrix (derived
    here) is

    .. math::

        \omega_1 = \frac{1}{2 q_2} \sum_i (\bar{q}_{i2} \omega_{i,x} -
        \bar{q}_{i1} \omega_{i,y}) - \frac{q_1}{q_2} \frac{w_2 - w_1}{l},
        \qquad \omega_2 = -\frac{w_2 - w_1}{l}, \qquad \omega_3 = \frac{v_2
        - v_1}{l}

    with `\omega_{i,x}, \omega_{i,y}` the local spins of the nodes. The
    frame is undefined when `\boldsymbol{q}` becomes parallel to the
    chord, `q_2 \to 0`, see ``q2_min``; [2]_, Appendix B.3, redefine
    `\boldsymbol{q}` in that case, which a stateless formulation cannot do
    without a jump of the forces.

    `\boldsymbol{G}` depends on the nodal rotations, so the condition (37)
    of Felippa and Haugen [3]_ does not hold and their Eq. (47) misses a
    term of the order of the unbalanced force ([3]_, Eq. (48)). With
    ``exact_KGP=True`` the projection geometric stiffness is the exact
    variation of `\boldsymbol{P}^T \boldsymbol{p}` (derived here),

    .. math::

        \boldsymbol{K}_{GP} = -\frac{\partial (\boldsymbol{G}^T
        \boldsymbol{z})}{\partial \bar{\boldsymbol{v}}} - \boldsymbol{G}^T
        \boldsymbol{F}_n(\boldsymbol{p})^T \boldsymbol{P}, \qquad
        \boldsymbol{z} = \boldsymbol{S}^T \boldsymbol{p}

    with `\boldsymbol{p} = \bar{\boldsymbol{H}}^T \boldsymbol{K}_e
    \boldsymbol{d}` the unprojected force, `\delta l = \delta \bar{u}_2 -
    \delta \bar{u}_1` and `\delta \bar{\boldsymbol{q}}_i =
    \mathrm{Spin}(\bar{\boldsymbol{q}}_i)(\boldsymbol{G} \delta
    \bar{\boldsymbol{v}} - \delta \bar{\boldsymbol{\omega}}_i)`; it is
    checked by finite differences in ``tests/test_corotational_core.py``.

    With ``local='von_karman'`` (see :class:`CorotationalShell`), the
    deformational translations of the nodes are along the chord, `\bar{v} =
    \bar{w} = 0`. BeamLR interpolates `v` and `w` linearly, so its von
    Kármán terms vanish and the result is that of ``'linear'`` to
    round-off. BeamC interpolates them with the nodal rotations, and its
    von Kármán axial strain `\epsilon_{xx} = u_{,x} + \bar{\epsilon}_{NL}`,
    with `\bar{\epsilon}_{NL}` the element mean of `(v_{,x}^2 +
    w_{,x}^2)/2` (see :meth:`pyfe3d.beamc.BeamC.update_KCNL`), gives the
    local "higher-order terms" of [1]_, Section 17.1.5: with an infinite
    transverse shear stiffness, it is the strain of [1]_, Eqs.
    (17.56a)-(17.56b), for the initially straight element, cubic `v` and
    `w`, which is checked in ``tests/test_corotational_core.py``. Measured
    on the cantilever of Sze, Liu and Lo [4]_, Section 3.1, at the end
    force ``P_max``, against the elastica: with ``'linear'``, 8 and 16
    BeamC give errors of `-U` and `W` of -0.07 %, +0.17 % and -0.02 %,
    +0.05 %; with ``'von_karman'``, 4, 8 and 16 BeamC give -0.013 %,
    -0.010 %; +0.001 %, +0.006 % and +0.002 %, +0.007 %.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, Wiley, 1997.
    .. [2] Nour-Omid, B. and Rankin, C. C., Finite rotation analysis and
       consistent linearization using projectors, CMAME 93, 353-384, 1991,
       https://doi.org/10.1016/0045-7825(91)90248-5.
    .. [3] Felippa, C. A. and Haugen, B., A unified formulation of
       small-strain corotational finite elements: I. Theory, CMAME 194,
       2285-2335, 2005, https://doi.org/10.1016/j.cma.2004.07.035.
    .. [4] Sze, K. Y., Liu, X. H. and Lo, S. H., Popular benchmark problems
       for geometric nonlinear analysis of shells, Finite Elements in
       Analysis and Design, 40(11), 1551-1569, 2004,
       https://doi.org/10.1016/j.finel.2003.11.001.

    """
    _types = _BEAMS

    def __init__(self, elem, prop, x, q2_min=1.e-2, exact_KGP=True,
                 local='linear', **elem_kwargs):
        if type(elem) not in self._types:
            raise TypeError('%s is not a %s' % (
                type(elem).__name__,
                ' or '.join(sorted(c.__name__ for c in self._types))))
        self.q2_min = q2_min
        self.exact_KGP = exact_KGP
        self._warned = False
        super().__init__(elem, prop, x, local=local, **elem_kwargs)

    def _base_frame(self):
        T0 = _element_rotation_matrix(self.elem)
        d = self.X0[1] - self.X0[0]
        if (not np.allclose(T0.T @ T0, _I3, rtol=0, atol=1e-10)
                or not np.allclose(T0[:, 0], d/np.sqrt(d @ d), rtol=0,
                                   atol=1e-10)):
            raise ValueError('r11..r33 of the element do not match x: call '
                             'elem.update_rotation_matrix(vxyi, vxyj, vxyk, '
                             'x) with the same x before')
        self.q0 = T0[:, 1].copy()
        return T0

    def _frame_kind(self):
        return FRAME_MEAN_TRIAD

    def _kgp_kind(self):
        return KGP_BEAM if self.exact_KGP else KGP_FH


class CorotationalTruss(CorotationalBeam):
    r"""Co-rotational wrapper of a :class:`.Truss` element

    Usually created by :meth:`CorotationalModel.add`.

    Parameters
    ----------
    elem : Truss
        The element, with ``c1, c2`` set and ``update_rotation_matrix(x)``
        called with the same ``x``.
    prop : BeamProp
        Property of the element.
    x : array_like
        Initial nodal coordinates of the model, size ``3*num_nodes``.
    frame : str, optional
        - ``'chord'``: the axial force only, along the current axis of the
          element; the rotational degrees-of-freedom receive no force and
          no stiffness, as in the linear element;
        - ``'mean_triad'``: the frame of :class:`CorotationalBeam`, `x`
          along the current axis and `y` from the mean nodal triad, which
          also gives the torsion moment;
        - ``'auto'``: ``'chord'`` when the torsional stiffness `G J` of the
          element is zero, ``'mean_triad'`` otherwise.

        Default ``'auto'``.
    q2_min, exact_KGP : optional
        See :class:`CorotationalBeam`, used with ``frame='mean_triad'``.
    **elem_kwargs
        Passed to ``update_KC0`` of the element.

    Attributes
    ----------
    frame : str
        ``'chord'`` or ``'mean_triad'``. With ``'chord'``,
        :meth:`kinematics` gives the base frame turned by the smallest
        rotation that aligns its `x` axis with the current axis, and zero
        deformational rotations.
    l0 : float
        Initial length.

    The other attributes are those of :class:`CorotationalBeam`.

    Raises
    ------
    TypeError
        When ``elem`` is not a Truss.
    ValueError
        When ``frame`` is not one of the three, or is ``'chord'`` with a
        torsional stiffness.

    Notes
    -----
    With ``frame='chord'``, the element is the co-rotational truss of
    Crisfield [1]_, Section 3.6, with the engineering strain, Eqs.
    (3.123)-(3.124), written for the space truss of [1]_, Section 3.7: with
    `l_0` and `l` the initial and current lengths, `\boldsymbol{e}` the unit
    vector from node 1 to node 2 and `N = (E A/l_0) (l - l_0)`, the
    internal force and the tangent stiffness matrix are

    .. math::

        \boldsymbol{f} = N \begin{bmatrix} -\boldsymbol{e} \\
        \boldsymbol{e} \end{bmatrix}, \qquad \boldsymbol{K} =
        \begin{bmatrix} \boldsymbol{K}_t & -\boldsymbol{K}_t \\
        -\boldsymbol{K}_t & \boldsymbol{K}_t \end{bmatrix}, \qquad
        \boldsymbol{K}_t = \frac{E A}{l_0} \boldsymbol{e} \boldsymbol{e}^T +
        \frac{N}{l} (\boldsymbol{I} - \boldsymbol{e} \boldsymbol{e}^T)

    on the translations. The first term of `\boldsymbol{K}_t` is the
    material part ([1]_, Eq. (3.131)) and the second one the geometric part
    ([1]_, Eqs. (3.132)-(3.138) for the planar truss; the space form was
    written here and is checked by finite differences in
    ``tests/test_corotational_core.py``). `E A/l_0` is the axial entry of
    the local stiffness matrix of the element. With ``frame='mean_triad'``
    the torsion moment requires the nodal triads, and the frame is
    undefined when the mean of the nodal `y` axes becomes parallel to the
    axis, see ``q2_min`` of :class:`CorotationalBeam`; with the nodal
    rotations constrained, this happens after a rotation of the axis of
    about 90 degrees, which ``'chord'`` avoids.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 1: Essentials, Wiley, 1991.

    """
    _types = _TRUSSES

    def __init__(self, elem, prop, x, frame='auto', q2_min=1.e-2,
                 exact_KGP=True, **elem_kwargs):
        if frame not in ('auto', 'chord', 'mean_triad'):
            raise ValueError("frame must be 'auto', 'chord' or "
                             "'mean_triad', got %r" % (frame,))
        super().__init__(elem, prop, x, q2_min=q2_min, exact_KGP=exact_KGP,
                         **elem_kwargs)
        torsion = self.Ke[3, 3] != 0.
        if frame == 'auto':
            frame = 'mean_triad' if torsion else 'chord'
        if frame == 'chord' and torsion:
            raise ValueError("frame='chord' ignores the torsional stiffness "
                             "of the element: use frame='mean_triad'")
        self.frame = frame
        d0 = self.X0[1] - self.X0[0]
        self.l0 = np.sqrt(d0 @ d0)

    def _frame_kind(self):
        if getattr(self, 'frame', 'mean_triad') == 'mean_triad':
            return FRAME_MEAN_TRIAD
        # NOTE with 'chord', for the post-processing only: the base frame
        #      turned by the smallest rotation that aligns x with the axis,
        #      and zero deformational rotations, since the chord does not
        #      define the nodal rotations about it
        return FRAME_CHORD

    def evaluate(self, u, tangent=True):
        r"""Internal force vector and tangent stiffness matrix, see
        :meth:`CorotationalShell.evaluate`"""
        if self.frame == 'mean_triad':
            return CorotationalBeam.evaluate(self, u, tangent=tangent)
        # NOTE Crisfield vol. 1, Section 3.6, Eqs. (3.123)-(3.124), (3.131),
        #      (3.138), space truss of Section 3.7
        ue, _ = self._split(u)
        # NOTE relative to the base centroid, see _kinematics
        XD = (self.X0 - self.xC0) + ue
        dx = XD[1] - XD[0]
        l = np.sqrt(dx @ dx)
        e = dx/l
        k = self.Ke[0, 0]
        N = k*(l - self.l0)
        f = np.zeros(12)
        f[0:3] = -N*e
        f[6:9] = N*e
        if not tangent:
            return f
        ee = np.outer(e, e)
        KM = np.zeros((12, 12))
        KG = np.zeros((12, 12))
        for ra, rb, sign in ((0, 0, 1.), (0, 6, -1.), (6, 0, -1.),
                             (6, 6, 1.)):
            KM[ra:ra + 3, rb:rb + 3] = sign*k*ee
            KG[ra:ra + 3, rb:rb + 3] = sign*N/l*(_I3 - ee)
        return f, KM, KG


class CorotationalSpring(CorotationalBeam):
    r"""Co-rotational wrapper of a :class:`.Spring` element

    Usually created by :meth:`CorotationalModel.add`.

    Parameters
    ----------
    elem : Spring
        The element, with ``c1, c2`` and its stiffnesses set and
        ``update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)`` called.
    prop : None
        Unused; the stiffnesses are attributes of the element.
    x : array_like
        Initial nodal coordinates of the model, size ``3*num_nodes``.
    frame : str, optional
        - ``'axis'``: the frame of :class:`CorotationalBeam`, `x` along
          the current axis from node 1 to node 2 and `y` from the mean
          nodal triad, for a spring whose `x` axis is the line through its
          nodes;
        - ``'triad'``: the mean nodal triad, for any orientation and for
          coincident nodes, see the Notes;
        - ``'auto'``: ``'axis'`` when the nodes are distinct, the `x`
          axis of the element is parallel to the line through them and
          ``kye = kze = 0``, ``'triad'`` otherwise.

        With ``'axis'`` the `x` axis of the frame follows the nodes, so
        there is no transverse relative displacement and ``kye``, ``kze``
        must be zero.

        Default ``'auto'``.
    q2_min : float, optional
        Smallest value of `|\boldsymbol{a}|` and `\bar{b}_2` (``'triad'``)
        or of `\bar{q}_2` (``'axis'``), below which the CR frame is taken
        as undefined, see :class:`CorotationalBeam`. Default ``1.e-2``.
    **elem_kwargs
        Passed to ``update_KC0`` of the element.

    Attributes
    ----------
    frame : str
        ``'axis'`` or ``'triad'``.
    p0, q0 : np.ndarray
        Base `x` and `y` axes of the element, the first two columns of
        `\boldsymbol{T}_0`.

    The other attributes are those of :class:`CorotationalBeam`.

    Raises
    ------
    TypeError
        When ``elem`` is not a Spring.
    ValueError
        When ``frame`` is not one of the three, when ``r11..r33`` of the
        element are not orthonormal, or with ``frame='axis'`` when the
        nodes coincide, the `x` axis of the element is not along the line
        through them, or ``kye`` or ``kze`` is not zero.

    Notes
    -----
    The local stiffness matrix `\boldsymbol{K}_e` is that of the element,
    by the identity trick of :class:`CorotationalShell`: six uncoupled
    springs between the relative translations and rotations of the nodes,
    in the element axes `\boldsymbol{T}_0`, i.e. the user orientation. For
    distinct nodes with transverse stiffnesses, `\boldsymbol{K}_e` does not
    annihilate the rigid rotations, since the linear element does not
    balance the moment of the transverse forces; the projector adds that
    moment (Felippa and Haugen [1]_, Sections 4.2 and 6.2), so the tangent
    at the base configuration differs from ``KC0``.

    The ``'triad'`` frame is derived here. With `\boldsymbol{t}_1,
    \boldsymbol{t}_2` the first two columns of `\boldsymbol{T}_0` and
    `\boldsymbol{R}_i` the nodal rotations,

    .. math::

        \boldsymbol{a} = \frac{1}{2} (\boldsymbol{R}_1 + \boldsymbol{R}_2)
        \boldsymbol{t}_1, \qquad \boldsymbol{b} = \frac{1}{2}
        (\boldsymbol{R}_1 + \boldsymbol{R}_2) \boldsymbol{t}_2, \qquad
        \boldsymbol{e}_1 = \frac{\boldsymbol{a}}{|\boldsymbol{a}|}, \qquad
        \boldsymbol{e}_3 = \frac{\boldsymbol{e}_1 \times
        \boldsymbol{b}}{|\boldsymbol{e}_1 \times \boldsymbol{b}|}, \qquad
        \boldsymbol{e}_2 = \boldsymbol{e}_3 \times \boldsymbol{e}_1

    i.e. the frame of :class:`CorotationalBeam` with the chord replaced by
    the mean of the nodal `x` axes. It depends on the nodal rotations only,
    so it is defined for coincident nodes. With the local vectors
    `\bar{\boldsymbol{p}}_i = \boldsymbol{T}_R^T \boldsymbol{R}_i
    \boldsymbol{t}_1` and `\bar{\boldsymbol{q}}_i = \boldsymbol{T}_R^T
    \boldsymbol{R}_i \boldsymbol{t}_2`, their means `\bar{\boldsymbol{a}} =
    (A, 0, 0)` and `\bar{\boldsymbol{b}} = (B_1, B_2, 0)`, and
    `\bar{\boldsymbol{\omega}}_i` the local spins of the nodes, the
    spin-fitter matrix follows from the conditions that
    `\bar{\boldsymbol{a}}` stays along `x` and `\bar{\boldsymbol{b}}` in the
    `x y` plane:

    .. math::

        \omega_3 = -\frac{1}{2 A} \sum_i
        \mathrm{Spin}(\bar{\boldsymbol{p}}_i)_2 \bar{\boldsymbol{\omega}}_i,
        \qquad \omega_2 = \frac{1}{2 A} \sum_i
        \mathrm{Spin}(\bar{\boldsymbol{p}}_i)_3 \bar{\boldsymbol{\omega}}_i,
        \qquad \omega_1 = \frac{1}{B_2} \left(B_1 \omega_2 - \frac{1}{2}
        \sum_i \mathrm{Spin}(\bar{\boldsymbol{q}}_i)_3
        \bar{\boldsymbol{\omega}}_i\right)

    with `\mathrm{Spin}(\cdot)_k` the row `k` of the matrix. It satisfies
    `\boldsymbol{G} \boldsymbol{S} = \boldsymbol{I}` ([1]_, Eq. (17)). The
    projection geometric stiffness is the exact one of
    :class:`CorotationalBeam`, with `\partial(\boldsymbol{G}^T
    \boldsymbol{z})/\partial \bar{\boldsymbol{v}}` obtained from `\delta
    \bar{\boldsymbol{p}}_i = \mathrm{Spin}(\bar{\boldsymbol{p}}_i)
    (\boldsymbol{G} \delta \bar{\boldsymbol{v}} - \delta
    \bar{\boldsymbol{\omega}}_i)` and the same for
    `\bar{\boldsymbol{q}}_i`; the rest is the CSE formulation of
    :class:`CorotationalShell`. The deformational rotations of the two
    nodes are opposite, `\pm \boldsymbol{\phi}/2`, so the rotational
    springs act on `\boldsymbol{\phi}`, close to the relative rotation of
    the nodes. Zero internal force under rigid motions and the consistency
    of the tangent with central finite differences are checked in
    ``tests/test_corotational_core.py``. The frame is undefined when the
    relative rotation of the nodes approaches 180 degrees (`A \to 0` or
    `B_2 \to 0`), see ``q2_min``.

    References
    ----------
    .. [1] Felippa, C. A. and Haugen, B., A unified formulation of
       small-strain corotational finite elements: I. Theory, CMAME 194,
       2285-2335, 2005, https://doi.org/10.1016/j.cma.2004.07.035.

    """
    _types = _SPRINGS

    def __init__(self, elem, prop, x, frame='auto', q2_min=1.e-2,
                 **elem_kwargs):
        if frame not in ('auto', 'axis', 'triad'):
            raise ValueError("frame must be 'auto', 'axis' or 'triad', got "
                             "%r" % (frame,))
        self.frame = frame
        super().__init__(elem, prop, x, q2_min=q2_min, exact_KGP=True,
                         **elem_kwargs)

    def _base_frame(self):
        T0 = _element_rotation_matrix(self.elem)
        if not np.allclose(T0.T @ T0, _I3, rtol=0, atol=1e-10):
            raise ValueError('r11..r33 of the element are not orthonormal: '
                             'call elem.update_rotation_matrix(xi, xj, xk, '
                             'vxyi, vxyj, vxyk) before')
        d = self.X0[1] - self.X0[0]
        ld = np.sqrt(d @ d)
        along = bool(ld > 0. and np.allclose(_cross(T0[:, 0], d/ld), 0.,
                                             rtol=0, atol=1e-8))
        e = self.elem
        transverse = e.kye != 0. or e.kze != 0.
        if self.frame == 'auto':
            self.frame = 'axis' if along and not transverse else 'triad'
        elif self.frame == 'axis':
            if not along:
                raise ValueError("frame='axis' requires distinct nodes and "
                                 "the x axis of the spring along the line "
                                 "through them")
            if transverse:
                # NOTE the x axis of the frame follows the nodes, so there is
                #      no transverse relative displacement in that frame
                raise ValueError("frame='axis' gives no stiffness to kye "
                                 "and kze: use frame='triad'")
        self.p0 = T0[:, 0].copy()
        self.q0 = T0[:, 1].copy()
        return T0

    def _frame_kind(self):
        return FRAME_MEAN_TRIAD if self.frame == 'axis' else FRAME_SPRING_TRIAD

    def _kgp_kind(self):
        if self.frame == 'axis':
            return KGP_BEAM if self.exact_KGP else KGP_FH
        return KGP_SPRING


def _wrapper_class(elem):
    r"""Co-rotational wrapper class of an element"""
    t = type(elem)
    if t in _SHELLS:
        return CorotationalShell
    if t in _BEAMS:
        return CorotationalBeam
    if t in _TRUSSES:
        return CorotationalTruss
    return CorotationalSpring


# ---------------------------------------------------------------------------
# assembler
# ---------------------------------------------------------------------------
_KINEMATICS = ('corotational', 'corotational_von_karman', 'von_karman',
               'linear')
#: kinematics accepted by :meth:`CorotationalModel.add` for each element;
#: Spring and Truss have no non-linear (von Kármán) internal force vector of
#: their own: they reject ``'von_karman'``, and ``'corotational_von_karman'``
#: is treated as ``'corotational'``
_MODEL_KINEMATICS = {**{cls: _KINEMATICS for cls in {**_SHELLS, **_BEAMS}},
                     Spring: ('corotational', 'corotational_von_karman',
                              'linear'),
                     Truss: ('corotational', 'corotational_von_karman',
                             'linear')}
_MOMENT_KINDS = ('rotation_vector', 'fixed_axes', 'follower')


class CorotationalModel(object):
    r"""Model with co-rotational, von Kármán and linear elements, with the
    callables of :class:`pyfe3d.solver.Analysis`

    The elements are added with :meth:`add`, each with its kinematics. The
    vectors and matrices returned by :meth:`calc_fext`, :meth:`calc_fint`,
    :meth:`calc_kC` and :meth:`calc_kG` are those of all the ``N``
    degrees-of-freedom, to be used with the ``bu`` and ``uk`` of
    :class:`pyfe3d.solver.Analysis`. The rotational degrees-of-freedom are
    total rotation vectors at the nodes of the co-rotational elements, see
    :mod:`pyfe3d.corotational`.

    Parameters
    ----------
    x : array_like
        Initial nodal coordinates, size ``3*num_nodes``, as used by the
        elements.
    N : int or None, optional
        Number of degrees-of-freedom. ``None`` gives ``6*num_nodes``.
        Default ``None``.

    Attributes
    ----------
    x : np.ndarray
        Copy of the initial nodal coordinates.
    N : int
        Number of degrees-of-freedom.
    fext : np.ndarray
        Constant external load vector, size ``N``, used by
        :meth:`calc_fext`: forces, and moments conjugate to the rotation
        vectors, see :meth:`add_moment`. It may be edited.
    corotational : list
        The wrappers (:class:`CorotationalShell`, :class:`CorotationalBeam`,
        :class:`CorotationalTruss`, :class:`CorotationalSpring`) of the
        elements added with ``kinematics='corotational'`` or
        ``'corotational_von_karman'``, in order; their attribute ``local``
        tells the two apart.
    von_karman, linear : list
        Tuples ``(elem, prop, elem_kwargs)`` of the elements added with
        ``kinematics='von_karman'`` and ``'linear'``.

    See Also
    --------
    pyfe3d.solver.analysis.Analysis : the non-linear static solvers, which
        take the callables of the model.

    Notes
    -----
    The residual seen by the solvers is

    .. math::

        \boldsymbol{R}(\boldsymbol{c}, \lambda) =
        \boldsymbol{F}_{int}(\boldsymbol{c}) - \lambda \left(\boldsymbol{F}_0
        + \sum_k \boldsymbol{F}_k(\boldsymbol{c})\right)

    with `\boldsymbol{F}_0` the attribute ``fext`` and
    `\boldsymbol{F}_k(\boldsymbol{c})` the configuration-dependent moments
    of :meth:`add_moment`. As described in :mod:`pyfe3d.solver.callbacks`,
    :meth:`calc_fint` returns `\boldsymbol{F}_{int} - \lambda \sum_k
    (\boldsymbol{F}_k(\boldsymbol{c}) - \boldsymbol{F}_k(\boldsymbol{0}))`,
    :meth:`calc_kG` contains the load stiffness `-\lambda \sum_k \partial
    \boldsymbol{F}_k/\partial \boldsymbol{c}`, and ``calc_fext(c=c)``
    returns `\boldsymbol{F}_0 + \sum_k \boldsymbol{F}_k(\boldsymbol{c}) =
    -\partial \boldsymbol{R}/\partial \lambda`.

    The contributions of the kinematics are:

    - ``'corotational'``: :meth:`CorotationalShell.evaluate` or
      :meth:`CorotationalBeam.evaluate`, the material part in
      :meth:`calc_kC` and the geometric part in :meth:`calc_kG`;
    - ``'corotational_von_karman'``: the same, with the von Kármán strains
      of the element in its CR frame (``local='von_karman'`` of the
      wrappers, see :class:`CorotationalShell`); the local ``K_e +
      KCNL(d)`` enters the material part and the local ``KG(d)`` the
      geometric part;
    - ``'von_karman'``: ``update_fint(..., nonlinear=1)`` of the element,
      ``KC0 + KCNL`` in :meth:`calc_kC` and ``KG`` in :meth:`calc_kG`;
    - ``'linear'``: ``KC0 u`` and ``KC0`` in :meth:`calc_kC`.

    Limits of validity: ``'von_karman'`` keeps the rotations of the
    elements small to moderate, the assumption of the von Kármán strains;
    the two co-rotational kinematics allow large displacements and
    rotations with small strains, each nodal rotation vector staying below
    one full revolution, `|\boldsymbol{\psi}_a| < 2 \pi`, see
    :mod:`pyfe3d.corotational`. In the CR frame, the von Kármán terms of
    the shells and of BeamLR vanish, and ``'corotational_von_karman'``
    gives the results of ``'corotational'`` for them, to round-off; it
    changes the results of BeamC only, see :class:`CorotationalBeam`.

    The rotational degrees-of-freedom of a node shared by a co-rotational
    element and an element of another kinematics are interpreted by the
    latter as small rotations, valid only while the rotations of that node
    remain small; a warning is issued at the first evaluation.

    """
    def __init__(self, x, N=None):
        self.x = np.array(x, dtype=float).ravel()
        self.N = DOF*(self.x.size//3) if N is None else int(N)
        self.fext = np.zeros(self.N)
        self.corotational = []
        self.von_karman = []
        self.linear = []
        self._moments = []
        self._KC0_parts = {k: [] for k in _KINEMATICS}
        self._KC0 = {}
        self._cr_rc = None
        self._cache = None
        self._shared_checked = False

    # NOTE set-up
    def _element_KC0(self, elem, prop, elem_kwargs):
        r"""Entries ``(r, c, v)`` of the linear stiffness matrix of one
        element in global coordinates, from its own ``update_KC0``"""
        size = _MODEL_ELEMENTS[type(elem)]().KC0_SPARSE_SIZE
        r = np.zeros(size, dtype=INT)
        c = np.zeros(size, dtype=INT)
        v = np.zeros(size)
        saved = elem.init_k_KC0
        try:
            elem.init_k_KC0 = 0
            if isinstance(elem, Spring):
                # NOTE the Spring has no coordinates and no property
                elem.update_KC0(r, c, v, **elem_kwargs)
            else:
                elem.update_probe_xe(self.x)
                elem.update_KC0(r, c, v, prop, **elem_kwargs)
        finally:
            elem.init_k_KC0 = saved
        return r, c, v

    def add(self, elem, prop, kinematics='corotational', **elem_kwargs):
        r"""Add an element to the model

        Parameters
        ----------
        elem : Tria3R, Quad4, Quad4R, BeamC, Tria3DSG, BeamLR, Spring or Truss
            The element, with its node positions ``c1, c2, ...`` set and
            ``update_rotation_matrix`` called with the coordinates ``x`` of
            the model.
        prop : ShellProp, BeamProp or None
            Property of the element; ``None`` for a Spring, whose
            stiffnesses are attributes of the element.
        kinematics : str, optional
            ``'corotational'`` (EICR, large rotations, small strains),
            ``'corotational_von_karman'`` (EICR with the von Kármán strains
            of the element in its CR frame, see :class:`CorotationalShell`),
            ``'von_karman'`` (the non-linear strains of the element,
            moderate rotations) or ``'linear'``. Default
            ``'corotational'``. Spring and Truss, which have no von
            Kármán internal force vector of their own, do not accept
            ``'von_karman'``, and ``'corotational_von_karman'`` is treated
            as ``'corotational'`` for them, without warning.
        **elem_kwargs
            Passed to ``update_KC0`` and, with ``'von_karman'`` and
            ``'corotational_von_karman'``, to ``update_fint`` or
            ``update_probe_finte`` of the element, e.g. the ``hgfactor_*``
            of :class:`.Quad4R`; with ``'corotational'`` and
            ``'corotational_von_karman'``, the keyword arguments
            of :class:`CorotationalShell` (``warping``),
            :class:`CorotationalBeam` (``q2_min``, ``exact_KGP``),
            :class:`CorotationalTruss` (``frame``, ``q2_min``,
            ``exact_KGP``) and :class:`CorotationalSpring` (``frame``,
            ``q2_min``) are also accepted.

        Returns
        -------
        wrapper : object or None
            The wrapper of a co-rotational element, a
            :class:`CorotationalShell`, :class:`CorotationalBeam`,
            :class:`CorotationalTruss` or :class:`CorotationalSpring`, also
            appended to ``corotational``; ``None`` for the other
            kinematics.

        Raises
        ------
        ValueError
            When ``kinematics`` is not one of the four, or is not accepted
            by the element type; with ``'corotational'`` and
            ``'corotational_von_karman'``, also the errors of the wrapper
            classes, e.g. when ``update_rotation_matrix`` of the element was
            not called with the coordinates ``x`` of the model.
        TypeError
            When the element type is not supported.

        """
        if kinematics not in _KINEMATICS:
            raise ValueError('kinematics must be one of %s, got %r'
                             % (', '.join(_KINEMATICS), kinematics))
        if type(elem) not in _MODEL_ELEMENTS:
            raise TypeError('%s is not supported, only %s' % (
                type(elem).__name__,
                ', '.join(sorted(c.__name__ for c in _MODEL_ELEMENTS))))
        accepted = _MODEL_KINEMATICS[type(elem)]
        if kinematics not in accepted:
            # NOTE the alias 'corotational_von_karman' of Spring and Truss is
            #      named apart
            alias = type(elem) in {**_TRUSSES, **_SPRINGS}
            names = [k for k in accepted
                     if not (alias and k == 'corotational_von_karman')]
            raise ValueError('%s supports kinematics=%s only%s, got %r'
                             % (type(elem).__name__,
                                ' or '.join(repr(k) for k in names),
                                " ('corotational_von_karman' is treated as "
                                "'corotational')" if alias else '',
                                kinematics))
        # NOTE a new kinematics is added to _KINEMATICS and
        #      _MODEL_KINEMATICS, with its own branch below, its own list of
        #      elements and its terms in calc_fint, calc_kC and calc_kG;
        #      _KC0_parts is keyed by the kinematics
        if (kinematics == 'corotational_von_karman'
                and type(elem) in {**_TRUSSES, **_SPRINGS}):
            # NOTE no von Kármán strains of their own, see the docstring
            kinematics = 'corotational'
        wrapper = None
        kc0_kwargs = dict(elem_kwargs)
        if kinematics in ('corotational', 'corotational_von_karman'):
            cls = _wrapper_class(elem)
            if kinematics == 'corotational_von_karman':
                wrapper = cls(elem, prop, self.x, local='von_karman',
                              **elem_kwargs)
            else:
                wrapper = cls(elem, prop, self.x, **elem_kwargs)
            self.corotational.append(wrapper)
            kc0_kwargs = wrapper.elem_kwargs
            self._cr_rc = None
        elif kinematics == 'von_karman':
            self.von_karman.append((elem, prop, dict(elem_kwargs)))
        else:
            self.linear.append((elem, prop, dict(elem_kwargs)))
        if (wrapper is not None and hasattr(elem, 'warp_offsets')
                and wrapper.warping == 'offsets'):
            # NOTE KC0 with the element's own rigid offsets, the tangent of
            #      warping='offsets' at the base configuration, see
            #      CorotationalShell
            elem.warp_offsets = 1
            try:
                KC0_part = self._element_KC0(elem, prop, kc0_kwargs)
            finally:
                elem.warp_offsets = 0
        else:
            KC0_part = self._element_KC0(elem, prop, kc0_kwargs)
        self._KC0_parts[kinematics].append(KC0_part)
        self._KC0 = {}
        self._cache = None
        self._shared_checked = False
        return wrapper

    def add_moment(self, node, M, kind='rotation_vector'):
        r"""Apply a concentrated moment at a node

        Parameters
        ----------
        node : int
            Node index, whose rotational degrees-of-freedom are ``6*node +
            3`` to ``6*node + 5``.
        M : array_like
            Moment vector, size 3, in global components at the base
            configuration, for a unit load factor.
        kind : str, optional
            - ``'rotation_vector'``: added to ``fext``, conjugate to the
              rotation vector `\boldsymbol{\psi}` of the node, constant;
            - ``'fixed_axes'``: moment about fixed global axes,
              configuration-dependent;
            - ``'follower'``: moment that rotates with the node,
              `\boldsymbol{R}(\boldsymbol{\psi}) \boldsymbol{M}`,
              configuration-dependent.

            Default ``'rotation_vector'``.

        Raises
        ------
        ValueError
            When ``kind`` is not one of the three, or ``M`` is not of size
            3.

        Notes
        -----
        The virtual work of a spatial moment `\boldsymbol{m}` is
        `\boldsymbol{m} \cdot \delta \boldsymbol{\omega} =
        (\boldsymbol{\Lambda}^T \boldsymbol{m}) \cdot \delta
        \boldsymbol{\psi}`, see :func:`Lambda`. Hence:

        - ``'rotation_vector'``: `\boldsymbol{F} = \boldsymbol{M}`; it is
          the moment about fixed axes when the rotation vector keeps a
          fixed direction, since `\boldsymbol{\Lambda}^T \boldsymbol{n} =
          \boldsymbol{n}` for `\boldsymbol{\psi} \parallel \boldsymbol{n}`,
          e.g. in plane problems;
        - ``'fixed_axes'``: `\boldsymbol{F} =
          \boldsymbol{\Lambda}(\boldsymbol{\psi})^T \boldsymbol{M}`, with
          the load stiffness `-\partial \boldsymbol{F}/\partial
          \boldsymbol{\psi}`, see :func:`dLambdaT_m`. Moments about fixed
          axes are non-conservative (Crisfield [1]_, Section 17.5.2,
          Fig. 17.4), so the tangent stiffness matrix is unsymmetric;
        - ``'follower'``: `\boldsymbol{m} = \boldsymbol{R} \boldsymbol{M}`
          and `\boldsymbol{F} = \boldsymbol{\Lambda}^T \boldsymbol{R}
          \boldsymbol{M} = \boldsymbol{\Lambda} \boldsymbol{M}`, since
          `\boldsymbol{\Lambda}^T \boldsymbol{R} = \boldsymbol{\Lambda}`
          (derived here and checked in the tests); the load stiffness is
          `-\partial (\boldsymbol{\Lambda} \boldsymbol{M})/\partial
          \boldsymbol{\psi} = \partial (\boldsymbol{\Lambda}^T
          \boldsymbol{M})/\partial \boldsymbol{\psi}` evaluated at
          `-\boldsymbol{\psi}`, since `\boldsymbol{\Lambda}(\boldsymbol{\psi})
          = \boldsymbol{\Lambda}(-\boldsymbol{\psi})^T`. This is the moment
          counterpart of the follower force of [1]_, Eqs.
          (17.211)-(17.212), following the nodal triad; the follower
          moments of [1]_, Section 17.5.2, Eqs. (17.216)-(17.218), follow
          the element axes instead.

        The configuration-dependent moments enter :meth:`calc_fint`,
        :meth:`calc_kG` and :meth:`calc_fext` as described in
        :class:`CorotationalModel`; use :meth:`calc_fext` as the
        ``calc_fext`` of :class:`pyfe3d.solver.Analysis`.

        References
        ----------
        .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of
           Solids and Structures, Volume 2: Advanced Topics, Wiley, 1997.

        """
        M = np.array(M, dtype=float).ravel()
        if M.size != 3:
            raise ValueError('M must have size 3')
        if kind not in _MOMENT_KINDS:
            raise ValueError('kind must be one of %s, got %r'
                             % (', '.join(_MOMENT_KINDS), kind))
        node = int(node)
        if kind == 'rotation_vector':
            self.fext[DOF*node + 3:DOF*node + 6] += M
        else:
            self._moments.append((node, M, kind))

    # NOTE linear stiffness matrices
    def _sparse(self, r, c, v):
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def _KC0_of(self, *kinds):
        r"""Assembled linear stiffness matrix of the elements of the given
        kinematics"""
        key = tuple(kinds)
        if key not in self._KC0:
            parts = [p for k in kinds for p in self._KC0_parts[k]]
            if parts:
                r, c, v = (np.concatenate(a) for a in zip(*parts))
            else:
                r = c = np.zeros(0, dtype=INT)
                v = np.zeros(0)
            self._KC0[key] = self._sparse(r, c, v)
        return self._KC0[key]

    @property
    def KC0(self):
        r"""Linear stiffness matrix of all the elements, sparse, shape ``(N,
        N)``, from their ``update_KC0``"""
        return self._KC0_of(*_KINEMATICS)

    # NOTE evaluations
    def _check_shared_nodes(self):
        if self._shared_checked:
            return
        self._shared_checked = True
        if not self.corotational:
            return
        cr = set(np.concatenate([w.nodes for w in self.corotational]))
        other = set()
        for elem, _, _ in self.von_karman + self.linear:
            other.update(_element_dofs(elem)//DOF)
        shared = sorted(cr & other)
        if shared:
            warnings.warn('%d node(s) shared by co-rotational elements and '
                          'elements of another kinematics, e.g. node %d: '
                          'valid only for small rotations of these nodes'
                          % (len(shared), shared[0]), RuntimeWarning,
                          stacklevel=3)

    def _evaluate_cr(self, u, tangent):
        r"""Assembled force and, with ``tangent``, the values of the material
        and geometric matrices of the co-rotational elements, cached for the
        last configuration"""
        cdef double[::1] fv
        cdef const double[::1] fev
        cdef Py_ssize_t[::1] dofs
        cdef Py_ssize_t i
        self._check_shared_nodes()
        cache = self._cache
        if (cache is not None and (cache[1] or not tangent)
                and np.array_equal(cache[0], u)):
            return cache[2:]
        f = np.zeros(self.N)
        fv = f
        vM = []
        vG = []
        for w in self.corotational:
            if tangent:
                fe, KM, KG = w.evaluate(u, tangent=True)
                vM.append(KM.ravel())
                vG.append(KG.ravel())
            else:
                fe = w.evaluate(u, tangent=False)
            # NOTE the scatter of np.add.at(f, w.dofs, fe), in the same order
            fev = np.ascontiguousarray(fe, dtype=np.float64)
            dofs = w._dofs_c
            for i in range(dofs.shape[0]):
                fv[dofs[i]] += fev[i]
        if tangent:
            vM = np.concatenate(vM) if vM else np.zeros(0)
            vG = np.concatenate(vG) if vG else np.zeros(0)
        else:
            vM = vG = None
        self._cache = (np.array(u, copy=True), tangent, f, vM, vG)
        return f, vM, vG

    def _cr_indices(self):
        if self._cr_rc is None:
            rs, cs = [], []
            for w in self.corotational:
                n6 = w.dofs.size
                rs.append(np.repeat(w.dofs, n6))
                cs.append(np.tile(w.dofs, n6))
            if rs:
                self._cr_rc = (np.concatenate(rs), np.concatenate(cs))
            else:
                self._cr_rc = (np.zeros(0, dtype=INT),) * 2
        return self._cr_rc

    def _von_karman(self, u, what):
        r"""``fint``, ``KCNL`` or ``KG`` of the von Kármán elements at ``u``"""
        if what == 'fint':
            f = np.zeros(self.N)
            for elem, prop, kw in self.von_karman:
                # NOTE the probe can be shared amongst the elements
                elem.update_probe_xe(self.x)
                elem.update_probe_ue(u)
                elem.update_fint(f, prop, nonlinear=1, **kw)
            return f
        rs, cs, vs = [], [], []
        for elem, prop, kw in self.von_karman:
            data = _MODEL_ELEMENTS[type(elem)]()
            size = getattr(data, what + '_SPARSE_SIZE')
            r = np.zeros(size, dtype=INT)
            c = np.zeros(size, dtype=INT)
            v = np.zeros(size)
            attr = 'init_k_' + what
            saved = getattr(elem, attr)
            try:
                setattr(elem, attr, 0)
                elem.update_probe_xe(self.x)
                elem.update_probe_ue(u)
                getattr(elem, 'update_' + what)(r, c, v, prop)
            finally:
                setattr(elem, attr, saved)
            rs.append(r)
            cs.append(c)
            vs.append(v)
        if not rs:
            return csr_matrix((self.N, self.N))
        return self._sparse(np.concatenate(rs), np.concatenate(cs),
                            np.concatenate(vs))

    def _moment_loads(self, u, tangent=False):
        r"""Sum of the configuration-dependent moments `\sum_k
        \boldsymbol{F}_k(\boldsymbol{u})` and, with ``tangent``, the
        entries ``(r, c, v)`` of `\sum_k \partial \boldsymbol{F}_k/\partial
        \boldsymbol{u}`"""
        F = np.zeros(self.N)
        rs, cs, vs = [], [], []
        for node, M, kind in self._moments:
            ra = slice(DOF*node + 3, DOF*node + 6)
            psi = np.asarray(u[ra], dtype=float)
            if kind == 'fixed_axes':
                F[ra] += Lambda(psi).T @ M
                dF = dLambdaT_m(psi, M) if tangent else None
            else:
                F[ra] += Lambda(psi) @ M
                dF = -dLambdaT_m(-psi, M) if tangent else None
            if tangent:
                idx = np.arange(DOF*node + 3, DOF*node + 6)
                rs.append(np.repeat(idx, 3))
                cs.append(np.tile(idx, 3))
                vs.append(dF.ravel())
        if not tangent:
            return F
        if rs:
            return F, (np.concatenate(rs), np.concatenate(cs),
                       np.concatenate(vs))
        return F, None

    # NOTE callables of pyfe3d.solver.Analysis
    @property
    def calc_fext(self):
        r"""External load vector, the ``calc_fext`` of
        :class:`pyfe3d.solver.Analysis`

        A callable ``calc_fext(inc=1., silent=True, c=None)``, with:

        inc : float, optional
            Load factor that multiplies the returned vector, dimensionless.
            Default ``1.``.
        silent : bool, optional
            Unused, for the signature of :class:`pyfe3d.solver.Analysis`.
        c : np.ndarray or None, optional
            Configuration of all the ``N`` degrees-of-freedom, for the
            configuration-dependent moments of :meth:`add_moment`. ``None``
            gives the reference vector, with these moments at the base
            configuration. Default ``None``.

        It returns ``inc*(fext + sum_k F_k(c))``, size ``N``, see
        :class:`CorotationalModel`.

        Notes
        -----
        :class:`pyfe3d.solver.Analysis` treats the load as
        configuration-dependent when ``calc_fext`` declares ``c`` in its
        signature, see :func:`pyfe3d.solver.callbacks.accepts_kwarg`, and a
        linear static analysis then iterates on the load. The returned
        callable declares ``c`` only when configuration-dependent moments
        (``kind='fixed_axes'`` or ``'follower'`` of :meth:`add_moment`) are
        present; otherwise ``c`` is accepted as a keyword argument and
        ignored. The attribute is read when it is passed to
        :class:`pyfe3d.solver.Analysis`, so add the moments before.

        """
        if self._moments:
            return self._calc_fext_configuration
        return self._calc_fext_constant

    def _calc_fext_configuration(self, inc=1., silent=True, c=None):
        r"""``calc_fext`` with configuration-dependent moments"""
        f = self.fext.copy()
        f += self._moment_loads(np.zeros(self.N) if c is None else c)
        return inc*f

    def _calc_fext_constant(self, inc=1., silent=True, **kwargs):
        r"""``calc_fext`` without configuration-dependent moments; ``c`` is
        accepted and ignored, but not declared"""
        unknown = set(kwargs) - {'c'}
        if unknown:
            raise TypeError('calc_fext() got an unexpected keyword argument '
                            '%r' % sorted(unknown)[0])
        return inc*self.fext.copy()

    def calc_fint(self, c, inc=1., silent=True):
        r"""Internal force vector

        Parameters
        ----------
        c : np.ndarray
            Configuration of all the ``N`` degrees-of-freedom: translations
            and rotation vectors.
        inc : float, optional
            Load factor, dimensionless, which scales the
            configuration-dependent part of the moments of
            :meth:`add_moment`. Default ``1.``.
        silent : bool, optional
            Unused, for the signature of :class:`pyfe3d.solver.Analysis`.

        Returns
        -------
        fint : np.ndarray
            Size ``N``: the internal forces of all the elements, minus
            ``inc`` times the configuration-dependent part of the moments,
            see :class:`CorotationalModel`.

        """
        u = np.asarray(c, dtype=float)
        f = self._KC0_of('linear') @ u
        if self.von_karman:
            f += self._von_karman(u, 'fint')
        if self.corotational:
            f += self._evaluate_cr(u, tangent=False)[0]
        if self._moments:
            f -= inc*(self._moment_loads(u)
                      - self._moment_loads(np.zeros(self.N)))
        return f

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        r"""Constitutive (material) stiffness matrix

        Parameters
        ----------
        c : np.ndarray or None, optional
            Configuration of all the ``N`` degrees-of-freedom. Default
            ``None``.
        NLgeom : bool, optional
            ``False`` or ``c=None`` give the linear stiffness matrix
            :attr:`KC0`. Default ``False``.
        silent : bool, optional
            Unused, for the signature of :class:`pyfe3d.solver.Analysis`.

        Returns
        -------
        kC : csr_matrix
            Shape ``(N, N)``: :attr:`KC0`, or at ``c`` the linear
            ``KC0`` of the linear elements, ``KC0 + KCNL`` of the von
            Kármán elements and the material part of the co-rotational
            tangent stiffness matrix.

        """
        if c is None or not NLgeom:
            return self.KC0
        u = np.asarray(c, dtype=float)
        K = self._KC0_of('linear', 'von_karman')
        if self.von_karman:
            K = K + self._von_karman(u, 'KCNL')
        if self.corotational:
            vM = self._evaluate_cr(u, tangent=True)[1]
            K = K + self._sparse(*self._cr_indices(), vM)
        return csr_matrix(K)

    def calc_kG(self, c=None, NLgeom=False, inc=1., silent=True):
        r"""Geometric stiffness matrix, including the load stiffness

        Parameters
        ----------
        c : np.ndarray or None, optional
            Configuration of all the ``N`` degrees-of-freedom. ``None``
            gives a null matrix. Default ``None``.
        NLgeom : bool, optional
            ``False`` gives a null matrix. Default ``False``.
        inc : float, optional
            Load factor, dimensionless, which scales the load stiffness of
            the configuration-dependent moments. Default ``1.``.
        silent : bool, optional
            Unused, for the signature of :class:`pyfe3d.solver.Analysis`.

        Returns
        -------
        kG : csr_matrix
            Shape ``(N, N)``: ``KG`` of the von Kármán elements, the
            geometric part of the co-rotational tangent stiffness matrix
            and the load stiffness ``-inc*dF_k/dc`` of the
            configuration-dependent moments. With :meth:`calc_kC` it is
            the exact Jacobian of :meth:`calc_fint`.

        """
        if c is None or not NLgeom:
            return csr_matrix((self.N, self.N))
        u = np.asarray(c, dtype=float)
        K = csr_matrix((self.N, self.N))
        if self.von_karman:
            K = K + self._von_karman(u, 'KG')
        if self.corotational:
            vG = self._evaluate_cr(u, tangent=True)[2]
            K = K + self._sparse(*self._cr_indices(), vG)
        if self._moments:
            rcv = self._moment_loads(u, tangent=True)[1]
            K = K - inc*self._sparse(*rcv)
        return csr_matrix(K)

    # NOTE post-processing
    def nodal_rotations(self, c):
        r"""Rotation matrices of the nodes

        Parameters
        ----------
        c : np.ndarray
            Configuration of all the ``N`` degrees-of-freedom.

        Returns
        -------
        R : np.ndarray
            Shape ``(N//6, 3, 3)``, ``R[a] = expso3(c[6*a + 3:6*a + 6])``,
            the rotation of node ``a`` from the base configuration, see
            :func:`expso3`.

        """
        psi = np.asarray(c, dtype=float)[:DOF*(self.N//DOF)]
        return np.array([expso3(p) for p in psi.reshape(-1, DOF)[:, 3:]])

    def deformational_displacements(self, c):
        r"""Deformational displacements of the co-rotational elements

        Parameters
        ----------
        c : np.ndarray
            Configuration of all the ``N`` degrees-of-freedom.

        Returns
        -------
        ds : list
            For each element of ``corotational``, in order, the
            deformational displacements and rotations in its CR frame, size
            ``6*nn``, or ``None`` when its frame is undefined, see
            :meth:`CorotationalShell.kinematics`.

        """
        u = np.asarray(c, dtype=float)
        return [w.kinematics(u)[1] for w in self.corotational]
