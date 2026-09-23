#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Quad4 - Quadrilateral element with mixed integration (:mod:`pyfe3d.quad4`)
==============================================================================

.. currentmodule:: pyfe3d.quad4

The :class:`.Quad4` is the recommended quadrilateral plane stress finite
element.

Another option is the :class:`pyfe3d.Quad4R` with full reduced
integration, more efficient, but with an hourglass control that creates
artificial stiffness to compensate the reduced integration.

The :class:`.Quad4` element has 6 degrees-of-freedom (DOF): `u`, `v`, `w`,
`r_x`, `r_y`, `r_z`. All DOF are interpolated bi-linearly between the nodes,
such that any of the DOF gradients can be constant over the element when the
element is rectangular.

The stiffness for the degrees of freedom `w`, `r_x` and `r_y` is based on the
paper below, where `r_x = \theta_1` and `r_y = \theta_2`:

    Hughes T.J.R., Taylor R.L., Kanoknukulchai W. "A simple and efficient
    finite element for plate bending". International Journal of  Numerical
    Methods in Engineering, Volume 11, 1977.
    https://doi.org/10.1002/nme.1620111005

Hughes et al. (1977) proposed the following integration scheme:

- For thin plates, when `h/\ell < 1`, where `\ell` is the element
  characteristic length, here calculated as the square root of the element area
  `\ell = \sqrt{\text{area}}`

-- two-by-two quadrature for the bending energy terms

-- one-point quadrature for the transverse shear energy terms

- For thick plates, when `h/\ell >= 1`

-- two-by-two quadrature for the bending energy terms

-- two-by-two quadrature for the transverse shear terms with gradients

-- one-point quadrature for the transverse shear terms without gradients

The membrane stiffness terms, which are not specified in the paper of Hughes
et al. (1977), are integrated with a three-by-three quadrature when the
default drilling model is active and with a two-by-two quadrature otherwise,
for the reason explained under the drilling stiffness below. The bending term
always uses two-by-two, so that the drilling model does not change the
bending response of the element.

The transverse shear stiffnesses `A_{44}`, `A_{45}` and `A_{55}` are read
from the :class:`pyfe3d.shellprop.ShellProp` object with the shear correction
already applied, see :meth:`pyfe3d.shellprop.ShellProp.calc_transverse_shear_stiffness`,
and no shear correction factor is applied by the element. When a material
direction is defined, they are brought to the element coordinate system with
:meth:`pyfe3d.shellprop.ShellProp.calc_Ats_element`, which re-evaluates the
equilibrium-based stiffness of Rohwer (1988) with the plies rotated to the
element coordinate system, such that the assumed cylindrical bending states
are posed along the element axes.

Drilling stiffness
------------------

The FSDT kinematics contains no strain measure associated with `r_z`, so the
rows and columns of the drilling degree-of-freedom would be empty and a mesh
of coplanar elements would give a singular global stiffness matrix. Two
models are available, selected with the ``drilling_model`` attribute.

**Physics-based, the default** (``drilling_model = 0``). The in-plane
displacement field is enriched so that the drilling rotations produce
membrane strain energy, following Allman, and the independently interpolated
`r_z` is tied to the rotation of the membrane field by the regularised
functional of Hughes and Brezzi. The combination for the quadrilateral is the
one of Ibrahimbegovic, Taylor and Wilson:

    Allman, D. J., 1984, "A compatible triangular element including vertex
    rotations for plane elasticity analysis," Computers & Structures,
    19(1-2), pp. 1-8. https://doi.org/10.1016/0045-7949(84)90197-4

    Hughes, T. J. R., and Brezzi, F., 1989, "On drilling degrees of
    freedom," Computer Methods in Applied Mechanics and Engineering, 72(1),
    pp. 105-121. https://doi.org/10.1016/0045-7825(89)90124-2

    Ibrahimbegovic, A., Taylor, R. L., and Wilson, E. L., 1990, "A robust
    quadrilateral membrane finite element with drilling degrees of freedom,"
    International Journal for Numerical Methods in Engineering, 30(3), pp.
    445-457. https://doi.org/10.1002/nme.1620300305

Each edge `k`, joining nodes `i` and `j` and of length `\ell_k`, carries a
quadratic normal displacement whose end slopes are identified with the vertex
drilling rotations, giving the amplitude

.. math::

    a_k = \frac{\ell_k}{8}\left({r_z}_i - {r_z}_j\right)

so that the enrichment is not a new degree-of-freedom but the difference of
the two drilling rotations already present at the ends of the edge. The
in-plane field becomes

.. math::

    \left\{\begin{matrix} u \\ v \end{matrix}\right\}
    = \sum_i S_i \left\{\begin{matrix} u_i \\ v_i \end{matrix}\right\}
    + \sum_k N_k \frac{\ell_k}{8}\left({r_z}_i - {r_z}_j\right) \pmb{n}_k

with `\pmb{n}_k` the unit normal of edge `k` and `N_k` the hierarchical
bubble of that edge, equal to unity at its mid-point and zero at every
corner, which for the quadrilateral are the mid-side functions of the
eight-node serendipity element. In row form the enrichment populates the
drilling columns of the membrane operator, giving `\pmb{\tilde B}_m`, whereas
the curvature and transverse shear operators are untouched, because the edge
modes act only on the in-plane translations. The drilling residual becomes

.. math::

    \pmb{\tilde B}_{r_z} = \pmb{S}^{r_z}
    + \frac{1}{2}\pmb{\tilde S}^u_{,y} - \frac{1}{2}\pmb{\tilde S}^v_{,x}

and the contribution to the element stiffness matrix is

.. math::

    \pmb{K}_{r_z} = \gamma_{r_z} \iint_{\xi\eta}
    \pmb{\tilde B}_{r_z}^\top \pmb{\tilde B}_{r_z} \det \pmb{J} d\xi d\eta
    \qquad \text{with} \qquad \gamma_{r_z} = A_{66}

The value `\gamma_{r_z} = A_{66}` is a modulus and not a user parameter.
Hughes and Brezzi identify the shear modulus `G` as the natural
regularisation parameter of the isotropic problem, and the thickness
integration turns `G` into `hG = A_{66}`, which generalises to the `A_{66}`
of the laminate extensional stiffness matrix. The ``gamma_rz`` attribute
allows a different value to be used, which is only of interest for the
sensitivity study that the literature recommends.

Allman's enrichment on its own is rank-deficient by one, because the state
`u_i = v_i = 0` with `{r_z}_i = \omega_0` makes every amplitude `a_k` vanish
and therefore produces no membrane strain, although it is not a rigid-body
motion. The term above gives that state the energy
`\frac{1}{2}\gamma_{r_z} A_e \omega_0^2`, with `A_e` the element area, and
restores the rank, which is why the two ingredients are used together.

Two quadrature choices matter and both follow Ibrahimbegovic et al. (1990).
The membrane term is integrated with three points per direction: the edge
modes make `\pmb{\tilde B}_m` vary linearly, and with two points the
alternating pattern of the drilling rotations produces no membrane strain at
any of the four points, leaving a zero-energy mode that no value of
`\gamma_{r_z}` can remove. The constraint term `\pmb{K}_{r_z}` is integrated
with a single point at the centroid, which is what makes the element
insensitive to `\gamma_{r_z}`: a fully integrated constraint over-constrains
`r_z = \theta_z` and locks the membrane response as `\gamma_{r_z}` grows,
whereas with one point the response reaches an asymptote. One point also
gives the constraint rank one per element, exactly what is needed to remove
the uniform drilling mode.

Unlike the penalty below, the added term is consistent rather than
artificial. Stationarity with respect to `r_z` gives `\gamma_{r_z}(r_z -
\theta_z) = 0` pointwise, so the exact solution satisfies `r_z = \theta_z`
and the term contributes no energy, for any positive `\gamma_{r_z}`. The
nodal moments about the shell normal recovered in the internal force vector
are therefore physical, and in-plane moments can be transmitted between
shells and beams through the shared drilling degree-of-freedom. Note also
that when `\pmb{B} \neq \pmb{0}`, for an offset reference surface or an
unsymmetric laminate, the enriched membrane operator couples `r_z` to the
curvatures, which is physically correct.

**Fictitious penalty** (``drilling_model = 1``), the default before version
0.10.0, following the approach adopted in MSC Nastran and Autodesk Nastran
through their ``K6ROT`` parameter. It provides a small artificial stiffness
whose only purpose is to remove the singularity, so the forces associated
with it are spurious and any moment recovered about the shell normal is
meaningless. The penalty energy is defined per element as:

.. math::

    U_{drill} = \frac{1}{2} K6ROT \cdot 10^{-6} \cdot \int_A A_{66} (r_z - \theta_z)^2 dA

where `10^{-6}` is a scaling factor suggested by MSC Nastran's approach (CQUAD4) to make the artificial
drilling stiffness sufficiently small. AUTODESK NASTRAN's quick reference guide recommends `K6ROT = 100`
for static analysis. For modal solutions, `K6ROT = 10^4` is suggested. MSC NASTRAN's quick reference guide
states that `K6ROT > 100` should not be used, thus contradicting AUTODESK NASTRAN. The rotation `r_z` represents
the drilling degree-of-freedom in element's coordinates, whereas `\theta_z` the in-plane rotation strain, defined as:

.. math::

    \theta_z = \frac{1}{2}\left(\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right)

The first variation of U_{drill} then becomes:

.. math::

    \delta U_{drill} = K6ROT \cdot 10^{-6} \cdot \int_A A_{66} (r_z - \theta_z)(\delta r_z - \delta \theta_z) dA

which can be expressed in terms of the shape functions and element degrees-of-freedom (`u_e`) as:
`r_z = S^{r_z} u_e`, `u = S^u u_e` and `v = S_v u_e` as:

.. math::

    \delta U_{drill} = K6ROT \cdot 10^{-6} \cdot u_e^\top \int_A A_{66} (S^{r_z \top} + 1/2 S^{u \top}_{,y} - 1/2 S^{v \top}_{,x})(\delta S^{r_z} + 1/2 \delta S^u_{,y} - 1/2 S^v_{,x}) dA u_e

or simply as:

.. math::

    \delta U_{drill} = K6ROT \cdot 10^{-6} \cdot A_{66} u_e^\top \int_A B_{drill}^\top B_{drill} dA u_e

with:

.. math::

    B_{drill} = S^{r_z} + 1/2 S^u_{,y} - 1/2 S^v_{,x}

which is the same operator as `\pmb{\tilde B}_{r_z}` above, evaluated on the
unenriched field. Being built from that operator and not from an addition on
the diagonal terms is what keeps the penalty from stiffening a rigid
rotation of the element about its normal, so both models represent all
rigid-body motions and all constant-strain states exactly. `A_{66}` is
assumed constant over the element. This term is integrated with the
two-by-two quadrature. The approach herein presented is very similar to the
one presented in Eq. 2.20 of:

    Adam, F. M., Mohamed, A. E., and Hassaballa, A. E., 2013,
    \u201cDegenerated Four Nodes Shell Element with Drilling Degree of
    Freedom,\u201d IOSR J. Eng., 3(8), pp. 10\u201320.

**Choosing between them.** The physics-based model is the default because it
is the one that is correct when the drilling moment is part of the load path,
when shells are connected to beams or stiffeners that must transmit in-plane
moments, or when the mesh is too coarse for the unenriched membrane response
to be trusted. It is markedly more accurate in in-plane bending: on Cook's
skew membrane with a two-by-two mesh it gives 20.8 against the reference
23.9, where the penalty gives 11.8. The penalty remains available for
reproducing results obtained before 0.10.0, and it is cheaper, since it
leaves the membrane term on the two-by-two quadrature.

"""
#TODO bending stiffness vanishes when thickness -> zero, so a correction is applied:
#     maximum allowable aspect ratio for plate: 10^5/8, beyond which the shear
#     stiffness is multiplied by (thickness/h)^2 * (max aspect ratio allowed)^2,
#     where the max aspect ratio allowed for plates is 10^5/8.
from libc.math cimport fabs

import numpy as np

from .shellprop cimport ShellProp

cdef int DOF = 6
cdef int NUM_NODES = 4


#cdef int init_double(double* a, int size, double value) noexcept nogil:
    #cdef int i
    #for i in range(size):
        #a[i] = value


cdef void allman_enrichment(double *xe, double xi, double eta,
                            double j11, double j12, double j21,
                            double j22, double *d) noexcept nogil:
    r"""Cartesian derivatives of the Allman drilling enrichment

    Fills the buffer ``d`` with the contribution of the hierarchical
    quadratic edge modes of Allman (1984) to the Cartesian derivatives of
    the enriched in-plane displacement rows `\pmb{\tilde S}^u` and
    `\pmb{\tilde S}^v` of

    .. math::
        \pmb{\tilde S}^u = \pmb{S}^u + \frac{1}{8} \sum_k N_k (y_i - y_j)
                           (\pmb{e}_i - \pmb{e}_j)
        \
        \pmb{\tilde S}^v = \pmb{S}^v + \frac{1}{8} \sum_k N_k (x_j - x_i)
                           (\pmb{e}_i - \pmb{e}_j)

    evaluated at the `r_z` column of each node, where `\pmb{e}_i` selects
    `{r_z}_i`, edge `k` joins nodes `i` and `j`, and `N_k` is the
    hierarchical bubble of that edge, equal to unity at its mid-point and
    zero at every corner. For the quadrilateral these are the mid-side
    functions of the eight-node serendipity element:

    .. math::
        N_k = \frac{1}{2}(1 - \xi^2)(1 + \eta_k \eta)
        \quad \text{for the edges at } \eta = \eta_k = \pm 1
        \
        N_k = \frac{1}{2}(1 + \xi_k \xi)(1 - \eta^2)
        \quad \text{for the edges at } \xi = \xi_k = \pm 1

    with the edges numbered 1-2, 2-3, 3-4 and 4-1, consistently with the
    nodal connectivity of :class:`.Quad4`.

    Parameters
    ----------
    xi, eta : double
        Natural coordinates of the evaluation point.
    j11, j12, j21, j22 : double
        Terms of the inverse Jacobian, used to bring the derivatives of
        the bubbles from the natural to the Cartesian coordinates.
    d : double pointer
        Buffer of 16 positions that is filled in place, in the order
        `\tilde S^u_{,x}`, `\tilde S^u_{,y}`, `\tilde S^v_{,x}`,
        `\tilde S^v_{,y}`, each one for the four nodes, i.e. ``d[0:4]``
        holds `\tilde S^u_{,x}` at the `r_z` column of nodes 1 to 4.

    """
    cdef double x1, x2, x3, x4, y1, y2, y3, y4
    cdef double Nb1xi, Nb2xi, Nb3xi, Nb4xi
    cdef double Nb1eta, Nb2eta, Nb3eta, Nb4eta
    cdef double Nb1x, Nb2x, Nb3x, Nb4x
    cdef double Nb1y, Nb2y, Nb3y, Nb4y
    cdef double cu1, cu2, cu3, cu4
    cdef double cv1, cv2, cv3, cv4

    # NOTE ignoring z in local coordinates
    x1 = xe[0]
    y1 = xe[1]
    x2 = xe[3]
    y2 = xe[4]
    x3 = xe[6]
    y3 = xe[7]
    x4 = xe[9]
    y4 = xe[10]

    # NOTE (l_k/8)*n_k = (1/8)*{y_i - y_j, x_j - x_i}, with n_k the unit
    #      normal of edge k and l_k its length, such that the amplitude
    #      a_k = (l_k/8)*({r_z}_i - {r_z}_j) of the edge mode multiplies
    #      the difference of the two vertex drilling rotations
    cu1 = 0.125*(y1 - y2)
    cu2 = 0.125*(y2 - y3)
    cu3 = 0.125*(y3 - y4)
    cu4 = 0.125*(y4 - y1)
    cv1 = 0.125*(x2 - x1)
    cv2 = 0.125*(x3 - x2)
    cv3 = 0.125*(x4 - x3)
    cv4 = 0.125*(x1 - x4)

    # NOTE derivatives of the bubbles with respect to the natural
    #      coordinates. Edge 1 lies at eta = -1, edge 2 at xi = +1,
    #      edge 3 at eta = +1 and edge 4 at xi = -1
    Nb1xi = -xi*(1. - eta)
    Nb1eta = -0.5*(1. - xi*xi)
    Nb2xi = 0.5*(1. - eta*eta)
    Nb2eta = -eta*(1. + xi)
    Nb3xi = -xi*(1. + eta)
    Nb3eta = 0.5*(1. - xi*xi)
    Nb4xi = -0.5*(1. - eta*eta)
    Nb4eta = -eta*(1. - xi)

    Nb1x = j11*Nb1xi + j12*Nb1eta
    Nb2x = j11*Nb2xi + j12*Nb2eta
    Nb3x = j11*Nb3xi + j12*Nb3eta
    Nb4x = j11*Nb4xi + j12*Nb4eta

    Nb1y = j21*Nb1xi + j22*Nb1eta
    Nb2y = j21*Nb2xi + j22*Nb2eta
    Nb3y = j21*Nb3xi + j22*Nb3eta
    Nb4y = j21*Nb4xi + j22*Nb4eta

    # NOTE each node is the first vertex of one edge and the second vertex
    #      of the previous one, hence the two contributions with opposite
    #      signs
    d[0] = cu1*Nb1x - cu4*Nb4x
    d[1] = cu2*Nb2x - cu1*Nb1x
    d[2] = cu3*Nb3x - cu2*Nb2x
    d[3] = cu4*Nb4x - cu3*Nb3x

    d[4] = cu1*Nb1y - cu4*Nb4y
    d[5] = cu2*Nb2y - cu1*Nb1y
    d[6] = cu3*Nb3y - cu2*Nb2y
    d[7] = cu4*Nb4y - cu3*Nb3y

    d[8] = cv1*Nb1x - cv4*Nb4x
    d[9] = cv2*Nb2x - cv1*Nb1x
    d[10] = cv3*Nb3x - cv2*Nb2x
    d[11] = cv4*Nb4x - cv3*Nb3x

    d[12] = cv1*Nb1y - cv4*Nb4y
    d[13] = cv2*Nb2y - cv1*Nb1y
    d[14] = cv3*Nb3y - cv2*Nb2y
    d[15] = cv4*Nb4y - cv3*Nb3y


cdef class Quad4Data:
    r"""
    Used to allocate memory for the sparse matrices.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 576``

    KCNL_SPARSE_SIZE, : int
        ``KCNL_SPARSE_SIZE = 576``

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 144``

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 480``

    KA_BETA_SPARSE_SIZE, : int
        ``KA_BETA_SPARSE_SIZE = 144``

    KA_GAMMA_SPARSE_SIZE, : int
        ``KA_GAMMA_SPARSE_SIZE = 144``

    CA_SPARSE_SIZE, : int
        ``CA_SPARSE_SIZE = 144``

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KCNL_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE
    cdef public int KA_BETA_SPARSE_SIZE
    cdef public int KA_GAMMA_SPARSE_SIZE
    cdef public int CA_SPARSE_SIZE

    def __cinit__(Quad4Data self):
        self.KC0_SPARSE_SIZE = 576
        self.KCNL_SPARSE_SIZE = 576
        self.KG_SPARSE_SIZE = 144
        self.M_SPARSE_SIZE = 480
        self.KA_BETA_SPARSE_SIZE = 144
        self.KA_GAMMA_SPARSE_SIZE = 144
        self.CA_SPARSE_SIZE = 144


cdef class Quad4Probe:
    r"""
    Probe used for local coordinates, local displacements, local stiffness,
    local stresses etc...

    The idea behind using a probe is to avoid allocating larger memory buffers
    per finite element. The memory buffers are allocated per probe, and one
    probe can be shared amongst many finite elements, with the information
    being updated and retrieved on demand.

    .. note:: The probe can be shared amongst more than one finite element, 
              depending on how you defined them. Mind that the probe will
              always safe the values from the last udpate.


    Attributes
    ----------
    xe, : array-like
        Array of size ``NUM_NODES*DOF//2=12`` containing the nodal coordinates
        in the element coordinate system, in the following order `{x_e}_1,
        {y_e}_1, {z_e}_1, `{x_e}_2, {y_e}_2, {z_e}_2`, `{x_e}_3, {y_e}_3,
        {z_e}_3`, `{x_e}_4, {y_e}_4, {z_e}_4`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=24`` containing the element displacements
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1`, `{u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`, `{u_e}_3, {v_e}_3, {w_e}_3, {{r_x}_e}_3,
        {{r_y}_e}_3, {{r_z}_e}_3`, `{u_e}_4, {v_e}_4, {w_e}_4, {{r_x}_e}_4,
        {{r_y}_e}_4, {{r_z}_e}_4`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=24`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``.
    KC0ve, : array-like
        Local stiffness matrix stored as a 1D array of size
        ``(NUM_NODES*DOF)**2``.
    BLexx, BLeyy, BLgxy : array-like
        Arrays of size ``NUM_NODES*DOF=24`` containing the in-plane strain
        interpolation functions evaluated at a given natural coordinate point
        `\xi`, `\eta`.
    BLkxx, BLkyy, BLkxy : array-like
        Arrays of size ``NUM_NODES*DOF=24`` containing the bending strain
        interpolation functions evaluated at a given natural coordinate point
        `\xi`, `\eta`.
    BLgyz_grad, BLgyz_rot, BLgxz_grad, BLgxz_rot : array-like
        Arrays of size ``NUM_NODES*DOF=24`` containing the transverse shear
        strain interpolation functions evaluated at a given natural coordinate
        point `\xi`, `\eta`.
    BLdrilling : array-like
        Arrays of size ``NUM_NODES*DOF=24`` containing the drilling
        interpolation functions evaluated at a given natural coordinate
        point `\xi`, `\eta`.
    Gwx, Gwy : array-like
        Arrays of size ``NUM_NODES*DOF=24`` with the rows giving `w_{,x}` and
        `w_{,y}`, at the last evaluated integration point.
    KCNLve : array-like
        Array of size ``(NUM_NODES*DOF)**2=576`` with the nonlinear
        constitutive stiffness matrix KCNL in element coordinates, stored row
        by row.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    cdef public double [::1] KC0ve
    cdef public double [::1] BLexx
    cdef public double [::1] BLeyy
    cdef public double [::1] BLgxy
    cdef public double [::1] BLkxx
    cdef public double [::1] BLkyy
    cdef public double [::1] BLkxy
    cdef public double [::1] BLgyz_grad
    cdef public double [::1] BLgyz_rot
    cdef public double [::1] BLgxz_grad
    cdef public double [::1] BLgxz_rot
    cdef public double [::1] BLdrilling
    cdef public double [::1] Gwx
    cdef public double [::1] Gwy
    cdef public double [::1] KCNLve

    def __cinit__(Quad4Probe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KC0ve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)

        self.BLexx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLeyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)

        self.BLkxx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)

        self.BLgyz_grad = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgyz_rot = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxz_grad = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxz_rot = np.zeros(NUM_NODES*DOF, dtype=np.float64)

        self.BLdrilling = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Gwx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Gwy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KCNLve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)

    cpdef void update_BL(Quad4Probe self, double xi, double eta,
                         int drilling_model=0):
        r"""
        Update all components of the interpolation matrix `\pmb{B_L}` at a
        given natural coordinate point `\xi`, `\eta`.

        Parameters
        ----------
        xi, eta : double
            Natural coordinates of the evaluation point.
        drilling_model : int
            Must match the ``drilling_model`` attribute of the finite element
            that the probe is being used with, such that the recovered strains
            correspond to the displacement field that produced the stiffness
            matrix. The default ``0`` includes the Allman drilling enrichment
            in the membrane rows and in ``BLdrilling``; any other value gives
            the unenriched rows used by the ``K6ROT`` penalty model.

        """
        cdef int i
        cdef double x1, x2, x3, x4, y1, y2, y3, y4
        cdef double J11, J12, J21, J22
        cdef double j11, j12, j21, j22
        cdef double N1, N2, N3, N4
        cdef double N1x, N2x, N3x, N4x
        cdef double N1y, N2y, N3y, N4y
        cdef double denr[16]

        x1 = self.xe[0]
        y1 = self.xe[1]
        # z1 = self.xe[2]
        x2 = self.xe[3]
        y2 = self.xe[4]
        # z2 = self.xe[5]
        x3 = self.xe[6]
        y3 = self.xe[7]
        # z3 = self.xe[8]
        x4 = self.xe[9]
        y4 = self.xe[10]
        # z4 = self.xe[11]

        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

        j11 = J22/(J11*J22 - J12*J21)
        j12 = -J12/(J11*J22 - J12*J21)
        j21 = -J21/(J11*J22 - J12*J21)
        j22 = J11/(J11*J22 - J12*J21)

        N1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
        N2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
        N3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
        N4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

        N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
        N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
        N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
        N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

        N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
        N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
        N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
        N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

        self.BLexx[0] = N1x
        self.BLexx[6] = N2x
        self.BLexx[12] = N3x
        self.BLexx[18] = N4x

        self.BLeyy[1] = N1y
        self.BLeyy[7] = N2y
        self.BLeyy[13] = N3y
        self.BLeyy[19] = N4y

        self.BLgxy[0] = N1y
        self.BLgxy[6] = N2y
        self.BLgxy[12] = N3y
        self.BLgxy[18] = N4y
        self.BLgxy[1] = N1x
        self.BLgxy[7] = N2x
        self.BLgxy[13] = N3x
        self.BLgxy[19] = N4x

        self.BLkxx[4] = N1x
        self.BLkxx[10] = N2x
        self.BLkxx[16] = N3x
        self.BLkxx[22] = N4x

        self.BLkyy[3] = -N1y
        self.BLkyy[9] = -N2y
        self.BLkyy[15] = -N3y
        self.BLkyy[21] = -N4y

        self.BLkxy[3] = -N1x
        self.BLkxy[9] = -N2x
        self.BLkxy[15] = -N3x
        self.BLkxy[21] = -N4x
        self.BLkxy[4] = N1y
        self.BLkxy[10] = N2y
        self.BLkxy[16] = N3y
        self.BLkxy[22] = N4y

        self.BLgyz_grad[2] = N1y
        self.BLgyz_grad[8] = N2y
        self.BLgyz_grad[14] = N3y
        self.BLgyz_grad[20] = N4y

        self.BLgyz_rot[3] = -N1
        self.BLgyz_rot[9] = -N2
        self.BLgyz_rot[15] = -N3
        self.BLgyz_rot[21] = -N4

        self.BLgxz_grad[2] = N1x
        self.BLgxz_grad[8] = N2x
        self.BLgxz_grad[14] = N3x
        self.BLgxz_grad[20] = N4x

        self.BLgxz_rot[4] = N1
        self.BLgxz_rot[10] = N2
        self.BLgxz_rot[16] = N3
        self.BLgxz_rot[22] = N4

        self.BLdrilling[0] = N1y/2.
        self.BLdrilling[6] = N2y/2.
        self.BLdrilling[12] = N3y/2.
        self.BLdrilling[18] = N4y/2.

        self.BLdrilling[1] = -N1x/2.
        self.BLdrilling[7] = -N2x/2.
        self.BLdrilling[13] = -N3x/2.
        self.BLdrilling[19] = -N4x/2.

        self.BLdrilling[5] = N1
        self.BLdrilling[11] = N2
        self.BLdrilling[17] = N3
        self.BLdrilling[23] = N4

        # NOTE Allman drilling enrichment of the in-plane rows, see
        #      allman_enrichment() and the module documentation
        for i in range(NUM_NODES):
            self.BLexx[DOF*i + 5] = 0.
            self.BLeyy[DOF*i + 5] = 0.
            self.BLgxy[DOF*i + 5] = 0.
        if drilling_model == 0:
            allman_enrichment(&self.xe[0], xi, eta, j11, j12, j21, j22, denr)
            for i in range(NUM_NODES):
                self.BLexx[DOF*i + 5] = denr[i]
                self.BLeyy[DOF*i + 5] = denr[12 + i]
                self.BLgxy[DOF*i + 5] = denr[4 + i] + denr[8 + i]
                self.BLdrilling[DOF*i + 5] += 0.5*denr[4 + i] - 0.5*denr[8 + i]


cdef class Quad4:
    r"""
    Nodal connectivity for the plate element similar to Nastran's CQUAD4::

         ^ y
         |
        4 ________ 3
         |       |
         |       |   --> x
         |       |
         |_______|
        1         2

    The element coordinate system is determined identically what is explained
    in Nastran's quick reference guide for the CQUAD4 element, as illustrated
    below.

    .. image:: ../figures/nastran_cquad4.svg

    Attributes
    ----------
    eid, : int
        Element identification number.
    pid, : int
        Property identification number.
    area, : double
        Element area.
    drilling_model, : int
        Selects how the drilling degree-of-freedom `r_z` is given stiffness,
        see the module documentation. The default ``0`` is the physics-based
        stiffness, combining the in-plane enrichment of Allman (1984) with
        the regularisation of Hughes and Brezzi (1989), in the form given for
        the quadrilateral by Ibrahimbegovic et al. (1990). With it the
        drilling rotation is a kinematic variable that carries strain energy,
        the recovered nodal moments about the shell normal are physical, and
        no user parameter is involved. Any other value selects the
        fictitious penalty of MSC Nastran and Autodesk Nastran, which was the
        default up to version 0.9.0 and is controlled by ``K6ROT``. Setting
        ``elem.drilling_model = 1`` before calling :meth:`.update_KC0` is the
        way to reproduce results obtained before 0.10.0.

        .. note:: The attribute must be set before the element matrices are
                  updated, and the same value must be passed to
                  :meth:`.Quad4Probe.update_BL` when strains are recovered
                  from the probe.

    K6ROT, : double
        Dimensionless multiplier for the fictitious drilling stiffness, only
        read when ``drilling_model`` is not ``0``. It has no effect under the
        default physics-based model, which takes its regularisation parameter
        from the laminate stiffness instead, see ``gamma_rz``.
        AUTODESK NASTRAN's quick reference guide recommends ``K6ROT = 100.``
        for static analysis. For modal solutions, ``K6ROT=1.e4`` is suggested.
        MSC NASTRAN's quick reference guide states that ``K6ROT > 100.``
        should not be used, but this is contradicting AUTODESK NASTRAN.
    gamma_rz, : double
        Regularisation parameter `\gamma_{r_z}` of the physics-based drilling
        stiffness, only read when ``drilling_model`` is ``0``. The default is
        a negative value, which means that `A_{66}` of the laminate
        extensional stiffness matrix is used, the value identified by Hughes
        and Brezzi (1989). This is a modulus and not a parameter that needs
        tuning: the element response has a broad plateau of insensitivity
        around it, and the attribute is exposed for the sensitivity study
        that the literature recommends rather than for normal use. Very large
        values over-constrain `r_z = \theta_z`, and a zero value leaves the
        Allman enrichment rank-deficient by one.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Rotation matrix from local to global coordinates.
    m11, m12, m21, m22 : double
        Rotation matrix only for the constitutive relations. Used when a
        material direction is used instead of the element local coordinates.
    c1, c2, c3, c4 : int
        Position of each node in the global stiffness matrix.
    n1, n2, n3, n4 : int
        Node identification number.
    init_k_KC0, init_k_KCNL, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.
    init_k_KA_beta, init_k_KA_gamma, init_k_CA : int
        Position in the arrays storing the sparse data for the aerodynamic
        matrices based on the Piston theory.
    probe, : :class:`.Quad4Probe` object
        Pointer to the probe.

    """
    cdef public int eid, pid
    cdef public int n1, n2, n3, n4
    cdef public int c1, c2, c3, c4
    cdef public int init_k_KC0, init_k_KCNL, init_k_KG, init_k_M
    cdef public int init_k_KA_beta, init_k_KA_gamma, init_k_CA
    cdef public double area
    cdef public int drilling_model
    cdef public double K6ROT
    cdef public double gamma_rz
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public double m11, m12, m21, m22
    cdef public Quad4Probe probe

    def __cinit__(Quad4 self, Quad4Probe p):
        self.probe = p
        self.eid = -1
        self.pid = -1
        self.n1 = -1
        self.n2 = -1
        self.n3 = -1
        self.n4 = -1
        self.c1 = -1
        self.c2 = -1
        self.c3 = -1
        self.c4 = -1
        self.init_k_KC0 = 0
        self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.init_k_KA_beta = 0
        self.init_k_KA_gamma = 0
        self.init_k_CA = 0
        self.area = 0
        self.drilling_model = 0 # NOTE Allman + Hughes-Brezzi, the default
        self.K6ROT = 100. # NOTE default value in MSC Nastran
        self.gamma_rz = -1. # NOTE negative means "use A66"
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.
        self.m11 = 1.
        self.m12 = 0.
        self.m21 = 0.
        self.m22 = 1.


    cpdef void update_rotation_matrix(Quad4 self, double [::1] x,
            double xmati=0., double xmatj=0., double xmatk=0.):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated,
        corresponding to the rotation matrix from local to global coordinates.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The rotation matrix terms are calculated after solving 9 equations.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.

        xmati, xmatj, xmatk: array-like
            Vector in global coordinates representing the material direction.
            This vector is projected onto the plate element, thus becoming the
            material direction. The `ABD` matrix defining the constitutive
            behavior of the element is rotated from the material direction to
            the element `x` axis while calculating the stiffness matrices.

        """
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk
        cdef double x1i, x1j, x1k, x2i, x2j, x2k, x3i, x3j, x3k, x4i, x4j, x4k
        cdef double v13i, v13j, v13k, v42i, v42j, v42k
        cdef double tmp, xmatnorm, ymati, ymatj, ymatk
        cdef double tol

        with nogil:
            x1i = x[self.c1//2 + 0]
            x1j = x[self.c1//2 + 1]
            x1k = x[self.c1//2 + 2]
            x2i = x[self.c2//2 + 0]
            x2j = x[self.c2//2 + 1]
            x2k = x[self.c2//2 + 2]
            x3i = x[self.c3//2 + 0]
            x3j = x[self.c3//2 + 1]
            x3k = x[self.c3//2 + 2]
            x4i = x[self.c4//2 + 0]
            x4j = x[self.c4//2 + 1]
            x4k = x[self.c4//2 + 2]

            v13i = x3i - x1i
            v13j = x3j - x1j
            v13k = x3k - x1k
            v42i = x2i - x4i
            v42j = x2j - x4j
            v42k = x2k - x4k

            zi = v42j*v13k - v42k*v13j
            zj = -v42i*v13k + v42k*v13i
            zk = v42i*v13j - v42j*v13i
            tmp = (zi**2 + zj**2 + zk**2)**0.5
            zi /= tmp
            zj /= tmp
            zk /= tmp
            # NOTE defining tolerance to be 1/1e10 of normal vector norm
            tol = tmp/1e10

            xi = (v13i + v42i)/2.
            xj = (v13j + v42j)/2.
            xk = (v13k + v42k)/2.
            tmp = (xi**2 + xj**2 + xk**2)**0.5
            xi /= tmp
            xj /= tmp
            xk /= tmp

            # y = z X x
            yi = zj*xk - zk*xj
            yj = zk*xi - zi*xk
            yk = zi*xj - zj*xi
            tmp = (yi**2 + yj**2 + yk**2)**0.5
            yi /= tmp
            yj /= tmp
            yk /= tmp

            self.r11 = xi
            self.r21 = xj
            self.r31 = xk
            self.r12 = yi
            self.r22 = yj
            self.r32 = yk
            self.r13 = zi
            self.r23 = zj
            self.r33 = zk

            xmatnorm = (xmati**2 + xmatj**2 + xmatk**2)**0.5
            xmati /= xmatnorm
            xmatj /= xmatnorm
            xmatk /= xmatnorm

            if xmatnorm > tol:
                # Project X Material Vector into Element CSYS
                # ymat = z X xmat
                ymati = zj*xmatk - zk*xmatj
                ymatj = zk*xmati - zi*xmatk
                ymatk = zi*xmatj - zj*xmati
                tmp = (ymati**2 + ymatj**2 + ymatk**2)**0.5
                ymati /= tmp
                ymatj /= tmp
                ymatk /= tmp
                if tmp > tol:
                    # NOTE ovewriting xmati,xmatj,xmatk, they now represent the projected xmat axis
                    # xmat_projected = ymat X z
                    xmati = ymatj*zk - ymatk*zj
                    xmatj = ymatk*zi - ymati*zk
                    xmatk = ymati*zj - ymatj*zi
                    tmp = (xmati**2 + xmatj**2 + xmatk**2)**0.5
                    xmati /= tmp
                    xmatj /= tmp
                    xmatk /= tmp

                    # NOTE angle between xmat_projected and xelem
                    # NOTE assuming they are already normalized (no need to normalize)
                    self.m11 = xmati*xi + xmatj*xj + xmatk*xk # costheta
                    self.m22 = self.m11
                    # NOTE sign of costheta
                    #     - the sign is positive when rotating from the material
                    #       to the element coordinate
                    #     - the sign only affects sintheta
                    # xmat dot_product y
                    if (xmati*yi + xmatj*yj + xmatk*yk) > 0:
                        self.m12 = -(1 - self.m11**2)**0.5 # sintheta
                        self.m21 = -self.m12
                    else:
                        # NOTE theta is negative
                        self.m12 = (1 - self.m11**2)**0.5 # sintheta
                        self.m21 = -self.m12


    cpdef void update_probe_ue(Quad4 self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        .. note:: The ``ue`` attribute of object :class:`.Quad4Probe` is
                  updated, accessible using ``.probe.ue``.

        Parameters
        ----------
        u : array-like
            Array with global displacements, for a total of `M` nodes in
            the model, this array will be arranged as: `u_1, v_1, w_1, {r_x}_1,
            {r_y}_1, {r_z}_1, u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2, ...,
            u_M, v_M, w_M, {r_x}_M, {r_y}_M, {r_z}_M`.

        """
        cdef int i, j
        cdef int c[4]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3
            c[3] = self.c4

            # global to local transformation of displacements (R.T)
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            for j in range(NUM_NODES):
                for i in range(DOF):
                    self.probe.ue[j*DOF + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    # transforming translations
                    self.probe.ue[j*DOF + 0] += s1[i]*u[c[j] + 0 + i]
                    self.probe.ue[j*DOF + 1] += s2[i]*u[c[j] + 0 + i]
                    self.probe.ue[j*DOF + 2] += s3[i]*u[c[j] + 0 + i]
                    # transforming rotations
                    self.probe.ue[j*DOF + 3] += s1[i]*u[c[j] + 3 + i]
                    self.probe.ue[j*DOF + 4] += s2[i]*u[c[j] + 3 + i]
                    self.probe.ue[j*DOF + 5] += s3[i]*u[c[j] + 3 + i]


    cpdef void update_probe_xe(Quad4 self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        .. note:: The ``xe`` attribute of object :class:`.Quad4Probe` is
                  updated, accessible using ``.probe.xe``.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.

        """
        cdef int i, j
        cdef int c[4]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3
            c[3] = self.c4

            # global to local transformation of displacements (R.T)
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + 0] += s1[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 1] += s2[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 2] += s3[i]*x[c[j]//2 + i]

        self.update_area()


    cpdef void update_area(Quad4 self):
        r"""Update element area

        """
        cdef double x1, x2, x3, x4, y1, y2, y3, y4
        with nogil:
            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]
            self.area = 1/2.*fabs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))


    cdef void _update_probe_KC0ve(Quad4 self, ShellProp prop) noexcept nogil:
        r"""Update the probe vector for values of the constitutive stiffness matrix KC0

        The attribute ``KC0ve`` of the object :class:`.Quad4Probe` is updated,
        which corresponds to the values of the constitutive stiffness matrix in
        local coordinates. While using this function, mind that the probe can
        be shared amongst more than one finite element, depending how you
        defined them, meaning that the probe will always safe the values from
        the last udpate.

        .. note:: The ``KC0ve`` attribute of object :class:`.Quad4Probe` is
                  updated, accessible using ``.probe.KC0ve``.

        Parameters
        ----------
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.

        """
        cdef int i, j, k, ke
        cdef int node_i, node_j
        cdef double x1, x2, x3, x4, y1, y2, y3, y4
        cdef double Ae[9]
        cdef double Be[9]
        cdef double De[9]
        cdef double Atse[4]
        cdef double A44, A45, A55
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double D11, D12, D16, D22, D26, D66
        cdef double length
        cdef double m11, m12, m21, m22
        cdef double N1, N2, N3, N4
        cdef double N1x, N2x, N3x, N4x
        cdef double N1y, N2y, N3y, N4y
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double j11, j12, j21, j22
        cdef double points[2]
        cdef double qpoints[3]
        cdef double qweights[3]
        cdef int nq
        cdef int pti, ptj
        # NOTE MITC4 assumed transverse shear: the covariant tying-point
        #      rows, each with six non-zeros, and the interpolated operator
        cdef double qrA[24]
        cdef double qrB[24]
        cdef double qsC[24]
        cdef double qsD[24]
        cdef double BLgxz_a[24]
        cdef double BLgyz_a[24]
        cdef double qr, qs, fA, fB, fC, fD
        cdef int itie
        cdef double* BLexx
        cdef double* BLeyy
        cdef double* BLgxy
        cdef double* BLkxx
        cdef double* BLkyy
        cdef double* BLkxy
        cdef double* BLgyz_rot
        cdef double* BLgyz_grad
        cdef double* BLgxz_rot
        cdef double* BLgxz_grad
        cdef double* BLdrilling
        cdef double BLdrilling_i
        cdef double gamma_drill
        cdef int enriched
        cdef double denr[16]
        cdef double exx, eyy, gxy, kxx, kyy, kxy
        cdef double gyz_rot, gxz_rot, gyz_grad, gxz_grad

        with nogil:
            BLexx = &self.probe.BLexx[0]
            BLeyy = &self.probe.BLeyy[0]
            BLgxy = &self.probe.BLgxy[0]
            BLkxx = &self.probe.BLkxx[0]
            BLkyy = &self.probe.BLkyy[0]
            BLkxy = &self.probe.BLkxy[0]
            BLgyz_rot = &self.probe.BLgyz_rot[0]
            BLgyz_grad = &self.probe.BLgyz_grad[0]
            BLgxz_rot = &self.probe.BLgxz_rot[0]
            BLgxz_grad = &self.probe.BLgxz_grad[0]
            BLdrilling = &self.probe.BLdrilling[0]

            # NOTE ignoring z in local coordinates

            # NOTE constitutive matrices in the element coordinate system,
            #      the same function is used by all element methods
            prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, Ae, Be, De, Atse)
            A11 = Ae[0]
            A12 = Ae[1]
            A16 = Ae[2]
            A22 = Ae[4]
            A26 = Ae[5]
            A66 = Ae[8]
            B11 = Be[0]
            B12 = Be[1]
            B16 = Be[2]
            B22 = Be[4]
            B26 = Be[5]
            B66 = Be[8]
            D11 = De[0]
            D12 = De[1]
            D16 = De[2]
            D22 = De[4]
            D26 = De[5]
            D66 = De[8]
            # NOTE transverse shear stiffness with the shear correction already applied
            A44 = Atse[0]
            A45 = Atse[1]
            A55 = Atse[3]

            # NOTE regularisation modulus of the drilling stiffness, see the
            #      module documentation. The fictitious penalty of MSC Nastran
            #      and Autodesk Nastran is recovered with K6ROT*1e-6*A66
            enriched = 1 if self.drilling_model == 0 else 0
            if enriched:
                # NOTE Hughes-Brezzi regularisation parameter, a modulus and
                #      not a user parameter. Attribute gamma_rz allows the
                #      sensitivity study recommended in the literature
                if self.gamma_rz >= 0.:
                    gamma_drill = self.gamma_rz
                else:
                    gamma_drill = A66
            else:
                gamma_drill = self.K6ROT*1.e-6*A66

            length = self.area**0.5


            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # zeroing probe KC0ve attribute
            for i in range(24):
                for j in range(24):
                    ke = 24*i + j
                    self.probe.KC0ve[ke] = 0.

            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            # NOTE quadrature of the membrane, bending and coupling terms.
            #      Two-point Gauss-Legendre is enough for the unenriched
            #      field, but the Allman edge modes make the membrane
            #      operator vary linearly, and with two points the
            #      alternating pattern of the drilling rotations produces no
            #      membrane strain at any of the four points, leaving a
            #      zero-energy mode that the drilling term cannot remove. The
            #      three-point rule, which is the one used by Ibrahimbegovic
            #      et al. (1990), restores the rank of the element
            if enriched:
                nq = 3
                qpoints[0] = -0.7745966692414834042779
                qpoints[1] = 0.
                qpoints[2] = +0.7745966692414834042779
                qweights[0] = 0.5555555555555555555556
                qweights[1] = 0.8888888888888888888889
                qweights[2] = 0.5555555555555555555556
            else:
                nq = 2
                qpoints[0] = points[0]
                qpoints[1] = points[1]
                qweights[0] = 1.
                qweights[1] = 1.

            for pti in range(nq):
                xi = qpoints[pti]
                for ptj in range(nq):
                    eta = qpoints[ptj]
                    wij = qweights[pti]*qweights[ptj]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    N1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                    N2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                    N3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                    N4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

                    N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    BLexx[0] = N1x
                    BLexx[6] = N2x
                    BLexx[12] = N3x
                    BLexx[18] = N4x

                    BLeyy[1] = N1y
                    BLeyy[7] = N2y
                    BLeyy[13] = N3y
                    BLeyy[19] = N4y

                    BLgxy[0] = N1y
                    BLgxy[6] = N2y
                    BLgxy[12] = N3y
                    BLgxy[18] = N4y
                    BLgxy[1] = N1x
                    BLgxy[7] = N2x
                    BLgxy[13] = N3x
                    BLgxy[19] = N4x

                    # NOTE Allman enrichment, populating the drilling columns
                    #      of the membrane operator. The columns are always
                    #      assigned because the probe is shared amongst finite
                    #      elements that may use a different drilling model
                    if enriched:
                        allman_enrichment(&self.probe.xe[0], xi, eta, j11, j12, j21, j22, denr)
                        for i in range(NUM_NODES):
                            BLexx[DOF*i + 5] = denr[i]
                            BLeyy[DOF*i + 5] = denr[12 + i]
                            BLgxy[DOF*i + 5] = denr[4 + i] + denr[8 + i]
                    else:
                        for i in range(NUM_NODES):
                            BLexx[DOF*i + 5] = 0.
                            BLeyy[DOF*i + 5] = 0.
                            BLgxy[DOF*i + 5] = 0.

                    BLkxx[4] = N1x
                    BLkxx[10] = N2x
                    BLkxx[16] = N3x
                    BLkxx[22] = N4x

                    BLkyy[3] = -N1y
                    BLkyy[9] = -N2y
                    BLkyy[15] = -N3y
                    BLkyy[21] = -N4y

                    BLkxy[3] = -N1x
                    BLkxy[9] = -N2x
                    BLkxy[15] = -N3x
                    BLkxy[21] = -N4x
                    BLkxy[4] = N1y
                    BLkxy[10] = N2y
                    BLkxy[16] = N3y
                    BLkxy[22] = N4y

                    for i in range(24):
                        exx = BLexx[i]
                        eyy = BLeyy[i]
                        gxy = BLgxy[i]
                        kxx = BLkxx[i]
                        kyy = BLkyy[i]
                        kxy = BLkxy[i]
                        for j in range(24):
                            ke = 24*i + j
                            self.probe.KC0ve[ke] += wij*detJ*(
                            # membrane
                                exx*A11*BLexx[j] + exx*A12*BLeyy[j] + exx*A16*BLgxy[j]
                              + eyy*A12*BLexx[j] + eyy*A22*BLeyy[j] + eyy*A26*BLgxy[j]
                              + gxy*A16*BLexx[j] + gxy*A26*BLeyy[j] + gxy*A66*BLgxy[j]

                            # coupled membrane-bending
                              + exx*B11*BLkxx[j] + exx*B12*BLkyy[j] + exx*B16*BLkxy[j]
                              + eyy*B12*BLkxx[j] + eyy*B22*BLkyy[j] + eyy*B26*BLkxy[j]
                              + gxy*B16*BLkxx[j] + gxy*B26*BLkyy[j] + gxy*B66*BLkxy[j]

                              + kxx*B11*BLexx[j] + kxx*B12*BLeyy[j] + kxx*B16*BLgxy[j]
                              + kyy*B12*BLexx[j] + kyy*B22*BLeyy[j] + kyy*B26*BLgxy[j]
                              + kxy*B16*BLexx[j] + kxy*B26*BLeyy[j] + kxy*B66*BLgxy[j]

                            )

            # NOTE two-point quadrature of the bending term. The Allman
            #      enrichment acts only on the in-plane translations, leaving
            #      the curvature rows untouched, so this term keeps the rule
            #      it has always used and the bending response of the element
            #      is identical for the two drilling models
            wij = 1.
            for pti in range(2):
                xi = points[pti]
                for ptj in range(2):
                    eta = points[ptj]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    BLkxx[4] = N1x
                    BLkxx[10] = N2x
                    BLkxx[16] = N3x
                    BLkxx[22] = N4x

                    BLkyy[3] = -N1y
                    BLkyy[9] = -N2y
                    BLkyy[15] = -N3y
                    BLkyy[21] = -N4y

                    BLkxy[3] = -N1x
                    BLkxy[9] = -N2x
                    BLkxy[15] = -N3x
                    BLkxy[21] = -N4x
                    BLkxy[4] = N1y
                    BLkxy[10] = N2y
                    BLkxy[16] = N3y
                    BLkxy[22] = N4y

                    for i in range(24):
                        kxx = BLkxx[i]
                        kyy = BLkyy[i]
                        kxy = BLkxy[i]
                        for j in range(24):
                            ke = 24*i + j
                            self.probe.KC0ve[ke] += wij*detJ*(
                                kxx*D11*BLkxx[j] + kxx*D12*BLkyy[j] + kxx*D16*BLkxy[j]
                              + kyy*D12*BLkxx[j] + kyy*D22*BLkyy[j] + kyy*D26*BLkxy[j]
                              + kxy*D16*BLkxx[j] + kxy*D26*BLkyy[j] + kxy*D66*BLkxy[j]
                            )

            # NOTE two-point quadrature of the transverse shear gradient
            #      term of the thick elements, kept independent of the rule
            #      used above for the membrane
            if prop.h/length >= 1.: # thick elements
                wij = 1.
                for pti in range(2):
                    xi = points[pti]
                    for ptj in range(2):
                        eta = points[ptj]

                        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                        detJ = J11*J22 - J12*J21

                        j11 = J22/(J11*J22 - J12*J21)
                        j12 = -J12/(J11*J22 - J12*J21)
                        j21 = -J21/(J11*J22 - J12*J21)
                        j22 = J11/(J11*J22 - J12*J21)

                        N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                        N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                        N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                        N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                        N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                        N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                        N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                        N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                        BLgyz_grad[2] = N1y
                        BLgyz_grad[8] = N2y
                        BLgyz_grad[14] = N3y
                        BLgyz_grad[20] = N4y

                        BLgxz_grad[2] = N1x
                        BLgxz_grad[8] = N2x
                        BLgxz_grad[14] = N3x
                        BLgxz_grad[20] = N4x
                        for i in range(24):
                            gyz_grad = BLgyz_grad[i]
                            gxz_grad = BLgxz_grad[i]
                            for j in range(24):
                                ke = 24*i + j
                                self.probe.KC0ve[ke] += wij*detJ*(
                                # transverse shear (gradient term)
                                    gyz_grad*A44*BLgyz_grad[j] + gyz_grad*A45*BLgxz_grad[j]
                                  + gxz_grad*A45*BLgyz_grad[j] + gxz_grad*A55*BLgxz_grad[j]
                                )

            # NOTE the fictitious penalty of the K6ROT model is integrated
            #      with the same two-point rule used before 0.10.0, whereas
            #      the physics-based constraint term is integrated with a
            #      single point in the block further below, following
            #      Ibrahimbegovic et al. (1990). See the module documentation
            if not enriched:
                wij = 1.
                for pti in range(2):
                    xi = points[pti]
                    for ptj in range(2):
                        eta = points[ptj]

                        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                        detJ = J11*J22 - J12*J21

                        j11 = J22/(J11*J22 - J12*J21)
                        j12 = -J12/(J11*J22 - J12*J21)
                        j21 = -J21/(J11*J22 - J12*J21)
                        j22 = J11/(J11*J22 - J12*J21)

                        N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                        N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                        N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                        N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                        N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                        N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                        N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                        N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                        N1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                        N2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                        N3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                        N4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

                        BLdrilling[0] = N1y/2.
                        BLdrilling[6] = N2y/2.
                        BLdrilling[12] = N3y/2.
                        BLdrilling[18] = N4y/2.

                        BLdrilling[1] = -N1x/2.
                        BLdrilling[7] = -N2x/2.
                        BLdrilling[13] = -N3x/2.
                        BLdrilling[19] = -N4x/2.

                        BLdrilling[5] = N1
                        BLdrilling[11] = N2
                        BLdrilling[17] = N3
                        BLdrilling[23] = N4

                        for i in range(24):
                            BLdrilling_i = gamma_drill*BLdrilling[i]
                            for j in range(24):
                                ke = 24*i + j
                                self.probe.KC0ve[ke] += wij*detJ*BLdrilling_i*BLdrilling[j]
            
            # NOTE reduced integration with one point at the center
            wij = 4.
            xi = 0.
            eta = 0.

            J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
            J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
            J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
            J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

            detJ = J11*J22 - J12*J21

            j11 = J22/(J11*J22 - J12*J21)
            j12 = -J12/(J11*J22 - J12*J21)
            j21 = -J21/(J11*J22 - J12*J21)
            j22 = J11/(J11*J22 - J12*J21)

            N1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
            N2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
            N3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
            N4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

            N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
            N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
            N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
            N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

            N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
            N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
            N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
            N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

            BLgyz_grad[2] = N1y
            BLgyz_grad[8] = N2y
            BLgyz_grad[14] = N3y
            BLgyz_grad[20] = N4y

            BLgyz_rot[3] = -N1
            BLgyz_rot[9] = -N2
            BLgyz_rot[15] = -N3
            BLgyz_rot[21] = -N4

            BLgxz_grad[2] = N1x
            BLgxz_grad[8] = N2x
            BLgxz_grad[14] = N3x
            BLgxz_grad[20] = N4x

            BLgxz_rot[4] = N1
            BLgxz_rot[10] = N2
            BLgxz_rot[16] = N3
            BLgxz_rot[22] = N4

            # NOTE the Hughes-Brezzi constraint term is integrated with this
            #      single point at the centroid, following Ibrahimbegovic et
            #      al. (1990), who integrate their penalty matrix with one
            #      point while the enriched membrane is fully integrated. The
            #      rule is what makes the element insensitive to gamma_rz: a
            #      fully integrated constraint over-constrains r_z = theta_z
            #      and locks the membrane response as gamma_rz grows, whereas
            #      with one point the response reaches an asymptote. A single
            #      point gives the constraint term rank one per element, which
            #      is exactly what is needed to remove the uniform drilling
            #      mode of the Allman enrichment
            if enriched:
                allman_enrichment(&self.probe.xe[0], xi, eta, j11, j12, j21,
                                  j22, denr)

                BLdrilling[0] = N1y/2.
                BLdrilling[6] = N2y/2.
                BLdrilling[12] = N3y/2.
                BLdrilling[18] = N4y/2.

                BLdrilling[1] = -N1x/2.
                BLdrilling[7] = -N2x/2.
                BLdrilling[13] = -N3x/2.
                BLdrilling[19] = -N4x/2.

                BLdrilling[5] = N1 + 0.5*denr[4] - 0.5*denr[8]
                BLdrilling[11] = N2 + 0.5*denr[5] - 0.5*denr[9]
                BLdrilling[17] = N3 + 0.5*denr[6] - 0.5*denr[10]
                BLdrilling[23] = N4 + 0.5*denr[7] - 0.5*denr[11]

                for i in range(24):
                    BLdrilling_i = gamma_drill*BLdrilling[i]
                    for j in range(24):
                        ke = 24*i + j
                        self.probe.KC0ve[ke] += wij*detJ*BLdrilling_i*BLdrilling[j]

            if prop.h/length < 1.: # thin elements
                for i in range(24):
                    gyz_rot = BLgyz_rot[i]
                    gxz_rot = BLgxz_rot[i]
                    gyz_grad = BLgyz_grad[i]
                    gxz_grad = BLgxz_grad[i]
                    for j in range(24):
                        ke = 24*i + j
                        self.probe.KC0ve[ke] += wij*detJ*(
                        # transverse shear (gradient term)
                            gyz_grad*A44*BLgyz_grad[j] + gyz_grad*A45*BLgxz_grad[j]
                          + gxz_grad*A45*BLgyz_grad[j] + gxz_grad*A55*BLgxz_grad[j]

                        # transverse shear (coupled terms)
                          + gyz_grad*A44*BLgyz_rot[j] + gyz_grad*A45*BLgxz_rot[j]
                          + gxz_grad*A45*BLgyz_rot[j] + gxz_grad*A55*BLgxz_rot[j]

                          + gyz_rot*A44*BLgyz_grad[j] + gyz_rot*A45*BLgxz_grad[j]
                          + gxz_rot*A45*BLgyz_grad[j] + gxz_rot*A55*BLgxz_grad[j]

                        # transverse shear (rotation term)
                          + gyz_rot*A44*BLgyz_rot[j] + gyz_rot*A45*BLgxz_rot[j]
                          + gxz_rot*A45*BLgyz_rot[j] + gxz_rot*A55*BLgxz_rot[j]
                        )

            else: # thick elements
                for i in range(24):
                    gyz_rot = BLgyz_rot[i]
                    gxz_rot = BLgxz_rot[i]
                    gyz_grad = BLgyz_grad[i]
                    gxz_grad = BLgxz_grad[i]
                    for j in range(24):
                        ke = 24*i + j
                        self.probe.KC0ve[ke] += wij*detJ*(
                        # transverse shear (coupled terms)
                            gyz_grad*A44*BLgyz_rot[j] + gyz_grad*A45*BLgxz_rot[j]
                          + gxz_grad*A45*BLgyz_rot[j] + gxz_grad*A55*BLgxz_rot[j]

                          + gyz_rot*A44*BLgyz_grad[j] + gyz_rot*A45*BLgxz_grad[j]
                          + gxz_rot*A45*BLgyz_grad[j] + gxz_rot*A55*BLgxz_grad[j]

                        # transverse shear (rotation term)
                          + gyz_rot*A44*BLgyz_rot[j] + gyz_rot*A45*BLgxz_rot[j]
                          + gxz_rot*A45*BLgyz_rot[j] + gxz_rot*A55*BLgxz_rot[j]
                        )


    cpdef void update_probe_finte(Quad4 self, ShellProp prop, int nonlinear=0):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` of the object :class:`.Quad4Probe` is updated,
        which corresponds to the internal forces in local coordinates. While
        using this function, mind that the probe can be shared amongst more
        than one finite element, depending how you defined them, meaning that
        the probe will always safe the values from the last udpate.

        .. note:: The ``finte`` attribute of object :class:`.Quad4Probe` is
                  updated, accessible using ``.probe.finte``.

        Parameters
        ----------
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        nonlinear : int
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Karman strains,
            for which the exact Jacobian of the internal forces is ``KC0 + KCNL +
            KG``, see :meth:`.update_KCNL`.

        """
        cdef int i, j, ke

        with nogil:
            self._update_probe_KC0ve(prop)
            for i in range(24):
                self.probe.finte[i] = 0.
                for j in range(24):
                    ke = 24*i + j
                    self.probe.finte[i] += self.probe.KC0ve[ke] * self.probe.ue[j]

            if nonlinear:
                self._update_probe_finte_nonlinear(prop)
        

    cpdef void update_KC0(Quad4 self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          ShellProp prop,
                          int update_KC0v_only=0,
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0


        Parameters
        ----------
        KC0r : np.array
            Array to store row positions of sparse values
        KC0c : np.array
            Array to store column positions of sparse values
        KC0v : np.array
            Array to store sparse values
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.

        """
        cdef int i, j, node_i, node_j, k, ke, m, n
        cdef int c[4]
        cdef double r[6][6]

        with nogil:
            # local to global transformation
            # translation DOFs
            r[0][0] = self.r11
            r[0][1] = self.r12
            r[0][2] = self.r13
            r[1][0] = self.r21
            r[1][1] = self.r22
            r[1][2] = self.r23
            r[2][0] = self.r31
            r[2][1] = self.r32
            r[2][2] = self.r33
            # rotation DOFs
            r[0+3][0+3] = self.r11
            r[0+3][1+3] = self.r12
            r[0+3][2+3] = self.r13
            r[1+3][0+3] = self.r21
            r[1+3][1+3] = self.r22
            r[1+3][2+3] = self.r23
            r[2+3][0+3] = self.r31
            r[2+3][1+3] = self.r32
            r[2+3][2+3] = self.r33
            # coupled translation-rotation DOFs
            r[0][0+3] = 0.
            r[0][1+3] = 0.
            r[0][2+3] = 0.
            r[1][0+3] = 0.
            r[1][1+3] = 0.
            r[1][2+3] = 0.
            r[2][0+3] = 0.
            r[2][1+3] = 0.
            r[2][2+3] = 0.
            # coupled translation-rotation DOFs
            r[0+3][0] = 0.
            r[0+3][1] = 0.
            r[0+3][2] = 0.
            r[1+3][0] = 0.
            r[1+3][1] = 0.
            r[1+3][2] = 0.
            r[2+3][0] = 0.
            r[2+3][1] = 0.
            r[2+3][2] = 0.

            if update_KC0v_only == 0:
                # positions in the global stiffness matrix
                c[0] = self.c1
                c[1] = self.c2
                c[2] = self.c3
                c[3] = self.c4

                # initializing row and column indices
                #
                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = self.init_k_KC0 + 24*(node_i*DOF + m) + node_j*DOF + n
                                KC0r[k] = c[node_i] + m
                                KC0c[k] = c[node_j] + n

            self._update_probe_KC0ve(prop)

            # NOTE from element to global coordinates:
            #
            # Kg = R @ Ke @ R.T
            #
            # in tensor notation:
            #
            # Kg_{mn} = r_{mi} * Ke_{ij} * r_{nj}
            #
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = self.init_k_KC0 + 24*(node_i*DOF + m) + node_j*DOF + n
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 24*(node_i*DOF + i) + node_j*DOF + j
                                    KC0v[k] += r[m][i]*self.probe.KC0ve[ke]*r[n][j]


    cpdef void update_fint(Quad4 self, double [::1] fint, ShellProp prop, int nonlinear=0):
        r"""Update the internal force vector

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.Quad4Probe` with the
            internal forces in local coordinates.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        nonlinear : int
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Karman strains,
            for which the exact Jacobian of the internal forces is ``KC0 + KCNL +
            KG``, see :meth:`.update_KCNL`.

        """
        cdef double *finte

        self.update_probe_finte(prop, nonlinear)

        with nogil:
            finte = &self.probe.finte[0]

            fint[0+self.c1] += finte[0]*self.r11 + finte[1]*self.r12 + finte[2]*self.r13
            fint[1+self.c1] += finte[0]*self.r21 + finte[1]*self.r22 + finte[2]*self.r23
            fint[2+self.c1] += finte[0]*self.r31 + finte[1]*self.r32 + finte[2]*self.r33
            fint[3+self.c1] += finte[3]*self.r11 + finte[4]*self.r12 + finte[5]*self.r13
            fint[4+self.c1] += finte[3]*self.r21 + finte[4]*self.r22 + finte[5]*self.r23
            fint[5+self.c1] += finte[3]*self.r31 + finte[4]*self.r32 + finte[5]*self.r33
            fint[0+self.c2] += finte[6]*self.r11 + finte[7]*self.r12 + finte[8]*self.r13
            fint[1+self.c2] += finte[6]*self.r21 + finte[7]*self.r22 + finte[8]*self.r23
            fint[2+self.c2] += finte[6]*self.r31 + finte[7]*self.r32 + finte[8]*self.r33
            fint[3+self.c2] += finte[10]*self.r12 + finte[11]*self.r13 + finte[9]*self.r11
            fint[4+self.c2] += finte[10]*self.r22 + finte[11]*self.r23 + finte[9]*self.r21
            fint[5+self.c2] += finte[10]*self.r32 + finte[11]*self.r33 + finte[9]*self.r31
            fint[0+self.c3] += finte[12]*self.r11 + finte[13]*self.r12 + finte[14]*self.r13
            fint[1+self.c3] += finte[12]*self.r21 + finte[13]*self.r22 + finte[14]*self.r23
            fint[2+self.c3] += finte[12]*self.r31 + finte[13]*self.r32 + finte[14]*self.r33
            fint[3+self.c3] += finte[15]*self.r11 + finte[16]*self.r12 + finte[17]*self.r13
            fint[4+self.c3] += finte[15]*self.r21 + finte[16]*self.r22 + finte[17]*self.r23
            fint[5+self.c3] += finte[15]*self.r31 + finte[16]*self.r32 + finte[17]*self.r33
            fint[0+self.c4] += finte[18]*self.r11 + finte[19]*self.r12 + finte[20]*self.r13
            fint[1+self.c4] += finte[18]*self.r21 + finte[19]*self.r22 + finte[20]*self.r23
            fint[2+self.c4] += finte[18]*self.r31 + finte[19]*self.r32 + finte[20]*self.r33
            fint[3+self.c4] += finte[21]*self.r11 + finte[22]*self.r12 + finte[23]*self.r13
            fint[4+self.c4] += finte[21]*self.r21 + finte[22]*self.r22 + finte[23]*self.r23
            fint[5+self.c4] += finte[21]*self.r31 + finte[22]*self.r32 + finte[23]*self.r33


    cdef double _update_probe_BL_G(Quad4 self, double xi,
                                   double eta) noexcept nogil:
        r"""Update the probe rows of the linear strains and of the gradient of `w`

        Evaluated at the natural coordinates ``xi, eta``, in element coordinates:

        - ``BLexx, BLeyy, BLgxy``: membrane strains `\epsilon_{xx}, \epsilon_{yy},
          \gamma_{xy}`
        - ``BLkxx, BLkyy, BLkxy``: curvatures `\kappa_{xx}, \kappa_{yy},
          \kappa_{xy}`
        - ``Gwx, Gwy``: `w_{,x}` and `w_{,y}`

        Returns
        -------
        detJ : double
            Determinant of the Jacobian matrix at ``xi, eta``.

        """
        cdef int i
        cdef double x1, x2, x3, x4, y1, y2, y3, y4
        cdef double J11, J12, J21, J22, detJ
        cdef double j11, j12, j21, j22
        cdef double N1x, N2x, N3x, N4x, N1y, N2y, N3y, N4y
        cdef double denr[16]
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Gwx
        cdef double *Gwy

        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Gwx = &self.probe.Gwx[0]
        Gwy = &self.probe.Gwy[0]

        # NOTE ignoring z in local coordinates
        x1 = self.probe.xe[0]
        y1 = self.probe.xe[1]
        x2 = self.probe.xe[3]
        y2 = self.probe.xe[4]
        x3 = self.probe.xe[6]
        y3 = self.probe.xe[7]
        x4 = self.probe.xe[9]
        y4 = self.probe.xe[10]

        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

        detJ = J11*J22 - J12*J21

        j11 = J22/detJ
        j12 = -J12/detJ
        j21 = -J21/detJ
        j22 = J11/detJ

        N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
        N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
        N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
        N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

        N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
        N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
        N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
        N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

        for i in range(24):
            BLexx[i] = 0.
            BLeyy[i] = 0.
            BLgxy[i] = 0.
            BLkxx[i] = 0.
            BLkyy[i] = 0.
            BLkxy[i] = 0.
            Gwx[i] = 0.
            Gwy[i] = 0.

        # exx = u,x
        BLexx[0] = N1x
        BLexx[6] = N2x
        BLexx[12] = N3x
        BLexx[18] = N4x

        # eyy = v,y
        BLeyy[1] = N1y
        BLeyy[7] = N2y
        BLeyy[13] = N3y
        BLeyy[19] = N4y

        # gxy = u,y + v,x
        BLgxy[0] = N1y
        BLgxy[6] = N2y
        BLgxy[12] = N3y
        BLgxy[18] = N4y
        BLgxy[1] = N1x
        BLgxy[7] = N2x
        BLgxy[13] = N3x
        BLgxy[19] = N4x

        # kxx = ry,x
        BLkxx[4] = N1x
        BLkxx[10] = N2x
        BLkxx[16] = N3x
        BLkxx[22] = N4x

        # kyy = -rx,y
        BLkyy[3] = -N1y
        BLkyy[9] = -N2y
        BLkyy[15] = -N3y
        BLkyy[21] = -N4y

        # kxy = ry,y - rx,x
        BLkxy[3] = -N1x
        BLkxy[9] = -N2x
        BLkxy[15] = -N3x
        BLkxy[21] = -N4x
        BLkxy[4] = N1y
        BLkxy[10] = N2y
        BLkxy[16] = N3y
        BLkxy[22] = N4y

        # w,x
        Gwx[2] = N1x
        Gwx[8] = N2x
        Gwx[14] = N3x
        Gwx[20] = N4x

        # w,y
        Gwy[2] = N1y
        Gwy[8] = N2y
        Gwy[14] = N3y
        Gwy[20] = N4y

        # NOTE Allman enrichment of the membrane rows, keeping the nonlinear
        #      tangent of update_KCNL consistent with the enriched KC0. The
        #      curvature rows and the gradient of w are not enriched, because
        #      the edge modes act only on the in-plane translations
        if self.drilling_model == 0:
            allman_enrichment(&self.probe.xe[0], xi, eta, j11, j12, j21, j22, denr)
            for i in range(NUM_NODES):
                BLexx[DOF*i + 5] = denr[i]
                BLeyy[DOF*i + 5] = denr[12 + i]
                BLgxy[DOF*i + 5] = denr[4 + i] + denr[8 + i]

        return detJ


    cdef void _update_probe_KCNLve(Quad4 self, ShellProp prop) noexcept nogil:
        r"""Update the probe values of the nonlinear constitutive stiffness matrix

        The attribute ``KCNLve`` of the :class:`.Quad4Probe` is updated with
        KCNL = KC0L + KCL0 + KCLL + KGNL in element coordinates, stored row by
        row and evaluated at the displacements ``ue`` of the probe. See
        :meth:`.update_KCNL`.

        """
        cdef int i, j, a
        cdef int pti, ptj
        cdef double xi, eta
        cdef double points[2]
        cdef double wij, detJ, w_x, w_y
        cdef double A[9]
        cdef double B[9]
        cdef double NNL[3]
        # NOTE products stored row by row, with 3 rows and NUM_NODES*DOF columns
        cdef double ABL[72]
        cdef double BmL[72]
        cdef double ABmL[72]
        cdef double *ue
        cdef double *KCNLve
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Gwx
        cdef double *Gwy

        ue = &self.probe.ue[0]
        KCNLve = &self.probe.KCNLve[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Gwx = &self.probe.Gwx[0]
        Gwy = &self.probe.Gwy[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, A, B, NULL, NULL)

        for i in range(24*24):
            KCNLve[i] = 0.

        # NOTE same two-point Gauss-Legendre quadrature as in update_KG
        wij = 1.
        points[0] = -0.5773502691896257645092
        points[1] = +0.5773502691896257645092

        for pti in range(2):
            xi = points[pti]
            for ptj in range(2):
                eta = points[ptj]
                detJ = self._update_probe_BL_G(xi, eta)

                w_x = 0.
                w_y = 0.
                for i in range(24):
                    w_x += Gwx[i]*ue[i]
                    w_y += Gwy[i]*ue[i]

                # stress resultants of the nonlinear membrane strain,
                # epsNL = {w_x**2/2, w_y**2/2, w_x*w_y}
                for a in range(3):
                    NNL[a] = A[3*a]*w_x*w_x/2. + A[3*a + 1]*w_y*w_y/2. + A[3*a + 2]*w_x*w_y

                # BmL, the variation of the nonlinear membrane strain, and the products
                # A*Bm + B*Bb and A*BmL, all stored row by row with 3 rows
                for i in range(24):
                    BmL[i] = w_x*Gwx[i]
                    BmL[24 + i] = w_y*Gwy[i]
                    BmL[48 + i] = w_x*Gwy[i] + w_y*Gwx[i]
                    for a in range(3):
                        ABL[24*a + i] = (A[3*a]*BLexx[i] + A[3*a + 1]*BLeyy[i] + A[3*a + 2]*BLgxy[i]
                                       + B[3*a]*BLkxx[i] + B[3*a + 1]*BLkyy[i] + B[3*a + 2]*BLkxy[i])
                        ABmL[24*a + i] = A[3*a]*BmL[i] + A[3*a + 1]*BmL[24 + i] + A[3*a + 2]*BmL[48 + i]

                for i in range(24):
                    for j in range(24):
                        KCNLve[24*i + j] += wij*detJ*(
                            # KC0L = (Bm.T*A + Bb.T*B)*BmL
                              ABL[i]*BmL[j] + ABL[24 + i]*BmL[24 + j] + ABL[48 + i]*BmL[48 + j]
                            # KCL0 = BmL.T*(A*Bm + B*Bb)
                            + BmL[i]*ABL[j] + BmL[24 + i]*ABL[24 + j] + BmL[48 + i]*ABL[48 + j]
                            # KCLL = BmL.T*A*BmL
                            + BmL[i]*ABmL[j] + BmL[24 + i]*ABmL[24 + j] + BmL[48 + i]*ABmL[48 + j]
                            # KGNL = G.T*[NNL]*G
                            + Gwx[i]*(NNL[0]*Gwx[j] + NNL[2]*Gwy[j])
                            + Gwy[i]*(NNL[2]*Gwx[j] + NNL[1]*Gwy[j])
                        )


    cdef void _update_probe_finte_nonlinear(Quad4 self,
                                            ShellProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The attribute ``finte`` of the :class:`.Quad4Probe` receives the terms
        of the von Karman membrane strain `\{\epsilon_{NL}\} = \{w_{,x}^2/2,
        w_{,y}^2/2, w_{,x} w_{,y}\}^T`, evaluated at the displacements ``ue`` of
        the probe, such that ``finte`` becomes the gradient of the strain energy
        whose Hessian is KC0 + KCNL + KG. See :meth:`.update_KCNL`.

        """
        cdef int i, a
        cdef int pti, ptj
        cdef double xi, eta
        cdef double points[2]
        cdef double wij, detJ, w_x, w_y
        cdef double exx, eyy, gxy, kxx, kyy, kxy
        cdef double A[9]
        cdef double B[9]
        cdef double N[3]
        cdef double NNL[3]
        cdef double MNL[3]
        cdef double epsNL[3]
        cdef double *ue
        cdef double *finte
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Gwx
        cdef double *Gwy

        ue = &self.probe.ue[0]
        finte = &self.probe.finte[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Gwx = &self.probe.Gwx[0]
        Gwy = &self.probe.Gwy[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, A, B, NULL, NULL)

        # NOTE same two-point Gauss-Legendre quadrature as in update_KG
        wij = 1.
        points[0] = -0.5773502691896257645092
        points[1] = +0.5773502691896257645092

        for pti in range(2):
            xi = points[pti]
            for ptj in range(2):
                eta = points[ptj]
                detJ = self._update_probe_BL_G(xi, eta)

                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.
                w_x = 0.
                w_y = 0.
                for i in range(24):
                    exx += BLexx[i]*ue[i]
                    eyy += BLeyy[i]*ue[i]
                    gxy += BLgxy[i]*ue[i]
                    kxx += BLkxx[i]*ue[i]
                    kyy += BLkyy[i]*ue[i]
                    kxy += BLkxy[i]*ue[i]
                    w_x += Gwx[i]*ue[i]
                    w_y += Gwy[i]*ue[i]

                epsNL[0] = w_x*w_x/2.
                epsNL[1] = w_y*w_y/2.
                epsNL[2] = w_x*w_y

                for a in range(3):
                    # stress resultants of the linear strains, as in update_KG
                    N[a] = (A[3*a]*exx + A[3*a + 1]*eyy + A[3*a + 2]*gxy
                          + B[3*a]*kxx + B[3*a + 1]*kyy + B[3*a + 2]*kxy)
                    # stress resultants of the nonlinear membrane strain
                    NNL[a] = A[3*a]*epsNL[0] + A[3*a + 1]*epsNL[1] + A[3*a + 2]*epsNL[2]
                    MNL[a] = B[3*a]*epsNL[0] + B[3*a + 1]*epsNL[1] + B[3*a + 2]*epsNL[2]

                for i in range(24):
                    finte[i] += wij*detJ*(
                        # Bm.T*NNL + Bb.T*MNL
                          BLexx[i]*NNL[0] + BLeyy[i]*NNL[1] + BLgxy[i]*NNL[2]
                        + BLkxx[i]*MNL[0] + BLkyy[i]*MNL[1] + BLkxy[i]*MNL[2]
                        # BmL.T*(N + NNL)
                        + w_x*Gwx[i]*(N[0] + NNL[0])
                        + w_y*Gwy[i]*(N[1] + NNL[1])
                        + (w_x*Gwy[i] + w_y*Gwx[i])*(N[2] + NNL[2])
                    )


    cpdef void update_KCNL(Quad4 self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           ShellProp prop,
                           int update_KCNLv_only=0
                           ):
        r"""Update sparse vectors for the nonlinear constitutive stiffness matrix KCNL

        Assuming that KCNL = KC0L + KCL0 + KCLL + KGNL, built from the von Karman
        membrane strains

        .. math::
            \epsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2, \quad
            \epsilon_{yy} = v_{,y} + \frac{1}{2} w_{,y}^2, \quad
            \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

        whose nonlinear part is `\{\epsilon_{NL}\} = \frac{1}{2} [B_{mL}]
        \{u_e\}`, with `[B_{mL}]` its variation. With `[B_m]` and `[B_b]` the
        linear membrane and bending strain-displacement matrices, `[G]` the
        gradient of `w`, and `[A]`, `[B]` the laminate matrices:

        - KC0L = `[B_m]^T [A] [B_{mL}] + [B_b]^T [B] [B_{mL}]`
        - KCL0 = KC0L`^T`
        - KCLL = `[B_{mL}]^T [A] [B_{mL}]`
        - KGNL = `[G]^T [N_{NL}] [G]`, with `\{N_{NL}\} = [A] \{\epsilon_{NL}\}`

        The first three groups are the constitutive terms coupling the linear and
        the nonlinear parts of the membrane strain. KGNL is geometric, carrying the
        stress of the nonlinear membrane strain. It is collected here so that
        :meth:`.update_KG` stays homogeneous of degree one in the displacements,
        which is what a linear buckling analysis needs. With it here,

        .. math::
            K_T = K_{C0} + K_{CNL}(u) + K_G(u)

        is the exact Jacobian of the internal forces of :meth:`.update_fint` with
        ``nonlinear=1``, and a Newton-Raphson iteration built on them converges
        quadratically. The quadrature of :meth:`.update_KG` is used.

        Before this function is called, the probe :class:`.Quad4Probe` attribute
        of the :class:`.Quad4` object must be updated using
        :func:`.update_probe_ue` with the current displacements; and
        :func:`.update_probe_xe` with the node coordinates.

        Parameters
        ----------
        KCNLr : np.array
            Array to store row positions of sparse values
        KCNLc : np.array
            Array to store column positions of sparse values
        KCNLv : np.array
            Array to store sparse values
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes are
            read from.
        update_KCNLv_only : int
            The default ``0`` means that the row and column indices ``KCNLr`` and
            ``KCNLc`` should also be updated. Any other value will only update the
            stiffness matrix values ``KCNLv``.

        """
        cdef int i, j, node_i, node_j, k, ke, m, n
        cdef int c[4]
        cdef double r[6][6]

        with nogil:
            # local to global transformation
            # translation DOFs
            r[0][0] = self.r11
            r[0][1] = self.r12
            r[0][2] = self.r13
            r[1][0] = self.r21
            r[1][1] = self.r22
            r[1][2] = self.r23
            r[2][0] = self.r31
            r[2][1] = self.r32
            r[2][2] = self.r33
            # rotation DOFs
            r[0+3][0+3] = self.r11
            r[0+3][1+3] = self.r12
            r[0+3][2+3] = self.r13
            r[1+3][0+3] = self.r21
            r[1+3][1+3] = self.r22
            r[1+3][2+3] = self.r23
            r[2+3][0+3] = self.r31
            r[2+3][1+3] = self.r32
            r[2+3][2+3] = self.r33
            # coupled translation-rotation DOFs
            for i in range(3):
                for j in range(3):
                    r[i][j+3] = 0.
                    r[i+3][j] = 0.

            if update_KCNLv_only == 0:
                # positions in the global stiffness matrix
                c[0] = self.c1
                c[1] = self.c2
                c[2] = self.c3
                c[3] = self.c4

                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = self.init_k_KCNL + 24*(node_i*DOF + m) + node_j*DOF + n
                                KCNLr[k] = c[node_i] + m
                                KCNLc[k] = c[node_j] + n

            self._update_probe_KCNLve(prop)

            # NOTE from element to global coordinates:
            #
            # Kg = R @ Ke @ R.T
            #
            # in tensor notation:
            #
            # Kg_{mn} = r_{mi} * Ke_{ij} * r_{nj}
            #
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = self.init_k_KCNL + 24*(node_i*DOF + m) + node_j*DOF + n
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 24*(node_i*DOF + i) + node_j*DOF + j
                                    KCNLv[k] += r[m][i]*self.probe.KCNLve[ke]*r[n][j]


    cpdef void update_KG(Quad4 self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         ShellProp prop
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        Two-point Gauss-Legendre quadrature is used, which showed more accuracy
        for linear buckling load predictions.

        Before this function is called, the probe :class:`.Quad4Probe`
        attribute of the :class:`.Quad4` object must be updated using
        :func:`.update_probe_ue` with the correct pre-buckling (fundamental
        state) displacements; and :func:`.update_probe_xe` with the node
        coordinates.

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values
        KGc : np.array
           Array to store column positions of sparse values
        KGv : np.array
            Array to store sparse values
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.

        """
        cdef double *ue
        cdef int c1, c2, c3, c4, i, j, k
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double denr[16]
        cdef double dexx, deyy, dgxy
        cdef double points[2]
        cdef double Ae[9]
        cdef double Be[9]
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double m11, m12, m21, m22
        cdef double j11, j12, j21, j22
        cdef double N1x, N2x, N3x, N4x, N1y, N2y, N3y, N4y
        cdef double Nxx, Nyy, Nxy

        with nogil:
            # NOTE constitutive matrices in the element coordinate system,
            #      the same function is used by all element methods
            prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, Ae, Be, NULL, NULL)
            A11 = Ae[0]
            A12 = Ae[1]
            A16 = Ae[2]
            A22 = Ae[4]
            A26 = Ae[5]
            A66 = Ae[8]
            B11 = Be[0]
            B12 = Be[1]
            B16 = Be[2]
            B22 = Be[4]
            B26 = Be[5]
            B66 = Be[8]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            ue = &self.probe.ue[0]

            # positions of nodes 1,2,3,4 in the global matrix
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_KG
            KGr[k] = 0+c1
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    Nxx = ue[0]*(A11*N1x + A16*N1y) + ue[10]*(B11*N2x + B16*N2y) + ue[12]*(A11*N3x + A16*N3y) + ue[13]*(A12*N3y + A16*N3x) - ue[15]*(B12*N3y + B16*N3x) + ue[16]*(B11*N3x + B16*N3y) + ue[18]*(A11*N4x + A16*N4y) + ue[19]*(A12*N4y + A16*N4x) + ue[1]*(A12*N1y + A16*N1x) - ue[21]*(B12*N4y + B16*N4x) + ue[22]*(B11*N4x + B16*N4y) - ue[3]*(B12*N1y + B16*N1x) + ue[4]*(B11*N1x + B16*N1y) + ue[6]*(A11*N2x + A16*N2y) + ue[7]*(A12*N2y + A16*N2x) - ue[9]*(B12*N2y + B16*N2x)
                    Nyy = ue[0]*(A12*N1x + A26*N1y) + ue[10]*(B12*N2x + B26*N2y) + ue[12]*(A12*N3x + A26*N3y) + ue[13]*(A22*N3y + A26*N3x) - ue[15]*(B22*N3y + B26*N3x) + ue[16]*(B12*N3x + B26*N3y) + ue[18]*(A12*N4x + A26*N4y) + ue[19]*(A22*N4y + A26*N4x) + ue[1]*(A22*N1y + A26*N1x) - ue[21]*(B22*N4y + B26*N4x) + ue[22]*(B12*N4x + B26*N4y) - ue[3]*(B22*N1y + B26*N1x) + ue[4]*(B12*N1x + B26*N1y) + ue[6]*(A12*N2x + A26*N2y) + ue[7]*(A22*N2y + A26*N2x) - ue[9]*(B22*N2y + B26*N2x)
                    Nxy = ue[0]*(A16*N1x + A66*N1y) + ue[10]*(B16*N2x + B66*N2y) + ue[12]*(A16*N3x + A66*N3y) + ue[13]*(A26*N3y + A66*N3x) - ue[15]*(B26*N3y + B66*N3x) + ue[16]*(B16*N3x + B66*N3y) + ue[18]*(A16*N4x + A66*N4y) + ue[19]*(A26*N4y + A66*N4x) + ue[1]*(A26*N1y + A66*N1x) - ue[21]*(B26*N4y + B66*N4x) + ue[22]*(B16*N4x + B66*N4y) - ue[3]*(B26*N1y + B66*N1x) + ue[4]*(B16*N1x + B66*N1y) + ue[6]*(A16*N2x + A66*N2y) + ue[7]*(A26*N2y + A66*N2x) - ue[9]*(B26*N2y + B66*N2x)
                    # NOTE the Allman enrichment populates the drilling
                    #      columns of the membrane operator, so the membrane
                    #      stress resultants above, generated for the
                    #      unenriched field, need the contribution of the
                    #      edge modes. Without it KG would not be part of the
                    #      exact Jacobian of the internal forces, see
                    #      update_KCNL
                    if self.drilling_model == 0:
                        allman_enrichment(&self.probe.xe[0], xi, eta, j11,
                                          j12, j21, j22, denr)
                        dexx = 0.
                        deyy = 0.
                        dgxy = 0.
                        for i in range(NUM_NODES):
                            dexx += denr[i]*ue[DOF*i + 5]
                            deyy += denr[12 + i]*ue[DOF*i + 5]
                            dgxy += (denr[4 + i] + denr[8 + i])*ue[DOF*i + 5]
                        Nxx += A11*dexx + A12*deyy + A16*dgxy
                        Nyy += A12*dexx + A22*deyy + A26*dgxy
                        Nxy += A16*dexx + A26*deyy + A66*dgxy

                    k = self.init_k_KG
                    KGv[k] += r13**2*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))


    cpdef void update_KG_given_stress(Quad4 self,
                                      double Nxx, double Nyy, double Nxy,
                                      long [::1] KGr,
                                      long [::1] KGc,
                                      double [::1] KGv
                                      ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        .. note:: A constant stress state is assumed within the element,
                  according to the given values of `N_{xx}, N_{yy}, N_{xy}`.

        Two-point Gauss-Legendre quadrature is used, which showed more accuracy
        for linear buckling load predictions.

        Before this function is called, the probe :class:`.Quad4Probe`
        attribute of the :class:`.Quad4` object must be updated using
        :func:`.update_probe_xe` with the node coordinates.

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values
        KGc : np.array
           Array to store column positions of sparse values
        KGv : np.array
            Array to store sparse values

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double points[2]
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double j11, j12, j21, j22
        cdef double N1x, N2x, N3x, N4x, N1y, N2y, N3y, N4y

        with nogil:
            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # positions of nodes 1,2,3,4 in the global matrix
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_KG
            KGr[k] = 0+c1
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c1
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c1
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c1
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c2
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c2
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c2
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c3
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c3
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c3
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 0+c4
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 1+c4
            KGc[k] = 2+c4
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c1
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c1
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c1
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c2
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c2
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c2
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c3
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c3
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c3
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 0+c4
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 1+c4
            k += 1
            KGr[k] = 2+c4
            KGc[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    k = self.init_k_KG
                    KGv[k] += r13**2*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N1y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N2y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N3y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N1x*Nxx*detJ*wij + N1y*Nxy*detJ*wij) + N4y*(N1x*Nxy*detJ*wij + N1y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N1y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N2y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N3y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N2x*Nxx*detJ*wij + N2y*Nxy*detJ*wij) + N4y*(N2x*Nxy*detJ*wij + N2y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N1y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N2y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N3y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N3x*Nxx*detJ*wij + N3y*Nxy*detJ*wij) + N4y*(N3x*Nxy*detJ*wij + N3y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N1x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N1y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N2x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N2y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N3x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N3y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(N4x*(N4x*Nxx*detJ*wij + N4y*Nxy*detJ*wij) + N4y*(N4x*Nxy*detJ*wij + N4y*Nyy*detJ*wij))


    cpdef void update_M(Quad4 self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        ShellProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        Different integration schemes are available by means of the ``mtype``
        parameter.

        Parameters
        ----------
        Mr : np.array
            Array to store row positions of sparse values
        Mc : np.array
            Array to store column positions of sparse values
        Mv : np.array
            Array to store sparse values
        mtype : int, optional
            0 for consistent mass matrix using method from Brockman 1987
            1 for reduced integration mass matrix using method from Brockman 1987
            2 for lumped mass matrix using method from Brockman 1987

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double intrho, intrhoz, intrhoz2
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double h11, h12, h13, h14, h22, h23, h24, h33, h34, h44, valH1
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double N1, N2, N3, N4
        cdef double points[2]

        with nogil:
            intrho = prop.intrho
            intrhoz = prop.intrhoz
            intrhoz2 = prop.intrhoz2

            valH1 = 0.0625*self.area

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            if mtype == 0: # M_cons consistent mass matrix, using two-point Gauss-Legendre quadrature
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c4

                # NOTE two-point Gauss-Legendre quadrature
                wij = 1.
                points[0] = -0.5773502691896257645092
                points[1] = +0.5773502691896257645092
                h11 = 0.
                h12 = 0.
                h13 = 0.
                h14 = 0.
                h22 = 0.
                h23 = 0.
                h24 = 0.
                h33 = 0.
                h34 = 0.
                h44 = 0.
                for i in range(2):
                    xi = points[i]
                    for j in range(2):
                        eta = points[j]

                        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                        detJ = J11*J22 - J12*J21

                        N1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                        N2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                        N3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                        N4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

                        h11 += N1**2*detJ*wij
                        h12 += N1*N2*detJ*wij
                        h13 += N1*N3*detJ*wij
                        h14 += N1*N4*detJ*wij
                        h22 += N2**2*detJ*wij
                        h23 += N2*N3*detJ*wij
                        h24 += N2*N4*detJ*wij
                        h33 += N3**2*detJ*wij
                        h34 += N3*N4*detJ*wij
                        h44 += N4**2*detJ*wij

                k = self.init_k_M
                Mv[k] += h11*intrho*r11**2 + h11*intrho*r12**2 + h11*intrho*r13**2
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrhoz*r11*r22 - h11*intrhoz*r12*r21
                k += 1
                Mv[k] += h11*intrhoz*r11*r32 - h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r21**2 + h11*intrho*r22**2 + h11*intrho*r23**2
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += -h11*intrhoz*r11*r22 + h11*intrhoz*r12*r21
                k += 1
                Mv[k] += h11*intrhoz*r21*r32 - h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r31**2 + h11*intrho*r32**2 + h11*intrho*r33**2
                k += 1
                Mv[k] += -h11*intrhoz*r11*r32 + h11*intrhoz*r12*r31
                k += 1
                Mv[k] += -h11*intrhoz*r21*r32 + h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += -h11*intrhoz*r11*r22 + h11*intrhoz*r12*r21
                k += 1
                Mv[k] += -h11*intrhoz*r11*r32 + h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11**2 + h11*intrhoz2*r12**2
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11**2 + h12*intrhoz2*r12**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11**2 + h13*intrhoz2*r12**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11**2 + h14*intrhoz2*r12**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz*r11*r22 - h11*intrhoz*r12*r21
                k += 1
                Mv[k] += -h11*intrhoz*r21*r32 + h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r21**2 + h11*intrhoz2*r22**2
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r21**2 + h12*intrhoz2*r22**2
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r21**2 + h13*intrhoz2*r22**2
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r21**2 + h14*intrhoz2*r22**2
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz*r11*r32 - h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrhoz*r21*r32 - h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz2*r31**2 + h11*intrhoz2*r32**2
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r31**2 + h12*intrhoz2*r32**2
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r31**2 + h13*intrhoz2*r32**2
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r31**2 + h14*intrhoz2*r32**2
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrho*r11**2 + h22*intrho*r12**2 + h22*intrho*r13**2
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrhoz*r11*r22 - h22*intrhoz*r12*r21
                k += 1
                Mv[k] += h22*intrhoz*r11*r32 - h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r21**2 + h22*intrho*r22**2 + h22*intrho*r23**2
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += -h22*intrhoz*r11*r22 + h22*intrhoz*r12*r21
                k += 1
                Mv[k] += h22*intrhoz*r21*r32 - h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r31**2 + h22*intrho*r32**2 + h22*intrho*r33**2
                k += 1
                Mv[k] += -h22*intrhoz*r11*r32 + h22*intrhoz*r12*r31
                k += 1
                Mv[k] += -h22*intrhoz*r21*r32 + h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11**2 + h12*intrhoz2*r12**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h22*intrhoz*r11*r22 + h22*intrhoz*r12*r21
                k += 1
                Mv[k] += -h22*intrhoz*r11*r32 + h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11**2 + h22*intrhoz2*r12**2
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11**2 + h23*intrhoz2*r12**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11**2 + h24*intrhoz2*r12**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r21**2 + h12*intrhoz2*r22**2
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz*r11*r22 - h22*intrhoz*r12*r21
                k += 1
                Mv[k] += -h22*intrhoz*r21*r32 + h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r21**2 + h22*intrhoz2*r22**2
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r21**2 + h23*intrhoz2*r22**2
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r21**2 + h24*intrhoz2*r22**2
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r31**2 + h12*intrhoz2*r32**2
                k += 1
                Mv[k] += h22*intrhoz*r11*r32 - h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrhoz*r21*r32 - h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz2*r31**2 + h22*intrhoz2*r32**2
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r31**2 + h23*intrhoz2*r32**2
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r31**2 + h24*intrhoz2*r32**2
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrho*r11**2 + h33*intrho*r12**2 + h33*intrho*r13**2
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrhoz*r11*r22 - h33*intrhoz*r12*r21
                k += 1
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r21**2 + h33*intrho*r22**2 + h33*intrho*r23**2
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += -h33*intrhoz*r11*r22 + h33*intrhoz*r12*r21
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r31**2 + h33*intrho*r32**2 + h33*intrho*r33**2
                k += 1
                Mv[k] += -h33*intrhoz*r11*r32 + h33*intrhoz*r12*r31
                k += 1
                Mv[k] += -h33*intrhoz*r21*r32 + h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11**2 + h13*intrhoz2*r12**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11**2 + h23*intrhoz2*r12**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h33*intrhoz*r11*r22 + h33*intrhoz*r12*r21
                k += 1
                Mv[k] += -h33*intrhoz*r11*r32 + h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11**2 + h33*intrhoz2*r12**2
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11**2 + h34*intrhoz2*r12**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r21**2 + h13*intrhoz2*r22**2
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r21**2 + h23*intrhoz2*r22**2
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz*r11*r22 - h33*intrhoz*r12*r21
                k += 1
                Mv[k] += -h33*intrhoz*r21*r32 + h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r21**2 + h33*intrhoz2*r22**2
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r21**2 + h34*intrhoz2*r22**2
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r31**2 + h13*intrhoz2*r32**2
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r31**2 + h23*intrhoz2*r32**2
                k += 1
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz2*r31**2 + h33*intrhoz2*r32**2
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r31**2 + h34*intrhoz2*r32**2
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrho*r11**2 + h44*intrho*r12**2 + h44*intrho*r13**2
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrhoz*r11*r22 - h44*intrhoz*r12*r21
                k += 1
                Mv[k] += h44*intrhoz*r11*r32 - h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r21**2 + h44*intrho*r22**2 + h44*intrho*r23**2
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += -h44*intrhoz*r11*r22 + h44*intrhoz*r12*r21
                k += 1
                Mv[k] += h44*intrhoz*r21*r32 - h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r31**2 + h44*intrho*r32**2 + h44*intrho*r33**2
                k += 1
                Mv[k] += -h44*intrhoz*r11*r32 + h44*intrhoz*r12*r31
                k += 1
                Mv[k] += -h44*intrhoz*r21*r32 + h44*intrhoz*r22*r31
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11**2 + h14*intrhoz2*r12**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11**2 + h24*intrhoz2*r12**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11**2 + h34*intrhoz2*r12**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h44*intrhoz*r11*r22 + h44*intrhoz*r12*r21
                k += 1
                Mv[k] += -h44*intrhoz*r11*r32 + h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11**2 + h44*intrhoz2*r12**2
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r21**2 + h14*intrhoz2*r22**2
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r21**2 + h24*intrhoz2*r22**2
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r21**2 + h34*intrhoz2*r22**2
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz*r11*r22 - h44*intrhoz*r12*r21
                k += 1
                Mv[k] += -h44*intrhoz*r21*r32 + h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r21**2 + h44*intrhoz2*r22**2
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r31**2 + h14*intrhoz2*r32**2
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r31**2 + h24*intrhoz2*r32**2
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r31**2 + h34*intrhoz2*r32**2
                k += 1
                Mv[k] += h44*intrhoz*r11*r32 - h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrhoz*r21*r32 - h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz2*r31**2 + h44*intrhoz2*r32**2

            elif mtype == 1: # M_red mass matrix purely by reduced integration
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c4

                k = self.init_k_M
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1

            elif mtype == 2: # M_lump lumped mass matrix using two-point Gauss-Lobatto quadrature
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c4

                wij = 1.
                # NOTE two-point Gauss-Lobatto quadrature
                points[0] = -1.
                points[1] = +1.
                h11 = 0.
                h12 = 0.
                h13 = 0.
                h14 = 0.
                h22 = 0.
                h23 = 0.
                h24 = 0.
                h33 = 0.
                h34 = 0.
                h44 = 0.
                for i in range(2):
                    xi = points[i]
                    for j in range(2):
                        eta = points[j]

                        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                        detJ = J11*J22 - J12*J21

                        N1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                        N2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                        N3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                        N4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

                        h11 += N1**2*detJ*wij
                        h12 += N1*N2*detJ*wij
                        h13 += N1*N3*detJ*wij
                        h14 += N1*N4*detJ*wij
                        h22 += N2**2*detJ*wij
                        h23 += N2*N3*detJ*wij
                        h24 += N2*N4*detJ*wij
                        h33 += N3**2*detJ*wij
                        h34 += N3*N4*detJ*wij
                        h44 += N4**2*detJ*wij

                k = self.init_k_M
                Mv[k] += h11*intrho*r11**2 + h11*intrho*r12**2 + h11*intrho*r13**2
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r21**2 + h11*intrho*r22**2 + h11*intrho*r23**2
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r31**2 + h11*intrho*r32**2 + h11*intrho*r33**2
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += h11*intrhoz2*r11**2 + h11*intrhoz2*r12**2
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r11**2 + h12*intrhoz2*r12**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r11**2 + h13*intrhoz2*r12**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r11**2 + h14*intrhoz2*r12**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r21**2 + h11*intrhoz2*r22**2
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r21**2 + h12*intrhoz2*r22**2
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r21**2 + h13*intrhoz2*r22**2
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r21**2 + h14*intrhoz2*r22**2
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz2*r31**2 + h11*intrhoz2*r32**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r31**2 + h12*intrhoz2*r32**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r31**2 + h13*intrhoz2*r32**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r31**2 + h14*intrhoz2*r32**2
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r11**2 + h22*intrho*r12**2 + h22*intrho*r13**2
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r21**2 + h22*intrho*r22**2 + h22*intrho*r23**2
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r31**2 + h22*intrho*r32**2 + h22*intrho*r33**2
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += h12*intrhoz2*r11**2 + h12*intrhoz2*r12**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h22*intrhoz2*r11**2 + h22*intrhoz2*r12**2
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r11**2 + h23*intrhoz2*r12**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r11**2 + h24*intrhoz2*r12**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r21**2 + h12*intrhoz2*r22**2
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r21**2 + h22*intrhoz2*r22**2
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r21**2 + h23*intrhoz2*r22**2
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r21**2 + h24*intrhoz2*r22**2
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r31**2 + h12*intrhoz2*r32**2
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz2*r31**2 + h22*intrhoz2*r32**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r31**2 + h23*intrhoz2*r32**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r31**2 + h24*intrhoz2*r32**2
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r11**2 + h33*intrho*r12**2 + h33*intrho*r13**2
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r21**2 + h33*intrho*r22**2 + h33*intrho*r23**2
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r31**2 + h33*intrho*r32**2 + h33*intrho*r33**2
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += h13*intrhoz2*r11**2 + h13*intrhoz2*r12**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r11**2 + h23*intrhoz2*r12**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz2*r11**2 + h33*intrhoz2*r12**2
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r11**2 + h34*intrhoz2*r12**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r21**2 + h13*intrhoz2*r22**2
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r21**2 + h23*intrhoz2*r22**2
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r21**2 + h33*intrhoz2*r22**2
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r21**2 + h34*intrhoz2*r22**2
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r31**2 + h13*intrhoz2*r32**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r31**2 + h23*intrhoz2*r32**2
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz2*r31**2 + h33*intrhoz2*r32**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r31**2 + h34*intrhoz2*r32**2
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r11**2 + h44*intrho*r12**2 + h44*intrho*r13**2
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r21**2 + h44*intrho*r22**2 + h44*intrho*r23**2
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r31**2 + h44*intrho*r32**2 + h44*intrho*r33**2
                k += 1
                Mv[k] += h14*intrhoz2*r11**2 + h14*intrhoz2*r12**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r11**2 + h24*intrhoz2*r12**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r11**2 + h34*intrhoz2*r12**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h44*intrhoz2*r11**2 + h44*intrhoz2*r12**2
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r21**2 + h14*intrhoz2*r22**2
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r21**2 + h24*intrhoz2*r22**2
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r21**2 + h34*intrhoz2*r22**2
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r21**2 + h44*intrhoz2*r22**2
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r31**2 + h14*intrhoz2*r32**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r31**2 + h24*intrhoz2*r32**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r31**2 + h34*intrhoz2*r32**2
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz2*r31**2 + h44*intrhoz2*r32**2


    cpdef void update_KA_beta(Quad4 self,
                        long [::1] KA_betar,
                        long [::1] KA_betac,
                        double [::1] KA_betav,
                        ):
        r"""Update sparse vectors for piston-theory aerodynamic matrix `KA_{\beta}`

        Parameters
        ----------
        KA_betar : np.array
            Array to store row positions of sparse values
        KA_betac : np.array
            Array to store column positions of sparse values
        KA_betav : np.array
            Array to store sparse values

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double j11, j12, j21, j22
        cdef double N1, N2, N3, N4
        cdef double N1x, N2x, N3x, N4x
        cdef double N1y, N2y, N3y, N4y
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double points[2]

        with nogil:

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_KA_beta
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    N1 = 0.25*eta*xi - 0.25*eta - 0.25*xi + 0.25
                    N2 = -0.25*eta*xi - 0.25*eta + 0.25*xi + 0.25
                    N3 = 0.25*eta*xi + 0.25*eta + 0.25*xi + 0.25
                    N4 = -0.25*eta*xi + 0.25*eta - 0.25*xi + 0.25

                    N1x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    N2x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    N3x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    N4x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    N1y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    N2y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    N3y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    N4y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    k = self.init_k_KA_beta
                    KA_betav[k] += -N1*detJ*r13**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r33**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r33**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r33**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N1*detJ*r33**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r33**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r33**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r33**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N2*detJ*r33**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r33**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r33**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r33**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N3*detJ*r33**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r23*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23**2*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r33**2*wij*(N1x*r11 + N1y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r33**2*wij*(N2x*r11 + N2y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r33**2*wij*(N3x*r11 + N3y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r13*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r23*r33*wij*(N4x*r11 + N4y*r21)
                    k += 1
                    KA_betav[k] += -N4*detJ*r33**2*wij*(N4x*r11 + N4y*r21)


    cpdef void update_KA_gamma(Quad4 self,
                        long [::1] KA_gammar,
                        long [::1] KA_gammac,
                        double [::1] KA_gammav,
                        ):
        r"""Update sparse vectors for piston-theory aerodynamic matrix `KA_{\gamma}`

        Parameters
        ----------
        KA_gammar : np.array
            Array to store row positions of sparse values
        KA_gammac : np.array
            Array to store column positions of sparse values
        KA_gammav : np.array
            Array to store sparse values

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double N1, N2, N3, N4
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double points[2]

        with nogil:

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_KA_gamma
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    N1 = 0.25*eta*xi - 0.25*eta - 0.25*xi + 0.25
                    N2 = -0.25*eta*xi - 0.25*eta + 0.25*xi + 0.25
                    N3 = 0.25*eta*xi + 0.25*eta + 0.25*xi + 0.25
                    N4 = -0.25*eta*xi + 0.25*eta - 0.25*xi + 0.25

                    k = self.init_k_KA_gamma
                    KA_gammav[k] += N1**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1**2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2**2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N3**2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N1*N4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N2*N4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N3*N4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += N4**2*detJ*r33**2*wij


    cpdef void update_CA(Quad4 self,
                        long [::1] CAr,
                        long [::1] CAc,
                        double [::1] CAv,
                        ):
        r"""Update sparse vectors for piston-theory aerodynamic damping matrix `CA`

        Parameters
        ----------
        CAr : np.array
            Array to store row positions of sparse values
        CAc : np.array
            Array to store column positions of sparse values
        CAv : np.array
            Array to store sparse values

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double N1, N2, N3, N4
        cdef double xi, eta, wij, J11, J12, J21, J22, detJ
        cdef double points[2]

        with nogil:

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_CA
            CAr[k] = 0+c1
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    N1 = 0.25*eta*xi - 0.25*eta - 0.25*xi + 0.25
                    N2 = -0.25*eta*xi - 0.25*eta + 0.25*xi + 0.25
                    N3 = 0.25*eta*xi + 0.25*eta + 0.25*xi + 0.25
                    N4 = -0.25*eta*xi + 0.25*eta - 0.25*xi + 0.25

                    k = self.init_k_CA
                    CAv[k] += -N1**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1**2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2**2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N3**2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N1*N4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N2*N4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N3*N4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -N4**2*detJ*r33**2*wij


