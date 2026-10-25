#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
===========================================================================
Tria3R - Triangular element with reduced integration (:mod:`pyfe3d.tria3r`)
===========================================================================

.. currentmodule:: pyfe3d.tria3r

The module has three classes: :class:`.Tria3RData`, with the sizes of the
sparse arrays that one element fills; :class:`.Tria3RProbe`, with the
buffers in element coordinates that one or more elements share; and
:class:`.Tria3R`, the element. The ``update_*`` methods of :class:`.Tria3R`
write the element contributions in sparse row, column and value arrays, from
the positions given by its ``init_k_*`` attributes, or add them to a global
internal force vector.

Three-node triangular shell element with 6 degrees-of-freedom per node, `u`,
`v`, `w`, `r_x`, `r_y`, `r_z`, with reduced integration.

The transverse shear stiffnesses ``A44``, ``A45`` and ``A55`` of the
:class:`pyfe3d.shellprop.ShellProp` are reduced by a factor controlled by the
``alpha_shear_locking`` attribute of :class:`.Tria3R`, ``0.7`` by default,
which is a parameter to be verified per problem class and not a constant of
the element. :class:`pyfe3d.tria3dsg.Tria3DSG` takes no such parameter and is
the triangle to reach for by default; this one is kept for continuity with
results obtained before it, and for the comparison itself.

The drilling rotation `r_z` is given stiffness by one of two models, selected
with the ``drilling_model`` attribute of :class:`.Tria3R`: the physics-based
default, for which the nodal moments about the shell normal recovered in the
internal force vector are physical and no user parameter is involved, and a
fictitious penalty scaled by ``K6ROT``, the default before version 0.10.0,
whose only purpose is to remove the singularity of the drilling
degree-of-freedom. Geometrically nonlinear analyses with the von Karman
membrane strains are supported by :meth:`.Tria3R.update_fint` with
``nonlinear=1``, :meth:`.Tria3R.update_KCNL` and :meth:`.Tria3R.update_KG`.

Notes
-----
Triangular element with reduced integration, where a single point at
the centroid (`N1=N2=N3=1/3`) is evaluated with the weight `1/2` times the
Jacobian determinant `2 A_e`, i.e. the area `A_e`, preventing shear
locking. The drilling stiffness is evaluated with
full integration. Under the default ``drilling_model = 0`` this applies to
the terms of the Allman enrichment, the Hughes-Brezzi term being integrated
with a single point at the centroid, see the drilling stiffness below.

The transverse shear stiffnesses `A_{44}`, `A_{45}` and `A_{55}` are stabilised
with the scalar factor that Lyly, Stenberg and Vihinen [1]_ proposed for the
MITC4 quadrilateral, with the longest edge of the element as its size
(their Eq. (3.12)), called Stenberg's method by Bischoff and Bletzinger [2]_
(Sec. 3.1), in the form adopted by Castro et al. [3]_, with the longest edge
of the triangle as the element size, where the transverse shear terms are
changed as per Eq. (26) in Castro et al., here repeated for convenience:

.. math::

    \left[\begin{matrix}\hat{A}_{44}, \hat{A}_{45} \ \hat{A}_{45}, \hat{A}_{55}\end{matrix}\right] = \frac{1}{1 + factor}\left[\begin{matrix}A_{44}, A_{45} \ A_{45}, A_{55}\end{matrix}\right]

with `factor` defined as:

    factor = \frac{\alpha \ell^2}{h^2}

where `\alpha` is a positive constant parameters (see the ``alpha_shear_locking`` attribute), 
`\ell` is the longest edge of the corresponding triangle, and `h` the total thickness of the element.

Note that `factor \rightarrow 0` as the element size shrinks, so the true
stiffness is recovered under mesh refinement and the scheme is consistent.
The rate matters though: `\ell < 0.12 h` is needed for the stabilised
stiffness to be within 1 per cent of the true one, which for a thin shell is
not a mesh anyone would build. What makes the scheme legitimate all the same
is that in a load-driven problem the error it introduces is
`factor \times f_s`, with `f_s` the shear fraction of the response, and for a
plate or beam of span `L` discretised with `n` elements

.. math::

    factor \times f_s \sim \frac{\alpha \ell^2}{h^2} c \frac{h^2}{L^2}
                       = \frac{\alpha c}{n^2}

so the thickness cancels and the error behaves as an ordinary
`O(1/n^2)` discretisation error. This is measured in
``tests/test_transverse_shear_stiffness.py``.

.. warning:: The default `\alpha = 0.7` is roughly seven times the value
    of Stenberg's method in the references above, which is near `0.1`:
    Lyly et al. chose `0.1`, the value Bischoff and Bletzinger quote from
    them, and Castro et al.
    investigate `0.05` to `0.15` and recommend `0.07` to `0.09`, reporting
    that above `0.09` the linear buckling behaviour becomes overly soft and
    converges from below. Bischoff and Bletzinger show, however, that
    matching the deflection of a Timoshenko beam element gives `\alpha =
    \kappa/(2(1+\nu))`, i.e. `0.32` for `\kappa = 5/6` and `\nu = 0.3`
    (their Eq. (22)), about half the default, a value that works well when
    bending is dominant in one direction but that they find too flexible
    as soon as bending in two directions and twist are present (their
    Sec. 4.2). Their own algebraic subgrid scale stabilisation uses one
    parameter per direction of the element, `\alpha_\xi = 0.17
    (\ell_\eta/\ell_\xi)/(2(1+\nu))` and likewise `\alpha_\eta` (their
    Eqs. (23) and (24)), i.e. `0.065` for a square element with
    `\nu = 0.3`.

    The difference is not a retune but a consequence of where the factor is
    applied. In all three references it sits on top of an assumed-strain or
    discrete shear gap (DSG) field, the MITC4 field for Lyly et al. and a
    DSG field for Castro et al., whose operator already contains the cure
    of the shear locking, so there the factor is a mild stabilisation that
    improves stability and coarse-mesh accuracy. This
    element has no DSG: the transverse shear is taken from a single point at
    the centroid, which still locks, and the factor is therefore doing the
    unlocking. Reducing `\alpha` to the literature value makes this element
    stiffen sharply. On the plate of ``test_tria3r_natural_freq.py``, with
    the consistent mass matrix, the error in the first natural frequency
    goes

    .. math::

        +0.6\% \ (\alpha = 0.7) \quad
        +30.6\% \ (\alpha = 0.1) \quad
        +77.3\% \ (\alpha = 0.01) \quad
        +96.1\% \ (\alpha = 0)

    so with the stabilisation removed the element is nearly twice as stiff
    as it should be, which is the locking itself. For scale,
    :class:`pyfe3d.tria3dsg.Tria3DSG` on the identical mesh, with no
    parameter of any kind, gives `+2.8\%`.

    The price of the large `\alpha` is paid on the transverse shear itself.
    A field with `w` varying and all rotations zero has identically zero
    curvature, so it is resisted by transverse shear alone, and the
    stabilisation divides that resistance by `1 + factor`. Measured on an
    8 by 8 patch, this element's stiffness in that subspace is 0.33, 0.019
    and 0.0012 of the Quad4 and Quad4R value at `\ell/h` of 1.25, 6.25 and
    25. A Donnell geometric stiffness matrix works on `\partial w/\partial x`,
    which is exactly that subspace, so thin-shell buckling is where the
    scheme shows its cost, as measured for a cylinder in
    ``tests/test_quad4_linear_buckling_cylinder_displ.py``. The failure
    there is not a uniformly wrong load but a change of critical mode: with
    `factor` at 1852 the element admits a mode at two elements per
    wavelength, the mesh Nyquist limit, carrying 36.6 per cent of its
    energy in transverse shear, and that mesh artefact undercuts the
    physical one. At the literature `\alpha = 0.1` the physical mode is
    critical again and this element agrees with the quadrilateral and with
    :class:`.Tria3DSG` to 3 per cent on it, while that same `\alpha = 0.1`
    makes the plate above 30.6 per cent too stiff. No single value serves
    both problem classes.

    The proper fix is to give the element a locking-free transverse shear
    field, the DSG of Bletzinger, Bischoff and Ramm (2000) [8]_ being the
    triangular counterpart of the MITC4 treatment used for quadrilaterals,
    after which no `\alpha` is needed at all. That element now exists as
    :mod:`pyfe3d.tria3dsg`, and it is the triangle to reach for by default;
    this one is kept for continuity with results obtained before it, and
    for the comparison itself. Where it is used, ``alpha_shear_locking`` is
    a parameter to be verified per problem class and not a constant of the
    element.

The transverse shear stiffnesses `A_{44}`, `A_{45}` and `A_{55}` are read
from the :class:`pyfe3d.shellprop.ShellProp` object with the shear correction
already applied, see :meth:`pyfe3d.shellprop.ShellProp.calc_transverse_shear_stiffness`,
and brought to the element coordinate system with
:meth:`pyfe3d.shellprop.ShellProp.calc_Ats_element`, before the stabilisation
above is applied. With ``shear_correction='rohwer'`` and the plies available,
that method re-evaluates the equilibrium-based stiffness of Rohwer (1988) [9]_
with the plies rotated to the element coordinate system. As concluded by
Lyly et al. (Sec. 5) and recalled by Castro et al., the shear correction is
no longer a very relevant parameter when the stabilisation scheme presented
above is used.

.. rubric:: Drilling stiffness

The FSDT kinematics contains no strain measure associated with `r_z`, so the
rows and columns of the drilling degree-of-freedom would be empty and a mesh
of coplanar elements would give a singular global stiffness matrix. Two
models are available, selected with the ``drilling_model`` attribute, and the
formulation of each is the one documented for :mod:`pyfe3d.quad4`, to which
the reader is referred for the derivation.

**Physics-based, the default** (``drilling_model = 0``). The in-plane
displacement field is enriched with the hierarchical quadratic edge modes of
Allman [4]_, so that the drilling rotations produce membrane strain energy,
and the independently interpolated `r_z` is tied to the rotation of the
membrane field by the regularised functional of Hughes and Brezzi [5]_.

The amplitude of the mode of the edge `k` joining nodes `i` and `j` is `a_k =
\frac{\ell_k}{8}\left({r_z}_i - {r_z}_j\right)`, so the enrichment is the
difference of the two drilling rotations already present at the ends of the
edge and introduces no new degree-of-freedom. For the triangle the
hierarchical bubble of that edge is

.. math::

    \hat{S}_k = 4 S_i S_j

with `S_i` the area coordinates, equal to unity at the mid-point of the edge
and zero at every vertex. The enrichment populates the drilling columns of
the membrane operator, giving `\tilde{\boldsymbol{B}}_m`, and the drilling residual
becomes `\tilde{\boldsymbol{B}}_{r_z} = \boldsymbol{S}^{r_z} + \frac{1}{2}\boldsymbol{\tilde
S}^u_{,y} - \frac{1}{2}\tilde{\boldsymbol{S}}^v_{,x}`, whose contribution to the
element stiffness matrix is `\gamma_{r_z} \int_A \tilde{\boldsymbol{B}}_{r_z}^\top
\tilde{\boldsymbol{B}}_{r_z} dA` with `\gamma_{r_z} = A_{66}`, a modulus and not a
user parameter. The curvature and transverse shear operators are untouched,
the edge modes acting only on the in-plane translations, so the bending and
the transverse shear response of the element are identical for the two
drilling models.

Two quadrature choices matter. Because the derivatives of the bubbles are
linear in the area coordinates, the drilling columns of `\tilde{\boldsymbol{B}}_m`
vary linearly, and the terms quadratic in them are integrated with the
three-point rule of degree of precision 2, first row of Table I in Cowper
[10]_, the same one the element already used for its drilling terms. A
single point at the centroid would leave those terms
unsampled and would give the element five zero eigenvalues over its nine
in-plane degrees-of-freedom instead of three. The Hughes-Brezzi term itself
is integrated with a single point at the centroid, following Ibrahimbegovic
et al. (1990) [7]_, which is what makes the element insensitive to
`\gamma_{r_z}`.

Unlike the penalty below, the added term is consistent rather than
artificial: stationarity with respect to `r_z` gives `r_z = \theta_z`
pointwise, so it contributes no energy at the exact solution for any positive
`\gamma_{r_z}`, and the nodal moments about the shell normal recovered in the
internal force vector are physical. Note that the transverse shear
stabilisation documented above scales the transverse shear stiffness only
and does not interact with the drilling term, which draws its scale from
`A_{66}` of the extensional stiffness matrix.

**Fictitious penalty** (``drilling_model = 1``), the default before version
0.10.0, following the approach adopted in MSC Nastran and Autodesk Nastran
through their ``K6ROT`` parameter. It provides a small artificial stiffness
whose only purpose is to remove the singularity, so the forces associated
with it are spurious and any moment recovered about the shell normal is
meaningless. The penalty energy is defined per element as:

.. math::

    U_{r_z} = \frac{1}{2} K6ROT \cdot 10^{-6} \cdot \int_A A_{66} (r_z - \theta_z)^2 dA

where `10^{-6}` is a scaling factor suggested by MSC Nastran's approach
(CQUAD4) to make the artificial drilling stiffness sufficiently small.
AUTODESK NASTRAN's quick reference guide recommends `K6ROT = 100` for static
analysis. For modal solutions, `K6ROT = 10^4` is suggested. MSC NASTRAN's
quick reference guide states that `K6ROT > 100` should not be used, thus
contradicting AUTODESK NASTRAN. The rotation `r_z` represents the drilling
degree-of-freedom in element's coordinates, whereas `\theta_z` the in-plane
rotation strain, defined as `\theta_z = \frac{1}{2}\left(v_{,x} -
u_{,y}\right)`, such that the penalty is built from the operator

.. math::

    B_{r_z} = S^{r_z} + 1/2 S^u_{,y} - 1/2 S^v_{,x}

which is the same operator as `\tilde{\boldsymbol{B}}_{r_z}` above, evaluated on the
unenriched field. Being built from that operator and not from an addition on
the diagonal terms is what keeps the penalty from stiffening a rigid rotation
of the element about its normal, so both models represent all rigid-body
motions and all constant-strain states exactly. `A_{66}` is assumed constant
over the element. This term is integrated with the three-point rule. The
approach herein presented is very similar to the one presented in Eq. 2.20 of
[6]_.

**Choosing between them.** The physics-based model is the default because it
is the one that is correct when the drilling moment is part of the load path,
when shells are connected to beams or stiffeners that must transmit in-plane
moments, or when the mesh is too coarse for the unenriched membrane response
to be trusted. It is markedly more accurate in in-plane bending: on Cook's
skew membrane with a four by four mesh of split quadrilaterals it gives 20.7
against the reference 23.9, where the penalty gives 11.3. The penalty remains
available for reproducing results obtained before 0.10.0.

References
----------
.. [1] Lyly, M., Stenberg, R., and Vihinen, T., A stable bilinear element for
   the Reissner-Mindlin plate model, Computer Methods in Applied Mechanics
   and Engineering, 110(3-4), 343-357, 1993.
   https://doi.org/10.1016/0045-7825(93)90214-I
.. [2] Bischoff, M., and Bletzinger, K.-U., Improving stability and accuracy
   of Reissner-Mindlin plate finite elements via algebraic subgrid scale
   stabilization, Computer Methods in Applied Mechanics and Engineering,
   193(15-16), 1517-1528, 2004. https://doi.org/10.1016/j.cma.2003.12.036
.. [3] Castro, S. G. P., Donadon, M. V., and Guimaraes, T. A. M., ES-PIM
   applied to buckling of variable angle tow laminates, Composite
   Structures, 209, 67-78, 2019.
   https://doi.org/10.1016/j.compstruct.2018.10.058
.. [4] Allman, D. J., A compatible triangular element including vertex
   rotations for plane elasticity analysis, Computers & Structures,
   19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
.. [5] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
   Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
   1989. https://doi.org/10.1016/0045-7825(89)90124-2
.. [6] Adam, F. M., Mohamed, A. E., and Hassaballa, A. E., Degenerated Four
   Nodes Shell Element with Drilling Degree of Freedom, IOSR Journal of
   Engineering, 3(8), 10-20, 2013.
.. [7] Ibrahimbegovic, A., Taylor, R. L., and Wilson, E. L., A robust
   quadrilateral membrane finite element with drilling degrees of freedom,
   International Journal for Numerical Methods in Engineering, 30(3),
   445-457, 1990. https://doi.org/10.1002/nme.1620300305
.. [8] Bletzinger, K.-U., Bischoff, M., and Ramm, E., A unified approach for
   shear-locking-free triangular and rectangular shell finite elements,
   Computers & Structures, 75(3), 321-334, 2000.
   https://doi.org/10.1016/S0045-7949(99)00140-6
.. [9] Rohwer, K., Improved transverse shear stiffness for layered finite
   elements, DFVLR-FB 88-32, 1988.
.. [10] Cowper, G. R., Gaussian quadrature formulas for triangles,
   International Journal for Numerical Methods in Engineering, 7(3),
   405-408, 1973. https://doi.org/10.1002/nme.1620070316

"""
from libc.math cimport fabs

import numpy as np

from .shellprop cimport ShellProp

cdef int DOF = 6
cdef int NUM_NODES = 3


cdef void allman_enrichment(double *Nx, double *Ny, double *xe,
                            double S1, double S2, double S3,
                            double *d) noexcept nogil:
    r"""Cartesian derivatives of the Allman drilling enrichment, triangle

    Fills ``d`` with the contribution of the hierarchical quadratic edge
    modes of the Allman enrichment to the Cartesian derivatives of the
    enriched in-plane displacement rows, evaluated at the drilling column of
    each node, exactly as :func:`pyfe3d.quad4.allman_enrichment` does for
    the quadrilateral. The edges are numbered 1-2, 2-3 and 3-1, consistently
    with the nodal connectivity of :class:`.Tria3R`. Called by
    :func:`.allman_delta_Ke` and by the methods of :class:`.Tria3R` that
    use the enriched membrane rows when ``drilling_model = 0``.

    Parameters
    ----------
    Nx, Ny : double pointer
        The three constant Cartesian derivatives of the area coordinates.
    xe : double pointer
        Nodal coordinates in the element coordinate system.
    S1, S2, S3 : double
        Area coordinates of the evaluation point.
    d : double pointer
        Buffer of 12 positions, filled in place with `\tilde S^u_{,x}`,
        `\tilde S^u_{,y}`, `\tilde S^v_{,x}` and `\tilde S^v_{,y}`, each one
        for the three nodes, so ``d[0:3]`` holds `\tilde S^u_{,x}` at the
        drilling column of nodes 1 to 3.

    Notes
    -----
    The edge modes are those of Allman (1984) [1]_, see the module
    documentation. With the edge `k` joining nodes `i` and `j`, the enriched
    in-plane displacement rows are

    .. math::
        \tilde{\boldsymbol{S}}^u = \boldsymbol{S}^u + \frac{1}{8} \sum_k \hat{S}_k (y_i - y_j)
                           (\boldsymbol{e}_i - \boldsymbol{e}_j)
        \
        \tilde{\boldsymbol{S}}^v = \boldsymbol{S}^v + \frac{1}{8} \sum_k \hat{S}_k (x_j - x_i)
                           (\boldsymbol{e}_i - \boldsymbol{e}_j)

    where `\boldsymbol{e}_i` selects `{r_z}_i`, as for
    :func:`pyfe3d.quad4.allman_enrichment`. For the triangle the
    hierarchical bubble of the edge joining nodes `i` and `j` is `\hat{S}_k
    = 4 S_i S_j`, with `S_i` the area coordinates, so that

    .. math::
        \hat{S}_{k,x} = 4 \left( S_{i,x} S_j + S_i S_{j,x} \right)

    and `S_{i,x}` is the constant ``Nx[i]``. The derivatives of the bubbles
    are linear in the area coordinates, so the values filled in ``d`` vary
    linearly over the element.

    References
    ----------
    .. [1] Allman, D. J., A compatible triangular element including vertex
       rotations for plane elasticity analysis, Computers & Structures,
       19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4

    """
    cdef int k
    cdef double S[3]
    cdef double cu[3]
    cdef double cv[3]
    cdef double Nbx[3]
    cdef double Nby[3]
    cdef int ei[3]
    cdef int ej[3]

    S[0] = S1
    S[1] = S2
    S[2] = S3

    # edges 1-2, 2-3, 3-1
    ei[0] = 0
    ej[0] = 1
    ei[1] = 1
    ej[1] = 2
    ei[2] = 2
    ej[2] = 0

    for k in range(3):
        # NOTE (l_k/8)*n_k = (1/8)*{y_i - y_j, x_j - x_i}
        cu[k] = 0.125*(xe[3*ei[k] + 1] - xe[3*ej[k] + 1])
        cv[k] = 0.125*(xe[3*ej[k] + 0] - xe[3*ei[k] + 0])
        Nbx[k] = 4.*(Nx[ei[k]]*S[ej[k]] + S[ei[k]]*Nx[ej[k]])
        Nby[k] = 4.*(Ny[ei[k]]*S[ej[k]] + S[ei[k]]*Ny[ej[k]])

    # NOTE node m is the first vertex of edge m and the second vertex of the
    #      previous edge, hence the two contributions with opposite signs
    d[0] = cu[0]*Nbx[0] - cu[2]*Nbx[2]
    d[1] = cu[1]*Nbx[1] - cu[0]*Nbx[0]
    d[2] = cu[2]*Nbx[2] - cu[1]*Nbx[1]

    d[3] = cu[0]*Nby[0] - cu[2]*Nby[2]
    d[4] = cu[1]*Nby[1] - cu[0]*Nby[0]
    d[5] = cu[2]*Nby[2] - cu[1]*Nby[1]

    d[6] = cv[0]*Nbx[0] - cv[2]*Nbx[2]
    d[7] = cv[1]*Nbx[1] - cv[0]*Nbx[0]
    d[8] = cv[2]*Nbx[2] - cv[1]*Nbx[1]

    d[9] = cv[0]*Nby[0] - cv[2]*Nby[2]
    d[10] = cv[1]*Nby[1] - cv[0]*Nby[0]
    d[11] = cv[2]*Nby[2] - cv[1]*Nby[1]


cdef void allman_delta_Ke(double *Nx, double *Ny, double *xe, double area,
                          double *Ae, double *Be, double gamma,
                          double *Ke) noexcept nogil:
    r"""The drilling enrichment contribution to the element stiffness matrix

    Fills the 18 by 18 buffer ``Ke``, stored row by row, with everything that
    the drilling enrichment and the term tying `r_z` to the in-plane rotation
    add to the element matrix of :class:`.Tria3R` in element coordinates,
    and with nothing else. It is used when ``drilling_model = 0``, by
    :meth:`.Tria3R.update_KC0`, which rotates ``Ke`` to global coordinates,
    and by :meth:`.Tria3R.update_probe_finte`, which adds ``Ke`` times the
    element displacements to the internal forces. ``Nx`` and ``Ny`` are the
    constant Cartesian derivatives of the area coordinates, ``xe`` the nodal
    coordinates in the element coordinate system, ``area`` the element area,
    ``Ae`` and ``Be`` the 3 by 3 extensional and extension-bending coupling
    matrices in the element coordinate system stored row by row, and
    ``gamma`` the coefficient of the drilling term, with dimension of force
    per unit length.

    Notes
    -----
    The enrichment is the one of Allman [1]_ and the drilling term the one
    of Hughes and Brezzi [2]_, see the module documentation.

    Writing the enriched membrane operator as `\tilde{\boldsymbol{B}}_m = \boldsymbol{B}_m +
    \boldsymbol{\Delta}_m`, where `\boldsymbol{\Delta}_m` is non-zero only in the drilling
    columns, the membrane and coupling contributions expand as

    .. math::
        \tilde{\boldsymbol{B}}_m^\top \boldsymbol{A} \tilde{\boldsymbol{B}}_m
        + \tilde{\boldsymbol{B}}_m^\top \boldsymbol{B} \boldsymbol{B}_b
        + \boldsymbol{B}_b^\top \boldsymbol{B} \tilde{\boldsymbol{B}}_m
        = (\text{unenriched}) + \boldsymbol{B}_m^\top \boldsymbol{A} \boldsymbol{\Delta}_m
        + \boldsymbol{\Delta}_m^\top \boldsymbol{A} \boldsymbol{B}_m
        + \boldsymbol{\Delta}_m^\top \boldsymbol{A} \boldsymbol{\Delta}_m
        + \boldsymbol{\Delta}_m^\top \boldsymbol{B} \boldsymbol{B}_b
        + \boldsymbol{B}_b^\top \boldsymbol{B} \boldsymbol{\Delta}_m

    and only the five terms in `\boldsymbol{\Delta}_m` are assembled here, so the
    block of the translations and of the bending rotations is left exactly as
    the unenriched element computed it. Those terms are integrated with the
    three-point rule, which is needed because `\boldsymbol{\Delta}_m` varies
    linearly over the element, so that the term quadratic in it would
    otherwise be unsampled and would leave a zero-energy mode. The
    Hughes-Brezzi term is integrated with a single point at the centroid, as
    in :mod:`pyfe3d.quad4`.

    The three-point rule has the area coordinates `(2/3, 1/6, 1/6)` and their
    permutations, each with the weight ``area/3``, with `\boldsymbol{\Delta}_m`
    given by :func:`.allman_enrichment` at each point,

    .. math::
        \boldsymbol{\Delta}_m = \begin{bmatrix} \tilde S^u_{,x} \ \tilde S^v_{,y}
        \ \tilde S^u_{,y} + \tilde S^v_{,x} \end{bmatrix}

    in the drilling columns. The Hughes-Brezzi term is

    .. math::
        \gamma A_e \tilde{\boldsymbol{B}}_{r_z}^\top \tilde{\boldsymbol{B}}_{r_z}, \qquad
        \tilde{\boldsymbol{B}}_{r_z} = \boldsymbol{S}^{r_z} + \frac{1}{2}\boldsymbol{\tilde
        S}^u_{,y} - \frac{1}{2}\tilde{\boldsymbol{S}}^v_{,x}

    evaluated at the centroid, where `\boldsymbol{S}^{r_z} = 1/3` at each node,
    with `A_e` given by ``area``.

    References
    ----------
    .. [1] Allman, D. J., A compatible triangular element including vertex
       rotations for plane elasticity analysis, Computers & Structures,
       19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
    .. [2] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
       Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
       1989. https://doi.org/10.1016/0045-7825(89)90124-2

    """
    cdef int i, j, a, n
    cdef double S1, S2, S3, wij
    cdef double d[12]
    cdef double Bm[54]
    cdef double Bb[54]
    cdef double Dm[54]
    cdef double bdrill[18]
    cdef double ABm[54]
    cdef double ADm[54]
    cdef double BBb[54]
    cdef double BDm[54]
    cdef double points[3]

    for i in range(18*18):
        Ke[i] = 0.

    # NOTE the unenriched membrane and bending rows, which are constant
    for i in range(54):
        Bm[i] = 0.
        Bb[i] = 0.
    for i in range(3):
        Bm[0*18 + 6*i + 0] = Nx[i]          # exx = u,x
        Bm[1*18 + 6*i + 1] = Ny[i]          # eyy = v,y
        Bm[2*18 + 6*i + 0] = Ny[i]          # gxy = u,y + v,x
        Bm[2*18 + 6*i + 1] = Nx[i]
        Bb[0*18 + 6*i + 4] = Nx[i]          # kxx = ry,x
        Bb[1*18 + 6*i + 3] = -Ny[i]         # kyy = -rx,y
        Bb[2*18 + 6*i + 3] = -Nx[i]         # kxy = ry,y - rx,x
        Bb[2*18 + 6*i + 4] = Ny[i]

    # NOTE three-point rule of degree of precision 2, first row of Table I
    #      in Cowper (1973), the same one already used for the drilling
    #      terms of the unenriched element
    points[0] = 0.66666666666666666666666666666666666666666667
    points[1] = 0.16666666666666666666666666666666666666666667
    points[2] = 0.16666666666666666666666666666666666666666667
    wij = area/3.

    for n in range(3):
        if n == 0:
            S1 = points[0]
            S2 = points[1]
            S3 = points[2]
        elif n == 1:
            S1 = points[1]
            S2 = points[0]
            S3 = points[2]
        else:
            S1 = points[1]
            S2 = points[2]
            S3 = points[0]

        allman_enrichment(Nx, Ny, xe, S1, S2, S3, d)

        for i in range(54):
            Dm[i] = 0.
        for i in range(3):
            Dm[0*18 + 6*i + 5] = d[i]               # exx gains S~u,x
            Dm[1*18 + 6*i + 5] = d[9 + i]           # eyy gains S~v,y
            Dm[2*18 + 6*i + 5] = d[3 + i] + d[6 + i]  # gxy gains both

        # products with the constitutive matrices, stored row by row
        for a in range(3):
            for j in range(18):
                ABm[18*a + j] = (Ae[3*a + 0]*Bm[0*18 + j]
                               + Ae[3*a + 1]*Bm[1*18 + j]
                               + Ae[3*a + 2]*Bm[2*18 + j])
                ADm[18*a + j] = (Ae[3*a + 0]*Dm[0*18 + j]
                               + Ae[3*a + 1]*Dm[1*18 + j]
                               + Ae[3*a + 2]*Dm[2*18 + j])
                BBb[18*a + j] = (Be[3*a + 0]*Bb[0*18 + j]
                               + Be[3*a + 1]*Bb[1*18 + j]
                               + Be[3*a + 2]*Bb[2*18 + j])
                BDm[18*a + j] = (Be[3*a + 0]*Dm[0*18 + j]
                               + Be[3*a + 1]*Dm[1*18 + j]
                               + Be[3*a + 2]*Dm[2*18 + j])

        for i in range(18):
            for j in range(18):
                for a in range(3):
                    Ke[18*i + j] += wij*(
                        # Bm.T*A*Dm + Dm.T*A*Bm + Dm.T*A*Dm
                          Bm[18*a + i]*ADm[18*a + j]
                        + Dm[18*a + i]*ABm[18*a + j]
                        + Dm[18*a + i]*ADm[18*a + j]
                        # Dm.T*B*Bb + Bb.T*B*Dm
                        + Dm[18*a + i]*BBb[18*a + j]
                        + Bb[18*a + i]*BDm[18*a + j]
                    )

    # NOTE Hughes-Brezzi term, one point at the centroid
    allman_enrichment(Nx, Ny, xe, 1./3., 1./3., 1./3., d)
    for i in range(3):
        bdrill[6*i + 0] = Ny[i]/2.
        bdrill[6*i + 1] = -Nx[i]/2.
        bdrill[6*i + 2] = 0.
        bdrill[6*i + 3] = 0.
        bdrill[6*i + 4] = 0.
        bdrill[6*i + 5] = 1./3. + 0.5*d[3 + i] - 0.5*d[6 + i]
    for i in range(18):
        for j in range(18):
            Ke[18*i + j] += area*gamma*bdrill[i]*bdrill[j]


cdef class Tria3RData:
    r"""
    Used to allocate memory for the sparse matrices.

    Each attribute is the number of entries that one :class:`.Tria3R` writes
    in the row, column and value arrays of the corresponding sparse matrix,
    starting at the position given by the matching ``init_k_*`` attribute of
    the element. For ``n`` elements the arrays need ``n`` times these sizes.
    The constructor takes no parameters.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 324``

    KCNL_SPARSE_SIZE, : int
        ``KCNL_SPARSE_SIZE = 324``

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 81``

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 270``

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KCNL_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE

    def __cinit__(Tria3RData self):
        self.KC0_SPARSE_SIZE = 324
        self.KCNL_SPARSE_SIZE = 324
        self.KG_SPARSE_SIZE = 81
        self.M_SPARSE_SIZE = 270


cdef class Tria3RProbe:
    r"""
    Probe used for local coordinates, local displacements, local stresses etc

    The idea behind using a probe is to avoid allocating larger memory buffers
    per finite element. The memory buffers are allocated per probe, and one
    probe can be shared amongst many finite elements, with the information
    being updated and retrieved on demand.

    .. note:: The probe can be shared amongst more than one finite element, 
              depending on how you defined them. Mind that the probe will
              always safe the values from the last udpate.


    The constructor takes no parameters and allocates every array filled with
    zeros.

    Attributes
    ----------
    xe, : array-like
        Array of size ``NUM_NODES*DOF//2=9`` containing the nodal coordinates
        in the element coordinate system, in the following order
        `{x_e}_1, {y_e}_1, {z_e}_1, `{x_e}_2, {y_e}_2, {z_e}_2`, `{x_e}_3,
        {y_e}_3, {z_e}_3`.
        Dimension of length. Updated by :meth:`.Tria3R.update_probe_xe`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=18`` containing the element displacements
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1`, `{u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`, `{u_e}_3, {v_e}_3, {w_e}_3, {{r_x}_e}_3,
        {{r_y}_e}_3, {{r_z}_e}_3`.
        Translations with dimension of length, rotations dimensionless.
        Updated by :meth:`.Tria3R.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=18`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``.
        Forces, and moments (force times length) for the rotations. Updated
        by :meth:`.Tria3R.update_probe_finte`.
    BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy : array-like
        Arrays of size ``NUM_NODES*DOF=18`` with the rows of the linear
        strain-displacement matrix for the membrane strains and curvatures,
        at the last evaluated integration point.
        Their products with ``ue`` give the membrane strains, dimensionless,
        and the curvatures, with dimension of one over length.
    Sw_x, Sw_y : array-like
        Arrays of size ``NUM_NODES*DOF=18`` with the rows giving `w_{,x}` and
        `w_{,y}`, at the last evaluated integration point.
        ``Sw_x`` holds `\boldsymbol{G}_x = \partial \boldsymbol{S}^w/\partial x`
        and ``Sw_y`` holds `\boldsymbol{G}_y = \partial \boldsymbol{S}^w/\partial
        y`, the underscore denoting the partial derivative.
        Their products with ``ue`` are dimensionless.
    KCNLve : array-like
        Array of size ``(NUM_NODES*DOF)**2=324`` with the nonlinear
        constitutive stiffness matrix KCNL in element coordinates, stored row
        by row.
        Updated by :meth:`.Tria3R.update_KCNL`.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    cdef public double [::1] BLexx
    cdef public double [::1] BLeyy
    cdef public double [::1] BLgxy
    cdef public double [::1] BLkxx
    cdef public double [::1] BLkyy
    cdef public double [::1] BLkxy
    cdef public double [::1] Sw_x
    cdef public double [::1] Sw_y
    cdef public double [::1] KCNLve

    def __cinit__(Tria3RProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLexx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLeyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_x = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_y = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KCNLve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)


cdef class Tria3R:
    r"""
    Three-node triangular shell element with reduced integration

    Nodal connectivity for the triangular element similar to Nastran's CTRIA3::

        3
        |\
        | \    positive normal in CCW
        |  \
        |___\
        1    2

    The element coordinate system is determined identically what is explained
    in Nastran's quick reference guide for the CTRIA3 element, as illustrated
    below.

    .. image:: ../figures/nastran_ctria3.svg

    The constructor stores the probe and sets the identification numbers and
    the positions ``c1, c2, c3`` to ``-1``, the ``init_k_*`` positions to
    ``0``, the rotation matrix ``r11, ..., r33`` to zero and ``m11, m12, m21,
    m22`` to the identity. Before computing any element matrix, the caller
    sets ``c1, c2, c3`` and the ``init_k_*`` positions, and calls
    :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe`.

    Parameters
    ----------
    p : :class:`.Tria3RProbe`
        Probe used by the element, stored in ``probe``. One probe can be
        shared by many elements.

    Attributes
    ----------
    eid, : int
        Element identification number.
    pid, : int
        Property identification number.
    area, : double
        Element area.
        Dimension of length squared. Updated by :meth:`.update_area`.
    alpha_shear_locking, : double
        Factor used to prevent shear locking, affecting the transverse shear
        stiffness terms ``A44``, ``A45``, ``A55``, already in the element
        coordinate system and with the shear correction applied (see
        :meth:`.ShellProp.calc_Ats_element`), which are reduced by a factor
        that grows with the square of the ratio of the longest edge to the
        thickness. ``0.`` disables the reduction. The adopted default is
        ``alpha_shear_locking = 0.7``, chosen such that the linear buckling
        load of a simply supported plate approaches the one of the
        :class:`.Quad4R` element for an equivalent mesh (see the test case
        ``test_tria3r_linear_buckling_plate.py``). No single value serves
        every problem class, so this is a parameter to be verified per
        problem and not a constant of the element; :class:`.Tria3DSG` takes
        no such parameter.
        Dimensionless.
    drilling_model, : int
        Selects how the drilling degree-of-freedom `r_z` is given stiffness.
        The default ``0`` is the physics-based stiffness, for which the
        recovered nodal moments about the shell normal are physical, and no
        user parameter is involved. Any other value selects the fictitious
        penalty of MSC Nastran and Autodesk Nastran, which was the default up
        to version 0.9.0 and is controlled by ``K6ROT``. Setting
        ``elem.drilling_model = 1`` before calling :meth:`.update_KC0` is the
        way to reproduce results obtained before 0.10.0.
    K6ROT, : double
        Dimensionless multiplier for the fictitious drilling stiffness, only
        read when ``drilling_model`` is not ``0``. It has no effect under the
        default physics-based model, which takes its coefficient from the
        laminate stiffness instead, see ``gamma_rz``. Default ``100.``.
    gamma_rz, : double
        Coefficient `\gamma_{r_z}` of the physics-based drilling stiffness,
        only read when ``drilling_model`` is ``0``. The default is a negative
        value, which means that ``A66`` of the laminate extensional stiffness
        matrix is used. The attribute is exposed for sensitivity studies
        rather than for normal use.
        Dimension of force per unit length, as ``A66``.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}` from
        element to global coordinates, whose columns are the element axes.
    m11, m12, m21, m22 : double
        Rotation matrix only for the constitutive relations. Used when a
        material direction is used instead of the element local coordinates.
    c1, c2, c3 : int
        Position of each node in the global stiffness matrix.
        The coordinates of a node are read at the positions ``c1//2``,
        ``c2//2`` and ``c3//2`` of the coordinate array.
    n1, n2, n3 : int
        Node identification number.
    init_k_KC0, init_k_KCNL, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.
    probe, : :class:`.Tria3RProbe` object
        Pointer to the probe.

    Notes
    -----
    The factor used to prevent shear locking, ``alpha_shear_locking``, is
    adopted from the stabilised assumed-strain and discrete shear gap (DSG)
    elements: it is the factor of Lyly, Stenberg and Vihinen, called
    Stenberg's method by Bischoff and Bletzinger [1]_ (Sec. 3.1), in the
    form of Castro et al. [2]_ (Eq. (26)), see the module documentation,
    and scales the transverse shear stiffnesses as::

        maxl = max(edge_12, edge_23, edge_31)
        factor = alpha_shear_locking*maxl**2/thickness**2
        A44 = 1 / (1 + factor) * A44
        A45 = 1 / (1 + factor) * A45
        A55 = 1 / (1 + factor) * A55

    .. warning:: `\alpha = 0.7` is about seven times the value used
        in the references this scheme comes from, and it is the
        dominant error term on problems where transverse shear carries
        load. The reason, and the measurements, are in the Notes of the
        module documentation.

        No single value serves every problem class, so this is a
        parameter to be verified per problem and not a constant of the
        element. Two measurements bracket it, and they pull in opposite
        directions:

        - on the plate of ``tests/test_tria3r_natural_freq.py`` with
          the consistent mass matrix, `\alpha = 0.7` gives a first
          natural frequency 0.6 per cent above the analytical value
          while the literature `\alpha = 0.1` gives one 30.6 per cent
          above it, so here the default is much the better of the two.
          With the stabilisation off the error is 96.1 per cent, which
          is how much of this element's accuracy rests on `\alpha`;

        - on the cylinder of
          ``tests/test_quad4_linear_buckling_cylinder_displ.py`` at
          ``ntheta = 60``, where `factor` reaches 1852, the ordering
          reverses. At `\alpha = 0.7` the critical eigenvalue belongs
          to a mode at two elements per wavelength, the mesh Nyquist
          limit, carrying 36.6 per cent of its energy in transverse
          shear: a numerical mechanism rather than a physical mode. At
          `\alpha = 0.1` the physical eight-wave mode is critical
          instead, and there this element agrees with the
          quadrilateral and with the discrete shear gap triangle to
          3 per cent, 2.520 against 2.453 and 2.534 times the reference
          load, all three overpredicting at so coarse a mesh.

        So on a thin shell in buckling the default is not merely
        inaccurate, it can change which mode is critical, and a
        plausible-looking eigenvalue can belong to a mesh artefact.
        Where the answer matters, either sweep `\alpha` and confirm
        that the critical mode is physical and resolved by several
        elements per wavelength, or use :class:`.Tria3DSG`, whose
        discrete shear gap transverse shear field is locking-free and
        takes no such parameter.

    Under the default ``drilling_model = 0``, the physics-based stiffness is
    the one of Allman (1984) [3]_ and Hughes and Brezzi (1989) [4]_, for
    which the drilling rotation is a kinematic variable that carries strain
    energy, see the module documentation.

    For ``K6ROT``, AUTODESK NASTRAN's quick reference guide recommends
    ``K6ROT = 100.`` for static analysis. For modal solutions,
    ``K6ROT=1.e4`` is suggested. MSC NASTRAN's quick reference guide states
    that ``K6ROT > 100.`` should not be used, but this is contradicting
    AUTODESK NASTRAN.

    The default of ``gamma_rz``, `\gamma_{r_z} = A_{66}`, is the value
    identified by Hughes and Brezzi (1989) [4]_. This is a modulus and not a
    parameter that needs tuning: the element response has a broad plateau of
    insensitivity around it, and the attribute is exposed for the sensitivity
    study that the literature recommends rather than for normal use. Very
    large values over-constrain `r_z = \theta_z`, and a zero value leaves the
    Allman enrichment rank-deficient by one.

    References
    ----------
    .. [1] Bischoff, M., and Bletzinger, K.-U., Improving stability and
       accuracy of Reissner-Mindlin plate finite elements via algebraic
       subgrid scale stabilization, Computer Methods in Applied Mechanics and
       Engineering, 193(15-16), 1517-1528, 2004.
       https://doi.org/10.1016/j.cma.2003.12.036
    .. [2] Castro, S. G. P., Donadon, M. V., and Guimaraes, T. A. M., ES-PIM
       applied to buckling of variable angle tow laminates, Composite
       Structures, 209, 67-78, 2019.
       https://doi.org/10.1016/j.compstruct.2018.10.058
    .. [3] Allman, D. J., A compatible triangular element including vertex
       rotations for plane elasticity analysis, Computers & Structures,
       19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
    .. [4] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
       Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
       1989. https://doi.org/10.1016/0045-7825(89)90124-2

    """
    cdef public int eid, pid
    cdef public int n1, n2, n3
    cdef public int c1, c2, c3
    cdef public int init_k_KC0, init_k_KCNL, init_k_KG, init_k_M
    cdef public double area
    cdef public double K6ROT
    cdef public double alpha_shear_locking
    cdef public int drilling_model
    cdef public double gamma_rz
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public double m11, m12, m21, m22
    cdef public Tria3RProbe probe

    def __cinit__(Tria3R self, Tria3RProbe p):
        self.probe = p
        self.eid = -1
        self.pid = -1
        self.n1 = -1
        self.n2 = -1
        self.n3 = -1
        self.c1 = -1
        self.c2 = -1
        self.c3 = -1
        self.init_k_KC0 = 0
        self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.area = 0
        self.K6ROT = 100. # NOTE default value in MSC Nastran
        self.alpha_shear_locking = 0.7
        self.drilling_model = 0 # NOTE Allman + Hughes-Brezzi, the default
        self.gamma_rz = -1. # NOTE negative means "use A66"
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.
        self.m11 = 1.
        self.m12 = 0.
        self.m21 = 0.
        self.m22 = 1.


    cpdef void update_rotation_matrix(Tria3R self, double [::1] x,
            double xmati=0., double xmatj=0., double xmatk=0.):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated, with
        the entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}`
        from element to global coordinates, whose columns are the element axes.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The element ``x`` axis runs from node 1 to node 2 and the element
        ``z`` axis is along the cross product of the vectors from node 1 to
        node 2 and from node 1 to node 3. When a material direction is given,
        ``m11, m12, m21, m22`` are updated with the rotation from the material
        direction projected onto the element to the element ``x`` axis. When
        it is the zero vector, the default, or parallel to the element normal,
        ``m11, m12, m21, m22`` are not changed.


        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.
            1-D array of size ``3*M``, dimension of length. The coordinates
            of each node are read from the positions ``c1//2``, ``c2//2``
            and ``c3//2``.

        xmati, xmatj, xmatk : double, optional
            Vector in global coordinates representing the material direction.
            This vector is projected onto the plate element, thus becoming the
            material direction. The `ABD` matrix defining the constitutive
            behavior of the element is rotated from the material direction to
            the element `x` axis while calculating the stiffness matrices.
            Only the direction of the vector is used. Default ``0.``, the
            zero vector, which leaves ``m11, m12, m21, m22`` unchanged.

        """
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk
        cdef double x1i, x1j, x1k, x2i, x2j, x2k, x3i, x3j, x3k
        cdef double v12i, v12j, v12k, v13i, v13j, v13k
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

            v12i = x2i - x1i
            v12j = x2j - x1j
            v12k = x2k - x1k
            v13i = x3i - x1i
            v13j = x3j - x1j
            v13k = x3k - x1k

            zi = v12j*v13k - v12k*v13j
            zj = v12k*v13i - v12i*v13k
            zk = v12i*v13j - v12j*v13i
            tmp = (zi**2 + zj**2 + zk**2)**0.5
            zi /= tmp
            zj /= tmp
            zk /= tmp
            # NOTE defining tolerance to be 1/1e10 of normal vector norm
            tol = tmp/1e10

            xi = v12i
            xj = v12j
            xk = v12k
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


    cpdef void update_probe_ue(Tria3R self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        The displacements and rotations of the three nodes, read from ``u`` at
        the positions ``c1``, ``c2`` and ``c3``, are rotated to the element
        coordinate system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the
        transformation matrix and written to ``probe.ue``.
        :meth:`.update_rotation_matrix` must have been called before.

        .. note:: The ``ue`` attribute of object :class:`.Tria3RProbe` is
                  updated, accessible using ``.probe.ue``.

        Parameters
        ----------
        u : array-like
            Array with global displacements, for a total of `M` nodes in
            the model, this array will be arranged as: `u_1, v_1, w_1, {r_x}_1,
            {r_y}_1, {r_z}_1, u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2, ...,
            u_M, v_M, w_M, {r_x}_M, {r_y}_M, {r_z}_M`.
            1-D array of size ``DOF*M``, translations with dimension of
            length and rotations dimensionless.

        """
        cdef int i, j
        cdef int c[3]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            # global to local transformation of displacements (T.T)
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


    cpdef void update_probe_xe(Tria3R self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        The coordinates of the three nodes, read from ``x`` at the positions
        ``c1//2``, ``c2//2`` and ``c3//2``, are rotated to the element
        coordinate system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the
        transformation matrix and written to ``probe.xe``. :meth:`.update_area`
        is then called, updating ``area``. :meth:`.update_rotation_matrix` must
        have been called before.

        .. note:: The ``xe`` attribute of object :class:`.Tria3RProbe` is
                  updated, accessible using ``.probe.xe``.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.
            1-D array of size ``3*M``, dimension of length.

        """
        cdef int i, j
        cdef int c[3]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            # global to local transformation of coordinates (T.T)
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


    cpdef void update_area(Tria3R self):
        r"""Update element area

        The ``area`` attribute is updated with the area of the triangle given
        by the element coordinates ``probe.xe``, ignoring the ``z``
        coordinates. Called by :meth:`.update_probe_xe`.

        """
        cdef double x1, x2, x3, y1, y2, y3
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
            self.area = fabs((-x1 + x2)*(-y1 + y3)/2. + (x1 - x3)*(-y1 + y2)/2.)


    cpdef void update_probe_finte(Tria3R self,
                           ShellProp prop,
                           int nonlinear=0):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` is updated with the :class:`.Tria3RProbe` the
        internal forces in local coordinates. While using this function, mind
        that the probe can be shared amongst more than one finite element,
        depending how you defined them, meaning that the probe will always safe
        the values from the last udpate.

        The internal forces are computed from the probe coordinates ``xe``
        and displacements ``ue``, the ``area`` and the constitutive matrices
        of ``prop`` rotated with ``m11, m12, m21, m22``. Before this method is
        called, :meth:`.update_probe_xe` and :meth:`.update_probe_ue` must
        have been used.

        .. note:: The ``finte`` attribute of object :class:`.Tria3RProbe` is
                  updated, accessible using ``.probe.finte``.

        Parameters
        ----------
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Karman strains,
            see :meth:`._update_probe_finte_nonlinear` and :meth:`.update_KCNL`.

        Notes
        -----
        The linear internal forces are the product of the element stiffness
        matrix of :meth:`.update_KC0`, in element coordinates, with ``ue``,
        computed with the same quadrature: the generated terms with the
        transverse shear stiffnesses scaled according to
        ``alpha_shear_locking``, plus, for ``drilling_model = 0``, the
        contribution of :func:`.allman_delta_Ke`, which keeps ``finte``
        exactly equal to the product of the element matrix with the element
        displacements. With ``nonlinear=1`` the exact Jacobian of the
        internal forces is ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef double *finte
        cdef int i
        cdef double x1, x2, x3, y1, y2, y3
        cdef double wij, detJ
        cdef double Ae[9]
        cdef double Be[9]
        cdef double De[9]
        cdef double Atse[4]
        cdef double A44, A45, A55
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double D11, D12, D16, D22, D26, D66
        cdef double K6ROT
        cdef double points[3]
        cdef double N1x, N2x, N3x, N1y, N2y, N3y
        cdef double N1, N2
        cdef double factor, maxl, l12, l23, l31
        cdef int qi, qj
        cdef double gamma_drill
        cdef double Nxv[3]
        cdef double Nyv[3]
        cdef double Kedelta[324]

        cdef double KC0e0000, KC0e0001, KC0e0003, KC0e0004, KC0e0005, KC0e0006, KC0e0007, KC0e0009, KC0e0010, KC0e0011, KC0e0012, KC0e0013, KC0e0015, KC0e0016, KC0e0017
        cdef double KC0e0101, KC0e0103, KC0e0104, KC0e0105, KC0e0106, KC0e0107, KC0e0109, KC0e0110, KC0e0111, KC0e0112, KC0e0113, KC0e0115, KC0e0116, KC0e0117
        cdef double KC0e0202, KC0e0203, KC0e0204, KC0e0208, KC0e0209, KC0e0210, KC0e0214, KC0e0215, KC0e0216
        cdef double KC0e0303, KC0e0304, KC0e0306, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0312, KC0e0313, KC0e0314, KC0e0315, KC0e0316
        cdef double KC0e0404, KC0e0406, KC0e0407, KC0e0408, KC0e0409, KC0e0410, KC0e0412, KC0e0413, KC0e0414, KC0e0415, KC0e0416
        cdef double KC0e0505, KC0e0506, KC0e0507, KC0e0511, KC0e0512, KC0e0513, KC0e0517
        cdef double KC0e0606, KC0e0607, KC0e0609, KC0e0610, KC0e0611, KC0e0612, KC0e0613, KC0e0615, KC0e0616, KC0e0617
        cdef double KC0e0707, KC0e0709, KC0e0710, KC0e0711, KC0e0712, KC0e0713, KC0e0715, KC0e0716, KC0e0717
        cdef double KC0e0808, KC0e0809, KC0e0810, KC0e0814, KC0e0815, KC0e0816
        cdef double KC0e0909, KC0e0910, KC0e0912, KC0e0913, KC0e0914, KC0e0915, KC0e0916
        cdef double KC0e1010, KC0e1012, KC0e1013, KC0e1014, KC0e1015, KC0e1016
        cdef double KC0e1111, KC0e1112, KC0e1113, KC0e1117
        cdef double KC0e1212, KC0e1213, KC0e1215, KC0e1216, KC0e1217
        cdef double KC0e1313, KC0e1315, KC0e1316, KC0e1317
        cdef double KC0e1414, KC0e1415, KC0e1416
        cdef double KC0e1515, KC0e1516
        cdef double KC0e1616
        cdef double KC0e1717

        with nogil:
            ue = &self.probe.ue[0]
            finte = &self.probe.finte[0]

            detJ = 2*self.area

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

            l12 = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            l23 = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
            l31 = ((x3 - x1)**2 + (y3 - y1)**2)**0.5
            maxl = l12
            if l23 > maxl:
                maxl = l23
            if l31 > maxl:
                maxl = l31

            # NOTE strategy to prevent shear locking used in the stabilised DSG elements imported here...
            factor = self.alpha_shear_locking*maxl**2/prop.h**2
            A44 = 1 / (1 + factor) * A44
            A45 = 1 / (1 + factor) * A45
            A55 = 1 / (1 + factor) * A55

            # NOTE the fictitious penalty is switched off when the
            #      physics-based drilling stiffness is active, so that the
            #      generated block contributes no drilling term at all and
            #      the enrichment supplies everything, see the module
            #      documentation
            if self.drilling_model == 0:
                K6ROT = 0.
            else:
                K6ROT = self.K6ROT

            N1x = (y2 - y3)/(2*self.area)
            N2x = (-y1 + y3)/(2*self.area)
            N3x = (y1 - y2)/(2*self.area)
            N1y = (-x2 + x3)/(2*self.area)
            N2y = (x1 - x3)/(2*self.area)
            N3y = (-x1 + x2)/(2*self.area)

            wij = 0.5
            N1 = N2 = 0.333333333333333333333333333333333333333333333

            KC0e0000 = detJ*wij*(N1x*(A11*N1x + A16*N1y) + N1y*(A16*N1x + A66*N1y))
            KC0e0001 = detJ*wij*(N1x*(A16*N1x + A66*N1y) + N1y*(A12*N1x + A26*N1y))
            KC0e0003 = -detJ*wij*(N1x*(B16*N1x + B66*N1y) + N1y*(B12*N1x + B26*N1y))
            KC0e0004 = detJ*wij*(N1x*(B11*N1x + B16*N1y) + N1y*(B16*N1x + B66*N1y))
            KC0e0006 = detJ*wij*(N2x*(A11*N1x + A16*N1y) + N2y*(A16*N1x + A66*N1y))
            KC0e0007 = detJ*wij*(N2x*(A16*N1x + A66*N1y) + N2y*(A12*N1x + A26*N1y))
            KC0e0009 = -detJ*wij*(N2x*(B16*N1x + B66*N1y) + N2y*(B12*N1x + B26*N1y))
            KC0e0010 = detJ*wij*(N2x*(B11*N1x + B16*N1y) + N2y*(B16*N1x + B66*N1y))
            KC0e0012 = detJ*wij*(N3x*(A11*N1x + A16*N1y) + N3y*(A16*N1x + A66*N1y))
            KC0e0013 = detJ*wij*(N3x*(A16*N1x + A66*N1y) + N3y*(A12*N1x + A26*N1y))
            KC0e0015 = -detJ*wij*(N3x*(B16*N1x + B66*N1y) + N3y*(B12*N1x + B26*N1y))
            KC0e0016 = detJ*wij*(N3x*(B11*N1x + B16*N1y) + N3y*(B16*N1x + B66*N1y))
            KC0e0101 = detJ*wij*(N1x*(A26*N1y + A66*N1x) + N1y*(A22*N1y + A26*N1x))
            KC0e0103 = -detJ*wij*(N1x*(B26*N1y + B66*N1x) + N1y*(B22*N1y + B26*N1x))
            KC0e0104 = detJ*wij*(N1x*(B12*N1y + B16*N1x) + N1y*(B26*N1y + B66*N1x))
            KC0e0106 = detJ*wij*(N2x*(A12*N1y + A16*N1x) + N2y*(A26*N1y + A66*N1x))
            KC0e0107 = detJ*wij*(N2x*(A26*N1y + A66*N1x) + N2y*(A22*N1y + A26*N1x))
            KC0e0109 = -detJ*wij*(N2x*(B26*N1y + B66*N1x) + N2y*(B22*N1y + B26*N1x))
            KC0e0110 = detJ*wij*(N2x*(B12*N1y + B16*N1x) + N2y*(B26*N1y + B66*N1x))
            KC0e0112 = detJ*wij*(N3x*(A12*N1y + A16*N1x) + N3y*(A26*N1y + A66*N1x))
            KC0e0113 = detJ*wij*(N3x*(A26*N1y + A66*N1x) + N3y*(A22*N1y + A26*N1x))
            KC0e0115 = -detJ*wij*(N3x*(B26*N1y + B66*N1x) + N3y*(B22*N1y + B26*N1x))
            KC0e0116 = detJ*wij*(N3x*(B12*N1y + B16*N1x) + N3y*(B26*N1y + B66*N1x))
            KC0e0202 = detJ*wij*(N1x*(A45*N1y + A55*N1x) + N1y*(A44*N1y + A45*N1x))
            KC0e0203 = -N1*detJ*wij*(A44*N1y + A45*N1x)
            KC0e0204 = N1*detJ*wij*(A45*N1y + A55*N1x)
            KC0e0208 = detJ*wij*(N2x*(A45*N1y + A55*N1x) + N2y*(A44*N1y + A45*N1x))
            KC0e0209 = -N2*detJ*wij*(A44*N1y + A45*N1x)
            KC0e0210 = N2*detJ*wij*(A45*N1y + A55*N1x)
            KC0e0214 = detJ*wij*(N3x*(A45*N1y + A55*N1x) + N3y*(A44*N1y + A45*N1x))
            KC0e0215 = detJ*wij*(A44*N1y + A45*N1x)*(N1 + N2 - 1)
            KC0e0216 = -detJ*wij*(A45*N1y + A55*N1x)*(N1 + N2 - 1)
            KC0e0303 = detJ*wij*(D22*N1y**2 + 2*D26*N1x*N1y + D66*N1x**2 + A44*N1**2)
            KC0e0304 = -detJ*wij*(A45*N1**2 + N1x*(D12*N1y + D16*N1x) + N1y*(D26*N1y + D66*N1x))
            KC0e0306 = -detJ*wij*(N2x*(B12*N1y + B16*N1x) + N2y*(B26*N1y + B66*N1x))
            KC0e0307 = -detJ*wij*(N2x*(B26*N1y + B66*N1x) + N2y*(B22*N1y + B26*N1x))
            KC0e0308 = -N1*detJ*wij*(A44*N2y + A45*N2x)
            KC0e0309 = detJ*wij*(A44*N1*N2 + N2x*(D26*N1y + D66*N1x) + N2y*(D22*N1y + D26*N1x))
            KC0e0310 = -detJ*wij*(A45*N1*N2 + N2x*(D12*N1y + D16*N1x) + N2y*(D26*N1y + D66*N1x))
            KC0e0312 = -detJ*wij*(N3x*(B12*N1y + B16*N1x) + N3y*(B26*N1y + B66*N1x))
            KC0e0313 = -detJ*wij*(N3x*(B26*N1y + B66*N1x) + N3y*(B22*N1y + B26*N1x))
            KC0e0314 = -N1*detJ*wij*(A44*N3y + A45*N3x)
            KC0e0315 = detJ*wij*(-A44*N1*(N1 + N2 - 1) + N3x*(D26*N1y + D66*N1x) + N3y*(D22*N1y + D26*N1x))
            KC0e0316 = -detJ*wij*(-A45*N1*(N1 + N2 - 1) + N3x*(D12*N1y + D16*N1x) + N3y*(D26*N1y + D66*N1x))
            KC0e0404 = detJ*wij*(A55*N1**2 + N1x*(D11*N1x + D16*N1y) + N1y*(D16*N1x + D66*N1y))
            KC0e0406 = detJ*wij*(N2x*(B11*N1x + B16*N1y) + N2y*(B16*N1x + B66*N1y))
            KC0e0407 = detJ*wij*(N2x*(B16*N1x + B66*N1y) + N2y*(B12*N1x + B26*N1y))
            KC0e0408 = N1*detJ*wij*(A45*N2y + A55*N2x)
            KC0e0409 = -detJ*wij*(A45*N1*N2 + N2x*(D16*N1x + D66*N1y) + N2y*(D12*N1x + D26*N1y))
            KC0e0410 = detJ*wij*(A55*N1*N2 + N2x*(D11*N1x + D16*N1y) + N2y*(D16*N1x + D66*N1y))
            KC0e0412 = detJ*wij*(N3x*(B11*N1x + B16*N1y) + N3y*(B16*N1x + B66*N1y))
            KC0e0413 = detJ*wij*(N3x*(B16*N1x + B66*N1y) + N3y*(B12*N1x + B26*N1y))
            KC0e0414 = N1*detJ*wij*(A45*N3y + A55*N3x)
            KC0e0415 = -detJ*wij*(-A45*N1*(N1 + N2 - 1) + N3x*(D16*N1x + D66*N1y) + N3y*(D12*N1x + D26*N1y))
            KC0e0416 = detJ*wij*(-A55*N1*(N1 + N2 - 1) + N3x*(D11*N1x + D16*N1y) + N3y*(D16*N1x + D66*N1y))
            KC0e0606 = detJ*wij*(N2x*(A11*N2x + A16*N2y) + N2y*(A16*N2x + A66*N2y))
            KC0e0607 = detJ*wij*(N2x*(A16*N2x + A66*N2y) + N2y*(A12*N2x + A26*N2y))
            KC0e0609 = -detJ*wij*(N2x*(B16*N2x + B66*N2y) + N2y*(B12*N2x + B26*N2y))
            KC0e0610 = detJ*wij*(N2x*(B11*N2x + B16*N2y) + N2y*(B16*N2x + B66*N2y))
            KC0e0612 = detJ*wij*(N3x*(A11*N2x + A16*N2y) + N3y*(A16*N2x + A66*N2y))
            KC0e0613 = detJ*wij*(N3x*(A16*N2x + A66*N2y) + N3y*(A12*N2x + A26*N2y))
            KC0e0615 = -detJ*wij*(N3x*(B16*N2x + B66*N2y) + N3y*(B12*N2x + B26*N2y))
            KC0e0616 = detJ*wij*(N3x*(B11*N2x + B16*N2y) + N3y*(B16*N2x + B66*N2y))
            KC0e0707 = detJ*wij*(N2x*(A26*N2y + A66*N2x) + N2y*(A22*N2y + A26*N2x))
            KC0e0709 = -detJ*wij*(N2x*(B26*N2y + B66*N2x) + N2y*(B22*N2y + B26*N2x))
            KC0e0710 = detJ*wij*(N2x*(B12*N2y + B16*N2x) + N2y*(B26*N2y + B66*N2x))
            KC0e0712 = detJ*wij*(N3x*(A12*N2y + A16*N2x) + N3y*(A26*N2y + A66*N2x))
            KC0e0713 = detJ*wij*(N3x*(A26*N2y + A66*N2x) + N3y*(A22*N2y + A26*N2x))
            KC0e0715 = -detJ*wij*(N3x*(B26*N2y + B66*N2x) + N3y*(B22*N2y + B26*N2x))
            KC0e0716 = detJ*wij*(N3x*(B12*N2y + B16*N2x) + N3y*(B26*N2y + B66*N2x))
            KC0e0808 = detJ*wij*(N2x*(A45*N2y + A55*N2x) + N2y*(A44*N2y + A45*N2x))
            KC0e0809 = -N2*detJ*wij*(A44*N2y + A45*N2x)
            KC0e0810 = N2*detJ*wij*(A45*N2y + A55*N2x)
            KC0e0814 = detJ*wij*(N3x*(A45*N2y + A55*N2x) + N3y*(A44*N2y + A45*N2x))
            KC0e0815 = detJ*wij*(A44*N2y + A45*N2x)*(N1 + N2 - 1)
            KC0e0816 = -detJ*wij*(A45*N2y + A55*N2x)*(N1 + N2 - 1)
            KC0e0909 = detJ*wij*(D22*N2y**2 + 2*D26*N2x*N2y + D66*N2x**2 + A44*N2**2)
            KC0e0910 = -detJ*wij*(A45*N2**2 + N2x*(D12*N2y + D16*N2x) + N2y*(D26*N2y + D66*N2x))
            KC0e0912 = -detJ*wij*(N3x*(B12*N2y + B16*N2x) + N3y*(B26*N2y + B66*N2x))
            KC0e0913 = -detJ*wij*(N3x*(B26*N2y + B66*N2x) + N3y*(B22*N2y + B26*N2x))
            KC0e0914 = -N2*detJ*wij*(A44*N3y + A45*N3x)
            KC0e0915 = detJ*wij*(-A44*N2*(N1 + N2 - 1) + N3x*(D26*N2y + D66*N2x) + N3y*(D22*N2y + D26*N2x))
            KC0e0916 = -detJ*wij*(-A45*N2*(N1 + N2 - 1) + N3x*(D12*N2y + D16*N2x) + N3y*(D26*N2y + D66*N2x))
            KC0e1010 = detJ*wij*(A55*N2**2 + N2x*(D11*N2x + D16*N2y) + N2y*(D16*N2x + D66*N2y))
            KC0e1012 = detJ*wij*(N3x*(B11*N2x + B16*N2y) + N3y*(B16*N2x + B66*N2y))
            KC0e1013 = detJ*wij*(N3x*(B16*N2x + B66*N2y) + N3y*(B12*N2x + B26*N2y))
            KC0e1014 = N2*detJ*wij*(A45*N3y + A55*N3x)
            KC0e1015 = -detJ*wij*(-A45*N2*(N1 + N2 - 1) + N3x*(D16*N2x + D66*N2y) + N3y*(D12*N2x + D26*N2y))
            KC0e1016 = detJ*wij*(-A55*N2*(N1 + N2 - 1) + N3x*(D11*N2x + D16*N2y) + N3y*(D16*N2x + D66*N2y))
            KC0e1212 = detJ*wij*(N3x*(A11*N3x + A16*N3y) + N3y*(A16*N3x + A66*N3y))
            KC0e1213 = detJ*wij*(N3x*(A16*N3x + A66*N3y) + N3y*(A12*N3x + A26*N3y))
            KC0e1215 = -detJ*wij*(N3x*(B16*N3x + B66*N3y) + N3y*(B12*N3x + B26*N3y))
            KC0e1216 = detJ*wij*(N3x*(B11*N3x + B16*N3y) + N3y*(B16*N3x + B66*N3y))
            KC0e1313 = detJ*wij*(N3x*(A26*N3y + A66*N3x) + N3y*(A22*N3y + A26*N3x))
            KC0e1315 = -detJ*wij*(N3x*(B26*N3y + B66*N3x) + N3y*(B22*N3y + B26*N3x))
            KC0e1316 = detJ*wij*(N3x*(B12*N3y + B16*N3x) + N3y*(B26*N3y + B66*N3x))
            KC0e1414 = detJ*wij*(N3x*(A45*N3y + A55*N3x) + N3y*(A44*N3y + A45*N3x))
            KC0e1415 = detJ*wij*(A44*N3y + A45*N3x)*(N1 + N2 - 1)
            KC0e1416 = -detJ*wij*(A45*N3y + A55*N3x)*(N1 + N2 - 1)
            KC0e1515 = detJ*wij*(A44*(N1 + N2 - 1)**2 + N3x*(D26*N3y + D66*N3x) + N3y*(D22*N3y + D26*N3x))
            KC0e1516 = -detJ*wij*(A45*(N1 + N2 - 1)**2 + N3x*(D12*N3y + D16*N3x) + N3y*(D26*N3y + D66*N3x))
            KC0e1616 = detJ*wij*(A55*(N1 + N2 - 1)**2 + N3x*(D11*N3x + D16*N3y) + N3y*(D16*N3x + D66*N3y))

            # NOTE KC0e drilling terms with FULL integration

            KC0e0005 = 0
            KC0e0011 = 0
            KC0e0017 = 0
            KC0e0105 = 0
            KC0e0111 = 0
            KC0e0117 = 0
            KC0e0505 = 0
            KC0e0506 = 0
            KC0e0507 = 0
            KC0e0511 = 0
            KC0e0512 = 0
            KC0e0513 = 0
            KC0e0517 = 0
            KC0e0611 = 0
            KC0e0617 = 0
            KC0e0711 = 0
            KC0e0717 = 0
            KC0e1111 = 0
            KC0e1112 = 0
            KC0e1113 = 0
            KC0e1117 = 0
            KC0e1217 = 0
            KC0e1317 = 0
            KC0e1717 = 0

            # NOTE three-point rule of degree of precision 2, first row of Table I
            #      in Cowper, G. R., Gaussian quadrature formulas for triangles,
            #      IJNME 7(3), 405-408, 1973, doi:10.1002/nme.1620070316: area
            #      coordinates (2/3, 1/6, 1/6) and their permutations, weight 1/6
            #      times detJ = 2*area, i.e. area/3
            wij = 0.5*0.333333333333333333333333333333333333333333333
            points[0] = 0.66666666666666666666666666666666666666666667
            points[1] = 0.16666666666666666666666666666666666666666667
            points[2] = 0.16666666666666666666666666666666666666666667
            for i in range(3):
                if i == 0:
                    N1 = points[0]
                    N2 = points[1]
                elif i == 1:
                    N1 = points[1]
                    N2 = points[2]
                elif i == 2:
                    N1 = points[2]
                    N2 = points[0]

                KC0e0000 += 2.5e-7*A66*K6ROT*N1y**2*detJ*wij
                KC0e0001 += -2.5e-7*A66*K6ROT*N1x*N1y*detJ*wij
                KC0e0005 += 5.0e-7*A66*K6ROT*N1*N1y*detJ*wij
                KC0e0006 += 2.5e-7*A66*K6ROT*N1y*N2y*detJ*wij
                KC0e0007 += -2.5e-7*A66*K6ROT*N1y*N2x*detJ*wij
                KC0e0011 += 5.0e-7*A66*K6ROT*N1y*N2*detJ*wij
                KC0e0012 += 2.5e-7*A66*K6ROT*N1y*N3y*detJ*wij
                KC0e0013 += -2.5e-7*A66*K6ROT*N1y*N3x*detJ*wij
                KC0e0017 += 5.0e-7*A66*K6ROT*N1y*detJ*wij*(-N1 - N2 + 1)
                KC0e0101 += 2.5e-7*A66*K6ROT*N1x**2*detJ*wij
                KC0e0105 += -5.0e-7*A66*K6ROT*N1*N1x*detJ*wij
                KC0e0106 += -2.5e-7*A66*K6ROT*N1x*N2y*detJ*wij
                KC0e0107 += 2.5e-7*A66*K6ROT*N1x*N2x*detJ*wij
                KC0e0111 += -5.0e-7*A66*K6ROT*N1x*N2*detJ*wij
                KC0e0112 += -2.5e-7*A66*K6ROT*N1x*N3y*detJ*wij
                KC0e0113 += 2.5e-7*A66*K6ROT*N1x*N3x*detJ*wij
                KC0e0117 += 5.0e-7*A66*K6ROT*N1x*detJ*wij*(N1 + N2 - 1)
                KC0e0505 += 1.0e-6*A66*K6ROT*N1**2*detJ*wij
                KC0e0506 += 5.0e-7*A66*K6ROT*N1*N2y*detJ*wij
                KC0e0507 += -5.0e-7*A66*K6ROT*N1*N2x*detJ*wij
                KC0e0511 += 1.0e-6*A66*K6ROT*N1*N2*detJ*wij
                KC0e0512 += 5.0e-7*A66*K6ROT*N1*N3y*detJ*wij
                KC0e0513 += -5.0e-7*A66*K6ROT*N1*N3x*detJ*wij
                KC0e0517 += 1.0e-6*A66*K6ROT*N1*detJ*wij*(-N1 - N2 + 1)
                KC0e0606 += 2.5e-7*A66*K6ROT*N2y**2*detJ*wij
                KC0e0607 += -2.5e-7*A66*K6ROT*N2x*N2y*detJ*wij
                KC0e0611 += 5.0e-7*A66*K6ROT*N2*N2y*detJ*wij
                KC0e0612 += 2.5e-7*A66*K6ROT*N2y*N3y*detJ*wij
                KC0e0613 += -2.5e-7*A66*K6ROT*N2y*N3x*detJ*wij
                KC0e0617 += 5.0e-7*A66*K6ROT*N2y*detJ*wij*(-N1 - N2 + 1)
                KC0e0707 += 2.5e-7*A66*K6ROT*N2x**2*detJ*wij
                KC0e0711 += -5.0e-7*A66*K6ROT*N2*N2x*detJ*wij
                KC0e0712 += -2.5e-7*A66*K6ROT*N2x*N3y*detJ*wij
                KC0e0713 += 2.5e-7*A66*K6ROT*N2x*N3x*detJ*wij
                KC0e0717 += 5.0e-7*A66*K6ROT*N2x*detJ*wij*(N1 + N2 - 1)
                KC0e1111 += 1.0e-6*A66*K6ROT*N2**2*detJ*wij
                KC0e1112 += 5.0e-7*A66*K6ROT*N2*N3y*detJ*wij
                KC0e1113 += -5.0e-7*A66*K6ROT*N2*N3x*detJ*wij
                KC0e1117 += 1.0e-6*A66*K6ROT*N2*detJ*wij*(-N1 - N2 + 1)
                KC0e1212 += 2.5e-7*A66*K6ROT*N3y**2*detJ*wij
                KC0e1213 += -2.5e-7*A66*K6ROT*N3x*N3y*detJ*wij
                KC0e1217 += 5.0e-7*A66*K6ROT*N3y*detJ*wij*(-N1 - N2 + 1)
                KC0e1313 += 2.5e-7*A66*K6ROT*N3x**2*detJ*wij
                KC0e1317 += 5.0e-7*A66*K6ROT*N3x*detJ*wij*(N1 + N2 - 1)
                KC0e1717 += 1.0e-6*A66*K6ROT*detJ*wij*(N1 + N2 - 1)**2

            finte[0] = KC0e0000*ue[0] + KC0e0001*ue[1] + KC0e0003*ue[3] + KC0e0004*ue[4] + KC0e0005*ue[5] + KC0e0006*ue[6] + KC0e0007*ue[7] + KC0e0009*ue[9] + KC0e0010*ue[10] + KC0e0011*ue[11] + KC0e0012*ue[12] + KC0e0013*ue[13] + KC0e0015*ue[15] + KC0e0016*ue[16] + KC0e0017*ue[17]
            finte[1] = KC0e0001*ue[0] + KC0e0101*ue[1] + KC0e0103*ue[3] + KC0e0104*ue[4] + KC0e0105*ue[5] + KC0e0106*ue[6] + KC0e0107*ue[7] + KC0e0109*ue[9] + KC0e0110*ue[10] + KC0e0111*ue[11] + KC0e0112*ue[12] + KC0e0113*ue[13] + KC0e0115*ue[15] + KC0e0116*ue[16] + KC0e0117*ue[17]
            finte[2] = KC0e0202*ue[2] + KC0e0203*ue[3] + KC0e0204*ue[4] + KC0e0208*ue[8] + KC0e0209*ue[9] + KC0e0210*ue[10] + KC0e0214*ue[14] + KC0e0215*ue[15] + KC0e0216*ue[16]
            finte[3] = KC0e0003*ue[0] + KC0e0103*ue[1] + KC0e0203*ue[2] + KC0e0303*ue[3] + KC0e0304*ue[4] + KC0e0306*ue[6] + KC0e0307*ue[7] + KC0e0308*ue[8] + KC0e0309*ue[9] + KC0e0310*ue[10] + KC0e0312*ue[12] + KC0e0313*ue[13] + KC0e0314*ue[14] + KC0e0315*ue[15] + KC0e0316*ue[16]
            finte[4] = KC0e0004*ue[0] + KC0e0104*ue[1] + KC0e0204*ue[2] + KC0e0304*ue[3] + KC0e0404*ue[4] + KC0e0406*ue[6] + KC0e0407*ue[7] + KC0e0408*ue[8] + KC0e0409*ue[9] + KC0e0410*ue[10] + KC0e0412*ue[12] + KC0e0413*ue[13] + KC0e0414*ue[14] + KC0e0415*ue[15] + KC0e0416*ue[16]
            finte[5] = KC0e0005*ue[0] + KC0e0105*ue[1] + KC0e0505*ue[5] + KC0e0506*ue[6] + KC0e0507*ue[7] + KC0e0511*ue[11] + KC0e0512*ue[12] + KC0e0513*ue[13] + KC0e0517*ue[17]
            finte[6] = KC0e0006*ue[0] + KC0e0106*ue[1] + KC0e0306*ue[3] + KC0e0406*ue[4] + KC0e0506*ue[5] + KC0e0606*ue[6] + KC0e0607*ue[7] + KC0e0609*ue[9] + KC0e0610*ue[10] + KC0e0611*ue[11] + KC0e0612*ue[12] + KC0e0613*ue[13] + KC0e0615*ue[15] + KC0e0616*ue[16] + KC0e0617*ue[17]
            finte[7] = KC0e0007*ue[0] + KC0e0107*ue[1] + KC0e0307*ue[3] + KC0e0407*ue[4] + KC0e0507*ue[5] + KC0e0607*ue[6] + KC0e0707*ue[7] + KC0e0709*ue[9] + KC0e0710*ue[10] + KC0e0711*ue[11] + KC0e0712*ue[12] + KC0e0713*ue[13] + KC0e0715*ue[15] + KC0e0716*ue[16] + KC0e0717*ue[17]
            finte[8] = KC0e0208*ue[2] + KC0e0308*ue[3] + KC0e0408*ue[4] + KC0e0808*ue[8] + KC0e0809*ue[9] + KC0e0810*ue[10] + KC0e0814*ue[14] + KC0e0815*ue[15] + KC0e0816*ue[16]
            finte[9] = KC0e0009*ue[0] + KC0e0109*ue[1] + KC0e0209*ue[2] + KC0e0309*ue[3] + KC0e0409*ue[4] + KC0e0609*ue[6] + KC0e0709*ue[7] + KC0e0809*ue[8] + KC0e0909*ue[9] + KC0e0910*ue[10] + KC0e0912*ue[12] + KC0e0913*ue[13] + KC0e0914*ue[14] + KC0e0915*ue[15] + KC0e0916*ue[16]
            finte[10] = KC0e0010*ue[0] + KC0e0110*ue[1] + KC0e0210*ue[2] + KC0e0310*ue[3] + KC0e0410*ue[4] + KC0e0610*ue[6] + KC0e0710*ue[7] + KC0e0810*ue[8] + KC0e0910*ue[9] + KC0e1010*ue[10] + KC0e1012*ue[12] + KC0e1013*ue[13] + KC0e1014*ue[14] + KC0e1015*ue[15] + KC0e1016*ue[16]
            finte[11] = KC0e0011*ue[0] + KC0e0111*ue[1] + KC0e0511*ue[5] + KC0e0611*ue[6] + KC0e0711*ue[7] + KC0e1111*ue[11] + KC0e1112*ue[12] + KC0e1113*ue[13] + KC0e1117*ue[17]
            finte[12] = KC0e0012*ue[0] + KC0e0112*ue[1] + KC0e0312*ue[3] + KC0e0412*ue[4] + KC0e0512*ue[5] + KC0e0612*ue[6] + KC0e0712*ue[7] + KC0e0912*ue[9] + KC0e1012*ue[10] + KC0e1112*ue[11] + KC0e1212*ue[12] + KC0e1213*ue[13] + KC0e1215*ue[15] + KC0e1216*ue[16] + KC0e1217*ue[17]
            finte[13] = KC0e0013*ue[0] + KC0e0113*ue[1] + KC0e0313*ue[3] + KC0e0413*ue[4] + KC0e0513*ue[5] + KC0e0613*ue[6] + KC0e0713*ue[7] + KC0e0913*ue[9] + KC0e1013*ue[10] + KC0e1113*ue[11] + KC0e1213*ue[12] + KC0e1313*ue[13] + KC0e1315*ue[15] + KC0e1316*ue[16] + KC0e1317*ue[17]
            finte[14] = KC0e0214*ue[2] + KC0e0314*ue[3] + KC0e0414*ue[4] + KC0e0814*ue[8] + KC0e0914*ue[9] + KC0e1014*ue[10] + KC0e1414*ue[14] + KC0e1415*ue[15] + KC0e1416*ue[16]
            finte[15] = KC0e0015*ue[0] + KC0e0115*ue[1] + KC0e0215*ue[2] + KC0e0315*ue[3] + KC0e0415*ue[4] + KC0e0615*ue[6] + KC0e0715*ue[7] + KC0e0815*ue[8] + KC0e0915*ue[9] + KC0e1015*ue[10] + KC0e1215*ue[12] + KC0e1315*ue[13] + KC0e1415*ue[14] + KC0e1515*ue[15] + KC0e1516*ue[16]
            finte[16] = KC0e0016*ue[0] + KC0e0116*ue[1] + KC0e0216*ue[2] + KC0e0316*ue[3] + KC0e0416*ue[4] + KC0e0616*ue[6] + KC0e0716*ue[7] + KC0e0816*ue[8] + KC0e0916*ue[9] + KC0e1016*ue[10] + KC0e1216*ue[12] + KC0e1316*ue[13] + KC0e1416*ue[14] + KC0e1516*ue[15] + KC0e1616*ue[16]
            finte[17] = KC0e0017*ue[0] + KC0e0117*ue[1] + KC0e0517*ue[5] + KC0e0617*ue[6] + KC0e0717*ue[7] + KC0e1117*ue[11] + KC0e1217*ue[12] + KC0e1317*ue[13] + KC0e1717*ue[17]

            # NOTE the physics-based drilling stiffness, in element
            #      coordinates as the rest of finte. Adding the same delta
            #      that update_KC0 assembles keeps finte exactly equal to
            #      the product of the element matrix with the element
            #      displacements
            if self.drilling_model == 0:
                if self.gamma_rz >= 0.:
                    gamma_drill = self.gamma_rz
                else:
                    gamma_drill = A66
                Nxv[0] = N1x
                Nxv[1] = N2x
                Nxv[2] = N3x
                Nyv[0] = N1y
                Nyv[1] = N2y
                Nyv[2] = N3y
                allman_delta_Ke(Nxv, Nyv, &self.probe.xe[0], self.area,
                                Ae, Be, gamma_drill, Kedelta)
                for qi in range(18):
                    for qj in range(18):
                        finte[qi] += Kedelta[18*qi + qj]*ue[qj]

            if nonlinear:
                self._update_probe_finte_nonlinear(prop)


    cpdef void update_KC0(Tria3R self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          ShellProp prop,
                          int update_KC0v_only=0,
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        The element matrix is computed from the coordinates ``probe.xe``, the
        ``area`` and the constitutive matrices of ``prop`` rotated with
        ``m11, m12, m21, m22``, the transverse shear stiffnesses being scaled
        according to ``alpha_shear_locking``. Its values, in global
        coordinates, are added to ``KC0v`` at the positions ``init_k_KC0`` to
        ``init_k_KC0 + KC0_SPARSE_SIZE - 1``. With ``update_KC0v_only=0`` the
        global row and column indices are written to ``KC0r`` and ``KC0c`` at
        the same positions. Before this method is called,
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe` must have
        been used.


        Parameters
        ----------
        KC0r : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_KC0 +
            KC0_SPARSE_SIZE``.
        KC0c : np.array
            Array to store column positions of sparse values
            Same type and size as ``KC0r``.
        KC0v : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KC0r``. The values are
            added to the existing ones. Dimension of force per unit length
            between translations, of force between a translation and a
            rotation, and of force times length between rotations.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int, optional
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.

        Notes
        -----
        The membrane, bending, extension-bending coupling and transverse shear
        terms are integrated with a single point at the centroid, with the
        weight ``0.5`` times ``detJ = 2*area``, the transverse shear
        stiffnesses `A_{44}`, `A_{45}`, `A_{55}` divided by `1 + \alpha
        \ell^2/h^2`, with `\alpha` given by ``alpha_shear_locking``, `\ell`
        the longest edge and `h` the total thickness ``prop.h``, see the
        module documentation. The fictitious drilling penalty,

        .. math::
            K6ROT \cdot 10^{-6} A_{66} \int_A B_{r_z}^\top B_{r_z} dA,
            \qquad B_{r_z} = S^{r_z} + 1/2 S^u_{,y} - 1/2 S^v_{,x}

        is integrated with the three-point rule of degree of precision 2,
        first row of Table I in Cowper [1]_, with the area coordinates
        `(2/3, 1/6, 1/6)` and their permutations, each with the weight
        ``area/3``. With ``drilling_model = 0`` this penalty is
        switched off, ``K6ROT`` being replaced by zero, and the physics-based
        drilling stiffness of :func:`.allman_delta_Ke` is added instead,
        rotated to global coordinates with `K_{g,mn} = T_{mi} K_{e,ij}
        T_{nj}`. It is added separately because the enrichment couples the
        drilling rotation to the curvatures whenever the extension-bending
        matrix `B` is not zero.

        References
        ----------
        .. [1] Cowper, G. R., Gaussian quadrature formulas for triangles,
           International Journal for Numerical Methods in Engineering,
           7(3), 405-408, 1973. https://doi.org/10.1002/nme.1620070316

        """
        cdef int c1, c2, c3, i, k
        cdef double x1, x2, x3, y1, y2, y3
        cdef double wij, detJ
        cdef double Ae[9]
        cdef double Be[9]
        cdef double De[9]
        cdef double Atse[4]
        cdef double A44, A45, A55
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double D11, D12, D16, D22, D26, D66
        cdef double K6ROT
        cdef double points[3]
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double N1x, N2x, N3x, N1y, N2y, N3y
        cdef double N1, N2
        cdef double factor, maxl, l12, l23, l31
        cdef int bi, bj, qa, qb, qi, qj
        cdef double tmp, gamma_drill
        cdef double Nxv[3]
        cdef double Nyv[3]
        cdef double rmat[9]
        cdef double Kedelta[324]

        cdef double KC0e0000, KC0e0001, KC0e0003, KC0e0004, KC0e0005, KC0e0006, KC0e0007, KC0e0009, KC0e0010, KC0e0011, KC0e0012, KC0e0013, KC0e0015, KC0e0016, KC0e0017
        cdef double KC0e0101, KC0e0103, KC0e0104, KC0e0105, KC0e0106, KC0e0107, KC0e0109, KC0e0110, KC0e0111, KC0e0112, KC0e0113, KC0e0115, KC0e0116, KC0e0117
        cdef double KC0e0202, KC0e0203, KC0e0204, KC0e0208, KC0e0209, KC0e0210, KC0e0214, KC0e0215, KC0e0216
        cdef double KC0e0303, KC0e0304, KC0e0306, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0312, KC0e0313, KC0e0314, KC0e0315, KC0e0316
        cdef double KC0e0404, KC0e0406, KC0e0407, KC0e0408, KC0e0409, KC0e0410, KC0e0412, KC0e0413, KC0e0414, KC0e0415, KC0e0416
        cdef double KC0e0505, KC0e0506, KC0e0507, KC0e0511, KC0e0512, KC0e0513, KC0e0517
        cdef double KC0e0606, KC0e0607, KC0e0609, KC0e0610, KC0e0611, KC0e0612, KC0e0613, KC0e0615, KC0e0616, KC0e0617
        cdef double KC0e0707, KC0e0709, KC0e0710, KC0e0711, KC0e0712, KC0e0713, KC0e0715, KC0e0716, KC0e0717
        cdef double KC0e0808, KC0e0809, KC0e0810, KC0e0814, KC0e0815, KC0e0816
        cdef double KC0e0909, KC0e0910, KC0e0912, KC0e0913, KC0e0914, KC0e0915, KC0e0916
        cdef double KC0e1010, KC0e1012, KC0e1013, KC0e1014, KC0e1015, KC0e1016
        cdef double KC0e1111, KC0e1112, KC0e1113, KC0e1117
        cdef double KC0e1212, KC0e1213, KC0e1215, KC0e1216, KC0e1217
        cdef double KC0e1313, KC0e1315, KC0e1316, KC0e1317
        cdef double KC0e1414, KC0e1415, KC0e1416
        cdef double KC0e1515, KC0e1516
        cdef double KC0e1616
        cdef double KC0e1717

        with nogil:
            detJ = 2*self.area

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

            l12 = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            l23 = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
            l31 = ((x3 - x1)**2 + (y3 - y1)**2)**0.5
            maxl = l12
            if l23 > maxl:
                maxl = l23
            if l31 > maxl:
                maxl = l31

            # NOTE strategy to prevent shear locking used in the stabilised DSG elements imported here...
            factor = self.alpha_shear_locking*maxl**2/prop.h**2
            A44 = 1 / (1 + factor) * A44
            A45 = 1 / (1 + factor) * A45
            A55 = 1 / (1 + factor) * A55

            # NOTE the fictitious penalty is switched off when the
            #      physics-based drilling stiffness is active, so that the
            #      generated block contributes no drilling term at all and
            #      the enrichment supplies everything, see the module
            #      documentation
            if self.drilling_model == 0:
                K6ROT = 0.
            else:
                K6ROT = self.K6ROT

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

            if update_KC0v_only == 0:
                # positions in the global stiffness matrix
                c1 = self.c1
                c2 = self.c2
                c3 = self.c3

                k = self.init_k_KC0
                KC0r[k] = 0+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c3

            N1x = (y2 - y3)/(2*self.area)
            N2x = (-y1 + y3)/(2*self.area)
            N3x = (y1 - y2)/(2*self.area)
            N1y = (-x2 + x3)/(2*self.area)
            N2y = (x1 - x3)/(2*self.area)
            N3y = (-x1 + x2)/(2*self.area)

            # NOTE KC0e terms for REDUCED integration
            wij = 0.5
            N1 = N2 = 0.333333333333333333333333333333333333333333333

            KC0e0000 = detJ*wij*(N1x*(A11*N1x + A16*N1y) + N1y*(A16*N1x + A66*N1y))
            KC0e0001 = detJ*wij*(N1x*(A16*N1x + A66*N1y) + N1y*(A12*N1x + A26*N1y))
            KC0e0003 = -detJ*wij*(N1x*(B16*N1x + B66*N1y) + N1y*(B12*N1x + B26*N1y))
            KC0e0004 = detJ*wij*(N1x*(B11*N1x + B16*N1y) + N1y*(B16*N1x + B66*N1y))
            KC0e0006 = detJ*wij*(N2x*(A11*N1x + A16*N1y) + N2y*(A16*N1x + A66*N1y))
            KC0e0007 = detJ*wij*(N2x*(A16*N1x + A66*N1y) + N2y*(A12*N1x + A26*N1y))
            KC0e0009 = -detJ*wij*(N2x*(B16*N1x + B66*N1y) + N2y*(B12*N1x + B26*N1y))
            KC0e0010 = detJ*wij*(N2x*(B11*N1x + B16*N1y) + N2y*(B16*N1x + B66*N1y))
            KC0e0012 = detJ*wij*(N3x*(A11*N1x + A16*N1y) + N3y*(A16*N1x + A66*N1y))
            KC0e0013 = detJ*wij*(N3x*(A16*N1x + A66*N1y) + N3y*(A12*N1x + A26*N1y))
            KC0e0015 = -detJ*wij*(N3x*(B16*N1x + B66*N1y) + N3y*(B12*N1x + B26*N1y))
            KC0e0016 = detJ*wij*(N3x*(B11*N1x + B16*N1y) + N3y*(B16*N1x + B66*N1y))
            KC0e0101 = detJ*wij*(N1x*(A26*N1y + A66*N1x) + N1y*(A22*N1y + A26*N1x))
            KC0e0103 = -detJ*wij*(N1x*(B26*N1y + B66*N1x) + N1y*(B22*N1y + B26*N1x))
            KC0e0104 = detJ*wij*(N1x*(B12*N1y + B16*N1x) + N1y*(B26*N1y + B66*N1x))
            KC0e0106 = detJ*wij*(N2x*(A12*N1y + A16*N1x) + N2y*(A26*N1y + A66*N1x))
            KC0e0107 = detJ*wij*(N2x*(A26*N1y + A66*N1x) + N2y*(A22*N1y + A26*N1x))
            KC0e0109 = -detJ*wij*(N2x*(B26*N1y + B66*N1x) + N2y*(B22*N1y + B26*N1x))
            KC0e0110 = detJ*wij*(N2x*(B12*N1y + B16*N1x) + N2y*(B26*N1y + B66*N1x))
            KC0e0112 = detJ*wij*(N3x*(A12*N1y + A16*N1x) + N3y*(A26*N1y + A66*N1x))
            KC0e0113 = detJ*wij*(N3x*(A26*N1y + A66*N1x) + N3y*(A22*N1y + A26*N1x))
            KC0e0115 = -detJ*wij*(N3x*(B26*N1y + B66*N1x) + N3y*(B22*N1y + B26*N1x))
            KC0e0116 = detJ*wij*(N3x*(B12*N1y + B16*N1x) + N3y*(B26*N1y + B66*N1x))
            KC0e0202 = detJ*wij*(N1x*(A45*N1y + A55*N1x) + N1y*(A44*N1y + A45*N1x))
            KC0e0203 = -N1*detJ*wij*(A44*N1y + A45*N1x)
            KC0e0204 = N1*detJ*wij*(A45*N1y + A55*N1x)
            KC0e0208 = detJ*wij*(N2x*(A45*N1y + A55*N1x) + N2y*(A44*N1y + A45*N1x))
            KC0e0209 = -N2*detJ*wij*(A44*N1y + A45*N1x)
            KC0e0210 = N2*detJ*wij*(A45*N1y + A55*N1x)
            KC0e0214 = detJ*wij*(N3x*(A45*N1y + A55*N1x) + N3y*(A44*N1y + A45*N1x))
            KC0e0215 = detJ*wij*(A44*N1y + A45*N1x)*(N1 + N2 - 1)
            KC0e0216 = -detJ*wij*(A45*N1y + A55*N1x)*(N1 + N2 - 1)
            KC0e0303 = detJ*wij*(D22*N1y**2 + 2*D26*N1x*N1y + D66*N1x**2 + A44*N1**2)
            KC0e0304 = -detJ*wij*(A45*N1**2 + N1x*(D12*N1y + D16*N1x) + N1y*(D26*N1y + D66*N1x))
            KC0e0306 = -detJ*wij*(N2x*(B12*N1y + B16*N1x) + N2y*(B26*N1y + B66*N1x))
            KC0e0307 = -detJ*wij*(N2x*(B26*N1y + B66*N1x) + N2y*(B22*N1y + B26*N1x))
            KC0e0308 = -N1*detJ*wij*(A44*N2y + A45*N2x)
            KC0e0309 = detJ*wij*(A44*N1*N2 + N2x*(D26*N1y + D66*N1x) + N2y*(D22*N1y + D26*N1x))
            KC0e0310 = -detJ*wij*(A45*N1*N2 + N2x*(D12*N1y + D16*N1x) + N2y*(D26*N1y + D66*N1x))
            KC0e0312 = -detJ*wij*(N3x*(B12*N1y + B16*N1x) + N3y*(B26*N1y + B66*N1x))
            KC0e0313 = -detJ*wij*(N3x*(B26*N1y + B66*N1x) + N3y*(B22*N1y + B26*N1x))
            KC0e0314 = -N1*detJ*wij*(A44*N3y + A45*N3x)
            KC0e0315 = detJ*wij*(-A44*N1*(N1 + N2 - 1) + N3x*(D26*N1y + D66*N1x) + N3y*(D22*N1y + D26*N1x))
            KC0e0316 = -detJ*wij*(-A45*N1*(N1 + N2 - 1) + N3x*(D12*N1y + D16*N1x) + N3y*(D26*N1y + D66*N1x))
            KC0e0404 = detJ*wij*(A55*N1**2 + N1x*(D11*N1x + D16*N1y) + N1y*(D16*N1x + D66*N1y))
            KC0e0406 = detJ*wij*(N2x*(B11*N1x + B16*N1y) + N2y*(B16*N1x + B66*N1y))
            KC0e0407 = detJ*wij*(N2x*(B16*N1x + B66*N1y) + N2y*(B12*N1x + B26*N1y))
            KC0e0408 = N1*detJ*wij*(A45*N2y + A55*N2x)
            KC0e0409 = -detJ*wij*(A45*N1*N2 + N2x*(D16*N1x + D66*N1y) + N2y*(D12*N1x + D26*N1y))
            KC0e0410 = detJ*wij*(A55*N1*N2 + N2x*(D11*N1x + D16*N1y) + N2y*(D16*N1x + D66*N1y))
            KC0e0412 = detJ*wij*(N3x*(B11*N1x + B16*N1y) + N3y*(B16*N1x + B66*N1y))
            KC0e0413 = detJ*wij*(N3x*(B16*N1x + B66*N1y) + N3y*(B12*N1x + B26*N1y))
            KC0e0414 = N1*detJ*wij*(A45*N3y + A55*N3x)
            KC0e0415 = -detJ*wij*(-A45*N1*(N1 + N2 - 1) + N3x*(D16*N1x + D66*N1y) + N3y*(D12*N1x + D26*N1y))
            KC0e0416 = detJ*wij*(-A55*N1*(N1 + N2 - 1) + N3x*(D11*N1x + D16*N1y) + N3y*(D16*N1x + D66*N1y))
            KC0e0606 = detJ*wij*(N2x*(A11*N2x + A16*N2y) + N2y*(A16*N2x + A66*N2y))
            KC0e0607 = detJ*wij*(N2x*(A16*N2x + A66*N2y) + N2y*(A12*N2x + A26*N2y))
            KC0e0609 = -detJ*wij*(N2x*(B16*N2x + B66*N2y) + N2y*(B12*N2x + B26*N2y))
            KC0e0610 = detJ*wij*(N2x*(B11*N2x + B16*N2y) + N2y*(B16*N2x + B66*N2y))
            KC0e0612 = detJ*wij*(N3x*(A11*N2x + A16*N2y) + N3y*(A16*N2x + A66*N2y))
            KC0e0613 = detJ*wij*(N3x*(A16*N2x + A66*N2y) + N3y*(A12*N2x + A26*N2y))
            KC0e0615 = -detJ*wij*(N3x*(B16*N2x + B66*N2y) + N3y*(B12*N2x + B26*N2y))
            KC0e0616 = detJ*wij*(N3x*(B11*N2x + B16*N2y) + N3y*(B16*N2x + B66*N2y))
            KC0e0707 = detJ*wij*(N2x*(A26*N2y + A66*N2x) + N2y*(A22*N2y + A26*N2x))
            KC0e0709 = -detJ*wij*(N2x*(B26*N2y + B66*N2x) + N2y*(B22*N2y + B26*N2x))
            KC0e0710 = detJ*wij*(N2x*(B12*N2y + B16*N2x) + N2y*(B26*N2y + B66*N2x))
            KC0e0712 = detJ*wij*(N3x*(A12*N2y + A16*N2x) + N3y*(A26*N2y + A66*N2x))
            KC0e0713 = detJ*wij*(N3x*(A26*N2y + A66*N2x) + N3y*(A22*N2y + A26*N2x))
            KC0e0715 = -detJ*wij*(N3x*(B26*N2y + B66*N2x) + N3y*(B22*N2y + B26*N2x))
            KC0e0716 = detJ*wij*(N3x*(B12*N2y + B16*N2x) + N3y*(B26*N2y + B66*N2x))
            KC0e0808 = detJ*wij*(N2x*(A45*N2y + A55*N2x) + N2y*(A44*N2y + A45*N2x))
            KC0e0809 = -N2*detJ*wij*(A44*N2y + A45*N2x)
            KC0e0810 = N2*detJ*wij*(A45*N2y + A55*N2x)
            KC0e0814 = detJ*wij*(N3x*(A45*N2y + A55*N2x) + N3y*(A44*N2y + A45*N2x))
            KC0e0815 = detJ*wij*(A44*N2y + A45*N2x)*(N1 + N2 - 1)
            KC0e0816 = -detJ*wij*(A45*N2y + A55*N2x)*(N1 + N2 - 1)
            KC0e0909 = detJ*wij*(D22*N2y**2 + 2*D26*N2x*N2y + D66*N2x**2 + A44*N2**2)
            KC0e0910 = -detJ*wij*(A45*N2**2 + N2x*(D12*N2y + D16*N2x) + N2y*(D26*N2y + D66*N2x))
            KC0e0912 = -detJ*wij*(N3x*(B12*N2y + B16*N2x) + N3y*(B26*N2y + B66*N2x))
            KC0e0913 = -detJ*wij*(N3x*(B26*N2y + B66*N2x) + N3y*(B22*N2y + B26*N2x))
            KC0e0914 = -N2*detJ*wij*(A44*N3y + A45*N3x)
            KC0e0915 = detJ*wij*(-A44*N2*(N1 + N2 - 1) + N3x*(D26*N2y + D66*N2x) + N3y*(D22*N2y + D26*N2x))
            KC0e0916 = -detJ*wij*(-A45*N2*(N1 + N2 - 1) + N3x*(D12*N2y + D16*N2x) + N3y*(D26*N2y + D66*N2x))
            KC0e1010 = detJ*wij*(A55*N2**2 + N2x*(D11*N2x + D16*N2y) + N2y*(D16*N2x + D66*N2y))
            KC0e1012 = detJ*wij*(N3x*(B11*N2x + B16*N2y) + N3y*(B16*N2x + B66*N2y))
            KC0e1013 = detJ*wij*(N3x*(B16*N2x + B66*N2y) + N3y*(B12*N2x + B26*N2y))
            KC0e1014 = N2*detJ*wij*(A45*N3y + A55*N3x)
            KC0e1015 = -detJ*wij*(-A45*N2*(N1 + N2 - 1) + N3x*(D16*N2x + D66*N2y) + N3y*(D12*N2x + D26*N2y))
            KC0e1016 = detJ*wij*(-A55*N2*(N1 + N2 - 1) + N3x*(D11*N2x + D16*N2y) + N3y*(D16*N2x + D66*N2y))
            KC0e1212 = detJ*wij*(N3x*(A11*N3x + A16*N3y) + N3y*(A16*N3x + A66*N3y))
            KC0e1213 = detJ*wij*(N3x*(A16*N3x + A66*N3y) + N3y*(A12*N3x + A26*N3y))
            KC0e1215 = -detJ*wij*(N3x*(B16*N3x + B66*N3y) + N3y*(B12*N3x + B26*N3y))
            KC0e1216 = detJ*wij*(N3x*(B11*N3x + B16*N3y) + N3y*(B16*N3x + B66*N3y))
            KC0e1313 = detJ*wij*(N3x*(A26*N3y + A66*N3x) + N3y*(A22*N3y + A26*N3x))
            KC0e1315 = -detJ*wij*(N3x*(B26*N3y + B66*N3x) + N3y*(B22*N3y + B26*N3x))
            KC0e1316 = detJ*wij*(N3x*(B12*N3y + B16*N3x) + N3y*(B26*N3y + B66*N3x))
            KC0e1414 = detJ*wij*(N3x*(A45*N3y + A55*N3x) + N3y*(A44*N3y + A45*N3x))
            KC0e1415 = detJ*wij*(A44*N3y + A45*N3x)*(N1 + N2 - 1)
            KC0e1416 = -detJ*wij*(A45*N3y + A55*N3x)*(N1 + N2 - 1)
            KC0e1515 = detJ*wij*(A44*(N1 + N2 - 1)**2 + N3x*(D26*N3y + D66*N3x) + N3y*(D22*N3y + D26*N3x))
            KC0e1516 = -detJ*wij*(A45*(N1 + N2 - 1)**2 + N3x*(D12*N3y + D16*N3x) + N3y*(D26*N3y + D66*N3x))
            KC0e1616 = detJ*wij*(A55*(N1 + N2 - 1)**2 + N3x*(D11*N3x + D16*N3y) + N3y*(D16*N3x + D66*N3y))

            # NOTE KC0e drilling terms with FULL integration

            KC0e0005 = 0
            KC0e0011 = 0
            KC0e0017 = 0
            KC0e0105 = 0
            KC0e0111 = 0
            KC0e0117 = 0
            KC0e0505 = 0
            KC0e0506 = 0
            KC0e0507 = 0
            KC0e0511 = 0
            KC0e0512 = 0
            KC0e0513 = 0
            KC0e0517 = 0
            KC0e0611 = 0
            KC0e0617 = 0
            KC0e0711 = 0
            KC0e0717 = 0
            KC0e1111 = 0
            KC0e1112 = 0
            KC0e1113 = 0
            KC0e1117 = 0
            KC0e1217 = 0
            KC0e1317 = 0
            KC0e1717 = 0

            # NOTE three-point rule of degree of precision 2, first row of Table I
            #      in Cowper, G. R., Gaussian quadrature formulas for triangles,
            #      IJNME 7(3), 405-408, 1973, doi:10.1002/nme.1620070316: area
            #      coordinates (2/3, 1/6, 1/6) and their permutations, weight 1/6
            #      times detJ = 2*area, i.e. area/3
            wij = 0.5*0.333333333333333333333333333333333333333333333
            points[0] = 0.66666666666666666666666666666666666666666667
            points[1] = 0.16666666666666666666666666666666666666666667
            points[2] = 0.16666666666666666666666666666666666666666667
            for i in range(3):
                if i == 0:
                    N1 = points[0]
                    N2 = points[1]
                elif i == 1:
                    N1 = points[1]
                    N2 = points[2]
                elif i == 2:
                    N1 = points[2]
                    N2 = points[0]

                KC0e0000 += 2.5e-7*A66*K6ROT*N1y**2*detJ*wij
                KC0e0001 += -2.5e-7*A66*K6ROT*N1x*N1y*detJ*wij
                KC0e0005 += 5.0e-7*A66*K6ROT*N1*N1y*detJ*wij
                KC0e0006 += 2.5e-7*A66*K6ROT*N1y*N2y*detJ*wij
                KC0e0007 += -2.5e-7*A66*K6ROT*N1y*N2x*detJ*wij
                KC0e0011 += 5.0e-7*A66*K6ROT*N1y*N2*detJ*wij
                KC0e0012 += 2.5e-7*A66*K6ROT*N1y*N3y*detJ*wij
                KC0e0013 += -2.5e-7*A66*K6ROT*N1y*N3x*detJ*wij
                KC0e0017 += 5.0e-7*A66*K6ROT*N1y*detJ*wij*(-N1 - N2 + 1)
                KC0e0101 += 2.5e-7*A66*K6ROT*N1x**2*detJ*wij
                KC0e0105 += -5.0e-7*A66*K6ROT*N1*N1x*detJ*wij
                KC0e0106 += -2.5e-7*A66*K6ROT*N1x*N2y*detJ*wij
                KC0e0107 += 2.5e-7*A66*K6ROT*N1x*N2x*detJ*wij
                KC0e0111 += -5.0e-7*A66*K6ROT*N1x*N2*detJ*wij
                KC0e0112 += -2.5e-7*A66*K6ROT*N1x*N3y*detJ*wij
                KC0e0113 += 2.5e-7*A66*K6ROT*N1x*N3x*detJ*wij
                KC0e0117 += 5.0e-7*A66*K6ROT*N1x*detJ*wij*(N1 + N2 - 1)
                KC0e0505 += 1.0e-6*A66*K6ROT*N1**2*detJ*wij
                KC0e0506 += 5.0e-7*A66*K6ROT*N1*N2y*detJ*wij
                KC0e0507 += -5.0e-7*A66*K6ROT*N1*N2x*detJ*wij
                KC0e0511 += 1.0e-6*A66*K6ROT*N1*N2*detJ*wij
                KC0e0512 += 5.0e-7*A66*K6ROT*N1*N3y*detJ*wij
                KC0e0513 += -5.0e-7*A66*K6ROT*N1*N3x*detJ*wij
                KC0e0517 += 1.0e-6*A66*K6ROT*N1*detJ*wij*(-N1 - N2 + 1)
                KC0e0606 += 2.5e-7*A66*K6ROT*N2y**2*detJ*wij
                KC0e0607 += -2.5e-7*A66*K6ROT*N2x*N2y*detJ*wij
                KC0e0611 += 5.0e-7*A66*K6ROT*N2*N2y*detJ*wij
                KC0e0612 += 2.5e-7*A66*K6ROT*N2y*N3y*detJ*wij
                KC0e0613 += -2.5e-7*A66*K6ROT*N2y*N3x*detJ*wij
                KC0e0617 += 5.0e-7*A66*K6ROT*N2y*detJ*wij*(-N1 - N2 + 1)
                KC0e0707 += 2.5e-7*A66*K6ROT*N2x**2*detJ*wij
                KC0e0711 += -5.0e-7*A66*K6ROT*N2*N2x*detJ*wij
                KC0e0712 += -2.5e-7*A66*K6ROT*N2x*N3y*detJ*wij
                KC0e0713 += 2.5e-7*A66*K6ROT*N2x*N3x*detJ*wij
                KC0e0717 += 5.0e-7*A66*K6ROT*N2x*detJ*wij*(N1 + N2 - 1)
                KC0e1111 += 1.0e-6*A66*K6ROT*N2**2*detJ*wij
                KC0e1112 += 5.0e-7*A66*K6ROT*N2*N3y*detJ*wij
                KC0e1113 += -5.0e-7*A66*K6ROT*N2*N3x*detJ*wij
                KC0e1117 += 1.0e-6*A66*K6ROT*N2*detJ*wij*(-N1 - N2 + 1)
                KC0e1212 += 2.5e-7*A66*K6ROT*N3y**2*detJ*wij
                KC0e1213 += -2.5e-7*A66*K6ROT*N3x*N3y*detJ*wij
                KC0e1217 += 5.0e-7*A66*K6ROT*N3y*detJ*wij*(-N1 - N2 + 1)
                KC0e1313 += 2.5e-7*A66*K6ROT*N3x**2*detJ*wij
                KC0e1317 += 5.0e-7*A66*K6ROT*N3x*detJ*wij*(N1 + N2 - 1)
                KC0e1717 += 1.0e-6*A66*K6ROT*detJ*wij*(N1 + N2 - 1)**2

            k = self.init_k_KC0
            KC0v[k] += KC0e0202*r13**2 + r11*(KC0e0000*r11 + KC0e0001*r12) + r12*(KC0e0001*r11 + KC0e0101*r12)
            k += 1
            KC0v[k] += KC0e0202*r13*r23 + r21*(KC0e0000*r11 + KC0e0001*r12) + r22*(KC0e0001*r11 + KC0e0101*r12)
            k += 1
            KC0v[k] += KC0e0202*r13*r33 + r31*(KC0e0000*r11 + KC0e0001*r12) + r32*(KC0e0001*r11 + KC0e0101*r12)
            k += 1
            KC0v[k] += r11*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r12*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r13*(KC0e0005*r11 + KC0e0105*r12)
            k += 1
            KC0v[k] += r21*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r22*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r23*(KC0e0005*r11 + KC0e0105*r12)
            k += 1
            KC0v[k] += r31*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r32*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r33*(KC0e0005*r11 + KC0e0105*r12)
            k += 1
            KC0v[k] += KC0e0208*r13**2 + r11*(KC0e0006*r11 + KC0e0106*r12) + r12*(KC0e0007*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r21*(KC0e0006*r11 + KC0e0106*r12) + r22*(KC0e0007*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r31*(KC0e0006*r11 + KC0e0106*r12) + r32*(KC0e0007*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += r11*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r12*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r13*(KC0e0011*r11 + KC0e0111*r12)
            k += 1
            KC0v[k] += r21*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r22*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r23*(KC0e0011*r11 + KC0e0111*r12)
            k += 1
            KC0v[k] += r31*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r32*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r33*(KC0e0011*r11 + KC0e0111*r12)
            k += 1
            KC0v[k] += KC0e0214*r13**2 + r11*(KC0e0012*r11 + KC0e0112*r12) + r12*(KC0e0013*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r21*(KC0e0012*r11 + KC0e0112*r12) + r22*(KC0e0013*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r31*(KC0e0012*r11 + KC0e0112*r12) + r32*(KC0e0013*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += r11*(KC0e0015*r11 + KC0e0115*r12 + KC0e0215*r13) + r12*(KC0e0016*r11 + KC0e0116*r12 + KC0e0216*r13) + r13*(KC0e0017*r11 + KC0e0117*r12)
            k += 1
            KC0v[k] += r21*(KC0e0015*r11 + KC0e0115*r12 + KC0e0215*r13) + r22*(KC0e0016*r11 + KC0e0116*r12 + KC0e0216*r13) + r23*(KC0e0017*r11 + KC0e0117*r12)
            k += 1
            KC0v[k] += r31*(KC0e0015*r11 + KC0e0115*r12 + KC0e0215*r13) + r32*(KC0e0016*r11 + KC0e0116*r12 + KC0e0216*r13) + r33*(KC0e0017*r11 + KC0e0117*r12)
            k += 1
            KC0v[k] += KC0e0202*r13*r23 + r11*(KC0e0000*r21 + KC0e0001*r22) + r12*(KC0e0001*r21 + KC0e0101*r22)
            k += 1
            KC0v[k] += KC0e0202*r23**2 + r21*(KC0e0000*r21 + KC0e0001*r22) + r22*(KC0e0001*r21 + KC0e0101*r22)
            k += 1
            KC0v[k] += KC0e0202*r23*r33 + r31*(KC0e0000*r21 + KC0e0001*r22) + r32*(KC0e0001*r21 + KC0e0101*r22)
            k += 1
            KC0v[k] += r11*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r12*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r13*(KC0e0005*r21 + KC0e0105*r22)
            k += 1
            KC0v[k] += r21*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r22*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r23*(KC0e0005*r21 + KC0e0105*r22)
            k += 1
            KC0v[k] += r31*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r32*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r33*(KC0e0005*r21 + KC0e0105*r22)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r11*(KC0e0006*r21 + KC0e0106*r22) + r12*(KC0e0007*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23**2 + r21*(KC0e0006*r21 + KC0e0106*r22) + r22*(KC0e0007*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r31*(KC0e0006*r21 + KC0e0106*r22) + r32*(KC0e0007*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += r11*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r12*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r13*(KC0e0011*r21 + KC0e0111*r22)
            k += 1
            KC0v[k] += r21*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r22*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r23*(KC0e0011*r21 + KC0e0111*r22)
            k += 1
            KC0v[k] += r31*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r32*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r33*(KC0e0011*r21 + KC0e0111*r22)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r11*(KC0e0012*r21 + KC0e0112*r22) + r12*(KC0e0013*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23**2 + r21*(KC0e0012*r21 + KC0e0112*r22) + r22*(KC0e0013*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r31*(KC0e0012*r21 + KC0e0112*r22) + r32*(KC0e0013*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += r11*(KC0e0015*r21 + KC0e0115*r22 + KC0e0215*r23) + r12*(KC0e0016*r21 + KC0e0116*r22 + KC0e0216*r23) + r13*(KC0e0017*r21 + KC0e0117*r22)
            k += 1
            KC0v[k] += r21*(KC0e0015*r21 + KC0e0115*r22 + KC0e0215*r23) + r22*(KC0e0016*r21 + KC0e0116*r22 + KC0e0216*r23) + r23*(KC0e0017*r21 + KC0e0117*r22)
            k += 1
            KC0v[k] += r31*(KC0e0015*r21 + KC0e0115*r22 + KC0e0215*r23) + r32*(KC0e0016*r21 + KC0e0116*r22 + KC0e0216*r23) + r33*(KC0e0017*r21 + KC0e0117*r22)
            k += 1
            KC0v[k] += KC0e0202*r13*r33 + r11*(KC0e0000*r31 + KC0e0001*r32) + r12*(KC0e0001*r31 + KC0e0101*r32)
            k += 1
            KC0v[k] += KC0e0202*r23*r33 + r21*(KC0e0000*r31 + KC0e0001*r32) + r22*(KC0e0001*r31 + KC0e0101*r32)
            k += 1
            KC0v[k] += KC0e0202*r33**2 + r31*(KC0e0000*r31 + KC0e0001*r32) + r32*(KC0e0001*r31 + KC0e0101*r32)
            k += 1
            KC0v[k] += r11*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r12*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r13*(KC0e0005*r31 + KC0e0105*r32)
            k += 1
            KC0v[k] += r21*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r22*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r23*(KC0e0005*r31 + KC0e0105*r32)
            k += 1
            KC0v[k] += r31*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r32*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r33*(KC0e0005*r31 + KC0e0105*r32)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r11*(KC0e0006*r31 + KC0e0106*r32) + r12*(KC0e0007*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r21*(KC0e0006*r31 + KC0e0106*r32) + r22*(KC0e0007*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r33**2 + r31*(KC0e0006*r31 + KC0e0106*r32) + r32*(KC0e0007*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += r11*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r12*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r13*(KC0e0011*r31 + KC0e0111*r32)
            k += 1
            KC0v[k] += r21*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r22*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r23*(KC0e0011*r31 + KC0e0111*r32)
            k += 1
            KC0v[k] += r31*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r32*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r33*(KC0e0011*r31 + KC0e0111*r32)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r11*(KC0e0012*r31 + KC0e0112*r32) + r12*(KC0e0013*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r21*(KC0e0012*r31 + KC0e0112*r32) + r22*(KC0e0013*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r33**2 + r31*(KC0e0012*r31 + KC0e0112*r32) + r32*(KC0e0013*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += r11*(KC0e0015*r31 + KC0e0115*r32 + KC0e0215*r33) + r12*(KC0e0016*r31 + KC0e0116*r32 + KC0e0216*r33) + r13*(KC0e0017*r31 + KC0e0117*r32)
            k += 1
            KC0v[k] += r21*(KC0e0015*r31 + KC0e0115*r32 + KC0e0215*r33) + r22*(KC0e0016*r31 + KC0e0116*r32 + KC0e0216*r33) + r23*(KC0e0017*r31 + KC0e0117*r32)
            k += 1
            KC0v[k] += r31*(KC0e0015*r31 + KC0e0115*r32 + KC0e0215*r33) + r32*(KC0e0016*r31 + KC0e0116*r32 + KC0e0216*r33) + r33*(KC0e0017*r31 + KC0e0117*r32)
            k += 1
            KC0v[k] += r11*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r12*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r13*(KC0e0203*r11 + KC0e0204*r12)
            k += 1
            KC0v[k] += r21*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r22*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r23*(KC0e0203*r11 + KC0e0204*r12)
            k += 1
            KC0v[k] += r31*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r32*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r33*(KC0e0203*r11 + KC0e0204*r12)
            k += 1
            KC0v[k] += KC0e0505*r13**2 + r11*(KC0e0303*r11 + KC0e0304*r12) + r12*(KC0e0304*r11 + KC0e0404*r12)
            k += 1
            KC0v[k] += KC0e0505*r13*r23 + r21*(KC0e0303*r11 + KC0e0304*r12) + r22*(KC0e0304*r11 + KC0e0404*r12)
            k += 1
            KC0v[k] += KC0e0505*r13*r33 + r31*(KC0e0303*r11 + KC0e0304*r12) + r32*(KC0e0304*r11 + KC0e0404*r12)
            k += 1
            KC0v[k] += r11*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r12*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r13*(KC0e0308*r11 + KC0e0408*r12)
            k += 1
            KC0v[k] += r21*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r22*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r23*(KC0e0308*r11 + KC0e0408*r12)
            k += 1
            KC0v[k] += r31*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r32*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r33*(KC0e0308*r11 + KC0e0408*r12)
            k += 1
            KC0v[k] += KC0e0511*r13**2 + r11*(KC0e0309*r11 + KC0e0409*r12) + r12*(KC0e0310*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r21*(KC0e0309*r11 + KC0e0409*r12) + r22*(KC0e0310*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r31*(KC0e0309*r11 + KC0e0409*r12) + r32*(KC0e0310*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += r11*(KC0e0312*r11 + KC0e0412*r12 + KC0e0512*r13) + r12*(KC0e0313*r11 + KC0e0413*r12 + KC0e0513*r13) + r13*(KC0e0314*r11 + KC0e0414*r12)
            k += 1
            KC0v[k] += r21*(KC0e0312*r11 + KC0e0412*r12 + KC0e0512*r13) + r22*(KC0e0313*r11 + KC0e0413*r12 + KC0e0513*r13) + r23*(KC0e0314*r11 + KC0e0414*r12)
            k += 1
            KC0v[k] += r31*(KC0e0312*r11 + KC0e0412*r12 + KC0e0512*r13) + r32*(KC0e0313*r11 + KC0e0413*r12 + KC0e0513*r13) + r33*(KC0e0314*r11 + KC0e0414*r12)
            k += 1
            KC0v[k] += KC0e0517*r13**2 + r11*(KC0e0315*r11 + KC0e0415*r12) + r12*(KC0e0316*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r21*(KC0e0315*r11 + KC0e0415*r12) + r22*(KC0e0316*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r31*(KC0e0315*r11 + KC0e0415*r12) + r32*(KC0e0316*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += r11*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r12*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r13*(KC0e0203*r21 + KC0e0204*r22)
            k += 1
            KC0v[k] += r21*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r22*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r23*(KC0e0203*r21 + KC0e0204*r22)
            k += 1
            KC0v[k] += r31*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r32*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r33*(KC0e0203*r21 + KC0e0204*r22)
            k += 1
            KC0v[k] += KC0e0505*r13*r23 + r11*(KC0e0303*r21 + KC0e0304*r22) + r12*(KC0e0304*r21 + KC0e0404*r22)
            k += 1
            KC0v[k] += KC0e0505*r23**2 + r21*(KC0e0303*r21 + KC0e0304*r22) + r22*(KC0e0304*r21 + KC0e0404*r22)
            k += 1
            KC0v[k] += KC0e0505*r23*r33 + r31*(KC0e0303*r21 + KC0e0304*r22) + r32*(KC0e0304*r21 + KC0e0404*r22)
            k += 1
            KC0v[k] += r11*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r12*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r13*(KC0e0308*r21 + KC0e0408*r22)
            k += 1
            KC0v[k] += r21*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r22*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r23*(KC0e0308*r21 + KC0e0408*r22)
            k += 1
            KC0v[k] += r31*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r32*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r33*(KC0e0308*r21 + KC0e0408*r22)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r11*(KC0e0309*r21 + KC0e0409*r22) + r12*(KC0e0310*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23**2 + r21*(KC0e0309*r21 + KC0e0409*r22) + r22*(KC0e0310*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r31*(KC0e0309*r21 + KC0e0409*r22) + r32*(KC0e0310*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += r11*(KC0e0312*r21 + KC0e0412*r22 + KC0e0512*r23) + r12*(KC0e0313*r21 + KC0e0413*r22 + KC0e0513*r23) + r13*(KC0e0314*r21 + KC0e0414*r22)
            k += 1
            KC0v[k] += r21*(KC0e0312*r21 + KC0e0412*r22 + KC0e0512*r23) + r22*(KC0e0313*r21 + KC0e0413*r22 + KC0e0513*r23) + r23*(KC0e0314*r21 + KC0e0414*r22)
            k += 1
            KC0v[k] += r31*(KC0e0312*r21 + KC0e0412*r22 + KC0e0512*r23) + r32*(KC0e0313*r21 + KC0e0413*r22 + KC0e0513*r23) + r33*(KC0e0314*r21 + KC0e0414*r22)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r11*(KC0e0315*r21 + KC0e0415*r22) + r12*(KC0e0316*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23**2 + r21*(KC0e0315*r21 + KC0e0415*r22) + r22*(KC0e0316*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r31*(KC0e0315*r21 + KC0e0415*r22) + r32*(KC0e0316*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += r11*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r12*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r13*(KC0e0203*r31 + KC0e0204*r32)
            k += 1
            KC0v[k] += r21*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r22*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r23*(KC0e0203*r31 + KC0e0204*r32)
            k += 1
            KC0v[k] += r31*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r32*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r33*(KC0e0203*r31 + KC0e0204*r32)
            k += 1
            KC0v[k] += KC0e0505*r13*r33 + r11*(KC0e0303*r31 + KC0e0304*r32) + r12*(KC0e0304*r31 + KC0e0404*r32)
            k += 1
            KC0v[k] += KC0e0505*r23*r33 + r21*(KC0e0303*r31 + KC0e0304*r32) + r22*(KC0e0304*r31 + KC0e0404*r32)
            k += 1
            KC0v[k] += KC0e0505*r33**2 + r31*(KC0e0303*r31 + KC0e0304*r32) + r32*(KC0e0304*r31 + KC0e0404*r32)
            k += 1
            KC0v[k] += r11*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r12*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r13*(KC0e0308*r31 + KC0e0408*r32)
            k += 1
            KC0v[k] += r21*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r22*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r23*(KC0e0308*r31 + KC0e0408*r32)
            k += 1
            KC0v[k] += r31*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r32*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r33*(KC0e0308*r31 + KC0e0408*r32)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r11*(KC0e0309*r31 + KC0e0409*r32) + r12*(KC0e0310*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r21*(KC0e0309*r31 + KC0e0409*r32) + r22*(KC0e0310*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r33**2 + r31*(KC0e0309*r31 + KC0e0409*r32) + r32*(KC0e0310*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += r11*(KC0e0312*r31 + KC0e0412*r32 + KC0e0512*r33) + r12*(KC0e0313*r31 + KC0e0413*r32 + KC0e0513*r33) + r13*(KC0e0314*r31 + KC0e0414*r32)
            k += 1
            KC0v[k] += r21*(KC0e0312*r31 + KC0e0412*r32 + KC0e0512*r33) + r22*(KC0e0313*r31 + KC0e0413*r32 + KC0e0513*r33) + r23*(KC0e0314*r31 + KC0e0414*r32)
            k += 1
            KC0v[k] += r31*(KC0e0312*r31 + KC0e0412*r32 + KC0e0512*r33) + r32*(KC0e0313*r31 + KC0e0413*r32 + KC0e0513*r33) + r33*(KC0e0314*r31 + KC0e0414*r32)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r11*(KC0e0315*r31 + KC0e0415*r32) + r12*(KC0e0316*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r21*(KC0e0315*r31 + KC0e0415*r32) + r22*(KC0e0316*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r33**2 + r31*(KC0e0315*r31 + KC0e0415*r32) + r32*(KC0e0316*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0208*r13**2 + r11*(KC0e0006*r11 + KC0e0007*r12) + r12*(KC0e0106*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r21*(KC0e0006*r11 + KC0e0007*r12) + r22*(KC0e0106*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r31*(KC0e0006*r11 + KC0e0007*r12) + r32*(KC0e0106*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += r11*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r12*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r13*(KC0e0506*r11 + KC0e0507*r12)
            k += 1
            KC0v[k] += r21*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r22*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r23*(KC0e0506*r11 + KC0e0507*r12)
            k += 1
            KC0v[k] += r31*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r32*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r33*(KC0e0506*r11 + KC0e0507*r12)
            k += 1
            KC0v[k] += KC0e0808*r13**2 + r11*(KC0e0606*r11 + KC0e0607*r12) + r12*(KC0e0607*r11 + KC0e0707*r12)
            k += 1
            KC0v[k] += KC0e0808*r13*r23 + r21*(KC0e0606*r11 + KC0e0607*r12) + r22*(KC0e0607*r11 + KC0e0707*r12)
            k += 1
            KC0v[k] += KC0e0808*r13*r33 + r31*(KC0e0606*r11 + KC0e0607*r12) + r32*(KC0e0607*r11 + KC0e0707*r12)
            k += 1
            KC0v[k] += r11*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r12*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r13*(KC0e0611*r11 + KC0e0711*r12)
            k += 1
            KC0v[k] += r21*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r22*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r23*(KC0e0611*r11 + KC0e0711*r12)
            k += 1
            KC0v[k] += r31*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r32*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r33*(KC0e0611*r11 + KC0e0711*r12)
            k += 1
            KC0v[k] += KC0e0814*r13**2 + r11*(KC0e0612*r11 + KC0e0712*r12) + r12*(KC0e0613*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r21*(KC0e0612*r11 + KC0e0712*r12) + r22*(KC0e0613*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r31*(KC0e0612*r11 + KC0e0712*r12) + r32*(KC0e0613*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += r11*(KC0e0615*r11 + KC0e0715*r12 + KC0e0815*r13) + r12*(KC0e0616*r11 + KC0e0716*r12 + KC0e0816*r13) + r13*(KC0e0617*r11 + KC0e0717*r12)
            k += 1
            KC0v[k] += r21*(KC0e0615*r11 + KC0e0715*r12 + KC0e0815*r13) + r22*(KC0e0616*r11 + KC0e0716*r12 + KC0e0816*r13) + r23*(KC0e0617*r11 + KC0e0717*r12)
            k += 1
            KC0v[k] += r31*(KC0e0615*r11 + KC0e0715*r12 + KC0e0815*r13) + r32*(KC0e0616*r11 + KC0e0716*r12 + KC0e0816*r13) + r33*(KC0e0617*r11 + KC0e0717*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r11*(KC0e0006*r21 + KC0e0007*r22) + r12*(KC0e0106*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23**2 + r21*(KC0e0006*r21 + KC0e0007*r22) + r22*(KC0e0106*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r31*(KC0e0006*r21 + KC0e0007*r22) + r32*(KC0e0106*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += r11*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r12*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r13*(KC0e0506*r21 + KC0e0507*r22)
            k += 1
            KC0v[k] += r21*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r22*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r23*(KC0e0506*r21 + KC0e0507*r22)
            k += 1
            KC0v[k] += r31*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r32*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r33*(KC0e0506*r21 + KC0e0507*r22)
            k += 1
            KC0v[k] += KC0e0808*r13*r23 + r11*(KC0e0606*r21 + KC0e0607*r22) + r12*(KC0e0607*r21 + KC0e0707*r22)
            k += 1
            KC0v[k] += KC0e0808*r23**2 + r21*(KC0e0606*r21 + KC0e0607*r22) + r22*(KC0e0607*r21 + KC0e0707*r22)
            k += 1
            KC0v[k] += KC0e0808*r23*r33 + r31*(KC0e0606*r21 + KC0e0607*r22) + r32*(KC0e0607*r21 + KC0e0707*r22)
            k += 1
            KC0v[k] += r11*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r12*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r13*(KC0e0611*r21 + KC0e0711*r22)
            k += 1
            KC0v[k] += r21*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r22*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r23*(KC0e0611*r21 + KC0e0711*r22)
            k += 1
            KC0v[k] += r31*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r32*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r33*(KC0e0611*r21 + KC0e0711*r22)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r11*(KC0e0612*r21 + KC0e0712*r22) + r12*(KC0e0613*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23**2 + r21*(KC0e0612*r21 + KC0e0712*r22) + r22*(KC0e0613*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r31*(KC0e0612*r21 + KC0e0712*r22) + r32*(KC0e0613*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += r11*(KC0e0615*r21 + KC0e0715*r22 + KC0e0815*r23) + r12*(KC0e0616*r21 + KC0e0716*r22 + KC0e0816*r23) + r13*(KC0e0617*r21 + KC0e0717*r22)
            k += 1
            KC0v[k] += r21*(KC0e0615*r21 + KC0e0715*r22 + KC0e0815*r23) + r22*(KC0e0616*r21 + KC0e0716*r22 + KC0e0816*r23) + r23*(KC0e0617*r21 + KC0e0717*r22)
            k += 1
            KC0v[k] += r31*(KC0e0615*r21 + KC0e0715*r22 + KC0e0815*r23) + r32*(KC0e0616*r21 + KC0e0716*r22 + KC0e0816*r23) + r33*(KC0e0617*r21 + KC0e0717*r22)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r11*(KC0e0006*r31 + KC0e0007*r32) + r12*(KC0e0106*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r21*(KC0e0006*r31 + KC0e0007*r32) + r22*(KC0e0106*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r33**2 + r31*(KC0e0006*r31 + KC0e0007*r32) + r32*(KC0e0106*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += r11*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r12*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r13*(KC0e0506*r31 + KC0e0507*r32)
            k += 1
            KC0v[k] += r21*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r22*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r23*(KC0e0506*r31 + KC0e0507*r32)
            k += 1
            KC0v[k] += r31*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r32*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r33*(KC0e0506*r31 + KC0e0507*r32)
            k += 1
            KC0v[k] += KC0e0808*r13*r33 + r11*(KC0e0606*r31 + KC0e0607*r32) + r12*(KC0e0607*r31 + KC0e0707*r32)
            k += 1
            KC0v[k] += KC0e0808*r23*r33 + r21*(KC0e0606*r31 + KC0e0607*r32) + r22*(KC0e0607*r31 + KC0e0707*r32)
            k += 1
            KC0v[k] += KC0e0808*r33**2 + r31*(KC0e0606*r31 + KC0e0607*r32) + r32*(KC0e0607*r31 + KC0e0707*r32)
            k += 1
            KC0v[k] += r11*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r12*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r13*(KC0e0611*r31 + KC0e0711*r32)
            k += 1
            KC0v[k] += r21*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r22*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r23*(KC0e0611*r31 + KC0e0711*r32)
            k += 1
            KC0v[k] += r31*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r32*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r33*(KC0e0611*r31 + KC0e0711*r32)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r11*(KC0e0612*r31 + KC0e0712*r32) + r12*(KC0e0613*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r21*(KC0e0612*r31 + KC0e0712*r32) + r22*(KC0e0613*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r33**2 + r31*(KC0e0612*r31 + KC0e0712*r32) + r32*(KC0e0613*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += r11*(KC0e0615*r31 + KC0e0715*r32 + KC0e0815*r33) + r12*(KC0e0616*r31 + KC0e0716*r32 + KC0e0816*r33) + r13*(KC0e0617*r31 + KC0e0717*r32)
            k += 1
            KC0v[k] += r21*(KC0e0615*r31 + KC0e0715*r32 + KC0e0815*r33) + r22*(KC0e0616*r31 + KC0e0716*r32 + KC0e0816*r33) + r23*(KC0e0617*r31 + KC0e0717*r32)
            k += 1
            KC0v[k] += r31*(KC0e0615*r31 + KC0e0715*r32 + KC0e0815*r33) + r32*(KC0e0616*r31 + KC0e0716*r32 + KC0e0816*r33) + r33*(KC0e0617*r31 + KC0e0717*r32)
            k += 1
            KC0v[k] += r11*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r12*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r13*(KC0e0209*r11 + KC0e0210*r12)
            k += 1
            KC0v[k] += r21*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r22*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r23*(KC0e0209*r11 + KC0e0210*r12)
            k += 1
            KC0v[k] += r31*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r32*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r33*(KC0e0209*r11 + KC0e0210*r12)
            k += 1
            KC0v[k] += KC0e0511*r13**2 + r11*(KC0e0309*r11 + KC0e0310*r12) + r12*(KC0e0409*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r21*(KC0e0309*r11 + KC0e0310*r12) + r22*(KC0e0409*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r31*(KC0e0309*r11 + KC0e0310*r12) + r32*(KC0e0409*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += r11*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r12*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r13*(KC0e0809*r11 + KC0e0810*r12)
            k += 1
            KC0v[k] += r21*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r22*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r23*(KC0e0809*r11 + KC0e0810*r12)
            k += 1
            KC0v[k] += r31*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r32*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r33*(KC0e0809*r11 + KC0e0810*r12)
            k += 1
            KC0v[k] += KC0e1111*r13**2 + r11*(KC0e0909*r11 + KC0e0910*r12) + r12*(KC0e0910*r11 + KC0e1010*r12)
            k += 1
            KC0v[k] += KC0e1111*r13*r23 + r21*(KC0e0909*r11 + KC0e0910*r12) + r22*(KC0e0910*r11 + KC0e1010*r12)
            k += 1
            KC0v[k] += KC0e1111*r13*r33 + r31*(KC0e0909*r11 + KC0e0910*r12) + r32*(KC0e0910*r11 + KC0e1010*r12)
            k += 1
            KC0v[k] += r11*(KC0e0912*r11 + KC0e1012*r12 + KC0e1112*r13) + r12*(KC0e0913*r11 + KC0e1013*r12 + KC0e1113*r13) + r13*(KC0e0914*r11 + KC0e1014*r12)
            k += 1
            KC0v[k] += r21*(KC0e0912*r11 + KC0e1012*r12 + KC0e1112*r13) + r22*(KC0e0913*r11 + KC0e1013*r12 + KC0e1113*r13) + r23*(KC0e0914*r11 + KC0e1014*r12)
            k += 1
            KC0v[k] += r31*(KC0e0912*r11 + KC0e1012*r12 + KC0e1112*r13) + r32*(KC0e0913*r11 + KC0e1013*r12 + KC0e1113*r13) + r33*(KC0e0914*r11 + KC0e1014*r12)
            k += 1
            KC0v[k] += KC0e1117*r13**2 + r11*(KC0e0915*r11 + KC0e1015*r12) + r12*(KC0e0916*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r21*(KC0e0915*r11 + KC0e1015*r12) + r22*(KC0e0916*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r31*(KC0e0915*r11 + KC0e1015*r12) + r32*(KC0e0916*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += r11*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r12*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r13*(KC0e0209*r21 + KC0e0210*r22)
            k += 1
            KC0v[k] += r21*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r22*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r23*(KC0e0209*r21 + KC0e0210*r22)
            k += 1
            KC0v[k] += r31*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r32*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r33*(KC0e0209*r21 + KC0e0210*r22)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r11*(KC0e0309*r21 + KC0e0310*r22) + r12*(KC0e0409*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23**2 + r21*(KC0e0309*r21 + KC0e0310*r22) + r22*(KC0e0409*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r31*(KC0e0309*r21 + KC0e0310*r22) + r32*(KC0e0409*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += r11*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r12*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r13*(KC0e0809*r21 + KC0e0810*r22)
            k += 1
            KC0v[k] += r21*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r22*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r23*(KC0e0809*r21 + KC0e0810*r22)
            k += 1
            KC0v[k] += r31*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r32*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r33*(KC0e0809*r21 + KC0e0810*r22)
            k += 1
            KC0v[k] += KC0e1111*r13*r23 + r11*(KC0e0909*r21 + KC0e0910*r22) + r12*(KC0e0910*r21 + KC0e1010*r22)
            k += 1
            KC0v[k] += KC0e1111*r23**2 + r21*(KC0e0909*r21 + KC0e0910*r22) + r22*(KC0e0910*r21 + KC0e1010*r22)
            k += 1
            KC0v[k] += KC0e1111*r23*r33 + r31*(KC0e0909*r21 + KC0e0910*r22) + r32*(KC0e0910*r21 + KC0e1010*r22)
            k += 1
            KC0v[k] += r11*(KC0e0912*r21 + KC0e1012*r22 + KC0e1112*r23) + r12*(KC0e0913*r21 + KC0e1013*r22 + KC0e1113*r23) + r13*(KC0e0914*r21 + KC0e1014*r22)
            k += 1
            KC0v[k] += r21*(KC0e0912*r21 + KC0e1012*r22 + KC0e1112*r23) + r22*(KC0e0913*r21 + KC0e1013*r22 + KC0e1113*r23) + r23*(KC0e0914*r21 + KC0e1014*r22)
            k += 1
            KC0v[k] += r31*(KC0e0912*r21 + KC0e1012*r22 + KC0e1112*r23) + r32*(KC0e0913*r21 + KC0e1013*r22 + KC0e1113*r23) + r33*(KC0e0914*r21 + KC0e1014*r22)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r11*(KC0e0915*r21 + KC0e1015*r22) + r12*(KC0e0916*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23**2 + r21*(KC0e0915*r21 + KC0e1015*r22) + r22*(KC0e0916*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r31*(KC0e0915*r21 + KC0e1015*r22) + r32*(KC0e0916*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += r11*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r12*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r13*(KC0e0209*r31 + KC0e0210*r32)
            k += 1
            KC0v[k] += r21*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r22*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r23*(KC0e0209*r31 + KC0e0210*r32)
            k += 1
            KC0v[k] += r31*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r32*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r33*(KC0e0209*r31 + KC0e0210*r32)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r11*(KC0e0309*r31 + KC0e0310*r32) + r12*(KC0e0409*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r21*(KC0e0309*r31 + KC0e0310*r32) + r22*(KC0e0409*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r33**2 + r31*(KC0e0309*r31 + KC0e0310*r32) + r32*(KC0e0409*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += r11*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r12*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r13*(KC0e0809*r31 + KC0e0810*r32)
            k += 1
            KC0v[k] += r21*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r22*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r23*(KC0e0809*r31 + KC0e0810*r32)
            k += 1
            KC0v[k] += r31*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r32*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r33*(KC0e0809*r31 + KC0e0810*r32)
            k += 1
            KC0v[k] += KC0e1111*r13*r33 + r11*(KC0e0909*r31 + KC0e0910*r32) + r12*(KC0e0910*r31 + KC0e1010*r32)
            k += 1
            KC0v[k] += KC0e1111*r23*r33 + r21*(KC0e0909*r31 + KC0e0910*r32) + r22*(KC0e0910*r31 + KC0e1010*r32)
            k += 1
            KC0v[k] += KC0e1111*r33**2 + r31*(KC0e0909*r31 + KC0e0910*r32) + r32*(KC0e0910*r31 + KC0e1010*r32)
            k += 1
            KC0v[k] += r11*(KC0e0912*r31 + KC0e1012*r32 + KC0e1112*r33) + r12*(KC0e0913*r31 + KC0e1013*r32 + KC0e1113*r33) + r13*(KC0e0914*r31 + KC0e1014*r32)
            k += 1
            KC0v[k] += r21*(KC0e0912*r31 + KC0e1012*r32 + KC0e1112*r33) + r22*(KC0e0913*r31 + KC0e1013*r32 + KC0e1113*r33) + r23*(KC0e0914*r31 + KC0e1014*r32)
            k += 1
            KC0v[k] += r31*(KC0e0912*r31 + KC0e1012*r32 + KC0e1112*r33) + r32*(KC0e0913*r31 + KC0e1013*r32 + KC0e1113*r33) + r33*(KC0e0914*r31 + KC0e1014*r32)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r11*(KC0e0915*r31 + KC0e1015*r32) + r12*(KC0e0916*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r21*(KC0e0915*r31 + KC0e1015*r32) + r22*(KC0e0916*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r33**2 + r31*(KC0e0915*r31 + KC0e1015*r32) + r32*(KC0e0916*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e0214*r13**2 + r11*(KC0e0012*r11 + KC0e0013*r12) + r12*(KC0e0112*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r21*(KC0e0012*r11 + KC0e0013*r12) + r22*(KC0e0112*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r31*(KC0e0012*r11 + KC0e0013*r12) + r32*(KC0e0112*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += r11*(KC0e0312*r11 + KC0e0313*r12 + KC0e0314*r13) + r12*(KC0e0412*r11 + KC0e0413*r12 + KC0e0414*r13) + r13*(KC0e0512*r11 + KC0e0513*r12)
            k += 1
            KC0v[k] += r21*(KC0e0312*r11 + KC0e0313*r12 + KC0e0314*r13) + r22*(KC0e0412*r11 + KC0e0413*r12 + KC0e0414*r13) + r23*(KC0e0512*r11 + KC0e0513*r12)
            k += 1
            KC0v[k] += r31*(KC0e0312*r11 + KC0e0313*r12 + KC0e0314*r13) + r32*(KC0e0412*r11 + KC0e0413*r12 + KC0e0414*r13) + r33*(KC0e0512*r11 + KC0e0513*r12)
            k += 1
            KC0v[k] += KC0e0814*r13**2 + r11*(KC0e0612*r11 + KC0e0613*r12) + r12*(KC0e0712*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r21*(KC0e0612*r11 + KC0e0613*r12) + r22*(KC0e0712*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r31*(KC0e0612*r11 + KC0e0613*r12) + r32*(KC0e0712*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += r11*(KC0e0912*r11 + KC0e0913*r12 + KC0e0914*r13) + r12*(KC0e1012*r11 + KC0e1013*r12 + KC0e1014*r13) + r13*(KC0e1112*r11 + KC0e1113*r12)
            k += 1
            KC0v[k] += r21*(KC0e0912*r11 + KC0e0913*r12 + KC0e0914*r13) + r22*(KC0e1012*r11 + KC0e1013*r12 + KC0e1014*r13) + r23*(KC0e1112*r11 + KC0e1113*r12)
            k += 1
            KC0v[k] += r31*(KC0e0912*r11 + KC0e0913*r12 + KC0e0914*r13) + r32*(KC0e1012*r11 + KC0e1013*r12 + KC0e1014*r13) + r33*(KC0e1112*r11 + KC0e1113*r12)
            k += 1
            KC0v[k] += KC0e1414*r13**2 + r11*(KC0e1212*r11 + KC0e1213*r12) + r12*(KC0e1213*r11 + KC0e1313*r12)
            k += 1
            KC0v[k] += KC0e1414*r13*r23 + r21*(KC0e1212*r11 + KC0e1213*r12) + r22*(KC0e1213*r11 + KC0e1313*r12)
            k += 1
            KC0v[k] += KC0e1414*r13*r33 + r31*(KC0e1212*r11 + KC0e1213*r12) + r32*(KC0e1213*r11 + KC0e1313*r12)
            k += 1
            KC0v[k] += r11*(KC0e1215*r11 + KC0e1315*r12 + KC0e1415*r13) + r12*(KC0e1216*r11 + KC0e1316*r12 + KC0e1416*r13) + r13*(KC0e1217*r11 + KC0e1317*r12)
            k += 1
            KC0v[k] += r21*(KC0e1215*r11 + KC0e1315*r12 + KC0e1415*r13) + r22*(KC0e1216*r11 + KC0e1316*r12 + KC0e1416*r13) + r23*(KC0e1217*r11 + KC0e1317*r12)
            k += 1
            KC0v[k] += r31*(KC0e1215*r11 + KC0e1315*r12 + KC0e1415*r13) + r32*(KC0e1216*r11 + KC0e1316*r12 + KC0e1416*r13) + r33*(KC0e1217*r11 + KC0e1317*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r11*(KC0e0012*r21 + KC0e0013*r22) + r12*(KC0e0112*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23**2 + r21*(KC0e0012*r21 + KC0e0013*r22) + r22*(KC0e0112*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r31*(KC0e0012*r21 + KC0e0013*r22) + r32*(KC0e0112*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += r11*(KC0e0312*r21 + KC0e0313*r22 + KC0e0314*r23) + r12*(KC0e0412*r21 + KC0e0413*r22 + KC0e0414*r23) + r13*(KC0e0512*r21 + KC0e0513*r22)
            k += 1
            KC0v[k] += r21*(KC0e0312*r21 + KC0e0313*r22 + KC0e0314*r23) + r22*(KC0e0412*r21 + KC0e0413*r22 + KC0e0414*r23) + r23*(KC0e0512*r21 + KC0e0513*r22)
            k += 1
            KC0v[k] += r31*(KC0e0312*r21 + KC0e0313*r22 + KC0e0314*r23) + r32*(KC0e0412*r21 + KC0e0413*r22 + KC0e0414*r23) + r33*(KC0e0512*r21 + KC0e0513*r22)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r11*(KC0e0612*r21 + KC0e0613*r22) + r12*(KC0e0712*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23**2 + r21*(KC0e0612*r21 + KC0e0613*r22) + r22*(KC0e0712*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r31*(KC0e0612*r21 + KC0e0613*r22) + r32*(KC0e0712*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += r11*(KC0e0912*r21 + KC0e0913*r22 + KC0e0914*r23) + r12*(KC0e1012*r21 + KC0e1013*r22 + KC0e1014*r23) + r13*(KC0e1112*r21 + KC0e1113*r22)
            k += 1
            KC0v[k] += r21*(KC0e0912*r21 + KC0e0913*r22 + KC0e0914*r23) + r22*(KC0e1012*r21 + KC0e1013*r22 + KC0e1014*r23) + r23*(KC0e1112*r21 + KC0e1113*r22)
            k += 1
            KC0v[k] += r31*(KC0e0912*r21 + KC0e0913*r22 + KC0e0914*r23) + r32*(KC0e1012*r21 + KC0e1013*r22 + KC0e1014*r23) + r33*(KC0e1112*r21 + KC0e1113*r22)
            k += 1
            KC0v[k] += KC0e1414*r13*r23 + r11*(KC0e1212*r21 + KC0e1213*r22) + r12*(KC0e1213*r21 + KC0e1313*r22)
            k += 1
            KC0v[k] += KC0e1414*r23**2 + r21*(KC0e1212*r21 + KC0e1213*r22) + r22*(KC0e1213*r21 + KC0e1313*r22)
            k += 1
            KC0v[k] += KC0e1414*r23*r33 + r31*(KC0e1212*r21 + KC0e1213*r22) + r32*(KC0e1213*r21 + KC0e1313*r22)
            k += 1
            KC0v[k] += r11*(KC0e1215*r21 + KC0e1315*r22 + KC0e1415*r23) + r12*(KC0e1216*r21 + KC0e1316*r22 + KC0e1416*r23) + r13*(KC0e1217*r21 + KC0e1317*r22)
            k += 1
            KC0v[k] += r21*(KC0e1215*r21 + KC0e1315*r22 + KC0e1415*r23) + r22*(KC0e1216*r21 + KC0e1316*r22 + KC0e1416*r23) + r23*(KC0e1217*r21 + KC0e1317*r22)
            k += 1
            KC0v[k] += r31*(KC0e1215*r21 + KC0e1315*r22 + KC0e1415*r23) + r32*(KC0e1216*r21 + KC0e1316*r22 + KC0e1416*r23) + r33*(KC0e1217*r21 + KC0e1317*r22)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r11*(KC0e0012*r31 + KC0e0013*r32) + r12*(KC0e0112*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r21*(KC0e0012*r31 + KC0e0013*r32) + r22*(KC0e0112*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r33**2 + r31*(KC0e0012*r31 + KC0e0013*r32) + r32*(KC0e0112*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += r11*(KC0e0312*r31 + KC0e0313*r32 + KC0e0314*r33) + r12*(KC0e0412*r31 + KC0e0413*r32 + KC0e0414*r33) + r13*(KC0e0512*r31 + KC0e0513*r32)
            k += 1
            KC0v[k] += r21*(KC0e0312*r31 + KC0e0313*r32 + KC0e0314*r33) + r22*(KC0e0412*r31 + KC0e0413*r32 + KC0e0414*r33) + r23*(KC0e0512*r31 + KC0e0513*r32)
            k += 1
            KC0v[k] += r31*(KC0e0312*r31 + KC0e0313*r32 + KC0e0314*r33) + r32*(KC0e0412*r31 + KC0e0413*r32 + KC0e0414*r33) + r33*(KC0e0512*r31 + KC0e0513*r32)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r11*(KC0e0612*r31 + KC0e0613*r32) + r12*(KC0e0712*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r21*(KC0e0612*r31 + KC0e0613*r32) + r22*(KC0e0712*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r33**2 + r31*(KC0e0612*r31 + KC0e0613*r32) + r32*(KC0e0712*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += r11*(KC0e0912*r31 + KC0e0913*r32 + KC0e0914*r33) + r12*(KC0e1012*r31 + KC0e1013*r32 + KC0e1014*r33) + r13*(KC0e1112*r31 + KC0e1113*r32)
            k += 1
            KC0v[k] += r21*(KC0e0912*r31 + KC0e0913*r32 + KC0e0914*r33) + r22*(KC0e1012*r31 + KC0e1013*r32 + KC0e1014*r33) + r23*(KC0e1112*r31 + KC0e1113*r32)
            k += 1
            KC0v[k] += r31*(KC0e0912*r31 + KC0e0913*r32 + KC0e0914*r33) + r32*(KC0e1012*r31 + KC0e1013*r32 + KC0e1014*r33) + r33*(KC0e1112*r31 + KC0e1113*r32)
            k += 1
            KC0v[k] += KC0e1414*r13*r33 + r11*(KC0e1212*r31 + KC0e1213*r32) + r12*(KC0e1213*r31 + KC0e1313*r32)
            k += 1
            KC0v[k] += KC0e1414*r23*r33 + r21*(KC0e1212*r31 + KC0e1213*r32) + r22*(KC0e1213*r31 + KC0e1313*r32)
            k += 1
            KC0v[k] += KC0e1414*r33**2 + r31*(KC0e1212*r31 + KC0e1213*r32) + r32*(KC0e1213*r31 + KC0e1313*r32)
            k += 1
            KC0v[k] += r11*(KC0e1215*r31 + KC0e1315*r32 + KC0e1415*r33) + r12*(KC0e1216*r31 + KC0e1316*r32 + KC0e1416*r33) + r13*(KC0e1217*r31 + KC0e1317*r32)
            k += 1
            KC0v[k] += r21*(KC0e1215*r31 + KC0e1315*r32 + KC0e1415*r33) + r22*(KC0e1216*r31 + KC0e1316*r32 + KC0e1416*r33) + r23*(KC0e1217*r31 + KC0e1317*r32)
            k += 1
            KC0v[k] += r31*(KC0e1215*r31 + KC0e1315*r32 + KC0e1415*r33) + r32*(KC0e1216*r31 + KC0e1316*r32 + KC0e1416*r33) + r33*(KC0e1217*r31 + KC0e1317*r32)
            k += 1
            KC0v[k] += r11*(KC0e0015*r11 + KC0e0016*r12 + KC0e0017*r13) + r12*(KC0e0115*r11 + KC0e0116*r12 + KC0e0117*r13) + r13*(KC0e0215*r11 + KC0e0216*r12)
            k += 1
            KC0v[k] += r21*(KC0e0015*r11 + KC0e0016*r12 + KC0e0017*r13) + r22*(KC0e0115*r11 + KC0e0116*r12 + KC0e0117*r13) + r23*(KC0e0215*r11 + KC0e0216*r12)
            k += 1
            KC0v[k] += r31*(KC0e0015*r11 + KC0e0016*r12 + KC0e0017*r13) + r32*(KC0e0115*r11 + KC0e0116*r12 + KC0e0117*r13) + r33*(KC0e0215*r11 + KC0e0216*r12)
            k += 1
            KC0v[k] += KC0e0517*r13**2 + r11*(KC0e0315*r11 + KC0e0316*r12) + r12*(KC0e0415*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r21*(KC0e0315*r11 + KC0e0316*r12) + r22*(KC0e0415*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r31*(KC0e0315*r11 + KC0e0316*r12) + r32*(KC0e0415*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += r11*(KC0e0615*r11 + KC0e0616*r12 + KC0e0617*r13) + r12*(KC0e0715*r11 + KC0e0716*r12 + KC0e0717*r13) + r13*(KC0e0815*r11 + KC0e0816*r12)
            k += 1
            KC0v[k] += r21*(KC0e0615*r11 + KC0e0616*r12 + KC0e0617*r13) + r22*(KC0e0715*r11 + KC0e0716*r12 + KC0e0717*r13) + r23*(KC0e0815*r11 + KC0e0816*r12)
            k += 1
            KC0v[k] += r31*(KC0e0615*r11 + KC0e0616*r12 + KC0e0617*r13) + r32*(KC0e0715*r11 + KC0e0716*r12 + KC0e0717*r13) + r33*(KC0e0815*r11 + KC0e0816*r12)
            k += 1
            KC0v[k] += KC0e1117*r13**2 + r11*(KC0e0915*r11 + KC0e0916*r12) + r12*(KC0e1015*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r21*(KC0e0915*r11 + KC0e0916*r12) + r22*(KC0e1015*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r31*(KC0e0915*r11 + KC0e0916*r12) + r32*(KC0e1015*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += r11*(KC0e1215*r11 + KC0e1216*r12 + KC0e1217*r13) + r12*(KC0e1315*r11 + KC0e1316*r12 + KC0e1317*r13) + r13*(KC0e1415*r11 + KC0e1416*r12)
            k += 1
            KC0v[k] += r21*(KC0e1215*r11 + KC0e1216*r12 + KC0e1217*r13) + r22*(KC0e1315*r11 + KC0e1316*r12 + KC0e1317*r13) + r23*(KC0e1415*r11 + KC0e1416*r12)
            k += 1
            KC0v[k] += r31*(KC0e1215*r11 + KC0e1216*r12 + KC0e1217*r13) + r32*(KC0e1315*r11 + KC0e1316*r12 + KC0e1317*r13) + r33*(KC0e1415*r11 + KC0e1416*r12)
            k += 1
            KC0v[k] += KC0e1717*r13**2 + r11*(KC0e1515*r11 + KC0e1516*r12) + r12*(KC0e1516*r11 + KC0e1616*r12)
            k += 1
            KC0v[k] += KC0e1717*r13*r23 + r21*(KC0e1515*r11 + KC0e1516*r12) + r22*(KC0e1516*r11 + KC0e1616*r12)
            k += 1
            KC0v[k] += KC0e1717*r13*r33 + r31*(KC0e1515*r11 + KC0e1516*r12) + r32*(KC0e1516*r11 + KC0e1616*r12)
            k += 1
            KC0v[k] += r11*(KC0e0015*r21 + KC0e0016*r22 + KC0e0017*r23) + r12*(KC0e0115*r21 + KC0e0116*r22 + KC0e0117*r23) + r13*(KC0e0215*r21 + KC0e0216*r22)
            k += 1
            KC0v[k] += r21*(KC0e0015*r21 + KC0e0016*r22 + KC0e0017*r23) + r22*(KC0e0115*r21 + KC0e0116*r22 + KC0e0117*r23) + r23*(KC0e0215*r21 + KC0e0216*r22)
            k += 1
            KC0v[k] += r31*(KC0e0015*r21 + KC0e0016*r22 + KC0e0017*r23) + r32*(KC0e0115*r21 + KC0e0116*r22 + KC0e0117*r23) + r33*(KC0e0215*r21 + KC0e0216*r22)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r11*(KC0e0315*r21 + KC0e0316*r22) + r12*(KC0e0415*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23**2 + r21*(KC0e0315*r21 + KC0e0316*r22) + r22*(KC0e0415*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r31*(KC0e0315*r21 + KC0e0316*r22) + r32*(KC0e0415*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += r11*(KC0e0615*r21 + KC0e0616*r22 + KC0e0617*r23) + r12*(KC0e0715*r21 + KC0e0716*r22 + KC0e0717*r23) + r13*(KC0e0815*r21 + KC0e0816*r22)
            k += 1
            KC0v[k] += r21*(KC0e0615*r21 + KC0e0616*r22 + KC0e0617*r23) + r22*(KC0e0715*r21 + KC0e0716*r22 + KC0e0717*r23) + r23*(KC0e0815*r21 + KC0e0816*r22)
            k += 1
            KC0v[k] += r31*(KC0e0615*r21 + KC0e0616*r22 + KC0e0617*r23) + r32*(KC0e0715*r21 + KC0e0716*r22 + KC0e0717*r23) + r33*(KC0e0815*r21 + KC0e0816*r22)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r11*(KC0e0915*r21 + KC0e0916*r22) + r12*(KC0e1015*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23**2 + r21*(KC0e0915*r21 + KC0e0916*r22) + r22*(KC0e1015*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r31*(KC0e0915*r21 + KC0e0916*r22) + r32*(KC0e1015*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += r11*(KC0e1215*r21 + KC0e1216*r22 + KC0e1217*r23) + r12*(KC0e1315*r21 + KC0e1316*r22 + KC0e1317*r23) + r13*(KC0e1415*r21 + KC0e1416*r22)
            k += 1
            KC0v[k] += r21*(KC0e1215*r21 + KC0e1216*r22 + KC0e1217*r23) + r22*(KC0e1315*r21 + KC0e1316*r22 + KC0e1317*r23) + r23*(KC0e1415*r21 + KC0e1416*r22)
            k += 1
            KC0v[k] += r31*(KC0e1215*r21 + KC0e1216*r22 + KC0e1217*r23) + r32*(KC0e1315*r21 + KC0e1316*r22 + KC0e1317*r23) + r33*(KC0e1415*r21 + KC0e1416*r22)
            k += 1
            KC0v[k] += KC0e1717*r13*r23 + r11*(KC0e1515*r21 + KC0e1516*r22) + r12*(KC0e1516*r21 + KC0e1616*r22)
            k += 1
            KC0v[k] += KC0e1717*r23**2 + r21*(KC0e1515*r21 + KC0e1516*r22) + r22*(KC0e1516*r21 + KC0e1616*r22)
            k += 1
            KC0v[k] += KC0e1717*r23*r33 + r31*(KC0e1515*r21 + KC0e1516*r22) + r32*(KC0e1516*r21 + KC0e1616*r22)
            k += 1
            KC0v[k] += r11*(KC0e0015*r31 + KC0e0016*r32 + KC0e0017*r33) + r12*(KC0e0115*r31 + KC0e0116*r32 + KC0e0117*r33) + r13*(KC0e0215*r31 + KC0e0216*r32)
            k += 1
            KC0v[k] += r21*(KC0e0015*r31 + KC0e0016*r32 + KC0e0017*r33) + r22*(KC0e0115*r31 + KC0e0116*r32 + KC0e0117*r33) + r23*(KC0e0215*r31 + KC0e0216*r32)
            k += 1
            KC0v[k] += r31*(KC0e0015*r31 + KC0e0016*r32 + KC0e0017*r33) + r32*(KC0e0115*r31 + KC0e0116*r32 + KC0e0117*r33) + r33*(KC0e0215*r31 + KC0e0216*r32)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r11*(KC0e0315*r31 + KC0e0316*r32) + r12*(KC0e0415*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r21*(KC0e0315*r31 + KC0e0316*r32) + r22*(KC0e0415*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r33**2 + r31*(KC0e0315*r31 + KC0e0316*r32) + r32*(KC0e0415*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += r11*(KC0e0615*r31 + KC0e0616*r32 + KC0e0617*r33) + r12*(KC0e0715*r31 + KC0e0716*r32 + KC0e0717*r33) + r13*(KC0e0815*r31 + KC0e0816*r32)
            k += 1
            KC0v[k] += r21*(KC0e0615*r31 + KC0e0616*r32 + KC0e0617*r33) + r22*(KC0e0715*r31 + KC0e0716*r32 + KC0e0717*r33) + r23*(KC0e0815*r31 + KC0e0816*r32)
            k += 1
            KC0v[k] += r31*(KC0e0615*r31 + KC0e0616*r32 + KC0e0617*r33) + r32*(KC0e0715*r31 + KC0e0716*r32 + KC0e0717*r33) + r33*(KC0e0815*r31 + KC0e0816*r32)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r11*(KC0e0915*r31 + KC0e0916*r32) + r12*(KC0e1015*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r21*(KC0e0915*r31 + KC0e0916*r32) + r22*(KC0e1015*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r33**2 + r31*(KC0e0915*r31 + KC0e0916*r32) + r32*(KC0e1015*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += r11*(KC0e1215*r31 + KC0e1216*r32 + KC0e1217*r33) + r12*(KC0e1315*r31 + KC0e1316*r32 + KC0e1317*r33) + r13*(KC0e1415*r31 + KC0e1416*r32)
            k += 1
            KC0v[k] += r21*(KC0e1215*r31 + KC0e1216*r32 + KC0e1217*r33) + r22*(KC0e1315*r31 + KC0e1316*r32 + KC0e1317*r33) + r23*(KC0e1415*r31 + KC0e1416*r32)
            k += 1
            KC0v[k] += r31*(KC0e1215*r31 + KC0e1216*r32 + KC0e1217*r33) + r32*(KC0e1315*r31 + KC0e1316*r32 + KC0e1317*r33) + r33*(KC0e1415*r31 + KC0e1416*r32)
            k += 1
            KC0v[k] += KC0e1717*r13*r33 + r11*(KC0e1515*r31 + KC0e1516*r32) + r12*(KC0e1516*r31 + KC0e1616*r32)
            k += 1
            KC0v[k] += KC0e1717*r23*r33 + r21*(KC0e1515*r31 + KC0e1516*r32) + r22*(KC0e1516*r31 + KC0e1616*r32)
            k += 1
            KC0v[k] += KC0e1717*r33**2 + r31*(KC0e1515*r31 + KC0e1516*r32) + r32*(KC0e1516*r31 + KC0e1616*r32)

            # NOTE contribution of the physics-based drilling stiffness,
            #      added here rather than through the KC0e terms above
            #      because the enrichment couples the drilling rotation to
            #      the curvatures whenever the extension-bending matrix B is
            #      not zero, and the generated block has no variable for
            #      those. The sparse pattern is the full 18 by 18 element
            #      matrix stored row by row, so the entry of the element
            #      degrees-of-freedom i and j sits at init_k_KC0 + 18*i + j
            if self.drilling_model == 0:
                if self.gamma_rz >= 0.:
                    gamma_drill = self.gamma_rz
                else:
                    gamma_drill = A66
                Nxv[0] = N1x
                Nxv[1] = N2x
                Nxv[2] = N3x
                Nyv[0] = N1y
                Nyv[1] = N2y
                Nyv[2] = N3y
                allman_delta_Ke(Nxv, Nyv, &self.probe.xe[0], self.area,
                                Ae, Be, gamma_drill, Kedelta)
                rmat[0] = r11
                rmat[1] = r12
                rmat[2] = r13
                rmat[3] = r21
                rmat[4] = r22
                rmat[5] = r23
                rmat[6] = r31
                rmat[7] = r32
                rmat[8] = r33
                # NOTE from element to global coordinates, Kg_{mn} = r_{mi}
                #      Ke_{ij} r_{nj}, with the rotation block diagonal over
                #      the six groups of three components of the 18
                #      degrees-of-freedom
                for bi in range(6):
                    for bj in range(6):
                        for qi in range(3):
                            for qj in range(3):
                                tmp = 0.
                                for qa in range(3):
                                    for qb in range(3):
                                        tmp += (rmat[3*qi + qa]
                                                *Kedelta[18*(3*bi + qa) + 3*bj + qb]
                                                *rmat[3*qj + qb])
                                KC0v[self.init_k_KC0 + 18*(3*bi + qi)
                                     + 3*bj + qj] += tmp


    cpdef void update_fint(Tria3R self,
                           double [::1] fint,
                           ShellProp prop,
                           int nonlinear=0):
        r"""Update the internal force vector

        The element internal forces, computed in element coordinates with
        :meth:`.update_probe_finte`, are rotated to global coordinates and
        added to ``fint`` at the positions of the nodes, ``c1``, ``c2`` and
        ``c3``. Before this method is called, :meth:`.update_probe_xe` and
        :meth:`.update_probe_ue` must have been used.

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.Tria3RProbe` with the
            internal forces in local coordinates.
            1-D array of size ``DOF*M``, with forces for the translations and
            moments (force times length) for the rotations.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Karman strains,
            see :meth:`.update_KCNL`.

        Notes
        -----
        With ``nonlinear=1`` the exact Jacobian of the internal forces is
        ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

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
            fint[1+self.c2] += finte[6]*self.r21 + finte[7]*self.r22 + finte[8]*self.r23
            fint[0+self.c2] += finte[6]*self.r11 + finte[7]*self.r12 + finte[8]*self.r13
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


    cdef double _update_probe_BL_G(Tria3R self) noexcept nogil:
        r"""Update the probe rows of the linear strains and of the gradient of `w`

        The rows are constant within the element and are given in element coordinates:

        - ``BLexx, BLeyy, BLgxy``: membrane strains `\epsilon_{xx}, \epsilon_{yy},
          \gamma_{xy}`
        - ``BLkxx, BLkyy, BLkxy``: curvatures `\kappa_{xx}, \kappa_{yy},
          \kappa_{xy}`
        - ``Sw_x, Sw_y``: `\boldsymbol{G}_x` and `\boldsymbol{G}_y`, the rows of `w_{,x}` and `w_{,y}`

        With ``drilling_model = 0`` the membrane rows also carry, in their
        drilling columns, the Allman enrichment evaluated at the centroid,
        see :func:`.allman_enrichment`. The curvature rows and the rows of
        the gradient of `w` are not enriched. The ``area`` attribute must be
        up to date, see :meth:`.update_area`.

        Returns
        -------
        detJ : double
            Determinant of the Jacobian matrix, ``2*area``.

        Notes
        -----
        The rows are

        .. math::

            \epsilon_{xx} = u_{,x}, \quad
            \epsilon_{yy} = v_{,y}, \quad
            \gamma_{xy} = u_{,y} + v_{,x}, \quad
            \kappa_{xx} = {r_y}_{,x}, \quad
            \kappa_{yy} = -{r_x}_{,y}, \quad
            \kappa_{xy} = {r_y}_{,y} - {r_x}_{,x}

        with the constant derivatives of the linear shape functions,
        `S_{1,x} = (y_2 - y_3)/(2 A_e)`, `S_{1,y} = (x_3 - x_2)/(2 A_e)` and
        their cyclic permutations, `A_e` being ``area``. The Allman
        enrichment keeps the nonlinear tangent of :meth:`.update_KCNL` and
        the nonlinear internal forces consistent with the enriched KC0. The
        drilling columns are linear over the element, so evaluating them at
        the centroid integrates them exactly with the single point that those
        methods use.

        """
        cdef int i
        cdef double x1, x2, x3, y1, y2, y3
        cdef double N1x, N2x, N3x, N1y, N2y, N3y
        cdef double Nxv[3]
        cdef double Nyv[3]
        cdef double d[12]
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        # NOTE ignoring z in local coordinates
        x1 = self.probe.xe[0]
        y1 = self.probe.xe[1]
        x2 = self.probe.xe[3]
        y2 = self.probe.xe[4]
        x3 = self.probe.xe[6]
        y3 = self.probe.xe[7]

        N1x = (y2 - y3)/(2*self.area)
        N2x = (-y1 + y3)/(2*self.area)
        N3x = (y1 - y2)/(2*self.area)
        N1y = (-x2 + x3)/(2*self.area)
        N2y = (x1 - x3)/(2*self.area)
        N3y = (-x1 + x2)/(2*self.area)

        for i in range(18):
            BLexx[i] = 0.
            BLeyy[i] = 0.
            BLgxy[i] = 0.
            BLkxx[i] = 0.
            BLkyy[i] = 0.
            BLkxy[i] = 0.
            Sw_x[i] = 0.
            Sw_y[i] = 0.

        # exx = u,x
        BLexx[0] = N1x
        BLexx[6] = N2x
        BLexx[12] = N3x

        # eyy = v,y
        BLeyy[1] = N1y
        BLeyy[7] = N2y
        BLeyy[13] = N3y

        # gxy = u,y + v,x
        BLgxy[0] = N1y
        BLgxy[6] = N2y
        BLgxy[12] = N3y
        BLgxy[1] = N1x
        BLgxy[7] = N2x
        BLgxy[13] = N3x

        # kxx = ry,x
        BLkxx[4] = N1x
        BLkxx[10] = N2x
        BLkxx[16] = N3x

        # kyy = -rx,y
        BLkyy[3] = -N1y
        BLkyy[9] = -N2y
        BLkyy[15] = -N3y

        # kxy = ry,y - rx,x
        BLkxy[3] = -N1x
        BLkxy[9] = -N2x
        BLkxy[15] = -N3x
        BLkxy[4] = N1y
        BLkxy[10] = N2y
        BLkxy[16] = N3y

        # w,x
        Sw_x[2] = N1x
        Sw_x[8] = N2x
        Sw_x[14] = N3x

        # w,y
        Sw_y[2] = N1y
        Sw_y[8] = N2y
        Sw_y[14] = N3y

        # NOTE Allman enrichment of the membrane rows, which keeps the
        #      nonlinear tangent of update_KCNL and the nonlinear internal
        #      forces consistent with the enriched KC0. The drilling columns
        #      are linear over the element, so evaluating them at the
        #      centroid integrates them exactly with the single point that
        #      those methods use. The curvature rows and the gradient of w
        #      are not enriched, the edge modes acting only on the in-plane
        #      translations
        if self.drilling_model == 0:
            Nxv[0] = N1x
            Nxv[1] = N2x
            Nxv[2] = N3x
            Nyv[0] = N1y
            Nyv[1] = N2y
            Nyv[2] = N3y
            allman_enrichment(Nxv, Nyv, &self.probe.xe[0], 1./3., 1./3.,
                              1./3., d)
            for i in range(3):
                BLexx[6*i + 5] = d[i]
                BLeyy[6*i + 5] = d[9 + i]
                BLgxy[6*i + 5] = d[3 + i] + d[6 + i]

        return 2*self.area


    cdef void _update_probe_KCNLve(Tria3R self, ShellProp prop) noexcept nogil:
        r"""Update the probe values of the nonlinear constitutive stiffness matrix

        The attribute ``KCNLve`` of the :class:`.Tria3RProbe` is updated with
        the nonlinear constitutive stiffness matrix in element coordinates,
        stored row by row and evaluated at the displacements ``ue`` of the
        probe. See :meth:`.update_KCNL`.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, with the terms given in
        :meth:`.update_KCNL`, built from the rows of
        :meth:`._update_probe_BL_G` and integrated with one point, as in
        :meth:`.update_KG`, the strains being constant within the element.

        """
        cdef int i, j, a

        cdef double wij, detJ, w_x, w_y
        cdef double A[9]
        cdef double B[9]
        cdef double NNL[3]
        # NOTE products stored row by row, with 3 rows and NUM_NODES*DOF columns
        cdef double ABL[54]
        cdef double BNL[54]
        cdef double ABNL[54]
        cdef double *ue
        cdef double *KCNLve
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        ue = &self.probe.ue[0]
        KCNLve = &self.probe.KCNLve[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, A, B, NULL, NULL)

        for i in range(18*18):
            KCNLve[i] = 0.

        # NOTE the strains are constant within the element, one integration point as
        #     in update_KG
        wij = 0.5
        detJ = self._update_probe_BL_G()

        w_x = 0.
        w_y = 0.
        for i in range(18):
            w_x += Sw_x[i]*ue[i]
            w_y += Sw_y[i]*ue[i]

        # stress resultants of the nonlinear membrane strain,
        # epsNL = {w_x**2/2, w_y**2/2, w_x*w_y}
        for a in range(3):
            NNL[a] = A[3*a]*w_x*w_x/2. + A[3*a + 1]*w_y*w_y/2. + A[3*a + 2]*w_x*w_y

        # BNL, the variation of the nonlinear membrane strain, and the products
        # A*Bm + B*Bb and A*BNL, all stored row by row with 3 rows
        for i in range(18):
            BNL[i] = w_x*Sw_x[i]
            BNL[18 + i] = w_y*Sw_y[i]
            BNL[36 + i] = w_x*Sw_y[i] + w_y*Sw_x[i]
            for a in range(3):
                ABL[18*a + i] = (A[3*a]*BLexx[i] + A[3*a + 1]*BLeyy[i] + A[3*a + 2]*BLgxy[i]
                               + B[3*a]*BLkxx[i] + B[3*a + 1]*BLkyy[i] + B[3*a + 2]*BLkxy[i])
                ABNL[18*a + i] = A[3*a]*BNL[i] + A[3*a + 1]*BNL[18 + i] + A[3*a + 2]*BNL[36 + i]

        for i in range(18):
            for j in range(18):
                KCNLve[18*i + j] += wij*detJ*(
                    # KC0L = (Bm.T*A + Bb.T*B)*BNL
                      ABL[i]*BNL[j] + ABL[18 + i]*BNL[18 + j] + ABL[36 + i]*BNL[36 + j]
                    # KCL0 = BNL.T*(A*Bm + B*Bb)
                    + BNL[i]*ABL[j] + BNL[18 + i]*ABL[18 + j] + BNL[36 + i]*ABL[36 + j]
                    # KCLL = BNL.T*A*BNL
                    + BNL[i]*ABNL[j] + BNL[18 + i]*ABNL[18 + j] + BNL[36 + i]*ABNL[36 + j]
                    # KGNL = G.T*[NNL]*G
                    + Sw_x[i]*(NNL[0]*Sw_x[j] + NNL[2]*Sw_y[j])
                    + Sw_y[i]*(NNL[2]*Sw_x[j] + NNL[1]*Sw_y[j])
                )


    cdef void _update_probe_finte_nonlinear(Tria3R self,
                                            ShellProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The attribute ``finte`` of the :class:`.Tria3RProbe` receives the terms
        of the von Karman membrane strains, evaluated at the displacements
        ``ue`` of the probe and added to the linear internal forces already in
        ``finte``. See :meth:`.update_KCNL`.

        Notes
        -----
        With the von Karman membrane strain `\boldsymbol{\epsilon}_{NL} = \{w_{,x}^2/2,
        w_{,y}^2/2, w_{,x} w_{,y}\}^\top`, the linear stress resultants `\boldsymbol{N} =
        \boldsymbol{A} \boldsymbol{\epsilon} + \boldsymbol{B} \boldsymbol{\kappa}`, and `\boldsymbol{N}_{NL} =
        \boldsymbol{A} \boldsymbol{\epsilon}_{NL}`, `\boldsymbol{M}_{NL} = \boldsymbol{B} \boldsymbol{\epsilon}_{NL}`, the terms
        added are

        .. math::

            A_e \left( \boldsymbol{B}_m^\top \boldsymbol{N}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{M}_{NL}
            + \boldsymbol{B}_{NL}^\top (\boldsymbol{N} + \boldsymbol{N}_{NL}) \right)

        with `A_e` the element area and `\boldsymbol{B}_{NL}` the variation of
        `\boldsymbol{\epsilon}_{NL}`, see :meth:`.update_KCNL`, such that ``finte``
        becomes the gradient of the strain energy whose Hessian is
        KC0 + KCNL + KG. The rows are those of :meth:`._update_probe_BL_G`, integrated
        with one point, as in :meth:`.update_KG`, the strains being constant
        within the element.

        """
        cdef int i, a

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
        cdef double *Sw_x
        cdef double *Sw_y

        ue = &self.probe.ue[0]
        finte = &self.probe.finte[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, A, B, NULL, NULL)

        # NOTE the strains are constant within the element, one integration point as
        #     in update_KG
        wij = 0.5
        detJ = self._update_probe_BL_G()

        exx = 0.
        eyy = 0.
        gxy = 0.
        kxx = 0.
        kyy = 0.
        kxy = 0.
        w_x = 0.
        w_y = 0.
        for i in range(18):
            exx += BLexx[i]*ue[i]
            eyy += BLeyy[i]*ue[i]
            gxy += BLgxy[i]*ue[i]
            kxx += BLkxx[i]*ue[i]
            kyy += BLkyy[i]*ue[i]
            kxy += BLkxy[i]*ue[i]
            w_x += Sw_x[i]*ue[i]
            w_y += Sw_y[i]*ue[i]

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

        for i in range(18):
            finte[i] += wij*detJ*(
                # Bm.T*NNL + Bb.T*MNL
                  BLexx[i]*NNL[0] + BLeyy[i]*NNL[1] + BLgxy[i]*NNL[2]
                + BLkxx[i]*MNL[0] + BLkyy[i]*MNL[1] + BLkxy[i]*MNL[2]
                # BNL.T*(N + NNL)
                + w_x*Sw_x[i]*(N[0] + NNL[0])
                + w_y*Sw_y[i]*(N[1] + NNL[1])
                + (w_x*Sw_y[i] + w_y*Sw_x[i])*(N[2] + NNL[2])
            )


    cpdef void update_KCNL(Tria3R self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           ShellProp prop,
                           int update_KCNLv_only=0
                           ):
        r"""Update sparse vectors for the nonlinear constitutive stiffness matrix KCNL

        The nonlinear constitutive stiffness matrix KCNL is the part of the
        tangent stiffness matrix, besides :meth:`.update_KC0` and
        :meth:`.update_KG`, that depends on the displacements through the von
        Karman membrane strains, evaluated at the displacements ``ue`` of the
        probe. The sum of the three is the tangent stiffness matrix of the
        internal forces of :meth:`.update_fint` with ``nonlinear=1``. With
        ``drilling_model = 0`` the membrane rows include the Allman
        enrichment, see :meth:`._update_probe_BL_G`.

        Before this function is called, the probe :class:`.Tria3RProbe` attribute
        of the :class:`.Tria3R` object must be updated using
        :meth:`.update_probe_ue` with the current displacements; and
        :meth:`.update_probe_xe` with the node coordinates.

        The element matrix is computed in element coordinates into the probe
        attribute ``KCNLve``, rotated to global coordinates, and its values
        are added to ``KCNLv`` at the positions ``init_k_KCNL`` to
        ``init_k_KCNL + KCNL_SPARSE_SIZE - 1``. With ``update_KCNLv_only=0``
        the global row and column indices are written to ``KCNLr`` and
        ``KCNLc`` at the same positions.

        Parameters
        ----------
        KCNLr : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_KCNL +
            KCNL_SPARSE_SIZE``.
        KCNLc : np.array
            Array to store column positions of sparse values
            Same type and size as ``KCNLr``.
        KCNLv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KCNLr``. The values are
            added to the existing ones, with the dimensions given for
            ``KC0v`` in :meth:`.update_KC0`.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes are
            read from.
        update_KCNLv_only : int, optional
            The default ``0`` means that the row and column indices ``KCNLr`` and
            ``KCNLc`` should also be updated. Any other value will only update the
            stiffness matrix values ``KCNLv``.

        Notes
        -----
        Assuming that KCNL = KC0L + KCL0 + KCLL + KGNL, built from the von Karman
        membrane strains

        .. math::
            \epsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2, \quad
            \epsilon_{yy} = v_{,y} + \frac{1}{2} w_{,y}^2, \quad
            \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

        whose nonlinear part is `\boldsymbol{\epsilon}_{NL} = \frac{1}{2} \boldsymbol{B}_{NL}
        \boldsymbol{u}_e`, with `\boldsymbol{B}_{NL}` its variation. With `\boldsymbol{B}_m` and `\boldsymbol{B}_b` the
        linear membrane and bending strain-displacement matrices, `\boldsymbol{G}_x = \partial \boldsymbol{S}^w / \partial x` and `\boldsymbol{G}_y = \partial \boldsymbol{S}^w / \partial y`
        the gradients of `w`, and `\boldsymbol{A}`, `\boldsymbol{B}` the laminate matrices:

        - KC0L = `\boldsymbol{B}_m^\top \boldsymbol{A} \boldsymbol{B}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{B} \boldsymbol{B}_{NL}`
        - KCL0 = KC0L`^\top`
        - KCLL = `\boldsymbol{B}_{NL}^\top \boldsymbol{A} \boldsymbol{B}_{NL}`
        - KGNL = `N_{NL,xx} \boldsymbol{G}_x^\top \boldsymbol{G}_x + N_{NL,yy}
          \boldsymbol{G}_y^\top \boldsymbol{G}_y + N_{NL,xy} (\boldsymbol{G}_x^\top
          \boldsymbol{G}_y + \boldsymbol{G}_y^\top \boldsymbol{G}_x)`, with
          `\boldsymbol{N}_{NL} = \{N_{NL,xx}, N_{NL,yy}, N_{NL,xy}\}^\top
          = \boldsymbol{A} \boldsymbol{\epsilon}_{NL}`

        The first three groups are the constitutive terms coupling the linear and
        the nonlinear parts of the membrane strain. KGNL is geometric, carrying the
        stress of the nonlinear membrane strain. It is collected here so that
        :meth:`.update_KG` stays homogeneous of degree one in the displacements,
        which is what a linear buckling analysis needs. With it here,

        .. math::
            \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

        is the exact Jacobian of the internal forces of :meth:`.update_fint` with
        ``nonlinear=1``, and a Newton-Raphson iteration built on them converges
        quadratically. The quadrature of :meth:`.update_KG` is used.

        """
        cdef int i, j, node_i, node_j, k, ke, m, n
        cdef int c[3]
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

                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = self.init_k_KCNL + 18*(node_i*DOF + m) + node_j*DOF + n
                                KCNLr[k] = c[node_i] + m
                                KCNLc[k] = c[node_j] + n

            self._update_probe_KCNLve(prop)

            # NOTE from element to global coordinates:
            #
            # Kg = T @ Ke @ T.T
            #
            # in tensor notation:
            #
            # Kg_{mn} = r_{mi} * Ke_{ij} * r_{nj}
            #
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = self.init_k_KCNL + 18*(node_i*DOF + m) + node_j*DOF + n
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 18*(node_i*DOF + i) + node_j*DOF + j
                                    KCNLv[k] += r[m][i]*self.probe.KCNLve[ke]*r[n][j]


    cpdef void update_KG(Tria3R self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         ShellProp prop,
                         int update_KGv_only=0,
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        The matrix is evaluated once, with no quadrature loop.

        Before this function is called, the probe :class:`.Tria3RProbe`
        attribute of the :class:`Tria3R` object must be updated using
        :meth:`.update_probe_ue` with the correct pre-buckling displacements;
        and :meth:`.update_probe_xe` with the node coordinates.

        The membrane stress resultants are obtained from the linear membrane
        strains and curvatures of the probe displacements ``ue``, including
        for ``drilling_model=0`` the contribution of the drilling rotations,
        and from the extensional and extension-bending coupling matrices of
        ``prop`` rotated with ``m11, m12, m21, m22``. Only the translational
        degrees-of-freedom appear. The values are added to ``KGv`` at the
        positions ``init_k_KG`` to ``init_k_KG + KG_SPARSE_SIZE - 1``. With
        ``update_KGv_only=0`` the global row and column indices are written
        to ``KGr`` and ``KGc`` at the same positions.

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values
           1-D array of integers (``INT``) of size at least ``init_k_KG +
           KG_SPARSE_SIZE``.
        KGc : np.array
           Array to store column positions of sparse values
           Same type and size as ``KGr``.
        KGv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KGr``. The values are
            added to the existing ones, with dimension of force per unit
            length.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        update_KGv_only : int, optional
            The default ``0`` means that the row and column indices ``KGr``
            and ``KGc`` are also updated. Any other value will only update the
            values ``KGv``.

        Notes
        -----
        The stress resultants are

        .. math::

            \{N_{xx}, N_{yy}, N_{xy}\}^\top = \boldsymbol{A}\{\epsilon_{xx}, \epsilon_{yy},
            \gamma_{xy}\}^\top + \boldsymbol{B}\{\kappa_{xx}, \kappa_{yy}, \kappa_{xy}\}^\top

        with the linear strains of the probe displacements, which for
        ``drilling_model = 0`` include the contribution of the Allman
        enrichment at the centroid, see :func:`.allman_enrichment`, since the
        edge modes populate the drilling columns of the membrane operator;
        without it KG would not be part of the exact Jacobian of the internal
        forces, see :meth:`.update_KCNL`. The matrix is then the one of
        :meth:`.update_KG_given_stress`. Only the linear strains contribute,
        which leaves this matrix homogeneous of degree one in the
        displacements, as a linear buckling analysis needs; the stress of the
        nonlinear membrane strain is carried by :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef int c1, c2, c3, k
        cdef double x1, x2, x3
        cdef double y1, y2, y3
        cdef double wij, detJ
        cdef double Ae[9]
        cdef double Be[9]
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double r13, r23, r33
        cdef double N1x, N2x, N3x, N1y, N2y, N3y
        cdef double Nxx, Nyy, Nxy
        cdef double Nxv[3]
        cdef double Nyv[3]
        cdef double denr[12]
        cdef double dexx, deyy, dgxy
        cdef int qi

        with nogil:
            detJ = 2*self.area

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
            r13 = self.r13
            r23 = self.r23
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

            ue = &self.probe.ue[0]

            if update_KGv_only == 0:
                # positions of nodes 1,2,3,4 in the global matrix
                c1 = self.c1
                c2 = self.c2
                c3 = self.c3

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

            wij = 0.5

            N1x = (y2 - y3)/(2*self.area)
            N2x = (-y1 + y3)/(2*self.area)
            N3x = (y1 - y2)/(2*self.area)
            N1y = (-x2 + x3)/(2*self.area)
            N2y = (x1 - x3)/(2*self.area)
            N3y = (-x1 + x2)/(2*self.area)

            Nxx = ue[0]*(A11*N1x + A16*N1y) + ue[10]*(B11*N2x + B16*N2y) + ue[12]*(A11*N3x + A16*N3y) + ue[13]*(A12*N3y + A16*N3x) - ue[15]*(B12*N3y + B16*N3x) + ue[16]*(B11*N3x + B16*N3y) + ue[1]*(A12*N1y + A16*N1x) - ue[3]*(B12*N1y + B16*N1x) + ue[4]*(B11*N1x + B16*N1y) + ue[6]*(A11*N2x + A16*N2y) + ue[7]*(A12*N2y + A16*N2x) - ue[9]*(B12*N2y + B16*N2x)
            Nyy = ue[0]*(A12*N1x + A26*N1y) + ue[10]*(B12*N2x + B26*N2y) + ue[12]*(A12*N3x + A26*N3y) + ue[13]*(A22*N3y + A26*N3x) - ue[15]*(B22*N3y + B26*N3x) + ue[16]*(B12*N3x + B26*N3y) + ue[1]*(A22*N1y + A26*N1x) - ue[3]*(B22*N1y + B26*N1x) + ue[4]*(B12*N1x + B26*N1y) + ue[6]*(A12*N2x + A26*N2y) + ue[7]*(A22*N2y + A26*N2x) - ue[9]*(B22*N2y + B26*N2x)
            Nxy = ue[0]*(A16*N1x + A66*N1y) + ue[10]*(B16*N2x + B66*N2y) + ue[12]*(A16*N3x + A66*N3y) + ue[13]*(A26*N3y + A66*N3x) - ue[15]*(B26*N3y + B66*N3x) + ue[16]*(B16*N3x + B66*N3y) + ue[1]*(A26*N1y + A66*N1x) - ue[3]*(B26*N1y + B66*N1x) + ue[4]*(B16*N1x + B66*N1y) + ue[6]*(A16*N2x + A66*N2y) + ue[7]*(A26*N2y + A66*N2x) - ue[9]*(B26*N2y + B66*N2x)

            # NOTE the Allman enrichment populates the drilling columns of
            #      the membrane operator, so the membrane stress resultants
            #      above, generated for the unenriched field, need the
            #      contribution of the edge modes. Without it KG would not
            #      be part of the exact Jacobian of the internal forces, see
            #      update_KCNL
            if self.drilling_model == 0:
                Nxv[0] = N1x
                Nxv[1] = N2x
                Nxv[2] = N3x
                Nyv[0] = N1y
                Nyv[1] = N2y
                Nyv[2] = N3y
                allman_enrichment(Nxv, Nyv, &self.probe.xe[0], 1./3., 1./3.,
                                  1./3., denr)
                dexx = 0.
                deyy = 0.
                dgxy = 0.
                for qi in range(3):
                    dexx += denr[qi]*ue[6*qi + 5]
                    deyy += denr[9 + qi]*ue[6*qi + 5]
                    dgxy += (denr[3 + qi] + denr[6 + qi])*ue[6*qi + 5]
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


    cpdef void update_KG_given_stress(Tria3R self,
                                      double Nxx, double Nyy, double Nxy,
                                      long [::1] KGr,
                                      long [::1] KGc,
                                      double [::1] KGv,
                                      int update_KGv_only=0,
                                      ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        .. note:: A constant stress state is assumed within the element,
                  according to the given values of `N_{xx}, N_{yy}, N_{xy}`.

        The matrix is evaluated once, with no quadrature loop.

        Before this function is called, the probe :class:`.Tria3RProbe`
        attribute of the :class:`.Tria3R` object must be updated using
        :meth:`.update_probe_xe` with the node coordinates.

        Only the translational degrees-of-freedom appear. The values are added
        to ``KGv`` at the positions ``init_k_KG`` to ``init_k_KG +
        KG_SPARSE_SIZE - 1``. With ``update_KGv_only=0`` the global row and
        column indices are written to ``KGr`` and ``KGc`` at the same
        positions. :meth:`.update_rotation_matrix` must also have been called
        before.

        Parameters
        ----------
        Nxx, Nyy, Nxy : double
            Membrane stress resultants in the element coordinate system,
            force per unit length.
        KGr : np.array
           Array to store row positions of sparse values
           1-D array of integers (``INT``) of size at least ``init_k_KG +
           KG_SPARSE_SIZE``.
        KGc : np.array
           Array to store column positions of sparse values
           Same type and size as ``KGr``.
        KGv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KGr``. The values are
            added to the existing ones, with dimension of force per unit
            length.
        update_KGv_only : int, optional
            The default ``0`` means that the row and column indices ``KGr``
            and ``KGc`` are also updated. Any other value will only update the
            values ``KGv``.

        Notes
        -----
        The geometric stiffness is the Donnell one, involving only the
        gradient of the element-frame `w`,

        .. math::

            \boldsymbol{K}_G = A_e \left( N_{xx} \boldsymbol{G}_x^\top \boldsymbol{G}_x
            + N_{yy} \boldsymbol{G}_y^\top \boldsymbol{G}_y
            + N_{xy} (\boldsymbol{G}_x^\top \boldsymbol{G}_y
            + \boldsymbol{G}_y^\top \boldsymbol{G}_x) \right)

        with `A_e` the element area and `\boldsymbol{G}_x`, `\boldsymbol{G}_y`
        the rows ``Sw_x``, ``Sw_y`` of
        :meth:`._update_probe_BL_G`, the element-frame `w` being obtained
        from the global translations with the element normal `\{T_{13},
        T_{23}, T_{33}\}`. The integrand is constant over the triangle, so the
        single evaluation is exact.

        """
        cdef int c1, c2, c3, k
        cdef double x1, x2, x3
        cdef double y1, y2, y3
        cdef double wij, detJ
        cdef double r13, r23, r33
        cdef double N1x, N2x, N3x, N1y, N2y, N3y

        with nogil:
            detJ = 2*self.area

            # local to global transformation
            r13 = self.r13
            r23 = self.r23
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

            if update_KGv_only == 0:
                # positions of nodes 1,2,3,4 in the global matrix
                c1 = self.c1
                c2 = self.c2
                c3 = self.c3

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

            wij = 0.5

            N1x = (y2 - y3)/(2*self.area)
            N2x = (-y1 + y3)/(2*self.area)
            N3x = (y1 - y2)/(2*self.area)
            N1y = (-x2 + x3)/(2*self.area)
            N2y = (x1 - x3)/(2*self.area)
            N3y = (-x1 + x2)/(2*self.area)

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


    cpdef void update_M(Tria3R self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        ShellProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        Different integration schemes are available by means of the ``mtype``
        parameter.

        The element matrix is computed from ``intrho``, ``intrhoz`` and
        ``intrhoz2`` of ``prop``, the coordinates ``probe.xe`` and the
        ``area``, rotated to global coordinates, and its values are added to
        ``Mv`` at the positions starting at ``init_k_M``; the global row and
        column indices are always written to ``Mr`` and ``Mc`` at the same
        positions. ``mtype=0`` and ``mtype=2`` write ``M_SPARSE_SIZE``
        entries and ``mtype=1`` writes the first 90 of them. Before this
        method is called, :meth:`.update_rotation_matrix` and
        :meth:`.update_probe_xe` must have been used.

        Parameters
        ----------
        Mr : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_M +
            M_SPARSE_SIZE``.
        Mc : np.array
            Array to store column positions of sparse values
            Same type and size as ``Mr``.
        Mv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``Mr``. The values are added
            to the existing ones. Dimension of mass between translations, of
            mass times length between a translation and a rotation, and of
            mass times length squared between rotations.
        prop : :class:`.ShellProp` object
            Shell property object, from which ``intrho``, ``intrhoz`` and
            ``intrhoz2`` are read.
        mtype : int, optional
            0 for consistent mass matrix
            1 for lumped mass matrix
            2 for reduced integration mass matrix
            Default ``0``.

        Raises
        ------
        ValueError
            If ``mtype`` is not 0, 1 or 2.

        Notes
        -----
        The element matrices follow from the kinetic energy of Brockman [1]_,
        Eq. (20), with the displacements `u + z r_y`, `v - z r_x` and `w`
        through the thickness,

        .. math::

            T = \frac{1}{2} \int_A \left[ I_\rho (\dot{u}^2 + \dot{v}^2 +
            \dot{w}^2) + 2 I_{\rho z} (\dot{u} \dot{r}_y - \dot{v} \dot{r}_x)
            + I_{\rho z^2} (\dot{r}_x^2 + \dot{r}_y^2) \right] dA

        with `I_\rho`, `I_{\rho z}` and `I_{\rho z^2}` given by ``intrho``,
        ``intrhoz`` and ``intrhoz2``, which include the offset. With `h_{ij}`
        the integral of the product of the shape functions `S_i S_j` over
        the element, the element matrix has, as in Eq. (21) of [1]_, `I_\rho
        h_{ij}` between the translations of nodes `i` and `j`, `I_{\rho z^2}
        h_{ij}` between the rotations `r_x` and between the rotations `r_y`,
        and `I_{\rho z} h_{ij}` coupling the translations to the rotations.
        The drilling rotation `r_z` has no inertia. The schemes differ in how
        `h_{ij}` is integrated:

        - ``mtype=0``: the three-point rule of degree of precision 2, first
          row of Table I in Cowper [2]_, with the area coordinates `(2/3,
          1/6, 1/6)` and their permutations, each with the weight `A_e/3`,
          which is exact for the quadratic `S_i S_j`, `h_{ii} = A_e/6` and
          `h_{ij} = A_e/12`
        - ``mtype=1``: the vertex rule, the three nodes, each with the weight
          `A_e/3`, `h_{ii} = A_e/3` and `h_{ij} = 0`
        - ``mtype=2``: a single point at the centroid, `h_{ij} = A_e/9`

        with `A_e` the element area. In element coordinates the coupling
        terms are `+I_{\rho z} h_{ij}` between `u` and `r_y` and `-I_{\rho z}
        h_{ij}` between `v` and `r_x`, consistently with the curvatures of
        :meth:`._update_probe_BL_G`.

        .. rubric:: Lumped mass matrix

        The vertex rule is a nodal quadrature. Applied to the whole integrand
        of `T`, in which `S_i` is 1 at node `i` and 0 at the other nodes, it
        gives no coupling between different nodes and, for node `i`, the
        block

        .. math::

            \frac{A_e}{3} \begin{bmatrix}
                I_\rho & 0 & 0 & 0 & I_{\rho z} & 0 \\
                0 & I_\rho & 0 & -I_{\rho z} & 0 & 0 \\
                0 & 0 & I_\rho & 0 & 0 & 0 \\
                0 & -I_{\rho z} & 0 & I_{\rho z^2} & 0 & 0 \\
                I_{\rho z} & 0 & 0 & 0 & I_{\rho z^2} & 0 \\
                0 & 0 & 0 & 0 & 0 & 0
            \end{bmatrix}

        for the DOF `u, v, w, r_x, r_y, r_z`. The coupling through `I_{\rho
        z}` stays in the nodal block: the matrix is block-diagonal per node,
        and diagonal only when `I_{\rho z} = 0`, as Brockman [1]_ notes for the
        Lobatto integration at the nodes of the quadrilateral. Since the
        weights sum to `A_e` and `\sum_i (A_e/3) \boldsymbol{x}_i = \int_A \boldsymbol{x}
        dA`, with `\boldsymbol{x}_i` the position of node `i`, the lumped matrix has
        the exact mass and the exact first moment of the mass, including the
        term `I_{\rho z} A_e` that places the centre of mass at the offset
        from the reference surface. Of the rotary inertia, the term `I_{\rho
        z^2} A_e` about the axes in the plane of the element is exact, while
        the part from the distribution of `I_\rho` over the area, about any
        axis, is concentrated at the nodes: 4 times the exact value about the
        centroid of the element, an error that decreases with the size of
        the elements.

        References
        ----------
        .. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate
           Element, International Journal for Numerical Methods in
           Engineering, 24(12), 2343-2356, 1987.
           https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208
        .. [2] Cowper, G. R., Gaussian quadrature formulas for triangles,
           International Journal for Numerical Methods in Engineering,
           7(3), 405-408, 1973. https://doi.org/10.1002/nme.1620070316

        """
        cdef int c1, c2, c3, i, k
        cdef double intrho, intrhoz, intrhoz2
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double h11, h12, h13, h22, h23, h33, valH1
        cdef double wij, detJ, N1, N2, N3
        cdef double points[3]

        # NOTE checked once per call, before the code without the GIL
        if mtype != 0 and mtype != 1 and mtype != 2:
            raise ValueError('mtype=%d is not supported by Tria3R.update_M, '
                             'use 0 (consistent), 1 (lumped) or 2 (reduced '
                             'integration)' % mtype)

        with nogil:
            intrho = prop.intrho
            intrhoz = prop.intrhoz
            intrhoz2 = prop.intrhoz2

            detJ = 2*self.area
            valH1 = detJ/18.

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

            if mtype == 0: # M_cons consistent mass matrix, three-point rule of Cowper (1973), see below
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

                # NOTE three-point rule of degree of precision 2, first row of Table I
                #      in Cowper, G. R., Gaussian quadrature formulas for triangles,
                #      IJNME 7(3), 405-408, 1973, doi:10.1002/nme.1620070316: area
                #      coordinates (2/3, 1/6, 1/6) and their permutations, weight 1/6
                #      times detJ = 2*area, i.e. area/3
                wij = 0.5*0.333333333333333333333333333333333333333333333
                points[0] = 0.66666666666666666666666666666666666666666667
                points[1] = 0.16666666666666666666666666666666666666666667
                points[2] = 0.16666666666666666666666666666666666666666667
                h11 = 0.
                h12 = 0.
                h13 = 0.
                h22 = 0.
                h23 = 0.
                h33 = 0.
                for i in range(3):
                    if i == 0:
                        N1 = points[0]
                        N2 = points[1]
                        N3 = points[2]
                    elif i == 1:
                        N1 = points[1]
                        N2 = points[2]
                        N3 = points[0]
                    elif i == 2:
                        N1 = points[2]
                        N2 = points[0]
                        N3 = points[1]

                    h11 += N1**2*detJ*wij
                    h12 += N1*N2*detJ*wij
                    h13 += N1*N3*detJ*wij
                    h22 += N2**2*detJ*wij
                    h23 += N2*N3*detJ*wij
                    h33 += N3**2*detJ*wij

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

            elif mtype == 2: # M_red mass matrix purely by reduced integration
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


            elif mtype == 1: # M_lump lumped mass matrix, vertex rule, see the Notes
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

                # NOTE vertex rule, area coordinates (1, 0, 0) and their permutations,
                #      weight 1/6 times detJ = 2*area, i.e. area/3; S_i is 1 at node i
                #      and 0 at the other nodes, such that h12 = h13 = h23 = 0
                wij = 0.5*0.3333333333333333333333333333333333333
                points[0] = 1.
                points[1] = 0.
                points[2] = 0.
                h11 = 0.
                h22 = 0.
                h33 = 0.
                for i in range(3):
                    if i == 0:
                        N1 = points[0]
                        N2 = points[1]
                        N3 = points[2]
                    elif i == 1:
                        N1 = points[1]
                        N2 = points[2]
                        N3 = points[0]
                    elif i == 2:
                        N1 = points[2]
                        N2 = points[0]
                        N3 = points[1]

                    h11 += N1**2*detJ*wij
                    h22 += N2**2*detJ*wij
                    h33 += N3**2*detJ*wij

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
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz2*r31**2 + h33*intrhoz2*r32**2

