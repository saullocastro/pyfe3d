Inertia relief
==============

A structure without supports, e.g. an aircraft or a launch vehicle in
flight, a spacecraft, or any free-free model, can move as a rigid body: its
stiffness matrix `\boldsymbol{K}_{C0}` is singular, with the rigid-body modes in
its null space, and the linear static system of :doc:`linear_analyses`
cannot be solved as it stands. Under steady loads that are not balanced,
e.g. the aerodynamic loads and the weights of an aircraft that is not
trimmed, or the thrust of a launch vehicle, the free structure accelerates
as a rigid body while it deforms elastically. Inertia relief balances the
loads with the d'Alembert forces of this rigid-body acceleration and
computes the displacements relative to a reference support, with
:func:`.apply_inertia_relief`.


Solvability of the singular system
----------------------------------

The `n_r` rigid-body modes, six for a free structure in 3D, are the columns
of `\boldsymbol{\Phi}`, with `\boldsymbol{K}_{C0} \boldsymbol{\Phi} = \boldsymbol{0}` and, by symmetry,
`\boldsymbol{\Phi}^\top \boldsymbol{K}_{C0} = \boldsymbol{0}`. Premultiplying `\boldsymbol{K}_{C0}
\boldsymbol{u} = \boldsymbol{F}_{ext}` by `\boldsymbol{\Phi}^\top` shows that a static solution
exists only for loads orthogonal to the rigid-body modes,

.. math::
    :label: ir-solvability

    \boldsymbol{\Phi}^\top \boldsymbol{F}_{ext} = \boldsymbol{0}

i.e. without a resultant force or moment (Farhat and Géradin 1998;
Cavaliere et al. 2021, Sec. 3), and that it is then defined up to a
rigid-body motion `\boldsymbol{\Phi} \boldsymbol{\alpha}`. A particular solution is
obtained by prescribing `n_r` null displacements, the reference support,
chosen such that they suppress the rigid-body modes without restraining any
deformation: a statically determinate support, for which the rows of
`\boldsymbol{\Phi}` at the reference dofs form a non-singular square matrix.


Rigid-body modes and accelerations
----------------------------------

The dofs are partitioned as in :doc:`boundary_conditions`, into the known
set `k`, the reference support ``bref`` together with the dofs ``bfix``
that the model itself constrains, e.g. a plane of symmetry, and the free
set `u`. With unit displacements at the reference dofs and zero at those of
``bfix``, the rigid-body modes follow from the stiffness matrix (Barnett et
al. 1995, p. 6, Eq. (19)):

.. math::
    :label: ir-modes

    \boldsymbol{\Phi}_k = \begin{bmatrix} \boldsymbol{I} \\ \boldsymbol{0} \end{bmatrix}, \qquad
    \boldsymbol{\Phi}_u = -\boldsymbol{K}_{{C0}_{uu}}^{-1} \boldsymbol{K}_{{C0}_{uk}} \boldsymbol{\Phi}_k

with the rows of ``bref`` first and those of ``bfix`` second in
`\boldsymbol{\Phi}_k`, where `\boldsymbol{K}_{{C0}_{uu}}` is positive definite only when the
supports suppress all the rigid-body modes (Bathe 2014, Sec. 8.2.5). Under
steady loads, the acceleration is that of a rigid body, `\ddot{\boldsymbol{u}} =
\boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}`; premultiplying the equations of motion
`\boldsymbol{M} \ddot{\boldsymbol{u}} + \boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{F}_{ext}` by
`\boldsymbol{\Phi}^\top` gives the rigid-body accelerations (Eqs. (21) and (22)),

.. math::
    :label: ir-acc

    \boldsymbol{\Phi}^\top \boldsymbol{M} \boldsymbol{\Phi} \, \ddot{\boldsymbol{\alpha}} = \boldsymbol{\Phi}^\top \boldsymbol{F}_{ext}

and the displacements relative to the reference support, `\boldsymbol{u}_k =
\boldsymbol{0}`, solve the free rows of the static system with the d'Alembert
forces added to the loads (Eqs. (23) and (24)):

.. math::
    :label: ir-u

    \boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{F}_{{ext}_u} - \boldsymbol{M}_u \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}

whose total load `\boldsymbol{F}_{ext} - \boldsymbol{M} \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}` satisfies
the solvability condition :eq:`ir-solvability`. The mass matrix must give
every rigid-body mode a mass, `\boldsymbol{\Phi}^\top \boldsymbol{M} \boldsymbol{\Phi}` positive
definite: e.g. a beam needs a rotary inertia about its axis, ``intrhoy2 +
intrhoz2`` of :class:`.BeamProp`. A gravity field enters
`\boldsymbol{F}_{ext}` as `\boldsymbol{M} \boldsymbol{a}_g`, with `\boldsymbol{a}_g` the uniform
acceleration at the translational dofs.


The reference support
---------------------

The reactions are the residual at the known dofs, `\boldsymbol{R} = \boldsymbol{K}_{C0}
\boldsymbol{u} - \boldsymbol{F}_{ext} + \boldsymbol{M} \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}`, as in
:eq:`bc-reactions`. Since `\boldsymbol{R}_u = \boldsymbol{0}` and `\boldsymbol{\Phi}^\top
\boldsymbol{K}_{C0} = \boldsymbol{0}`, premultiplying it by `\boldsymbol{\Phi}^\top` shows that the
reactions of a statically determinate reference support are zero: the
support carries no load and does not change the strains, the stresses and
the internal forces; another statically determinate support changes the
displacements by a rigid-body motion only. The displacements are thus
relative to the chosen support, which moves with the rigid-body
acceleration; a support at a point of interest, e.g. the centre of gravity
or a payload interface, makes them easier to read. Typical choices are the
six dofs of one node, or three translations of one node, two of a second
node and one of a third node that are not aligned; with ``bfix``, fewer,
e.g. heave and pitch for a half model of an airplane whose plane of
symmetry and fore-aft translation are constrained, see
``tests/test_inertia_relief_ha144a.py``.

:func:`.apply_inertia_relief` checks the choice: with too few reference
dofs, or dofs that leave a rigid-body mode free, e.g. two translations along
the line joining their nodes, `\boldsymbol{K}_{{C0}_{uu}}` is singular, which is
detected from the signs and the magnitudes of the pivots of its
factorization, see ``pivot_rtol``; with more dofs than the rigid-body modes,
the support is statically indeterminate, and a column of `\boldsymbol{\Phi}` of
:eq:`ir-modes` is not a rigid-body mode, `\boldsymbol{K}_{C0} \boldsymbol{\Phi}_j \ne
\boldsymbol{0}`, see ``rtol``. Both raise a :class:`numpy.linalg.LinAlgError`.
`\boldsymbol{K}_{{C0}_{uu}}` is factorized once, as in :func:`.linear_static`, and
the factorization gives both `\boldsymbol{\Phi}_u` and `\boldsymbol{u}_u`.


Balanced and unbalanced loads
-----------------------------

For balanced loads, e.g. self-equilibrated loads, the rigid-body
accelerations are zero up to the round-off and inertia relief reduces to a
fictitious support of the singular static problem: the displacements are
those of the structure supported by the reference support, which carries no
reaction. For unbalanced loads, the plain linear static solution with the
same support would carry the resultant of the loads at the support, with
unrealistic reactions and internal forces near it (Cavaliere et al. 2021,
Sec. 1). Inertia relief replaces these reactions by the d'Alembert forces
of the rigid-body acceleration, distributed as the mass. The element forces
of a distributed inertia load contain the element's own share of it, as
for any distributed load. The result is a quasi-static state, valid when
the loads are steady and the transient response has been damped out; it is
not a dynamic response.


Relation to the prescribed displacements
----------------------------------------

The reference support is a set of prescribed null displacements, the known
set of :doc:`boundary_conditions` with ``bk = bref | bfix`` and ``uk = 0``,
and the reactions are computed as there, with the d'Alembert forces added
to the external forces. The difference is the choice of these dofs: a
support of a linear static analysis restrains the structure and carries
the loads, whereas the reference support of inertia relief only fixes the
rigid-body motion and carries no load. The methods for large singular
systems without a reference support, which compute the null space of
`\boldsymbol{K}_{C0}` during the factorization, are discussed by Farhat and
Géradin (1998, Secs. 1-4).

``tests/test_inertia_relief.py`` checks the function against closed-form
solutions, three point masses joined by two springs, a chain of springs and
masses, free-free beams and bars, and ``tests/test_inertia_relief_ha144a.py``
against the published results of the MSC Nastran example HA144A of Rodden
and Johnson (1994).


References
----------

- Barnett, A. R., Widrick, T. W. and Ludwiczak, D. R., Closed-Form Static
  Analysis with Inertia Relief and Displacement-Dependent Loads Using a
  MSC/NASTRAN DMAP Alter, NASA TM-106836, NASA Lewis Research Center,
  Cleveland, OH, 1995. https://ntrs.nasa.gov/citations/19950013233
- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
  Watertown, MA, 2014.
- Cavaliere, F., Zlotnik, S., Sevilla, R., Larráyoz, X. and Díez, P.,
  Nonintrusive reduced order model for parametric solutions of inertia
  relief problems, International Journal for Numerical Methods in
  Engineering, 122(16), 4270-4291, 2021. https://doi.org/10.1002/nme.6702
- Farhat, C. and Géradin, M., On the general solution by a direct method of
  a large-scale singular system of linear equations: application to the
  analysis of floating structures, International Journal for Numerical
  Methods in Engineering, 41(4), 675-696, 1998.
  https://doi.org/10.1002/%28SICI%291097-0207%2819980228%2941%3A4%3C675%3A%3AAID-NME305%3E3.0.CO%3B2-8
- Rodden, W. P. and Johnson, E. H., MSC/NASTRAN Aeroelastic Analysis
  User's Guide, Version 68, The MacNeal-Schwendler Corporation, Los
  Angeles, 1994.
