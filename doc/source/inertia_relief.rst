Inertia relief
==============

A structure without supports, e.g. an aircraft or a launch vehicle in
flight, a spacecraft, or any free-free model, can move as a rigid body: its
stiffness matrix ``KC0`` is singular, with the rigid-body modes in its null
space, and the linear static system of :doc:`linear_analyses` cannot be
solved as it stands. A static solution exists only for balanced loads,
without a resultant force or moment. Under unbalanced loads, e.g. the
aerodynamic loads and the weights of an aircraft that is not trimmed, or
the thrust of a launch vehicle, the free structure accelerates; under
steady loads it deforms elastically while it accelerates as a rigid body.
Inertia relief balances the loads with the d'Alembert forces of this
rigid-body acceleration and computes the displacements relative to a
reference support. In pyfe3d this is done by
:func:`.inertia_relief`, whose Notes give the equations and the
references.


Workflow
--------

1. Assemble ``KC0`` as for a linear static analysis, see :doc:`overview`,
   without any support.
2. Assemble the mass matrix ``M`` with the ``update_M`` methods of the
   elements, consistent (``mtype=0``) or lumped (``mtype=1``), see
   :doc:`element_choice`, and add the point masses on the diagonal. The
   mass matrix must give every rigid-body mode a mass, e.g. a beam needs a
   rotary inertia about its axis, ``intrhoy2 + intrhoz2`` of
   :class:`.BeamProp`.
3. Build the load vector ``fext``. A gravity field is ``M @ a_g``, with
   ``a_g`` the uniform acceleration field of the gravity at the
   translational dofs.
4. Choose the reference support, the Boolean array ``bref``, and, when the
   model itself is constrained, e.g. by a plane of symmetry or by the dofs
   that a model in one or two dimensions does not use, the Boolean array
   ``bfix`` of these constraints.
5. Solve::

       from pyfe3d.solver import inertia_relief
       u, acc, Phi, R = inertia_relief(KC0, M, fext, bref, bfix,
                                       return_reactions=True)

   ``u`` are the displacements relative to the reference support, ``acc``
   the rigid-body accelerations of the reference dofs, ``Phi`` the
   rigid-body modes, with which ``Phi @ acc`` is the acceleration of every
   dof and ``-M @ Phi @ acc`` the d'Alembert forces, and ``R`` the residual,
   whose values at ``bref`` are the reactions of the reference support,
   zero up to the round-off.
6. Post-process ``u`` as in a linear static analysis, e.g. with
   ``update_probe_ue`` and ``update_probe_finte`` of the elements. The
   element forces of a distributed inertia load contain the element's own
   share of it, as for any distributed load.


The reference support
---------------------

The reference support sets to zero as many dofs as there are rigid-body
modes, six for a free structure in 3D, fewer with ``bfix``, e.g. two, heave
and pitch, for the half model of the airplane of
:doc:`tut-inertia-relief-aircraft-ha144a`, whose plane of symmetry and
fore-aft translation are constrained by ``bfix``. It must be
statically determinate: it suppresses every rigid-body mode without
restraining any deformation, i.e. the rows of the rigid-body modes at the
reference dofs form a non-singular square matrix. Typical choices are the
six dofs of one node, or three translations of one node, two of a second
node and one of a third node that are not aligned. :func:`.inertia_relief`
checks the choice:

- with too few dofs, or dofs that leave a rigid-body mode free, e.g. two
  translations along the line joining their nodes, the stiffness matrix of
  the free dofs, `\boldsymbol{K}_{{C0}_{uu}}`, is singular, which raises a
  :class:`numpy.linalg.LinAlgError`, as for insufficient boundary
  conditions in :func:`.linear_static`;
- with more dofs than the rigid-body modes, the support is statically
  indeterminate: it would restrain the deformation and carry reactions,
  which raises a :class:`numpy.linalg.LinAlgError` too.

With a statically determinate support, the reactions are zero and the
support does not change the solution: another choice changes the
displacements by a rigid-body motion only, while the strains, the stresses
and the internal forces are the same. The displacements are thus always
relative to the chosen support, which moves with the rigid-body
acceleration; a support at a point of interest, e.g. the centre of gravity
or a payload interface, makes them easier to read.


Balanced and unbalanced loads
-----------------------------

For balanced loads, e.g. a structure under its own weight balanced by
other loads, or self-equilibrated loads, the rigid-body accelerations are
zero up to the round-off and inertia relief reduces to a fictitious support
of the singular static problem: the displacements are those of the
structure supported by the reference support, which carries no reaction.

For unbalanced loads, the plain linear static solution with the same
support would carry the resultant of the loads at the support, with
unrealistic reactions and internal forces near it. Inertia relief replaces
these reactions by the d'Alembert forces of the rigid-body acceleration,
distributed as the mass. The result is a quasi-static state, valid when the
loads are steady and the transient response has been damped out; it is not
a dynamic response.


Relation to the prescribed displacements
----------------------------------------

The reference support is a set of prescribed null displacements, the known
set `k` of the partition of :doc:`linear_analyses`, with ``bk = bref |
bfix`` and ``uk = 0``, and the reactions are computed as there, with the
d'Alembert forces added to the external forces. The difference is the
choice of these dofs: a support of a linear static analysis restrains the
structure and carries the loads, whereas the reference support of inertia
relief only fixes the rigid-body motion and carries no load.


Worked examples
---------------

- :doc:`tut-inertia-relief`: springs and point masses, with the closed-form
  solutions, a free beam under a balanced load set, and a free beam and bar
  under an end force, against closed-form solutions;
- :doc:`tut-inertia-relief-aircraft-ha144a`: the stick model of an airplane
  in trimmed level flight, the MSC Nastran example HA144A, against the
  published results.


References
----------

- Barnett, A. R., Widrick, T. W. and Ludwiczak, D. R., Closed-Form Static
  Analysis with Inertia Relief and Displacement-Dependent Loads Using a
  MSC/NASTRAN DMAP Alter, NASA TM-106836, NASA Lewis Research Center,
  Cleveland, OH, 1995. https://ntrs.nasa.gov/citations/19950013233
- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
  Watertown, MA, 2014.
