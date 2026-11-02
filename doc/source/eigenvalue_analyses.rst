Linear buckling and natural frequencies
=======================================

The eigenvalue analyses of pyfe3d linearize the behaviour of the structure
about a reference state: the linear buckling analysis seeks the load
multipliers at which the stiffness of a linear pre-buckling state vanishes,
the natural frequency analysis the free vibrations about a state at rest,
possibly pre-stressed, and the complex frequency analysis the free
vibrations of a damped system. They use the free-free blocks of the
assembled matrices, e.g. ``KC0uu = KC0[bu, :][:, bu]``, see
:doc:`boundary_conditions`, and are solved with :func:`.linear_buckling`,
:func:`.natural_frequency` and :func:`.complex_frequency` of
:mod:`pyfe3d.solver.eigen`, which return the eigenvectors with the size of
the matrices received, to be expanded by the user, ``u[bu] = eigvecs[:,
i]``. The algorithms that solve and verify them are described in
:doc:`eigenvalue_solvers`.


Linear buckling
---------------

At a stable equilibrium the second variation of the total potential energy
is positive for every variation of the configuration; the neutral
equilibrium criterion, `\delta^2 V = \delta \boldsymbol{u}^\top \boldsymbol{K}_T \,
\delta \boldsymbol{u} = 0` for at least one `\delta \boldsymbol{u} \ne \boldsymbol{0}`, marks
the loss of stability, where the tangent stiffness matrix of
:doc:`nonlinear_analyses` becomes singular. The linear buckling analysis
takes a linear pre-buckling state: the solution `\boldsymbol{u}_0` of the linear
static system of a reference load, `\boldsymbol{K}_{C0} \boldsymbol{u}_0 =
\boldsymbol{F}_{ext}`, with the pre-buckling rotations neglected, such that
`\boldsymbol{K}_{CNL} = \boldsymbol{0}` and `\boldsymbol{K}_{GNL} = \boldsymbol{0}`. The geometric
stiffness matrix of that state, ``KG``, is linear in its stresses, hence in
the load, and the tangent stiffness matrix along the pre-buckling path is
`\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G`. It is singular at the load multipliers
of

.. math::
    :label: ev-buckling

    (\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}

whose lowest positive eigenvalue `\lambda_1` gives the critical load
`\lambda_1 \boldsymbol{F}_{ext}` and whose eigenvector is the buckling mode. The
analysis has three steps:

1. the linear static solution `\boldsymbol{u}_0` of the reference load, see
   :doc:`linear_analyses`;
2. the geometric stiffness matrix of `\boldsymbol{u}_0`: for each element,
   ``update_probe_xe(x)``, ``update_probe_ue(u0)`` and ``update_KG(KGr,
   KGc, KGv, prop)``; for the shells, a known constant membrane state can
   be given instead with ``update_KG_given_stress(Nxx, Nyy, Nxy, KGr, KGc,
   KGv)``, with the stress resultants in the element coordinate system of
   each element, see :doc:`conventions`;
3. the eigenvalue problem :eq:`ev-buckling` of the free dofs,
   ``eigvals, eigvecs = linear_buckling(KC0uu, KGuu, num_eigvalues=...)``.

:func:`.linear_buckling` returns the positive load multipliers first, in
increasing order, followed, when fewer than ``num_eigvalues`` positive ones
exist, by the negative ones closest to zero, which are the critical load
multipliers of the reversed reference load: a tensile reference load, which
has no positive multiplier, thus gives the critical multipliers of the
compressive one. The infinite load multipliers, e.g. of dofs where ``KG``
has null rows, are discarded by the sparse solvers of symmetric matrices,
and returned as ``-inf``, last, by the dense solvers.

**Preloads and combined loads.** A load that is not scaled by `\lambda`,
e.g. a preload with the geometric stiffness matrix `\boldsymbol{K}_{G0}`, belongs
to the first matrix, since the geometric stiffness matrix is linear in the
stresses and the stresses of a linear pre-buckling state add up:

.. math::
    :label: ev-preload

    (\boldsymbol{K}_{C0} + \boldsymbol{K}_{G0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}

solved with ``linear_buckling(KC0uu + KG0uu, KGuu)``, which is how the
critical value of one load component is found with the other components
kept constant, e.g. the compressive load of a panel under a given shear
load. A preload beyond its own buckling load makes the first matrix
indefinite, which :func:`.linear_buckling` reports, see
:doc:`eigenvalue_solvers`.

**Elements.** The beams have ``update_KG`` for the axial force of the
pre-buckling state and take the slopes of their geometric stiffness
matrices from the interpolations of the transverse displacements, such that
the buckling load includes the reduction due to shear deformation, Eq.
(2-57) of Timoshenko and Gere (1961, Sec. 2.17, p. 133), see
:class:`.BeamC` and :class:`.BeamLR`; :class:`.Truss` and :class:`.Spring`
have no geometric stiffness matrix. The integration of the geometric
stiffness matrices of the shells, the slopes at the centroid of the
elements with reduced transverse shear and the eccentricity term of the
rigid offsets of warped quadrilaterals are described with each element, see
:meth:`.Quad4.update_KG`, :meth:`.Quad4R.update_KG` and
:doc:`element_choice`.

**Limits.** The linear pre-buckling state neglects the pre-buckling
rotations and the change of the stresses along the path. The critical load
is that of a bifurcation from the linear path; a structure whose
pre-buckling path is markedly non-linear, or that is sensitive to
imperfections, has a limit load that is traced with the non-linear
analyses of :doc:`nonlinear_analyses` and :doc:`arc_length_methods`. A
geometric stiffness matrix built at a converged non-linear state,
``calc_kG(c, NLgeom=True)``, linearizes the stability about that state
instead.


Follower loads and flutter
--------------------------

With a configuration-dependent load, the load stiffness matrix of the
reference load, `\boldsymbol{K}_{F_{ref}}`, see
:doc:`configuration_dependent_loads`, adds to the geometric stiffness
matrix:

.. math::
    :label: ev-follower

    \left(\boldsymbol{K}_{C0} + \lambda (\boldsymbol{K}_G + \boldsymbol{K}_{F_{ref}})\right) \boldsymbol{u} = \boldsymbol{0}

an unsymmetric problem when `\boldsymbol{K}_{F_{ref}}` is, whose load multipliers
may be complex; :func:`.linear_buckling` detects the unsymmetric matrices
and uses its solvers of unsymmetric matrices, which find the lowest
positive real load multiplier, the divergence load. A non-conservative
system may lose its stability by flutter, a dynamic instability that this
static criterion does not detect: Ziegler's double pendulum under a
follower force has no finite load multiplier, its stiffness matrix being
singular for no load, and loses its stability by flutter, see
``tests/test_solver_linear_buckling.py``. Its stability is studied with the
dynamic method below. For a uniform pressure on a closed surface, or on a
surface whose edges satisfy the conditions of
:doc:`configuration_dependent_loads`, `\boldsymbol{K}_{F_{ref}}` is symmetric and
the static criterion applies.


Natural frequencies
-------------------

The free vibrations `\boldsymbol{u} \sin(\omega_n t)` about a state at rest give
the generalized eigenvalue problem

.. math::
    :label: ev-frequency

    (\boldsymbol{K} - \omega_n^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}

solved by ``omegan, eigvecs = natural_frequency(Kuu, Muu,
num_eigvalues=...)``, which returns the circular frequencies `\omega_n`, in
rad/s with consistent units, in increasing order. `\boldsymbol{K}` is
`\boldsymbol{K}_{C0}`, plus the geometric stiffness matrix of a pre-stress when it
applies, e.g. ``KC0uu + KGuu``: a compressive pre-stress lowers the
frequencies, and the lowest one vanishes at the critical load multiplier of
:eq:`ev-buckling`, see ``tests/test_solver_natural_frequency.py``. A
pre-stress beyond the buckling load gives negative `\omega_n^2`, returned as
null frequencies. The mass matrix is assembled with ``update_M``, whose
``mtype`` selects the consistent, lumped or reduced-integration mass matrix
of each element, see :doc:`element_choice`. A lumped mass matrix without
rotary inertia is singular: its massless dofs have infinite eigenvalues,
which the solvers, seeking the lowest ones, are not troubled by, but which
the lumped matrix places where its factorization puts them, so a consistent
mass matrix is preferable when the lowest frequencies matter. The
unconstrained rigid-body modes of a free structure give null frequencies.

**Dynamic stability.** With `\boldsymbol{K}` the tangent stiffness matrix at a
load level, including the load stiffness of a follower load, e.g. with the
aerodynamic stiffness matrix of the piston theory, `\boldsymbol{K}` is
unsymmetric and the eigenvalues `\omega_n^2` may become complex; ``omegan``
is then the principal square root of the complex eigenvalues. Below the
flutter load the `\omega_n^2` are real and positive; at the flutter load two
frequencies coalesce, a double root of the frequency equation, and above it
they become a complex conjugate pair, one of which corresponds to a growing
motion. This is the dynamic method of Bolotin (1963, Introduction, Secs. 6
and 7, pp. 12-17), where `\omega_n` is the frequency `\Omega` of the
characteristic exponent `s = i\Omega`; a real negative `\omega_n^2`, i.e. a
real positive `s`, indicates divergence (p. 17). The flutter load of
Ziegler's double pendulum and the frequencies below it are checked against
their analytical values in ``tests/test_solver_natural_frequency.py``.


Damped frequencies
------------------

With a damping matrix `\boldsymbol{C}`, the free vibrations `\boldsymbol{u} e^{\lambda
t}` of `\boldsymbol{M} \ddot{\boldsymbol{q}} + \boldsymbol{C} \dot{\boldsymbol{q}} + \boldsymbol{K}
\boldsymbol{q} = \boldsymbol{0}` give the quadratic eigenvalue problem

.. math::
    :label: ev-qep

    (\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}

solved by ``eigvals, eigvecs = complex_frequency(Kuu, Cuu, Muu,
num_eigvalues=...)``, which returns the complex eigenvalues `\lambda`, one
per complex conjugate pair, with `\mathrm{Im}(\lambda) \ge 0`, and the
complex eigenvectors. With `\lambda = -\zeta \omega_n \pm i \omega_d`, the
damped circular frequency is `\omega_d = |\mathrm{Im}(\lambda)|`,
``abs(eigvals.imag)``, and the damping ratio `\zeta =
-\mathrm{Re}(\lambda)/|\lambda|`, ``-eigvals.real/abs(eigvals)``: a single
dof, `m \lambda^2 + c \lambda + k = 0`, has the roots `\lambda = -\zeta
\omega_n \pm i \omega_n \sqrt{1 - \zeta^2}` for `\zeta < 1`, with `\omega_n
= \sqrt{k/m} = |\lambda|` and `\zeta = c/(2\sqrt{k m})`. A real `\lambda`,
e.g. of an overdamped mode, gives `\omega_d = 0`, and a positive
`\mathrm{Re}(\lambda)` a growing motion, i.e. an unstable system, by
flutter when `\lambda` is complex. With `\boldsymbol{C} = \boldsymbol{0}`, `\lambda = i
\omega_n`, with the `\omega_n` of :func:`.natural_frequency`. This
`\zeta` is not the `\zeta` of the convergence criterion of the non-linear
analyses.

The damping matrix is assembled by the user, e.g. Rayleigh damping
`\boldsymbol{C} = \alpha \boldsymbol{M} + \beta \boldsymbol{K}`, or the aerodynamic damping of
the piston theory, ``Cuu = -mu*CAuu`` with ``update_CA`` of :class:`.Quad4`
and :class:`.Quad4R`, together with the aerodynamic stiffness in ``Kuu =
KC0uu - beta*KA_betauu - gamma*KA_gammauu``, see ``update_KA_beta``. A
flutter analysis then follows the largest `\mathrm{Re}(\lambda)` over the
flow speed: the aerodynamic damping moves the flutter onset beyond the
coalescence of the frequencies of :func:`.natural_frequency`, see
``tests/test_quad4_piston_theory.py``. ``C`` may also contain gyroscopic
terms; ``K``, ``C`` and ``M`` may be unsymmetric.


Choice of the analysis
----------------------

.. list-table::
    :header-rows: 1

    * - Question
      - Analysis
      - Remarks
    * - critical load of a conservative load, linear pre-buckling state
      - :func:`.linear_buckling`, Eq. :eq:`ev-buckling`
      - preloads in the first matrix, Eq. :eq:`ev-preload`
    * - divergence load under a follower load
      - :func:`.linear_buckling` with `\boldsymbol{K}_G + \boldsymbol{K}_{F_{ref}}`
      - unsymmetric; does not detect flutter
    * - frequencies, with or without pre-stress
      - :func:`.natural_frequency`, Eq. :eq:`ev-frequency`
      - verified to miss no frequency for symmetric matrices
    * - flutter of an undamped non-conservative system
      - :func:`.natural_frequency` over the load level
      - coalescence of two frequencies
    * - damped frequencies, flutter with damping
      - :func:`.complex_frequency`, Eq. :eq:`ev-qep`
      - growth rate `\mathrm{Re}(\lambda)`; no count of missed eigenvalues
    * - stability along a non-linear path, imperfections
      - :doc:`nonlinear_analyses` with :doc:`arc_length_methods`
      - limit points of the load


References
----------

- Bolotin, V. V., Nonconservative Problems of the Theory of Elastic
  Stability, Pergamon Press, Oxford, 1963.
- Timoshenko, S. P. and Gere, J. M., Theory of Elastic Stability, 2nd ed.,
  McGraw-Hill, New York, 1961.
- Ziegler, H., Die Stabilitätskriterien der Elastomechanik,
  Ingenieur-Archiv, 20(1), 49-56, 1952. https://doi.org/10.1007/BF00536796
