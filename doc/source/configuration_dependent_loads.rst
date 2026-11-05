Load- and configuration-dependent loads
=======================================

A dead load keeps its magnitude and direction during the deformation. A
follower load does not: a pressure normal to the deformed surface and
acting on the deformed area, a force that follows the rotation of the node
it is applied to, or the aerodynamic load of the piston theory. The load
vector then depends on the configuration, its derivative adds the load
stiffness matrix to the tangent stiffness matrix, and the arc-length
methods must follow the load vector of the current configuration. A load
vector may also depend on the load factor, which is how prescribed
displacements are imposed exactly. This page gives the theory and how such
loads are passed to :class:`.Analysis`, through the optional keyword
arguments of :mod:`pyfe3d.solver.callbacks`.


The load stiffness matrix
-------------------------

With a load `\lambda \boldsymbol{F}(\boldsymbol{u})` proportional to the load factor,
the residual and its derivatives are

.. math::
    :label: cl-residual

    \boldsymbol{R}(\boldsymbol{u}, \lambda) = \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}(\boldsymbol{u}),
    \qquad
    \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{u}} = \boldsymbol{K}_C + \boldsymbol{K}_G + \boldsymbol{K}_F,
    \qquad
    -\frac{\partial \boldsymbol{R}}{\partial \lambda} = \boldsymbol{F}(\boldsymbol{u})

with the load stiffness matrix `\boldsymbol{K}_F = -\lambda \, \partial
\boldsymbol{F}/\partial \boldsymbol{u} = \lambda \boldsymbol{K}_{F_{ref}}`, of the sign that
makes a load growing along the displacement it produces reduce the tangent
stiffness. The integrand of `\boldsymbol{K}_F` has the interpolation of the
virtual displacements on one side and the derivative of the load, e.g.
through the slopes of the loaded surface, on the other, and `\boldsymbol{K}_F` is
symmetric only when the load derives from a potential, i.e. when
`\partial F_i/\partial u_j = \partial F_j/\partial u_i`.

For a pressure normal to the deformed surface, Hibbitt (1979, p. 938)
splits the load stiffness of a uniform pressure into a symmetric surface
integral and an antisymmetric integral along the boundary of the loaded
surface, which vanishes where the displacements are prescribed: a uniform
pressure on a closed surface, or on a surface whose boundaries have
prescribed displacements, gives a symmetric `\boldsymbol{K}_F`, whereas free edges,
where the pressure changes, and a non-uniform pressure give unsymmetric
terms (p. 939). In Table 2 of Schweizerhof and Ramm (1984), the domain terms
are symmetric for a uniform pressure, and for a non-uniform one only when
the pressure field is fixed in space, and the boundary terms vanish under
one of the conditions of their Sec. 3.2.1, Eqs. (32)-(35): the pressure is
zero on the whole boundary; at least two displacement components are
prescribed at the edge; the edge is restrained normal to a continuous
surface, e.g. the undeformed surface, `w = 0` for a plate on the `XY`
plane; or the boundary terms add up to zero.

pyfe3d does not symmetrise `\boldsymbol{K}_F`. The solvers factorize the tangent
stiffness matrix with a sparse LU factorization, which accepts unsymmetric
matrices, such that a consistent `\boldsymbol{K}_F` keeps the quadratic
convergence of the Newton-Raphson method, verified by
``test_newton_raphson_quadratic_with_follower_load`` of
``tests/test_solver_follower_loads.py``. The symmetric part of
`\boldsymbol{K}_F` would lead to the same converged solution, controlled by the
residual alone, with a linear convergence.


Passing the load to the solvers
-------------------------------

The callables of :class:`.Analysis` may declare three optional keyword
arguments, which the solvers pass only to the callables that declare them
explicitly in their signature, see
:func:`pyfe3d.solver.callbacks.accepts_kwarg`; a ``**kwargs`` does not
count, and the simplest callables are called with the minimum set of
arguments:

- ``inc``, the current load factor, passed to ``calc_fint``, ``calc_kC``
  and ``calc_kG``, see :func:`pyfe3d.solver.callbacks.calc_fint` and
  :func:`pyfe3d.solver.callbacks.calc_kT`. ``calc_fint(c, inc=lbd)``
  then contains the configuration-dependent part of the load, `-\lambda
  (\boldsymbol{F}(\boldsymbol{c}) - \boldsymbol{F}_{ext})`, with `\boldsymbol{F}_{ext}` the vector
  of ``calc_fext(inc=1.)``, and ``calc_kC + calc_kG`` its derivative, the
  load stiffness matrix `\boldsymbol{K}_F`. The residual that the solvers form,
  ``calc_fint(c, inc=lbd) - lbd*calc_fext(inc=1.)``, is then
  :eq:`cl-residual`.
- ``c``, the current configuration, passed to ``calc_fext``, which then
  returns the load vector of that configuration for a unit load factor,
  `\boldsymbol{F}(\boldsymbol{c}) = -\partial \boldsymbol{R}/\partial \lambda`, see
  :func:`pyfe3d.solver.callbacks.calc_load_vector` and
  :func:`pyfe3d.solver.callbacks.is_load_configuration_dependent`. The
  arc-length methods use it in the predictor and in the bordered system of
  each iteration, see :doc:`arc_length_methods`, and the linear analysis
  uses it in its load correction iterations, see :doc:`linear_analyses`.
- ``lbd``, the current load factor, passed to ``calc_fext`` together with
  ``c``, for a vector `-\partial \boldsymbol{R}/\partial \lambda` that also
  depends on the load factor, see
  :func:`pyfe3d.solver.callbacks.is_load_factor_dependent`. ``inc`` keeps
  its meaning, the factor that multiplies the returned vector. The vector
  is then evaluated together with the tangent stiffness matrix, at the same
  configuration and load factor, right after ``calc_kC`` and ``calc_kG``,
  such that the callables may share their computations; its linear
  solution at `\boldsymbol{c} = \boldsymbol{0}`, `\lambda = 0` gives the predictor of
  the first Newton-Raphson increment and the scaling of the arc length, and
  `\lambda` times it is a reference of the convergence criteria, see
  :doc:`convergence_criteria`. The linear analysis never passes ``lbd``.

With the elements, ``inc`` and ``c`` are the arguments of a follower load;
``lbd`` is used by :class:`.FreeDofs` for the prescribed displacements.


Prescribed displacements as a load-factor-dependent load
--------------------------------------------------------

A linear constraint imposed exactly, `\boldsymbol{u} = \boldsymbol{Z} \boldsymbol{c} +
\boldsymbol{u}_0 + \lambda \boldsymbol{u}_1`, gives the reduced residual and its
derivative with respect to the load factor

.. math::
    :label: cl-reduced

    \boldsymbol{R}_r(\boldsymbol{c}, \lambda) = \boldsymbol{Z}^\top \left(
    \boldsymbol{F}_{int}(\boldsymbol{Z} \boldsymbol{c} + \boldsymbol{u}_0 + \lambda \boldsymbol{u}_1) -
    \lambda \boldsymbol{F}_{ext} \right), \qquad
    -\frac{\partial \boldsymbol{R}_r}{\partial \lambda} = \boldsymbol{Z}^\top \left(
    \boldsymbol{F}_{ext} - \boldsymbol{K}_T \boldsymbol{u}_1 \right)

with `\boldsymbol{K}_T` at the current configuration and load factor: the vector
that ``calc_fext(inc=1., c=c, lbd=lbd)`` returns. Under a prescribed
displacement alone, the reference vector `\boldsymbol{F}_{ext}` and the reduced
internal force vector both vanish at equilibrium, and only this vector
gives the solvers a scale. It is not the reaction: it acts on the free dofs,
whereas the reaction is the residual of the full model at the constrained
dofs. :class:`.FreeDofs` implements the case of `\boldsymbol{Z}` made of columns
of the identity, `\boldsymbol{u}_0 = \boldsymbol{0}` and `\boldsymbol{u}_1 = \bar{\boldsymbol{u}}_k`,
see :doc:`boundary_conditions`; ``tests/test_solver_load_factor_dependent_load.py``
checks a general constraint on hardening springs in series against their
exact solution.


Follower loads of pyfe3d
------------------------

- The aerodynamic load of the piston theory, `\lambda \beta
  \boldsymbol{K}_{A_\beta} \boldsymbol{u}`, assembled with ``update_KA_beta`` of
  :class:`.Quad4` and :class:`.Quad4R`, is linear in the configuration and
  gives the constant, unsymmetric load stiffness `-\beta
  \boldsymbol{K}_{A_\beta}` per unit load factor, see
  ``test_piston_theory_load_unsymmetric_tangent`` of
  ``tests/test_solver_nonlinear_static.py``; its stability is studied with
  the eigenvalue analyses of :doc:`eigenvalue_analyses`.
- The concentrated moments of :meth:`.CorotationalModel.add_moment` with
  ``kind='fixed_axes'``, about the fixed global axes, which are
  non-conservative and give an unsymmetric tangent, or
  ``kind='follower'``, rotating with the nodal triad, depend on the
  configuration and enter ``calc_fint``, ``calc_kG`` and ``calc_fext(c=...)``
  of the model as described above; the moments in the attribute ``fext``
  of the model, conjugate to the rotation vectors, are constant.
- A pressure or any other follower load of the user enters the callables
  in the same way; when it is affine in the displacements, `\boldsymbol{F}(\boldsymbol{u})
  = \boldsymbol{F}_0 - \boldsymbol{K}_{F_{ref}} \boldsymbol{u}`, its load stiffness is constant and
  the geometrically linear problem is the unsymmetric linear system of
  :doc:`linear_analyses`.

In a linear buckling analysis, the load stiffness of the reference load
adds to the geometric stiffness matrix, and a non-symmetric `\boldsymbol{K}_F` may
lead to flutter instead of divergence, see :doc:`eigenvalue_analyses`.


References
----------

- Hibbitt, H. D., Some follower forces and load stiffness, International
  Journal for Numerical Methods in Engineering, 14(6), 937-941, 1979.
  https://doi.org/10.1002/nme.1620140613
- Schweizerhof, K. and Ramm, E., Displacement dependent pressure loads in
  nonlinear finite element analyses, Computers & Structures, 18(6),
  1099-1114, 1984. https://doi.org/10.1016/0045-7949(84)90154-8
