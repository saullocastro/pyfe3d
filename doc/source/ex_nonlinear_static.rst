Non-linear static analysis with pyfe3d.solver
=============================================

The geometrically non-linear analyses of :class:`.Analysis` receive the
callables of a model, see :doc:`nonlinear_analyses`. A simply supported
plate with immovable edges under a uniform pressure, whose linear
deflection is about three times the thickness, such that the membrane
action stiffens the plate:

.. literalinclude:: ../../tests/test_nonlinear_static_newton_raphson.py
    :pyobject: plate

The class below wraps a list of elements into the four callables, with the
vectors and matrices of all the dofs: ``calc_fint`` with ``update_fint(...,
nonlinear=1)``, ``calc_kC`` with ``KC0`` and ``KCNL``, ``calc_kG`` with
``KG``. The scaling ``S`` changes the units of the rotations, which leaves
Crisfield's convergence criterion unchanged:

.. literalinclude:: ../../tests/test_solver_nonlinear_static.py
    :pyobject: Model

The full load in a single increment, with the free dofs given by ``bu``:
the solution is the one of a Newton-Raphson loop written by hand with the
element matrices, and the convergence criterion falls quadratically:

.. literalinclude:: ../../tests/test_solver_nonlinear_static.py
    :pyobject: test_newton_raphson_reproduces_the_hand_written_loop


Prescribed displacements
------------------------

The central deflection of the load-controlled analysis, prescribed with
``uk`` at a constrained dof, with the Newton-Raphson and the Riks methods:
the displacement field is the same and the reaction equals the load:

.. literalinclude:: ../../tests/test_solver_nonlinear_static.py
    :pyobject: point_loaded_plate

.. literalinclude:: ../../tests/test_solver_nonlinear_static.py
    :pyobject: test_prescribed_displacement_and_reaction


Configuration-dependent load
----------------------------

The aerodynamic load of the piston theory, `\lambda \beta
\boldsymbol{K}_{A_\beta} \boldsymbol{u}`, from ``update_KA_beta`` of :class:`.Quad4R`, enters
``calc_fint`` with the load factor ``inc``, and its unsymmetric load
stiffness enters ``calc_kC``; ``calc_fext(c=...)`` gives the load vector of
the current configuration:

.. literalinclude:: ../../tests/test_solver_nonlinear_static.py
    :pyobject: piston_theory_plate

.. literalinclude:: ../../tests/test_solver_nonlinear_static.py
    :pyobject: test_piston_theory_load_unsymmetric_tangent


Limit points with the arc-length methods
----------------------------------------

The shallow von Mises truss of ``tests/analytic_problems.py``, loaded
through a linear spring beyond its limit load: the arc-length methods trace the
snap-through, along which the load factor becomes negative, until the load
factor 1.0 on the inverted configuration:

.. literalinclude:: ../../tests/analytic_problems.py
    :pyobject: von_mises_truss

.. literalinclude:: ../../tests/test_solver_arc_length.py
    :pyobject: test_arc_length_snap_through

The previous converged point is at the distance of the arc length from the
last one, and the iterations of a step may converge to it, especially with
long steps or near a limit point: such a step, whose increment points against
the tangent predictor, is repeated with a shorter arc length, such that the
path is not traced backwards:

.. literalinclude:: ../../tests/test_solver_arc_length.py
    :pyobject: test_arc_length_no_doubling_back


The snap-through of a low sinusoidal arch of :class:`.BeamC` elements, hinged
at both ends under a sinusoidal load, against the limit loads of Fung and
Kaplan (1952), at the default tolerances: the null load between the limit
points is passed thanks to the round-off floor of the convergence criteria,
see :doc:`nonlinear_analyses`:

.. literalinclude:: ../../tests/test_solver_arc_length_fung_kaplan.py
    :pyobject: arch

.. literalinclude:: ../../tests/test_solver_arc_length_fung_kaplan.py
    :pyobject: test_fung_kaplan_beam_arch
