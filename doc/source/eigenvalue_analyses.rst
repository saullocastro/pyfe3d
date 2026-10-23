Linear buckling and natural frequencies
=======================================

The eigenvalue analyses use the matrices of the free dofs, e.g.
``KC0uu = KC0[bu, :][:, bu]``, and are solved with
:func:`.linear_buckling` and :func:`.natural_frequency` of
:mod:`pyfe3d.solver.eigen`, which return the eigenvectors with the size of
the matrices received, to be expanded by the user, ``u[bu] =
eigvecs[:, i]``.


Linear buckling
---------------

A linear buckling analysis has three steps:

1. the linear static solution `\{u_0\}` of a reference load `\{F_{ref}\}`,
   `[K_{C0}]_{uu}\{u_0\}_u = \{F_{ref}\}_u`, see :doc:`overview`;
2. the geometric stiffness matrix of the prestress of `\{u_0\}`: for each
   element, ``update_probe_xe(x)``, ``update_probe_ue(u0)`` and
   ``update_KG(KGr, KGc, KGv, prop)``. For the shells, a known constant
   membrane state can be given instead with ``update_KG_given_stress(Nxx,
   Nyy, Nxy, KGr, KGc, KGv)``, with the stress resultants in the element
   coordinate system of each element, see :doc:`conventions`;
3. the eigenvalue problem

   .. math::

       ([K_{C0}] + \lambda [K_G])\{u\} = \{0\}

   solved by ``eigvals, eigvecs = linear_buckling(KC0uu, KGuu,
   num_eigvalues=...)``.

`[K_G]` is linear in `\{u_0\}`, hence in the reference load, and the
critical load is `\lambda_1 \{F_{ref}\}`, with `\lambda_1` = ``eigvals[0]``.
:func:`.linear_buckling` returns the positive load multipliers first, in
increasing order, followed, when fewer than ``num_eigvalues`` positive ones
exist, by the negative ones closest to zero, which are the critical load
multipliers of the reversed reference load. A load that is not scaled by
`\lambda`, e.g. a preload with the geometric stiffness matrix `[K_{G0}]`,
belongs to the first matrix, ``linear_buckling(KC0uu + KG0uu, KGuu)``,
which is also how the critical value of one load component is found under
the other components kept constant.

The beams have ``update_KG`` for the axial force of the prestress, see
``tests/test_beam_linear_buckling_3D_column.py``; :class:`.Truss` and
:class:`.Spring` have no geometric stiffness matrix.


Natural frequencies
-------------------

The frequency problem

.. math::

    ([K] - \omega_n^2 [M])\{u\} = \{0\}

is solved by ``omegan, eigvecs = natural_frequency(Kuu, Muu,
num_eigvalues=...)``, which returns the circular frequencies `\omega_n`,
in rad/s with consistent units, in increasing order. `[K]` is
`[K_{C0}]`, plus the geometric stiffness matrix of a prestress when it
applies, e.g. ``KC0uu + KGuu``, see
``tests/test_quad4r_natural_freq_pre_stress.py``. The mass matrix is
assembled with ``update_M``, whose ``mtype`` selects the consistent or the
lumped mass matrix of each element.

With an unsymmetric `[K]`, e.g. with the aerodynamic stiffness matrix of
the piston theory or the load stiffness of a follower load, the
frequencies may become complex; ``omegan`` is then the principal square
root of the complex eigenvalues. Two frequencies that coalesce and become
complex indicate the loss of stability by flutter, which
:func:`.linear_buckling` does not detect: this is the dynamic method of
Bolotin (1963), see :func:`.natural_frequency`.


Verification and dense fallback
-------------------------------

Both functions remove the null rows and columns of the first matrix, see
:func:`.remove_null_cols`, equilibrate the diagonal of the matrices with
:func:`.diagonal_preconditioner`, which does not change the eigenvalues,
and select the solvers of symmetric or unsymmetric matrices with
:func:`.is_symmetric`. Then:

- the sparse solvers of SciPy run first, in shift-invert or Cayley mode,
  with a shift estimated from the matrices for the linear buckling,
  :func:`.estimate_cayley_sigma`;
- every result is verified: the relative residual of each eigenpair and,
  for symmetric matrices, the absence of an eigenvalue missed below the
  highest one returned, from the count of the negative pivots of a shifted
  matrix, see :func:`.check_eigenpairs` and :func:`.is_positive_definite`;
- when the sparse solvers fail, a dense solver of SciPy is used as a last
  resort, when the dense problem has at most ``max_dense_size`` dofs, with a
  :class:`.DenseFallbackWarning` giving the reasons; otherwise a
  ``RuntimeError`` lists every failure, such that no unverified result is
  returned silently;
- ``sparse_solver=False`` solves the full problem with a dense solver,
  returning all the eigenvalues, as a reference for validations.

The fallback can be turned into an error, e.g. in a test suite::

    import warnings
    from pyfe3d.solver import DenseFallbackWarning
    warnings.simplefilter('error', DenseFallbackWarning)

With Intel MKL 2024.2.0 to 2025.0.0, the symmetric solver of ARPACK
returns wrong eigenpairs when its number of Lanczos vectors exceeds 32;
these versions are detected and the number is capped, see
:mod:`pyfe3d.solver.arpackutils`, also for the override with the
environment variable ``PYFE3D_ARPACK_MAX_NCV``.


Examples
--------

The reference load, a preload and the reversed reference load of a
:class:`.Quad4` plate:

.. literalinclude:: ../../tests/test_solver_linear_buckling.py
    :pyobject: test_quad4_plate_reference_load

The frequencies of the same plate under a compressive prestress, whose
lowest one vanishes at the critical load multiplier:

.. literalinclude:: ../../tests/test_solver_natural_frequency.py
    :pyobject: test_quad4_plate_prestress

Complete models are in :doc:`ex_linear_buckling_plate`,
:doc:`ex_linear_buckling_cylinder_torsion`, :doc:`ex_natural_frequency_plate`
and :doc:`ex_natural_frequency_cylinder`.
