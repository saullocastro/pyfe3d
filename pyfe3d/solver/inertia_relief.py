r"""
=========================================================================
Static analysis with inertia relief (:mod:`pyfe3d.solver.inertia_relief`)
=========================================================================

.. currentmodule:: pyfe3d.solver.inertia_relief

:func:`apply_inertia_relief` is the linear static analysis of a structure
without supports, e.g. an aircraft or a spacecraft in flight, whose loads
need not be balanced: the rigid-body accelerations are computed from the
mass matrix, and the displacements relative to a statically determinate
reference support from the loads plus the d'Alembert forces of these
accelerations. The user guide is :doc:`inertia_relief`.

"""
import numpy as np
from numpy.linalg import LinAlgError
from scipy.sparse import csc_matrix

from .sparseutils import (check_null_rows_cols, is_symmetric,
                          _not_positive_definite_message)


def _bool_dofs(b, N, name):
    b = np.asarray(b)
    if b.dtype != bool or b.shape != (N,):
        raise ValueError('%s must be a Boolean array of shape (%d,), the '
                         'number of rows of KC0, got dtype %s and shape %s'
                         % (name, N, b.dtype, b.shape))
    return b


def apply_inertia_relief(KC0, M, fext, bref, bfix=None,
                         return_reactions=False, rtol=1.e-6,
                         pivot_rtol=1.e-13):
    r"""Linear static analysis with inertia relief

    Solves the linear static problem of an unsupported structure, whose
    stiffness matrix ``KC0`` is singular, under loads ``fext`` that need
    not be balanced. The rigid-body modes are computed from ``KC0`` with
    the reference support ``bref``, the rigid-body accelerations from the
    mass matrix ``M``, and the displacements relative to the reference
    support from ``fext`` plus the d'Alembert (inertia) forces of these
    accelerations. The reference support must be statically determinate,
    which is checked.

    Parameters
    ----------
    KC0 : sparse_matrix
        Stiffness matrix of all the ``N`` degrees-of-freedom, shape ``(N,
        N)``, symmetric, without supports, e.g. as assembled by the
        ``update_KC0`` methods of the elements. Force per unit
        displacement.
    M : sparse_matrix or np.ndarray
        Mass matrix of the same degrees-of-freedom, shape ``(N, N)``, e.g.
        as assembled by the ``update_M`` methods of the elements, consistent
        or lumped, plus any point masses. Mass, in units consistent with
        ``KC0`` and ``fext``.
    fext : np.ndarray
        External force vector, 1-D array of size ``N``. Force.
    bref : np.ndarray
        Boolean array of size ``N``, ``True`` at the reference
        degrees-of-freedom, a statically determinate support whose
        displacements are set to zero: as many as the rigid-body modes left
        by ``bfix``, e.g. six for a free structure in 3D.
    bfix : np.ndarray or None, optional
        Boolean array of size ``N``, ``True`` at the degrees-of-freedom
        constrained by the model itself, set to zero, e.g. a plane of
        symmetry of a half model, or the degrees-of-freedom of a model in
        one or two dimensions that are not used. It must not overlap
        ``bref``. Default ``None``, no such constraint.
    return_reactions : bool, optional
        Also return the residual ``R``. Default ``False``.
    rtol : float, optional
        Relative tolerance of the check that the columns of ``Phi`` are
        rigid-body modes, ``max|KC0 @ Phi[:, j]| <= rtol*max|KC0|*max|Phi[:,
        j]|``. Default ``1.e-6``.
    pivot_rtol : float, optional
        ``KC0uu`` is taken as singular when the smallest pivot of its
        factorization is at most ``pivot_rtol`` times the largest. Default
        ``1.e-13``: a rigid-body mode left free by the supports gives a
        pivot of the order of the round-off, below ``1.e-15`` times the
        largest, whereas the beam models of the tests, up to 200 elements,
        and the airplane of ``tests/test_inertia_relief_ha144a.py`` have
        ratios above ``1.e-7``.

    Returns
    -------
    u : np.ndarray
        Displacements relative to the reference support, 1-D array of size
        ``N``, zero at ``bref`` and ``bfix``.
    acc : np.ndarray
        Rigid-body accelerations `\ddot{\boldsymbol{\alpha}}`, 1-D array of
        size ``nr = bref.sum()``: the accelerations of the reference
        degrees-of-freedom, in their order in ``KC0``, translational or
        angular.
    Phi : np.ndarray
        Rigid-body modes, shape ``(N, nr)``, with ``Phi[bref]`` the identity
        matrix: column ``j`` is the rigid-body motion with a unit
        displacement at the ``j``-th reference degree-of-freedom and zero at
        the others. The rigid-body accelerations of all the
        degrees-of-freedom are ``Phi @ acc``, and the d'Alembert forces
        ``-M @ Phi @ acc``.
    R : np.ndarray
        Only with ``return_reactions=True``. The residual ``KC0 @ u - fext +
        M @ Phi @ acc``, 1-D array of size ``N``: zero at the free
        degrees-of-freedom; at ``bref``, the reactions of the reference
        support, zero up to the round-off; at ``bfix``, the reactions of the
        constraints of the model.

    Raises
    ------
    ValueError
        When the shapes do not match, when ``bref`` is empty or overlaps
        ``bfix``, or when ``KC0`` is not symmetric, see
        :func:`.is_symmetric`.
    numpy.linalg.LinAlgError
        When the reference support does not suppress all the rigid-body
        modes: ``KC0uu``, the stiffness matrix of the free
        degrees-of-freedom, has a null row or column, see
        :func:`.check_null_rows_cols`, is not positive definite, with
        the message of :func:`.linear_static`, or is singular up to the
        round-off, see ``pivot_rtol``; when the reference support
        is statically indeterminate, i.e. a column of ``Phi`` is not a
        rigid-body mode, since ``bref`` has more degrees-of-freedom than
        the rigid-body modes; or when ``Phi.T @ M @ Phi`` is not positive
        definite, i.e. a rigid-body mode has no mass.

    See Also
    --------
    pyfe3d.solver.static.linear_static : the linear static analysis of a
        supported structure.

    Notes
    -----
    The degrees-of-freedom are partitioned as in :doc:`linear_analyses`:
    the known set `k`, ``bk = bref | bfix``, with zero displacements, and
    the free set `u`, ``bu = ~bk``. The `n_r` rigid-body modes, the columns
    of `\boldsymbol{\Phi}`, satisfy `\boldsymbol{K}_{C0} \boldsymbol{\Phi}
    = \boldsymbol{0}`, and a static solution of `\boldsymbol{K}_{C0}
    \boldsymbol{u} = \boldsymbol{F}_{ext}` exists only for loads with
    `\boldsymbol{\Phi}^\top \boldsymbol{F}_{ext} = \boldsymbol{0}`, i.e.
    without a resultant force or moment [3]_, [4]_. With unit
    displacements at the reference degrees-of-freedom and zero at those of
    ``bfix``, the rigid-body modes follow from the stiffness matrix, [1]_,
    p. 6, Eq. (19):

    .. math::
        \boldsymbol{\Phi}_k = \begin{bmatrix} \boldsymbol{I} \\
        \boldsymbol{0} \end{bmatrix}, \qquad
        \boldsymbol{\Phi}_u = -\boldsymbol{K}_{{C0}_{uu}}^{-1}
        \boldsymbol{K}_{{C0}_{uk}} \boldsymbol{\Phi}_k

    with the rows of ``bref`` first and those of ``bfix`` second in
    `\boldsymbol{\Phi}_k`, where `\boldsymbol{K}_{{C0}_{uu}}` is positive
    definite only when the supports suppress all the rigid-body modes [2]_,
    Sec. 8.2.5, pp. 726-727. Under steady loads, the structure deforms elastically and
    accelerates as a rigid body, `\ddot{\boldsymbol{u}} = \boldsymbol{\Phi}
    \ddot{\boldsymbol{\alpha}}`; premultiplying `\boldsymbol{M}
    \ddot{\boldsymbol{u}} + \boldsymbol{K}_{C0} \boldsymbol{u} =
    \boldsymbol{F}_{ext}` by `\boldsymbol{\Phi}^\top` gives the rigid-body
    accelerations [1]_, p. 6, Eqs. (21) and (22):

    .. math::
        \boldsymbol{\Phi}^\top \boldsymbol{M} \boldsymbol{\Phi} \,
        \ddot{\boldsymbol{\alpha}} = \boldsymbol{\Phi}^\top
        \boldsymbol{F}_{ext}

    and the displacements relative to the reference support, `\boldsymbol{u}_k
    = \boldsymbol{0}`, are [1]_, p. 7, Eqs. (23) and (24):

    .. math::
        \boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{F}_{{ext}_u}
        - \boldsymbol{M}_u \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}

    the free rows of the linear static system with the d'Alembert forces
    `-\boldsymbol{M} \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}`, whose
    sum with `\boldsymbol{F}_{ext}` satisfies `\boldsymbol{\Phi}^\top
    (\boldsymbol{F}_{ext} - \boldsymbol{M} \boldsymbol{\Phi}
    \ddot{\boldsymbol{\alpha}}) = \boldsymbol{0}`. The reactions are the
    residual at the known degrees-of-freedom, as in [2]_, Sec. 4.2.2, Eq.
    (4.45). Since the residual is zero at the free degrees-of-freedom and
    `\boldsymbol{\Phi}^\top \boldsymbol{K}_{C0} = \boldsymbol{0}`,
    premultiplying it by `\boldsymbol{\Phi}^\top` gives that the reactions
    of the reference support are zero, and the support does not change the
    strains and the internal forces: another statically determinate
    support changes `\boldsymbol{u}` by a rigid-body motion
    `\boldsymbol{\Phi} \boldsymbol{\alpha}` only. For balanced loads,
    `\ddot{\boldsymbol{\alpha}} = \boldsymbol{0}` and the reference support
    is a fictitious support of the singular static problem.

    ``KC0uu`` is factorized once, as in :func:`.linear_static`, and the
    factorization gives both `\boldsymbol{\Phi}_u` and
    `\boldsymbol{u}_u`. The methods for large singular systems without a
    reference support, which compute the null space of `\boldsymbol{K}_{C0}`
    during the factorization, are discussed in [3]_, Secs. 1-4.

    References
    ----------
    .. [1] Barnett, A. R., Widrick, T. W. and Ludwiczak, D. R., Closed-Form
       Static Analysis with Inertia Relief and Displacement-Dependent Loads
       Using a MSC/NASTRAN DMAP Alter, NASA TM-106836, NASA Lewis Research
       Center, Cleveland, OH, 1995.
       https://ntrs.nasa.gov/citations/19950013233
    .. [2] Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
       Watertown, MA, 2014, ISBN 978-0-9790049-5-7.
    .. [3] Farhat, C. and Géradin, M., On the general solution by a direct
       method of a large-scale singular system of linear equations:
       application to the analysis of floating structures, International
       Journal for Numerical Methods in Engineering, 41(4), 675-696, 1998.
       https://doi.org/10.1002/%28SICI%291097-0207%2819980228%2941%3A4%3C675%3A%3AAID-NME305%3E3.0.CO%3B2-8
    .. [4] Cavaliere, F., Zlotnik, S., Sevilla, R., Larráyoz, X. and Díez,
       P., Nonintrusive reduced order model for parametric solutions of
       inertia relief problems, International Journal for Numerical Methods
       in Engineering, 122(16), 4270-4291, 2021, Sec. 3, pp. 4274-4275.
       https://doi.org/10.1002/nme.6702

    Examples
    --------
    Three point masses along `x`, ``m = (1, 2, 3)``, joined by the springs
    ``k1`` and ``k2``, with the force ``F`` at the third mass and the
    reference support at the first one: the rigid-body acceleration is
    ``F/(m1 + m2 + m3)``, and the spring forces ``k1*u[1] = m1*acc`` and
    ``k2*(u[2] - u[1]) = (m1 + m2)*acc`` accelerate the masses on their
    left, see :doc:`tut-inertia-relief`.

    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from pyfe3d.solver import apply_inertia_relief
    >>> k1, k2, F = 2000., 500., 12.
    >>> KC0 = csc_matrix([[k1, -k1, 0.], [-k1, k1 + k2, -k2], [0., -k2, k2]])
    >>> M = csc_matrix(np.diag([1., 2., 3.]))
    >>> bref = np.array([True, False, False])
    >>> fext = np.array([0., 0., F])
    >>> u, acc, Phi = apply_inertia_relief(KC0, M, fext, bref)
    >>> print(acc, u, Phi.ravel())
    [2.] [0.    0.001 0.013] [1. 1. 1.]

    """
    KC0 = csc_matrix(KC0)
    N = KC0.shape[0]
    if KC0.shape != (N, N):
        raise ValueError('KC0 must be square, got shape %s' % (KC0.shape,))
    if M.shape != (N, N):
        raise ValueError('M must have the shape of KC0, (%d, %d), got %s'
                         % (N, N, M.shape))
    fext = np.asarray(fext, dtype=float)
    if fext.shape != (N,):
        raise ValueError('fext must be a 1-D array of size %d, got shape %s'
                         % (N, fext.shape))
    bref = _bool_dofs(bref, N, 'bref')
    bfix = (np.zeros(N, dtype=bool) if bfix is None
            else _bool_dofs(bfix, N, 'bfix'))
    nr = int(bref.sum())
    if nr == 0:
        raise ValueError('bref has no degree-of-freedom: the reference '
                         'support must suppress the rigid-body modes')
    if np.any(bref & bfix):
        raise ValueError('bref and bfix overlap at the degrees-of-freedom %s'
                         % np.flatnonzero(bref & bfix).tolist())
    if not is_symmetric(KC0):
        raise ValueError('KC0 must be symmetric')

    from .eigen import _symmetric_lu
    bu = ~(bref | bfix)
    Kuu = KC0[bu, :][:, bu]
    check_null_rows_cols(Kuu, 'KC0uu')
    lu, count = _symmetric_lu(Kuu)
    hint = ('. In inertia relief, the reference support bref, with bfix, '
            'must suppress every rigid-body mode: too few reference '
            'degrees-of-freedom, or ones that do not restrain all the '
            'rigid-body modes, leave KC0uu singular')
    if count != 0:
        raise LinAlgError(
            _not_positive_definite_message('Inertia relief', 'KC0uu', count)
            + hint)
    d = np.abs(lu.U.diagonal())
    if d.min() <= pivot_rtol*d.max():
        raise LinAlgError(
            'Inertia relief: the stiffness matrix KC0uu is singular up to '
            'the round-off, its smallest pivot is %.1e times the largest, '
            'below pivot_rtol = %g, i.e. a deformation mode without strain '
            'energy' % (d.min()/d.max(), pivot_rtol) + hint)

    # rigid-body modes, Phi_k = I at bref
    Kuref = KC0[bu, :][:, bref].toarray()
    Phi = np.zeros((N, nr))
    Phi[bref] = np.eye(nr)
    Phi[bu] = -lu.solve(np.asfortranarray(Kuref)).reshape(bu.sum(), nr)
    KPhi = np.abs(KC0 @ Phi).max(axis=0)
    scale = abs(KC0).max()*np.abs(Phi).max(axis=0)
    bad = np.flatnonzero(KPhi > rtol*scale)
    if bad.size:
        raise LinAlgError(
            'Inertia relief: the reference support is statically '
            'indeterminate: the columns %s of Phi, with a unit displacement '
            'at the reference degrees-of-freedom %s, are not rigid-body '
            'modes, max|KC0 @ Phi[:, j]|/(max|KC0| max|Phi[:, j]|) = %s > '
            'rtol = %g. Choose as many reference degrees-of-freedom as '
            'rigid-body modes, such that they suppress all of them'
            % (bad.tolist(), np.flatnonzero(bref)[bad].tolist(),
               np.array2string(KPhi[bad]/scale[bad], precision=1), rtol))

    # rigid-body accelerations
    Mrr = Phi.T @ (M @ Phi)
    try:
        np.linalg.cholesky(Mrr)
    except LinAlgError as e:
        raise LinAlgError(
            'Inertia relief: Phi.T @ M @ Phi is not positive definite, a '
            'rigid-body mode has no mass, e.g. a rotation about the axis '
            'of a beam without rotary inertia: %s' % e) from e
    acc = np.linalg.solve(Mrr, Phi.T @ fext)

    # displacements relative to the reference support
    ftot = fext - M @ (Phi @ acc)
    u = np.zeros(N)
    u[bu] = lu.solve(np.ascontiguousarray(ftot[bu]))
    if return_reactions:
        R = KC0 @ u - ftot
        return u, acc, Phi, R
    return u, acc, Phi
