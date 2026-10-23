r"""
=============================================================================
Static analyses of a model given by callables (:mod:`pyfe3d.solver.analysis`)
=============================================================================

.. currentmodule:: pyfe3d.solver.analysis

The class :class:`Analysis` receives the callables that compute the force
vectors and the stiffness matrices of a model and runs a linear static
analysis, or a non-linear static analysis with the Newton-Raphson solver of
:mod:`.newton_raphson` or the arc-length solvers of :mod:`.arc_length`, with
Crisfield's convergence criterion by default, see :mod:`.convergence`. The
optional keyword arguments of the callables are described in
:mod:`.callbacks`. With ``bu``, the callables work with the vectors and
matrices of all the degrees-of-freedom, as assembled by the elements of
this package, and :class:`FreeDofs` restricts them to the free ones and
imposes the prescribed displacements ``uk``.

"""
import numpy as np
from scipy.sparse import csr_matrix

from .logger import msg, warn
from .static import solve, _factorize
from .convergence import check_convergence
from .newton_raphson import _solver_NR, _weights
from .arc_length import _solver_arc_length
from . import callbacks


class FreeDofs(object):
    r"""Callables of the free degrees-of-freedom, with prescribed
    displacements imposed exactly

    Wraps the callables of the full model, which receive and return the
    vectors and matrices of all the ``N`` degrees-of-freedom, such that the
    solvers see the free degrees-of-freedom ``bu`` only. The displacements of
    the constrained degrees-of-freedom are ``lbd*uk[~bu]``, prescribed and
    proportional to the load factor ``lbd``, or null when ``uk`` is
    ``None``. Used by :class:`Analysis` when ``bu`` is given.

    The wrapped ``calc_fint``, ``calc_kC`` and ``calc_kG`` always declare
    ``inc``, which they need to build the full configuration, and pass it
    to the callables of the full model only when they declare it, see
    :mod:`.callbacks`. The wrapped ``calc_fext`` declares ``c`` when the one
    of the full model declares it or when displacements are prescribed, and
    ``lbd`` when the one of the full model declares it or when displacements
    are prescribed.

    Parameters
    ----------
    calc_fext, calc_fint, calc_kC, calc_kG : callable
        The callables of the full model, see :class:`Analysis`.
    bu : array_like
        Boolean mask of the free degrees-of-freedom, 1-D array of size ``N``.
    uk : array_like or None, optional
        Prescribed displacements for a unit load factor, 1-D array of size
        ``N`` whose entries at the free degrees-of-freedom are ignored, of
        the dimension of the displacements and rotations. ``None`` for null
        prescribed displacements. Default ``None``.

    Attributes
    ----------
    free, known : np.ndarray
        Indices of the free and of the constrained degrees-of-freedom.
    prescribed : bool
        Whether a prescribed displacement is not null.

    Raises
    ------
    ValueError
        When ``bu`` is not a boolean array, or when ``uk`` does not have the
        size of ``bu``.

    Notes
    -----
    With `f` the free and `k` the constrained degrees-of-freedom, the
    configuration of the full model is

    .. math::

        \{u\} = [T]\{c\} + \lambda \{u_k\}, \qquad
        [T] = \begin{bmatrix} [I_{ff}] \\ [0_{kf}] \end{bmatrix}

    the null-space reduction of the constraints `\{u_k\} = \lambda
    \{\bar{u}_k\}` imposed exactly, see :mod:`.callbacks`. The solvers see
    the reduced residual `\{R_f\} = \lambda \{F_{ext}\}_f - \{F_{int}(u)\}_f`,
    its Jacobian `[K_T]_{ff}` and

    .. math::

        \frac{\partial R_f}{\partial \lambda} = \{F_{ext}\}_f -
        [K_T]_{fk}\{\bar{u}_k\}

    with `[K_T]` at the current configuration and load factor, reused from
    the last evaluation of the tangent stiffness matrix when it was done at
    the same configuration and load factor. The norms of the convergence
    criteria are thus over the free degrees-of-freedom only. The reaction
    forces are `\{F_{int}(u)\}_k - \lambda \{F_{ext}\}_k`, given by the
    callables of the full model at a converged configuration.

    """
    def __init__(self, calc_fext, calc_fint, calc_kC, calc_kG, bu, uk=None):
        bu = np.asarray(bu)
        if bu.dtype != bool or bu.ndim != 1:
            raise ValueError('bu must be a 1-D boolean array')
        self.size = bu.size
        self.free = np.flatnonzero(bu)
        self.known = np.flatnonzero(~bu)
        if uk is None:
            self.uk = np.zeros(self.known.size)
        else:
            uk = np.asarray(uk, dtype=float)
            if uk.shape != bu.shape:
                raise ValueError('uk must have the size of bu')
            self.uk = uk[self.known]
        self.prescribed = bool(np.any(self.uk != 0))
        self.full_fext = calc_fext
        self.full_fint = calc_fint
        self.full_kC = calc_kC
        self.full_kG = calc_kG
        self._kC = None
        self._kG = None
        fext_c = (calc_fext is not None
                  and callbacks.accepts_kwarg(calc_fext, 'c'))
        fext_lbd = (calc_fext is not None
                    and callbacks.accepts_kwarg(calc_fext, 'lbd'))
        #: whether the external load of the full model depends on the
        #: configuration
        self.config_dependent = fext_c
        if self.prescribed or fext_lbd:
            def wrapped_fext(inc=1., silent=True, c=None, lbd=None):
                return self._fext(inc, silent, c, lbd)
        elif fext_c:
            def wrapped_fext(inc=1., silent=True, c=None):
                return self._fext(inc, silent, c, None)
        else:
            def wrapped_fext(inc=1., silent=True):
                return self._fext(inc, silent, None, None)
        self.calc_fext = wrapped_fext

    def full(self, c, lbd):
        r"""Configuration of the full model

        Parameters
        ----------
        c : np.ndarray
            Configuration of the free degrees-of-freedom.
        lbd : float
            Load factor, dimensionless.

        Returns
        -------
        u : np.ndarray
            Configuration of all the degrees-of-freedom, 1-D array of size
            ``N``, with ``lbd*uk`` at the constrained ones.

        """
        c = np.asarray(c)
        u = np.zeros(self.size, dtype=np.result_type(c, float))
        u[self.free] = c
        if self.prescribed:
            u[self.known] = lbd*self.uk
        return u

    def _reduce(self, A):
        r"""Free-free block of a matrix of the full model, in CSR format"""
        A = csr_matrix(A)
        return A[self.free, :][:, self.free]

    def calc_fint(self, c, silent=True, inc=1.):
        r"""Internal force vector of the free degrees-of-freedom

        Parameters
        ----------
        c : np.ndarray
            Configuration of the free degrees-of-freedom.
        silent : bool, optional
            Passed to the callable of the full model. Default ``True``.
        inc : float, optional
            Load factor, dimensionless. Default ``1.``.

        Returns
        -------
        fint : np.ndarray
            ``calc_fint`` of the full model at :meth:`full`, restricted to
            the free degrees-of-freedom.

        """
        kwargs = (dict(inc=inc)
                  if callbacks.accepts_kwarg(self.full_fint, 'inc') else {})
        fint = self.full_fint(c=self.full(c, inc), silent=silent,
                              **kwargs)
        return np.asarray(fint)[self.free]

    def _matrix(self, func, c, NLgeom, silent, inc):
        r"""Matrix of the full model ``func`` at :meth:`full`, or the linear
        one when ``c`` is ``None`` or ``NLgeom=False``"""
        if c is None or not NLgeom:
            return func(silent=silent), None
        kwargs = dict(inc=inc) if callbacks.accepts_kwarg(func, 'inc') else {}
        A = func(c=self.full(c, inc), NLgeom=True, silent=silent, **kwargs)
        return A, (np.array(c, copy=True), inc)

    def calc_kC(self, c=None, NLgeom=False, silent=True, inc=1.):
        r"""Constitutive stiffness matrix of the free degrees-of-freedom

        Parameters
        ----------
        c : np.ndarray or None, optional
            Configuration of the free degrees-of-freedom. ``None`` gives the
            linear stiffness matrix. Default ``None``.
        NLgeom : bool, optional
            Passed to the callable of the full model; ``False`` gives the
            linear stiffness matrix. Default ``False``.
        silent : bool, optional
            Passed to the callable of the full model. Default ``True``.
        inc : float, optional
            Load factor, dimensionless. Default ``1.``.

        Returns
        -------
        kC : sparse_matrix
            Free-free block of ``calc_kC`` of the full model.

        """
        A, key = self._matrix(self.full_kC, c, NLgeom, silent, inc)
        self._kC = (key, A) if key is not None else None
        return self._reduce(A)

    def calc_kG(self, c=None, NLgeom=False, silent=True, inc=1.):
        r"""Geometric stiffness matrix of the free degrees-of-freedom

        Parameters
        ----------
        c : np.ndarray or None, optional
            Configuration of the free degrees-of-freedom. Default ``None``.
        NLgeom : bool, optional
            Passed to the callable of the full model. Default ``False``.
        silent : bool, optional
            Passed to the callable of the full model. Default ``True``.
        inc : float, optional
            Load factor, dimensionless. Default ``1.``.

        Returns
        -------
        kG : sparse_matrix
            Free-free block of ``calc_kG`` of the full model.

        """
        A, key = self._matrix(self.full_kG, c, NLgeom, silent, inc)
        self._kG = (key, A) if key is not None else None
        return self._reduce(A)

    def _same(self, cached, c, lbd):
        r"""Whether a cached matrix was evaluated at ``c`` and ``lbd``"""
        if cached is None:
            return False
        (c0, lbd0), _ = cached
        return lbd0 == lbd and np.array_equal(c0, c)

    def _kT_fk(self, c, lbd):
        r"""Free-constrained block of the tangent stiffness matrix of the
        full model at ``c`` and ``lbd``, reused from the last evaluation of
        ``calc_kC`` and ``calc_kG`` when done there"""
        if self._same(self._kC, c, lbd) and self._same(self._kG, c, lbd):
            kT = csr_matrix(self._kC[1]) + csr_matrix(self._kG[1])
        else:
            u = self.full(c, lbd)
            kwC = (dict(inc=lbd)
                   if callbacks.accepts_kwarg(self.full_kC, 'inc') else {})
            kwG = (dict(inc=lbd)
                   if callbacks.accepts_kwarg(self.full_kG, 'inc') else {})
            kT = (csr_matrix(self.full_kC(c=u, NLgeom=True, silent=True,
                                          **kwC))
                  + csr_matrix(self.full_kG(c=u, NLgeom=True, silent=True,
                                            **kwG)))
        return kT[self.free, :][:, self.known]

    def _fext(self, inc, silent, c, lbd):
        r"""Load vector of the free degrees-of-freedom, see
        :class:`FreeDofs`"""
        kwargs = {}
        if c is not None and self.config_dependent:
            kwargs['c'] = self.full(c, 0. if lbd is None else lbd)
        if (c is not None and lbd is not None
                and callbacks.accepts_kwarg(self.full_fext, 'lbd')):
            kwargs['lbd'] = lbd
        f = np.asarray(self.full_fext(inc=1., silent=silent,
                                      **kwargs))[self.free]
        if c is not None and lbd is not None and self.prescribed:
            f = f - self._kT_fk(c, lbd) @ self.uk
        return inc*f

    def linear_fext(self, c=None):
        r"""Load vector of the linear static analysis, at a unit load factor

        Parameters
        ----------
        c : np.ndarray or None, optional
            Configuration of the free degrees-of-freedom, passed to the
            ``calc_fext`` of the full model when it declares ``c``. Default
            ``None``.

        Returns
        -------
        f : np.ndarray
            `\{F_{ext}\}_f - [K_{C_0}]_{fk}\{\bar{u}_k\}`, with `[K_{C_0}]`
            the linear stiffness matrix of the full model.

        """
        kwargs = {}
        if c is not None and self.config_dependent:
            kwargs['c'] = self.full(c, 1.)
        f = np.asarray(self.full_fext(inc=1., silent=True,
                                      **kwargs))[self.free]
        if self.prescribed:
            K0 = csr_matrix(self.full_kC(silent=True))
            f = f - K0[self.free, :][:, self.known] @ self.uk
        return f


class Analysis(object):
    r"""Linear and non-linear static analyses of a model given by callables

    The structural model is defined by the callables passed to the
    constructor, which return the force vectors and the stiffness matrices.
    The static analysis, linear or non-linear, is run with :meth:`static`,
    whose solution is stored in the attributes ``increments`` and ``cs``.

    The non-linear analyses find the solution vectors ``c`` for which the
    internal force vector ``calc_fint(c)`` balances the external force
    vector ``calc_fext(inc=1.)`` scaled by the load factor, using the
    tangent stiffness matrix ``calc_kC(c, NLgeom=True) + calc_kG(c,
    NLgeom=True)``. With the elements of this package, ``calc_kC`` normally
    returns ``KC0 + KCNL`` and ``calc_kG`` returns ``KG``, both updated at
    ``c``, and ``calc_fint`` the vector of ``update_fint(...,
    nonlinear=1)``.

    The analysis parameters are attributes of this class, described in the
    following tables together with their default values:

    ========================  ==================================================
    Non-Linear Algorithm      Description
    ========================  ==================================================
    ``NL_method``             ``str``, ``'NR'`` (default) for the
                              Newton-Raphson method, see
                              :func:`.newton_raphson._solver_NR`, or
                              ``'arc_length_riks'`` and
                              ``'arc_length_crisfield'`` for the arc-length
                              methods, see
                              :func:`.arc_length._solver_arc_length`
    ``line_search``           ``bool``, activates a safeguarding line-search,
                              for the Newton-Raphson method only, see
                              :func:`.newton_raphson._NR_iterations`. Default
                              is ``False``
    ``max_iter_line_search``  ``int``, maximum number of iterations of the
                              line-search. Default is ``20``
    ``modified_NR``           ``bool``, activates the modified Newton-Raphson
                              method, where the tangent stiffness matrix is
                              not updated at every iteration. Default is
                              ``False``, i.e. full Newton-Raphson
    ``compute_every_n``       ``int``, if ``modified_NR=True``, the tangent
                              stiffness matrix is updated every `n`
                              iterations. Default is ``6``
    ``kT_initial_state``      ``bool``, if ``modified_NR=True``, tells if the
                              tangent stiffness matrix should be calculated
                              already at the first iteration of the analysis,
                              which is required for example when initial
                              imperfections take place. Otherwise the linear
                              constitutive stiffness matrix is used. Default
                              is ``True``
    ========================  ==================================================

    ================   =================================================
    Incrementation     Description
    ================   =================================================
    ``initialInc``     initial load increment. In the arc-length
                       methods it defines the initial arc-length
                       increment, corresponding to a load factor
                       increment of ``initialInc`` along the initial
                       tangent. Default is ``0.1``
    ``minInc``         minimum increment; the analysis stops when the
                       load increment (Newton-Raphson) or the arc-length
                       increment (arc-length methods) becomes smaller
                       than ``minInc``. Default is ``1.e-4``
    ``maxInc``         maximum load increment, or maximum arc-length
                       increment for the arc-length methods. Default is
                       ``1.``
    ``maxArcLength``   maximum cumulative arc length covered by the
                       arc-length methods, dimensionless. Default is
                       ``18``
    ================   =================================================

    ====================    ============================================
    Convergence Criteria    Description
    ====================    ============================================
    ``convergence``         ``str``, ``'crisfield'`` (default) for
                            Crisfield's criterion, the residual scaled
                            with the diagonal of the linear stiffness
                            matrix over the free degrees-of-freedom, or
                            ``'relative'`` for the Euclidean norm of the
                            residual, see :mod:`.convergence`
    ``relTOL``              tolerance `\epsilon` of the relative force
                            imbalance: Crisfield's `\zeta \le` ``relTOL``,
                            or, with ``'relative'``, the norm of the
                            residual force vector smaller than ``relTOL``
                            times the largest norm between the external
                            and internal force vectors. Not used if
                            ``None``. Default is ``1.e-9``, the tight
                            tolerance of Crisfield's criterion with a
                            consistent tangent, see :mod:`.convergence`;
                            the same default applies with ``'relative'``
    ``absTOL``              the convergence is also achieved when the
                            maximum absolute residual force is smaller
                            than this value, which depends on the units of
                            the model. Not used if ``None``. Default is
                            ``None``
    ``maxNumIter``          maximum number of iterations (corrections) of a
                            step; if achieved the increment is reduced.
                            Default is ``30``
    ``too_slow_TOL``        a step is considered too slow when the smallest
                            residual norm is not reduced by this fraction
                            over the last iterations; the increment is then
                            reduced. Default is ``0.005``
    ====================    ============================================

    Parameters
    ----------
    calc_fext : callable, optional
        ``calc_fext(inc=1., silent=False)``, must return a 1-D array with the
        external force vector. Required for linear and non-linear static
        analyses. The non-linear solvers call it with ``inc=1.`` and scale the
        returned vector by the load factor. It may declare the optional
        keyword arguments ``c``, the current configuration, and ``lbd``, the
        current load factor, and must then return the load vector of that
        configuration and load factor for a unit load factor, ``inc=1.``,
        see :mod:`.callbacks`. The linear static analysis never passes
        ``lbd``.
    calc_fint : callable, optional
        ``calc_fint(c, silent=False)``, must return a 1-D array with the
        internal force vector for the solution vector ``c``. Required for
        non-linear analyses. It may declare the optional keyword argument
        ``inc``, which then receives the current load factor, see
        :mod:`.callbacks`.
    calc_kC : callable, optional
        ``calc_kC(c=None, NLgeom=False, silent=False)``, must return a sparse
        matrix with the constitutive stiffness matrix. With ``c=None`` and
        ``NLgeom=False`` it must return the linear stiffness matrix, which
        also gives the scaling of Crisfield's criterion, and with
        ``NLgeom=True`` the constitutive part of the tangent stiffness matrix
        at ``c``. Required for linear and non-linear static analyses. It may
        declare the optional keyword argument ``inc``, which then receives
        the current load factor in the calls with ``NLgeom=True``, see
        :mod:`.callbacks`.
    calc_kG : callable, optional
        ``calc_kG(c=None, NLgeom=False, silent=False)``, must return a sparse
        matrix with the geometric stiffness matrix at ``c``. It is called
        with ``NLgeom=True``. Required for non-linear analyses. It may
        declare the optional keyword argument ``inc``, which then receives
        the current load factor, see :mod:`.callbacks`. The sum of
        ``calc_kC`` and ``calc_kG`` may be unsymmetric.
    bu : array_like or None, optional
        Boolean mask of the free degrees-of-freedom, 1-D array of size
        ``N``. With ``bu``, the callables receive and return the vectors and
        matrices of all the ``N`` degrees-of-freedom, e.g. assembled with the
        elements of this package, and the solvers work with the free ones
        only, see :class:`FreeDofs`. With ``None``, the callables work with
        the vectors and matrices of the solver, whose null rows and columns
        of the stiffness matrices, if any, are the constrained
        degrees-of-freedom. Default ``None``.
    uk : array_like or None, optional
        Prescribed displacements for a unit load factor, 1-D array of size
        ``N``, used with ``bu``: the displacements of the constrained
        degrees-of-freedom are ``lbd*uk[~bu]``, with ``lbd`` the load
        factor. ``None`` for null prescribed displacements. Default ``None``.

    Attributes
    ----------
    increments : list
        Load factors of the converged increments, filled by :meth:`static`.
    cs : list
        Solution vectors of the converged increments, filled by
        :meth:`static`, of size ``N`` with ``bu``.
    last_analysis : str
        Type of the last analysis run, ``'static'`` after :meth:`static`.

    Notes
    -----
    For non-linear analyses, the internal force vector
    `\{F_{int}(c)\}` must be in equilibrium with the external force
    vector scaled by the load factor `\lambda`:

    .. math::

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}

    and the tangent stiffness matrix `[K_T] = [K_C] + [K_G]`, given by the
    callables ``calc_kC`` and ``calc_kG``, should be the exact derivative of
    `\{F_{int}\}` with respect to `\{c\}`, such that the Newton-Raphson
    iterations converge quadratically ([1]_, Section 1.2.2, p. 9). The
    convergence is checked by default with Crisfield's criterion, see
    :mod:`.convergence`.

    The line-search tries the full step first and only reduces it when it
    fails a sufficient-decrease test on the residual norm, see
    :func:`.newton_raphson._NR_iterations`.

    The arc length is dimensionless, with displacements scaled by the
    linear solution for a load factor of 1, such that in the linear regime
    an arc length of about ``sqrt(2)`` corresponds to a load factor
    increment of 1, see :func:`.arc_length._solver_arc_length`.

    .. rubric:: Configuration-dependent loads

    Configuration-dependent loads, e.g. follower pressures [2]_, are
    supported through optional keyword arguments, passed only to the
    callables that declare them explicitly, see :mod:`.callbacks`:

    - ``calc_fint(c, inc=lbd)``, ``calc_kC(c, NLgeom=True, inc=lbd)`` and
      ``calc_kG(c, NLgeom=True, inc=lbd)`` receive the current load factor,
      such that ``calc_fint`` may contain the configuration-dependent part of
      the load, scaled by ``inc``, and ``calc_kC + calc_kG`` its derivative,
      the load stiffness matrix.
    - ``calc_fext(inc=1., c=c)`` must return the load vector of the
      configuration ``c`` for a unit load factor, the derivative of the
      residual with respect to the load factor, used by the arc-length
      solvers.
    - ``calc_fext(inc=1., c=c, lbd=lbd)`` also receives the current load
      factor when it declares ``lbd``, for a derivative `\partial
      R/\partial \lambda` that depends on the load factor, e.g. prescribed
      displacements imposed exactly as linear constraints, where the
      reference vector ``calc_fext(inc=1.)`` may be null. The solvers then
      use `\partial R/\partial \lambda` for the displacement scale of the
      arc-length methods, for the predictor of the first Newton-Raphson
      increment and as a reference of the convergence criteria. The linear
      static analysis never passes ``lbd``.

    The tangent stiffness matrix may then be unsymmetric, which the solvers
    support, since they use LU factorizations. With ``bu`` and ``uk``, the
    prescribed displacements are imposed with ``lbd``, see
    :class:`FreeDofs`.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.
    .. [2] Schweizerhof, K. and Ramm, E., Displacement dependent pressure
       loads in nonlinear finite element analyses, Computers & Structures,
       18(6), 1099-1114, 1984.

    """
    __slots__ = ['NL_method', 'line_search', 'max_iter_line_search',
                 'modified_NR', 'compute_every_n', 'kT_initial_state',
                 'initialInc', 'minInc', 'maxInc', 'maxArcLength',
                 'convergence', 'absTOL', 'relTOL', 'maxNumIter',
                 'too_slow_TOL', 'increments', 'cs', 'last_analysis',
                 'calc_fext', 'calc_kC', 'calc_fint', 'calc_kG', 'bu', 'uk']

    def __init__(self, calc_fext=None, calc_fint=None, calc_kC=None,
                 calc_kG=None, bu=None, uk=None):
        # NOTE non-linear algorithm
        self.NL_method = 'NR'
        self.line_search = False
        self.max_iter_line_search = 20
        self.modified_NR = False
        self.compute_every_n = 6
        self.kT_initial_state = True
        # NOTE incrementation
        self.initialInc = 0.1
        self.minInc = 1.e-4
        self.maxInc = 1.
        self.maxArcLength = 18
        # NOTE convergence criteria
        self.convergence = 'crisfield'
        self.absTOL = None
        self.relTOL = 1.e-9
        self.maxNumIter = 30
        self.too_slow_TOL = 0.005

        # NOTE callables of the model
        self.calc_fext = calc_fext
        self.calc_fint = calc_fint
        self.calc_kC = calc_kC
        self.calc_kG = calc_kG
        self.bu = bu
        self.uk = uk

        # NOTE outputs to be filled
        self.increments = None
        self.cs = None

        # NOTE flag telling the last analysis
        self.last_analysis = ''

    def _problem(self):
        r"""Object with the callables seen by the solvers: the analysis
        itself, or a :class:`FreeDofs` when ``bu`` is given"""
        if self.bu is None:
            if self.uk is not None:
                raise ValueError('uk requires bu')
            return self
        return FreeDofs(self.calc_fext, self.calc_fint, self.calc_kC,
                        self.calc_kG, self.bu, self.uk)

    def static(self, NLgeom=False, silent=False):
        r"""Linear or non-linear static analysis

        The linear analysis solves the system of the linear stiffness matrix
        ``calc_kC()`` and the external force vector ``calc_fext()`` using
        :func:`.solve`. When the load depends on the configuration, i.e.
        ``calc_fext`` declares the keyword argument ``c``, see
        :mod:`.callbacks`, the load is evaluated in the current configuration
        and corrected by iterations that reuse one factorization of the
        linear stiffness matrix, starting from the solution of the reference
        load, until the convergence criterion is satisfied, see
        :func:`.check_convergence`, or ``maxNumIter`` iterations are done; a
        warning is printed when they do not converge and the last iterate is
        returned. The non-linear analysis uses the solver selected by the
        ``NL_method`` attribute.

        Parameters
        ----------
        NLgeom : bool, optional
            Whether a non-linear analysis is performed. Default ``False``.
        silent : bool, optional
            Whether the log messages are suppressed. Default ``False``.

        Returns
        -------
        increments : list
            Load factors of the converged increments, dimensionless. A
            linear analysis returns ``[1.]``. Non-linear analyses finish at a
            load factor of exactly ``1.`` unless they are stopped earlier,
            see the documentation of each solver.
        cs : list
            Solution vectors of the converged increments, 1-D arrays, of
            size ``N`` with ``bu``, including the prescribed displacements.

        Raises
        ------
        ValueError
            When ``NLgeom=True`` and the ``NL_method`` attribute is not
            ``'NR'``, ``'arc_length_riks'`` or ``'arc_length_crisfield'``,
            or when ``uk`` is given without ``bu``.

        Notes
        -----
        The linear analysis solves `[K_{C_0}]\{c\} = \{F_{ext}\}`; with
        ``bu`` and ``uk``, `[K_{C_0}]_{ff}\{c\} = \{F_{ext}\}_f -
        [K_{C_0}]_{fk}\{\bar{u}_k\}`. When the load depends on the
        configuration, the structure is still geometrically linear but the
        load is evaluated in the current configuration, `[K_{C_0}]\{c\} =
        \{F_{ext}(c)\}`, which is solved by the load correction iterations

        .. math::

            \{c_{k+1}\} = \{c_k\} + [K_{C_0}]^{-1}(\{F_{ext}(c_k)\}
                          - [K_{C_0}]\{c_k\})

        starting from `\{c_0\} = [K_{C_0}]^{-1}\{F_{ext}\}`, the solution of
        the reference load ``calc_fext()``, with `\{F_{ext}(c_k)\}` given by
        ``calc_fext(inc=1., c=c_k)``. This is a fixed-point iteration
        without the load stiffness in the iteration matrix. For a follower
        load whose force vector is affine in `c`, `\{F(c)\} = \{F_0\} -
        [K_L]\{c\}`, this is the solution of `([K_{C_0}] + [K_L])\{c\} =
        \{F_0\}`, the error being multiplied by `-[K_{C_0}]^{-1}[K_L]` at
        each iteration, whose spectral radius is `1/\min|\lambda|` over the
        load multipliers `\lambda` of `([K_{C_0}] + \lambda [K_L])\{u\} =
        \{0\}`, including the negative and complex ones; beyond the load of
        the smallest `|\lambda|` the iterations diverge.

        """
        self.increments = []
        self.cs = []
        pb = self._problem()

        if NLgeom:
            self.maxInc = max(self.initialInc, self.maxInc)
            msg('Started Non-Linear Static Analysis', silent=silent)
            if self.NL_method == 'NR':
                _solver_NR(self, pb, silent=silent)
            elif self.NL_method == 'arc_length_riks':
                _solver_arc_length(self, pb, 'riks', silent=silent)
            elif self.NL_method == 'arc_length_crisfield':
                _solver_arc_length(self, pb, 'crisfield', silent=silent)
            else:
                raise ValueError('{0} is an invalid NL_method'.format(
                                 self.NL_method))

        else:
            msg('Started Linear Static Analysis', silent=silent)
            k0 = pb.calc_kC(silent=silent)
            if isinstance(pb, FreeDofs):
                fext = pb.linear_fext()
                follower = pb.config_dependent
            else:
                fext = pb.calc_fext(silent=silent)
                follower = callbacks.is_load_configuration_dependent(pb)
            c = solve(k0, fext)
            if follower:
                c = self._linear_static_follower(pb, k0, c, silent)
            self.cs.append(c)
            self.increments.append(1.)
            msg('Finished Linear Static Analysis', silent=silent)

        if isinstance(pb, FreeDofs):
            self.cs = [pb.full(c, lbd) for c, lbd in zip(self.cs,
                                                         self.increments)]
        self.last_analysis = 'static'

        return self.increments, self.cs

    def _linear_static_follower(self, pb, k0, c, silent):
        r"""Load correction iterations of the linear static analysis with a
        configuration-dependent load, see :meth:`static`

        Parameters
        ----------
        pb : object
            Object with the callables, the analysis itself or a
            :class:`FreeDofs`.
        k0 : sparse_matrix
            Linear stiffness matrix of the solver.
        c : np.ndarray
            Solution of the reference load, the first iterate.
        silent : bool
            Whether the log messages are suppressed.

        Returns
        -------
        c : np.ndarray
            The converged iterate, or the last one with a warning.

        Notes
        -----
        The iterations and their convergence are described in
        :meth:`static`; the convergence criterion is the one of the
        non-linear solvers, see :func:`.check_convergence`, with the
        internal force vector `[K_{C_0}]\{c_k\}`.

        """
        msg('Configuration-dependent load: load correction iterations',
            level=2, silent=silent)
        w = _weights(self, k0)
        # NOTE k0 is factorized once for all the corrections
        solve_k0 = None
        for iteration in range(1, self.maxNumIter + 1):
            if isinstance(pb, FreeDofs):
                fext = pb.linear_fext(c)
            else:
                fext = pb.calc_fext(inc=1., c=c, silent=True)
            fint = k0 @ c
            R = fext - fint
            conv, Rnorm, zeta = check_convergence(self, R, fint, fext, w=w)
            msg('Iteration: %d, %s = %1.3e' % (iteration, 'zeta' if
                self.convergence == 'crisfield' else 'relative ||R||', zeta),
                level=3, silent=silent)
            if conv:
                return c
            if not np.isfinite(Rnorm):
                break
            if solve_k0 is None:
                try:
                    solve_k0 = _factorize(k0)
                except RuntimeError:
                    break
            c = c + solve_k0(R)
        warn('The load correction iterations did not converge, the load may '
             'exceed the critical load of the linearized problem', level=1,
             silent=silent)
        return c
