"""
===========================================================
Shell property utils module (:mod:`pyfe3d.shellprop_utils`)
===========================================================

Highly based on the `composites <https://saullocastro.github.io/composites/>`_. module.

.. currentmodule:: pyfe3d.shellprop_utils

Functions that create a :class:`.ShellProp` from the stacking sequence and
the ply data (:func:`.laminated_plate`) or for an isotropic plate
(:func:`.isotropic_plate`), and a :class:`.MatLamina` from a tuple of
material properties (:func:`.read_laminaprop`).

"""
from .shellprop import Lamina, MatLamina, ShellProp


def read_laminaprop(laminaprop, rho=0):
    r"""Returns a :class:`.MatLamina` object based on an input ``laminaprop`` tuple

    The entries of ``laminaprop`` and ``rho`` are stored in a new
    :class:`.MatLamina` object, the Poisson's ratios ``nu21``, ``nu31`` and
    ``nu32`` are computed from ``nu12``, ``nu13``, ``nu23`` and the Young
    moduli, and :meth:`.MatLamina.rebuild` is called. With 2 entries, ``g12 =
    g13 = g23 = e/(2*(1+nu))``; with 2 or 6 entries, ``e3``, ``nu13`` and
    ``nu23`` are zero. Another number of entries fails an ``assert``
    statement.

    Parameters
    ----------
    laminaprop : list or tuple
        For the most general case of tri-axial stress, use a tuple containing
        the following entries::

            laminaprop = (e1, e2, nu12, g12, g13, g23, e3, nu13, nu23)

        For isotropic materials aiming calculations with tri-axial stresses,
        use::

            g = e/(2*(1+nu))
            laminaprop = (e, e, nu, g, g, g, e, nu, nu)

        For orthotropic materials with in-plane stresses the user can only
        supply::

            laminaprop = (e1, e2, nu12, g12, g13, g23)

        For isotropic materials with in-plane stresses the user can only
        supply::

            laminaprop = (e, nu) # new


        ======  ==============================
        symbol  value
        ======  ==============================
        e1      Young Modulus in direction 1
        e2      Young Modulus in direction 2
        nu12    12 Poisson's ratio
        g12     12 Shear Modulus
        g13     13 Shear Modulus
        g23     23 Shear Modulus
        e3      Young Modulus in direction 3
        nu13    13 Poisson's ratio
        nu23    23 Poisson's ratio
        ======  ==============================

        The Young and shear moduli have the dimension of force per unit area,
        the Poisson's ratios are dimensionless.


    rho : float, optional
        Material density, mass per unit volume. Default ``0``.


    Returns
    -------
    matlam : MatLamina
        A :class:`.MatLamina` object.

    Notes
    -----
    The Poisson's ratios not given follow from the reciprocity relations of
    an orthotropic material, `\nu_{ij}/E_i = \nu_{ji}/E_j`, which make its
    compliance matrix symmetric [1]_:

    .. math::

        \nu_{21} = \nu_{12} \frac{E_2}{E_1}, \quad
        \nu_{31} = \nu_{13} \frac{E_3}{E_1}, \quad
        \nu_{32} = \nu_{23} \frac{E_3}{E_2}

    and for an isotropic material `G = E/(2(1 + \nu))`. With 2 or 6 entries,
    ``e3 = 0`` makes the constitutive constants of :meth:`.MatLamina.rebuild`
    the plane-stress reduced stiffnesses.

    References
    ----------
    .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and Shells:
       Theory and Analysis, Second Edition, CRC Press, 2003.
       https://doi.org/10.1201/b12409

    """
    matlam = MatLamina()

    # laminaProp = (e1, e2, nu12, g12, g13, g23, e3, nu13, nu23)
    assert len(laminaprop) in (2, 6, 9), ('Invalid entry for laminaprop: ' +
                                          str(laminaprop))
    if len(laminaprop) == 2: # ISOTROPIC in-plane stress new
        e = laminaprop[0]
        nu = laminaprop[1]
        g = e/(2*(1+nu))
        laminaprop = (e, e, nu, g, g, g, 0, 0, 0)
    elif len(laminaprop) == 6: # ORTHOTROPIC in-plane stress
        laminaprop = tuple(list(laminaprop) + [0, 0, 0])
    matlam.e1 = laminaprop[0]
    matlam.e2 = laminaprop[1]
    matlam.e3 = laminaprop[6]
    matlam.nu12 = laminaprop[2]
    matlam.nu13 = laminaprop[7]
    matlam.nu23 = laminaprop[8]
    matlam.nu21 = matlam.nu12 * matlam.e2 / matlam.e1
    matlam.nu31 = matlam.nu13 * matlam.e3 / matlam.e1
    matlam.nu32 = matlam.nu23 * matlam.e3 / matlam.e2
    matlam.g12 = laminaprop[3]
    matlam.g13 = laminaprop[4]
    matlam.g23 = laminaprop[5]
    matlam.rho = rho
    matlam.rebuild()

    return matlam


def laminated_plate(stack, plyt=None, laminaprop=None, rho=0., plyts=None,
        laminaprops=None, rhos=None, offset=0., shear_correction='rohwer'):
    r"""Read a laminate stacking sequence data.

    :class:`.ShellProp` object is returned based on the inputs given.

    One :class:`.Lamina` is created for each entry of ``stack``, with its
    thickness, its :class:`.MatLamina` given by :func:`.read_laminaprop` with
    its ``laminaprop`` and density, and :meth:`.Lamina.rebuild` called. Then
    :meth:`.ShellProp.calc_constitutive_matrix` and
    :meth:`.ShellProp.calc_equivalent_properties` are called.

    Parameters
    ----------
    stack : list
        Angles of the stacking sequence in degrees, from the bottom to the top
        ply.
    plyt : float, optional
        When all plies have the same thickness, ``plyt`` can be supplied.
        ``plyt`` or ``plyts`` must be supplied. Length. Default ``None``.
    laminaprop : tuple, optional
        When all plies have the same material properties, ``laminaprop``
        can be supplied. ``laminaprop`` or ``laminaprops`` must be supplied.
        For orthotropic plies, the ``laminaprop`` should be::

            laminaprop = (E11, E22, nu12, G12, G13, G23)

        For isotropic plies, the ``laminaprop`` should be::

            laminaprop = (E, nu)

        See :func:`.read_laminaprop`. Default ``None``.
    rho : float, optional
        Uniform material density to be used for all plies, mass per unit
        volume. Default ``0.``.
    plyts : list, optional
        A list of floats with the thickness of each ply. When given,
        ``plyt`` is ignored. Length. Default ``None``.
    laminaprops : list, optional
        A list of tuples with a laminaprop for each ply. When given,
        ``laminaprop`` is ignored. Default ``None``.
    rhos : list, optional
        A list of floats with the material density of each ply. When given,
        ``rho`` is ignored. Mass per unit volume. Default ``None``.
    offset : float, optional
        Offset along the normal axis about the mid-surface, which influences
        the laminate properties. Length. Default ``0.``.
    shear_correction : str or None, optional
        Method used to compute the transverse shear stiffnesses ``A44``,
        ``A45``, ``A55``, which are returned with the correction already
        applied: ``'rohwer'`` (default, equilibrium approach), ``'vlachoutsis'``,
        ``'constant'`` (5/6) or ``None`` (no correction). See
        :meth:`.ShellProp.calc_transverse_shear_stiffness`.

    Returns
    -------
    prop : :class:`.ShellProp`
        The laminate, with its stiffnesses and equivalent properties
        computed.

    Raises
    ------
    ValueError
        If neither ``plyt`` nor ``plyts`` is supplied, or if neither
        ``laminaprop`` nor ``laminaprops`` is supplied. Also if ``stack``,
        ``plyts``, ``laminaprops`` and ``rhos`` do not have the same
        length.

    """
    prop = ShellProp()
    prop.offset = offset
    prop.stack = list(stack)
    prop.shear_correction = shear_correction

    if plyts is None:
        if plyt is None:
            raise ValueError('plyt or plyts must be supplied')
        else:
            plyts = [plyt for i in stack]

    if laminaprops is None:
        if laminaprop is None:
            raise ValueError('laminaprop or laminaprops must be supplied')
        else:
            laminaprops = [laminaprop for i in stack]

    if rhos is None:
        rhos = [rho for i in stack]

    if not (len(stack) == len(plyts) == len(laminaprops) == len(rhos)):
        raise ValueError('stack, plyts, laminaprops and rhos must have the same length')

    plies = []
    prop.h = 0.
    for plyt, laminaprop, thetadeg, rho in zip(plyts, laminaprops, stack, rhos):
        ply = Lamina()
        ply.thetadeg = float(thetadeg)
        ply.h = plyt
        prop.h += ply.h
        ply.matlamina = read_laminaprop(laminaprop, rho)
        ply.rebuild()
        plies.append(ply)
    prop.plies = plies

    prop.calc_constitutive_matrix()
    prop.calc_equivalent_properties()

    return prop


def isotropic_plate(thickness, E, nu, offset=0., rho=0.,
        shear_correction='rohwer'):
    """Read data for an isotropic plate

    :class:`.ShellProp` object is returned based on the inputs given.

    Calls :func:`.laminated_plate` with a single ply of angle ``0``,
    thickness ``thickness`` and ``laminaprop=(E, nu)``.

    Parameters
    ----------
    thickness : float
        Plate thickness, length.
    E : float
        Young modulus, force per unit area.
    nu : float
        Poisson's ratio, dimensionless.
    rho : float, optional
        Material density, mass per unit volume. Default ``0.``.
    offset : float, optional
        Offset along the normal axis about the mid-surface, which influences
        the extension-bending coupling (B matrix). Length. Default ``0.``.
    shear_correction : str or None, optional
        See :func:`.laminated_plate`. For an isotropic plate, ``'rohwer'``,
        ``'vlachoutsis'`` and ``'constant'`` give ``A44 = A55 = 5/6 G h``.
        Default ``'rohwer'``.

    Returns
    -------
    prop : :class:`.ShellProp`
        The plate, with its stiffnesses and equivalent properties computed.

    See Also
    --------
    laminated_plate

    """
    return laminated_plate(plyt=thickness, stack=[0], laminaprop=(E, nu),
            rho=rho, offset=offset, shear_correction=shear_correction)
