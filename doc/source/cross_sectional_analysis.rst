Cross-sectional analysis of beams
=================================

The beam elements :class:`.BeamC` and :class:`.BeamLR` assume a rigid
cross section: the section only enters through the integrals of a
:class:`.BeamProp`, and the torsional stiffness ``GJ`` and the shear
moduli ``G12`` and ``G13``, with their correction factors, must be given by
the user, see :doc:`shell_properties`. The module :mod:`pyfe3d.vabs`
computes these stiffnesses, and the couplings between them, for an
arbitrary cross section made of anisotropic and non-homogeneous materials,
with the variational asymptotic beam sectional analysis (VABS) of Cesnik
and Hodges (1997) and Yu et al. (2002, 2012): the three-dimensional
elasticity problem of a slender prismatic beam is split into a
two-dimensional problem over the cross section, solved here with finite
elements, and a one-dimensional beam problem, governed by the resulting
sectional stiffness matrix.


Steps of an analysis
--------------------

1. **Mesh.** Node coordinates `x_2`, `x_3`, the `y` and `z` of the beam
   elements, and the connectivity of 3- or 6-node triangles and 4-, 8- or
   9-node quadrilaterals, numbered counter-clockwise, see
   :class:`.CrossSection`. The origin of `x_2`, `x_3` is the reference axis
   of the beam: all results refer to it, and it need not be the centroid.
2. **Materials.** One `6 \times 6` material matrix per material, in its
   material axes, e.g. from :func:`.isotropic` or :func:`.orthotropic`, and
   per element the material index and two angles: `\theta_1`, which
   orients the local axes of the element, e.g. along a wall, and the ply
   angle `\theta_3`, see :func:`.rotate_material`.
3. **Classical model.** :meth:`.CrossSection.calc_stiffness` solves the
   warping of the cross section for the four classical strain measures,
   extension, twist and two bending curvatures, and condenses it into the
   `4 \times 4` stiffness ``Kc``. This is the generalized Euler-Bernoulli
   model: the warping softens the rigid section, gives the Saint-Venant
   torsional stiffness, and populates the couplings of anisotropic
   sections.
4. **Generalized Timoshenko model.** The same factorization gives the
   first correction of the warping, driven by the derivatives of the
   classical strains along the beam, and the second-order energy is
   transformed into the `6 \times 6` stiffness ``Kt``, which adds the two
   transverse shear strains. Its shear entries contain the shear
   correction factors and the location of the shear centre, which are not
   assumed. The theory of these two steps is in the notes of
   :meth:`.CrossSection.calc_stiffness`, and the discretization in those
   of the kernel ``_integrate`` of :mod:`pyfe3d.vabs`.
5. **Derived quantities.** The flexibilities ``Fc`` and ``Ft``, the tension
   and shear centres, and, with :meth:`.CrossSection.calc_mass`, the mass
   matrix and the mass integrals of :class:`.BeamProp`.
6. **Use in the beam elements.** :func:`.yu_to_pyfe3d` converts the
   matrices to the convention of pyfe3d, and :func:`.beamprop_from_vabs`
   creates the :class:`.BeamProp`, when the matrix can be represented by
   one, see below.

The analysis is linear and per station: a beam whose section varies along
its length needs one analysis per station.


Axes, angles and centres
------------------------

.. figure:: ../figures/vabs_axes.svg

   (a) Flared channel of constant thickness, made of one isotropic
   material, with the tension and shear centres computed by
   :meth:`.CrossSection.calc_stiffness`, and an element of the upper
   flange with its local axes; (b) that element seen from the tip of
   `y_3`, with its material axes.

The beam axis `x_1` points to the viewer of the cross section and `x_2`,
`x_3` are the `y`, `z` of :class:`.BeamC` and :class:`.BeamLR`, with the
origin at the reference axis. Each element has two angles, see
:func:`.rotate_material`:

- `\theta_1` (``theta1``) rotates the local axes `y_2`, `y_3` of the
  element about `y_1 = x_1`, positive from `x_2` towards `x_3`; for a
  wall, `y_2` is usually taken along the wall and `y_3` across it;
- `\theta_3` (``theta3``) rotates the material axes `m_1`, `m_2` about
  `m_3 = y_3`, positive from `y_1` towards `y_2`; for a laminated wall,
  it is the ply angle measured from the beam axis.

The tension centre `(a_2, a_3)` is the point where an axial force `F_1`
gives no bending curvature, `\kappa_2 = \kappa_3 = 0`, and the shear
centre `(e_2, e_3)` the point where the transverse forces `F_2`, `F_3`
give no twist, `\kappa_1 = 0`. Both are measured from the reference axis.
For the homogeneous channel of the figure, the tension centre is the
centroid and the shear centre lies outside the section, on the side of
the web opposite to the flanges.

This convention of the angles reproduces all the signs of the box beam
CUS computed with VABS 3.2 by Yu et al. (2012). With it, two other
published examples give three couplings each with the published magnitude
and the opposite sign, which is attributed to the labelling of their axes
and ply angles: the box beam CAS1 of Yu et al. (2002) and the pipe of Yu
and Hodges (2005), see the notes of :func:`.rotate_material` and
``tests/test_vabs_composite.py``.


Two conventions
---------------

The matrices of :class:`.CrossSection` are written in the convention of
the VABS literature, which orders the strains as `\{\gamma_{11},
2\gamma_{12}, 2\gamma_{13}, \kappa_1, \kappa_2, \kappa_3\}` and defines
the curvatures as the components of the curvature vector, i.e. the rates
of rotation about `x_1`, `x_2` and `x_3`. The beam elements of pyfe3d use
`\{\bar{\varepsilon}, \bar{\gamma}_y, \bar{\gamma}_z, \kappa_x, \kappa_y,
\kappa_z\}`, with `\kappa_y` and `\kappa_z` named after the coordinate
they multiply in the axial strain, see the notes of :class:`.BeamC`. The
two agree on the extension, the shear strains and the twist, while
`\kappa_2 = \kappa_z` and `\kappa_3 = -\kappa_y`; the moments are `M_2 =
M_y` and `M_3 = -M_z`. Hence, when the reference axis is offset from the
centroid by `d_2` along `x_2`, the coupling of the extension with
`\kappa_3` is `-E A d_2` in ``Kt`` and the coupling with `\kappa_y` is `E
A d_2 = E Q_z` after :func:`.yu_to_pyfe3d`, as in the rigid section of
:class:`.BeamC`; this is checked in ``tests/test_vabs_isotropic.py``.

The ordering of the classical model is the same without the shear
strains, and the classical model is recovered from the Timoshenko one by
a condensation of the flexibility, :func:`.classical_from_timoshenko`,
not by deleting rows and columns of the stiffness.


From the sectional matrix to the beam elements
----------------------------------------------

A :class:`.BeamProp` has 11 cross-section parameters, against the 21
independent entries of a `6 \times 6` stiffness matrix: the elements of
pyfe3d cannot represent the extension-shear, extension-twist,
bending-shear and bending-twist couplings, and their shear centre is the
tension centre. :func:`.beamprop_from_vabs` therefore converts a matrix
only when it has the pattern of a :class:`.BeamProp`, e.g. for isotropic
sections whose shear centre is the tension centre, and raises an error
otherwise; the element code is not changed. For such sections the
computed ``G12``, ``G13`` and ``GJ`` replace the shear correction factors
and the torsion constant that the user would otherwise supply.


Plotting the warping
--------------------

After :meth:`.CrossSection.calc_stiffness`, :meth:`.CrossSection.warping`
returns the warping displacements `w_1, w_2, w_3` of the nodes for given
strains of the beam, and :meth:`.CrossSection.triangles` a triangulation of
the mesh through all its nodes. These two arrays are all that a plotting
library needs, and pyfe3d does not depend on any. The warping satisfies
the constraints chosen in :meth:`.CrossSection.calc_stiffness`, by default
zero average and zero average rotation, i.e. it is the warping relative
to the rigid motion of the section about the reference axis.

With matplotlib, for a thin-walled channel, the out-of-plane warping
`w_1` per unit twist rate `\kappa_1`, with the tension centre (circle)
and the shear centre (cross), and the in-plane warping per unit curvature
`\kappa_2`, i.e. the anticlastic deformation due to the Poisson effect,
drawn to scale on the undeformed mesh: the flange in tension, `x_3 > 0`,
contracts, and the flange in compression expands. On the web, which
contains the reference axis, the torsional warping is almost zero; it is
the warping relative to the rotation about the reference axis, not about
the shear centre.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.tri import Triangulation

   from pyfe3d.vabs import CrossSection, isotropic

   # channel: web of height h along x3, flanges of width b along x2,
   # thickness t, meshed with bilinear quadrilaterals of side t/2
   h, b, t = 1., 0.5, 0.05
   d = t/2
   xs = np.linspace(0., b, round(b/d) + 1)
   ys = np.linspace(-h/2, h/2, round(h/d) + 1)
   X2, X3 = np.meshgrid(xs, ys)
   nx = xs.size
   elements = []
   for j in range(ys.size - 1):
       for i in range(nx - 1):
           xc, yc = (xs[i] + xs[i + 1])/2, (ys[j] + ys[j + 1])/2
           if xc < t or abs(yc) > h/2 - t:
               n = j*nx + i
               elements.append([n, n + 1, n + 1 + nx, n + nx])
   used = np.unique(elements)
   new = -np.ones(X2.size, dtype=int)
   new[used] = np.arange(used.size)
   x2, x3 = X2.ravel()[used], X3.ravel()[used]
   elements = new[np.array(elements)]

   sec = CrossSection(x2, x3, elements, materials=[isotropic(70e9, 0.3)],
                      mat_ids=[0]*len(elements))
   sec.calc_stiffness()
   tri = Triangulation(sec.x2, sec.x3, sec.triangles())

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
   w = sec.warping([0., 1., 0., 0.])  # unit twist rate kappa1
   cs = ax1.tricontourf(tri, w[:, 0], levels=21, cmap='RdBu_r')
   fig.colorbar(cs, ax=ax1, label=r'$w_1$ per unit $\kappa_1$')
   ax1.plot(*sec.tension_centre, 'ko', mfc='none')
   ax1.plot(*sec.shear_centre, 'kx')
   ax1.set_title('torsion')

   w = sec.warping([0., 0., 1., 0.])  # unit curvature kappa2
   ax2.triplot(tri, color='0.8', lw=0.5)
   ax2.triplot(sec.x2 + w[:, 1], sec.x3 + w[:, 2], sec.triangles(),
               color='C0', lw=0.5)
   ax2.set_title('bending, in-plane warping')
   for ax in (ax1, ax2):
       ax.set_aspect('equal')
       ax.set_xlabel('$x_2$')
       ax.set_ylabel('$x_3$')
   fig.tight_layout()

The same arrays give a three-dimensional view with plotly. The script
below draws the channel as a beam of length 3 under a uniform twist rate:
the sections rotate about the reference axis and warp along it, the
warping being exaggerated and coloured by `w_1`. The figure can be
rotated, panned and zoomed:

.. literalinclude:: _scripts/vabs_warping_3d.py
   :start-at: import numpy as np
   :end-before: # NOTE written for the documentation

.. raw:: html
   :file: _generated/vabs_warping_3d.html


Validation
----------

The tests compare the results with elasticity solutions and with
published VABS results:

- ``tests/test_vabs_isotropic.py``: rectangles, circles and tubes, with the
  axial and bending stiffnesses, the Saint-Venant torsion constant and the
  shear correction factors of the elasticity solutions, Eq. (25) of Yu and
  Hodges (2005) for the rectangle; the isotropic square of Yu et al.
  (2002), the T-section of Yu and Hodges (2005), and the couplings of an
  offset reference axis;
- ``tests/test_vabs_composite.py``: the box beam CUS of Yu et al. (2012),
  reproduced to the four digits of their VABS 3.2 results, the
  anisotropic pipe of Yu and Hodges (2005), with its mass properties, and
  the box beam CAS1 of Yu et al. (2002), with the opposite signs of three
  couplings of the last two examples asserted explicitly, see above;
- ``tests/test_vabs_elements.py``: integration rules, shape functions,
  material rotation and conventions.
- ``tests/test_vabs_warping.py``: the torsional warping of an ellipse
  against the Saint-Venant solution, and the triangulation of every
  element type.


Limitations
-----------

- Prismatic beams only: initially twisted and curved beams, whose theory
  is in Yu et al. (2002, 2012), are not implemented, and non-zero ``k1``,
  ``k2``, ``k3`` of :class:`.CrossSection` raise ``NotImplementedError``.
- No applied loads in the sectional analysis, Yu et al. (2012), Section
  5.3, and no recovery of the three-dimensional strains and stresses,
  although the warping is available, see :meth:`.CrossSection.warping`.
- No Vlasov refinement for thin-walled open sections.


References
----------

Cesnik, C. E. S., and Hodges, D. H., "VABS: A New Concept for Composite
Rotor Blade Cross-Sectional Modeling", Journal of the American Helicopter
Society, 42(1), pp. 27-38, 1997. https://doi.org/10.4050/JAHS.42.27

Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
Timoshenko-like modeling of initially curved and twisted composite beams",
International Journal of Solids and Structures, 39(19), pp. 5101-5121,
2002. https://doi.org/10.1016/S0020-7683(02)00399-2

Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
Variational Asymptotic Beam Sectional Analysis", Journal of the American
Helicopter Society, 50(1), pp. 46-55, 2005.
https://doi.org/10.4050/1.3092842

Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic beam
sectional analysis -- An updated version", International Journal of
Engineering Science, 59, pp. 40-64, 2012.
https://doi.org/10.1016/j.ijengsci.2012.03.006
