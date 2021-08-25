Finite elements for 3D problems in Python/Cython
================================================

Github Actions status:

[![Actions Status](https://github.com/saullocastro/pyfe3d/workflows/pytest/badge.svg)](https://github.com/saullocastro/pyfe3d/actions)

Coverage status:

[![Codecov Status](https://codecov.io/gh/saullocastro/pyfe3d/branch/master/graph/badge.svg?token=KVZCRIACL7)](https://codecov.io/gh/saullocastro/pyfe3d)

Important features of this library
----------------------------------
- efficient and simple for linear and nonlinear analyses
- importable and cimportable code
- 6 degrees-of-freedom per node

Elements implemented
--------------------
- 'Quad4R' - 4-node plate with linear interpolation, equivalent to Abaqus' S4R
finite element.

- 'BeamLR' - 2-node Timoshenko beam element with linear interpolation and
reduced integration.

- 'BeamC' - 2-node Timoshenko beam element with consistent shape functions and
analytical integration.

- 'Spring' - 2-node spring element with 6 stiffenesses defined in the element
  coordinate system.

- 'Truss' - 2-node truss element with only axial and torsion stiffness. I
recommend using the BeamLR instead, which has physical meaning.

Documentation
-------------

The documentation is available on: https://saullocastro.github.io/pyfe3d.

License
-------
Distrubuted in the 2-Clause BSD license (https://raw.github.com/saullocastro/pyfe3d/master/LICENSE).

Contact: S.G.P.Castro@tudelft.nl

