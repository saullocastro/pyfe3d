General-purpose finite element solver - pyfe3d
==============================================

Github Actions status:

[![Actions Status](https://github.com/saullocastro/pyfe3d/workflows/pytest/badge.svg)](https://github.com/saullocastro/pyfe3d/actions)

Coverage status:

[![Codecov Status](https://codecov.io/gh/saullocastro/pyfe3d/branch/main/graph/badge.svg?token=KVZCRIACL7)](https://codecov.io/gh/saullocastro/pyfe3d)

About pyfe3d
------------

The ``pyfe3d`` module is a general-purpose finite element solver for structural
analysis and optimization based on Python and Cython. The main principles
guiding the development of ``pyfe3d`` are: simplicity, efficiency and
compatibility. The aimed level of compatibility allows one to run this solver
in any platform, including the Google Colab environment.


Citing this library
-------------------

Saullo G. P. Castro. (2026). General-purpose finite element solver based on Python and Cython (Version 0.8.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.6573489.


Documentation
-------------

The documentation is available on: https://saullocastro.github.io/pyfe3d.


Important features of this library
----------------------------------
- efficient and simple for linear and nonlinear analyses
- importable and cimportable code
- 6 degrees-of-freedom per node


Available finite elements
-------------------------
- 'Quad4' - 4-node plate with linear interpolation, equivalent to Nastran's
  CQUAD4. The recommended quadrilateral plate element.

- 'Quad4R' - 4-node plate with linear interpolation, equivalent to Abaqus' S4R.
  It has a not very robust hourglass control.

- 'Tria3R' - 3-node plate with linear interpolation, equivalent to Abaqus' S3R
or Nastran's CTRIA3.

- 'BeamC' - 2-node Timoshenko beam element with consistent shape functions and
analytical integration. The recommended beam element.

- 'BeamLR' - 2-node Timoshenko beam element with linear interpolation and
reduced integration. 

- 'Spring' - 2-node spring element with 6 stiffenesses defined in the element
  coordinate system.

- 'Truss' - 2-node truss element with only axial and torsion stiffness. I
recommend using the BeamLR instead, which is physically more consistent.


Development history
-------------------

See [CHANGELOG.md](CHANGELOG.md) for the details of each version.

| Main versions | Changes |
| - | - |
| 0.8.0 | KCNL matrix for all elements, with more accurate nonlinear analyses; fixed KC0 and KG for beams; fixed KC0 for Tria3R. |
| 0.7.0 | Drilling stiffness re-coupled with in-plane rotation using K6ROT; fixed drilling stiffness integration; restored missing `pid` attribute. |
| 0.6.4 | Internal force vectors (finte) for all elements; fixed drilling stiffness (K6ROT) scaling; added Python 3.14 and removed Python 3.8 support; Linux wheels limited to 64-bit;  Exposed update_probe_finte to Python; new documentation examples. |
| 0.5.2 | New Quad4 element; K6ROT drilling stiffness; probe.update_BL; removed alg3dpy dependency; topology optimization tutorial  |
| 0.4.28 | Piston theory aerodynamics for Quad4R; fixed array casting on some systems; Quad4R less sensitive to small element sizes; new `pid` attribute for elements; exposed element orientation vectors; Python 3.12 support; fixed build for Linux and macOS wheels. |
| 0.3.24 | New Tria3R element; fixed BeamC formulation; fixed beam torsion stiffness; fixed lamination parameters and added their gradients; fixed lumped mass for Tria3R; removed dependency on the NumPy C API; OpenMP build fixes; improved Zenodo metadata and documentation layout; Python 3.10 support. |
| 0.2.8 | Material coordinate systems for plate elements; Fixed Truss mass matrix; more aggressive Cython directives; license changed to 3-clause BSD; license info in README, author metadata updates. |
| 0.1.16 | New Spring element; Rotation matrices using direct `rij` terms instead of Euler angles; renamed `update_xe` to `update_probe_xe`; `probe` made a public attribute; KG with a given stress state; renamed `update_ue` to `update_probe_ue` |
| 0.1.3 | First release: Quad4R, BeamC, BeamLR and Truss elements; shell properties; linear buckling |



Installing pyfe3d
-----------------

First, you should try to install from the distributed binaries by simply doing:

```
python -m pip install pyfe3d
```

If a distribution could not be found, you can try to install from the source
code using:

```
python -m pip install .
```

Another alternative is the following:

```
python -m pip install -r requirements.txt
python setup.py install
```

If none of the above alternatives worked for you, this link shares some
information on how to set up a C compiler on different operating systems: 

https://cython2.readthedocs.io/en/latest/src/quickstart/install.html


License
-------
Distrubuted under the 3-Clause BSD license
(https://raw.github.com/saullocastro/pyfe3d/main/LICENSE):

    Copyright (c) 2021-2026, Saullo G. P. Castro (S.G.P.Castro@tudelft.nl)
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Contacts:
- Saullo G. P. Castro, S.G.P.Castro@tudelft.nl

