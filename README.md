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

Saullo G. P. Castro. (2023). General-purpose finite element solver based on Python and Cython (Version 0.4.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.6573489.


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
- 'Quad4R' - 4-node plate with linear interpolation, equivalent to Abaqus' S4R
or Nastran's CQUAD4.

- 'Tria3R' - 3-node plate with linear interpolation, equivalent to Abaqus' S3R
or Nastran's CTRIA3.

- 'BeamLR' - 2-node Timoshenko beam element with linear interpolation and
reduced integration.

- 'BeamC' - 2-node Timoshenko beam element with consistent shape functions and
analytical integration.

- 'Spring' - 2-node spring element with 6 stiffenesses defined in the element
  coordinate system.

- 'Truss' - 2-node truss element with only axial and torsion stiffness. I
recommend using the BeamLR instead, which is physically more consistent.


License
-------
Distrubuted under the 3-Clause BSD license
(https://raw.github.com/saullocastro/pyfe3d/main/LICENSE):

    Copyright (c) 2021-2023, Saullo G. P. Castro (S.G.P.Castro@tudelft.nl)
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Contacts:
- Saullo G. P. Castro, S.G.P.Castro@tudelft.nl

