Available finite elements
=========================

For shells, :mod:`pyfe3d.quad4` is the recommended quadrilateral and
:mod:`pyfe3d.tria3dsg` the recommended triangle. Tria3DSG takes its
transverse shear from a discrete shear gap field and is therefore free of
shear locking by construction, with no tuning parameter.
:mod:`pyfe3d.tria3r` is the older triangle, kept for continuity with results
obtained before Tria3DSG existed; its transverse shear is sampled at the
centroid and locks, and the ``alpha_shear_locking`` stabilisation that
unlocks it has no single value that serves every problem class, so it has to
be verified per problem. For beams, :mod:`pyfe3d.beamc` is recommended over
:mod:`pyfe3d.beamlr`, and over :mod:`pyfe3d.truss` where a truss would
otherwise be used.

.. toctree::
    :titlesonly:

    quad4.rst
    quad4r.rst
    tria3dsg.rst
    tria3r.rst
    beamc.rst
    beamlr.rst
    spring.rst
    truss.rst
