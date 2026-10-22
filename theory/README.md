# theory

SymPy scripts that generate the expanded element code of pyfe3d. Each script
under `deriving_equations/<element>/` prints the sparsity pattern and the value
lines of one element matrix, which are then pasted into the corresponding
`.pyx`.

These scripts were kept in the separate repository `pyfe3d-dev` until its
commit 9b9971c, from which they were copied here unchanged; the later changes
are in the history of this folder.

## Elements covered

`beamc`, `beamLR`, `beamcurvedLR`, `quad4r`, `tria3r`, `spring`, `truss`.

`quad4` is **not** covered any more. Its constitutive and internal-force
methods in `pyfe3d/quad4.pyx` were rewritten as explicit quadrature loops
over the `Quad4Probe` rows (`_update_probe_KC0ve`, `_update_probe_BL_G`,
`_update_probe_KCNLve`, `_update_probe_finte_nonlinear`), so there is no
generated block left for a generator to be the source of truth of. The
scripts had drifted out of step with the `.pyx` and were removed in 0.10.0
rather than kept as a misleading reference. `update_KG`,
`update_KG_given_stress`, `update_M` and the Piston theory methods of
`Quad4` are still expanded code, produced by those scripts before their
removal; recover them from the history of the former `pyfe3d-dev`
repository if they ever need regenerating.

## References

Two papers that the plate and shell element derivations follow (kept out of
the repository, cite them):

- T. J. R. Hughes, R. L. Taylor and W. Kanoknukulchai, "A simple and efficient
  finite element for plate bending", International Journal for Numerical
  Methods in Engineering, 11(10), 1529-1543, 1977, doi:10.1002/nme.1620111005.
- T. J. R. Hughes and T. E. Tezduyar, "Finite elements based upon Mindlin plate
  theory with particular reference to the four-node bilinear isoparametric
  element", Journal of Applied Mechanics, 48(3), 587-596, 1981,
  doi:10.1115/1.3157679.
