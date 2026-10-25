# theory

SymPy scripts that generate the expanded element code of pyfe3d. Each script
under `deriving_equations/<element>/` prints the sparsity pattern and the value
lines of one element matrix, which are then pasted into the corresponding
`.pyx`, except `tria3r/derive_tria3r_mass.py` and
`quad4r/derive_quad4r_mass.py`, which generate no code. These two check the integrals through the thickness, `intrho`,
`intrhoz` and `intrhoz2`, and the matrix `h_ij` of the integrals of the
products of the shape functions, exactly and with the quadrature rules of
`update_M`. The `derive_*_M_sparse.py` scripts of the shells print the mass
matrices with the symbols `h11`, `h12`, ... and `valH1`, whose values come
from the quadrature written by hand in `update_M` before the generated lines.
`quad4r/derive_quad4r_KC0_sensitivy.py`, a draft of the derivatives of the
stiffness and mass matrices with respect to the nodal coordinates, generates
no code either: no method of pyfe3d computes these sensitivities, and the
matrices it prints are never filled.

These scripts were kept in the separate repository `pyfe3d-dev` until its
commit 9b9971c, from which they were copied here unchanged; the later changes
are in the history of this folder.

SymPy prints the factors and terms sorted by the names of the symbols.
Since the shape-function derivatives are named `S1_x`, `S1_y`, ... instead
of `N1x`, `N1y`, ..., 14 lines of the drilling penalty of `Quad4R` and
`Tria3R` are printed with another order of the factors, e.g.
`5.0e-7*A66*K6ROT*N2*S1_y*detJ*wij` instead of
`5.0e-7*A66*K6ROT*S1_y*N2*detJ*wij`: `KC0e0011`, `KC0e0017`, `KC0e0023`,
`KC0e0111`, `KC0e0117`, `KC0e0123`, `KC0e0617`, `KC0e0623`, `KC0e0717`,
`KC0e0723`, `KC0e1223` and `KC0e1323` of `quad4r/derive_quad4r_KC0_sparse.py`
and `quad4r/derive_quad4r_fint.py`, `KC0e0011` and `KC0e0111` of
`tria3r/derive_tria3r_KC0_sparse.py` and `tria3r/derive_tria3r_fint.py`.
The `.pyx` keep the former order, whose products are evaluated in the same
sequence as before the renaming; the new order changes the last bits of
the results. The lines of the geometric stiffness whose order changed,
printed by `quad4r/derive_quad4r_KG_sparse.py` and
`tria3r/derive_tria3r_KG_sparse.py`, are pasted in the new order, which
gives the same bits, in `Quad4R`, `Tria3R` and in `Quad4`, whose lines are
the same.

## Elements covered

`beamc`, `beamLR`, `beamcurvedLR`, `quad4r`, `tria3r`, `spring`, `truss`.

`quad4` is **not** covered any more. Its constitutive and internal-force
methods in `pyfe3d/quad4.pyx` were rewritten as explicit quadrature loops
over the `Quad4Probe` rows (`_update_probe_KC0ve`, `_update_probe_BL_G`,
`_update_probe_KCNLve`, `_update_probe_finte_nonlinear`), so there is no
generated block left for a generator to be the source of truth of. The
scripts had drifted out of step with the `.pyx` and were removed in 0.10.0
rather than kept as a misleading reference. `update_KG`,
`update_KG_given_stress` and the Piston theory methods of `Quad4` are still
expanded code, produced by those scripts before their removal; recover them
from the history of the former `pyfe3d-dev` repository if they ever need
regenerating. The generated lines of `Quad4.update_M` are those of
`quad4r/derive_quad4r_M_sparse.py`, as in `Quad4R.update_M`.

`tria3dsg` is not covered: its methods are explicit loops in
`pyfe3d/tria3dsg.pyx`.

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
