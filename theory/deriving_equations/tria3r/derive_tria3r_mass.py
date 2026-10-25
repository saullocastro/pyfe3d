r"""Check of the integrals used by the mass matrices of Tria3R and Tria3DSG

This script generates no code. It prints:

- the integrals through the thickness, per unit density, ``intrho``,
  ``intrhoz`` and ``intrhoz2`` of ShellProp, for a reference surface at the
  distance ``offset`` from the mid-surface;
- the matrix `h_{ij} = \int_A N_i N_j dA` of the linear triangle, exactly,
  and with the three rules of Tria3R.update_M, for a right triangle and for
  a distorted one.

With the area coordinates `N_1, N_2, N_3 = 1 - N_1 - N_2`, the reference
triangle `0 \le N_2 \le 1 - N_1`, `0 \le N_1 \le 1` has the area 1/2 and the
Jacobian determinant of the map to an element of area `A` is ``detJ = 2*A``.
The weights of a rule in the reference triangle therefore sum to 1/2. The
weights `w_i` of Table I of Cowper (1973), doi:10.1002/nme.1620070316, sum to
1 because they multiply the area, `\int_A f dA = A \sum_i w_i f_i`, his
Eq. (1): a weight `w_i` of Cowper is `w_i/2` here.

"""
import numpy as np
import sympy

# integrals through the thickness
offset, rho, h, z = sympy.symbols('offset, rho, h, z', real=True)
for name, f in (('intrho', 1), ('intrhoz', z), ('intrhoz2', z**2)):
    val = sympy.integrate(rho*f, (z, -h/2 + offset, h/2 + offset))
    print(name, '=', sympy.factor(sympy.simplify(val)))
print()

# exact h_ij = int_A N_i N_j dA = detJ * int_ref N_i N_j dN2 dN1
A = sympy.Symbol('A', positive=True)
N1, N2 = sympy.symbols('N1, N2')
N = sympy.Matrix([N1, N2, 1 - N1 - N2])
H_exact = 2*A*(N*N.T).applyfunc(
    lambda f: sympy.integrate(f, (N2, 0, 1 - N1), (N1, 0, 1)))
print('exact H =', H_exact)
assert sympy.simplify(H_exact - A/12*sympy.Matrix([[2, 1, 1],
                                                   [1, 2, 1],
                                                   [1, 1, 2]])) == sympy.zeros(3, 3)
print()

# rules in the reference triangle, area coordinates (N1, N2, N3) and weights
# summing to 1/2, the area of the reference triangle
third = sympy.Rational(1, 3)
rules = {
    # Cowper (1973), Table I, first row: (2/3, 1/6, 1/6) and permutations,
    # Cowper weight 1/3, here 1/6; degree of precision 2, Tria3R mtype=0
    'three-point, Cowper Table I': (
        [(sympy.Rational(2, 3), sympy.Rational(1, 6), sympy.Rational(1, 6)),
         (sympy.Rational(1, 6), sympy.Rational(2, 3), sympy.Rational(1, 6)),
         (sympy.Rational(1, 6), sympy.Rational(1, 6), sympy.Rational(2, 3))],
        [sympy.Rational(1, 6)]*3),
    # centroid, weight 1/2; degree of precision 1, Tria3R mtype=2
    'one-point, centroid': ([(third, third, third)], [sympy.Rational(1, 2)]),
    # vertices, the nodal quadrature, weight 1/6 each; degree of precision
    # 1, Tria3R mtype=1 and Tria3DSG mtype=1
    'vertex rule': ([(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                    [sympy.Rational(1, 6)]*3),
}

geometries = {
    'right': np.array([[1., 0.5], [2.5, 0.5], [2.5, 3.1]]),
    'distorted': np.array([[1., 0.5], [2.5, 0.7], [3.5, 3.1]]),
}

for gname, xy in geometries.items():
    (x1, y1), (x2, y2), (x3, y3) = xy
    Anum = ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))/2
    detJ = 2*Anum
    print('%s triangle, A = %g' % (gname, Anum))
    Hex = np.array(H_exact.subs(A, Anum), dtype=float)
    for rname, (points, weights) in rules.items():
        assert sum(weights) == sympy.Rational(1, 2)
        H = np.zeros((3, 3))
        for p, w in zip(points, weights):
            Np = np.array(p, dtype=float)
            H += float(w)*detJ*np.outer(Np, Np)
        print('  %-28s H/A =' % rname, (H/Anum).round(6).tolist())
        if rname.startswith('three-point'):
            assert np.allclose(H, Hex)
        elif rname.startswith('one-point'):
            assert np.allclose(H, Anum/9*np.ones((3, 3)))
        else:
            assert np.allclose(H, Anum/3*np.eye(3))
        # every rule keeps the area and the first moment of the area, since
        # sum_i N_i = 1: sum_ij h_ij = A and sum_ij h_ij x_j = int_A x dA
        assert np.isclose(H.sum(), Anum)
        assert np.allclose(H.sum(axis=0) @ xy, Anum*xy.mean(axis=0))
    print()
