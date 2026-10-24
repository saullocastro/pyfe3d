r"""Check of the integrals used by the mass matrices of Quad4R and Quad4

This script generates no code. It prints:

- the integrals through the thickness, per unit density, ``intrho``,
  ``intrhoz`` and ``intrhoz2`` of ShellProp, for a reference surface at the
  distance ``offset`` from the mid-surface;
- the matrix `h_{ij} = \int_A N_i N_j dA` of the bilinear quadrilateral
  computed with the Gauss rules of `n` points in each direction of the
  natural coordinates `\xi, \eta \in [-1, 1]`, whose weights sum to 2, and
  the Jacobian determinant of the element, for a rectangle and for a
  distorted quadrilateral, and, for the rectangle, the closed form of
  Brockman (1987), Eq. (22).

Gauss-Legendre with 2 points per direction (Quad4R and Quad4 ``mtype=0``) is
exact; the 2-point Gauss-Lobatto rule (``mtype=2``), whose points are the
nodes, gives `h_{ii}` equal to the Jacobian determinant at node `i` and
`h_{ij} = 0` for `i \neq j`.

"""
import numpy as np
import sympy

# integrals through the thickness
offset, rho, h, z = sympy.symbols('offset, rho, h, z', real=True)
for name, f in (('intrho', 1), ('intrhoz', z), ('intrhoz2', z**2)):
    val = sympy.integrate(rho*f, (z, -h/2 + offset, h/2 + offset))
    print(name, '=', sympy.factor(sympy.simplify(val)))
print()

geometries = {
    'rectangle': np.array([[1., 0.5], [2.5, 0.5], [2.5, 3.1], [1., 3.1]]),
    'distorted': np.array([[1., 0.5], [2.5, 0.7], [3.5, 3.1], [1.5, 3.2]]),
}

# points and weights in [-1, 1], the weights summing to 2
leg = np.polynomial.legendre.leggauss
rules = {
    'Gauss-Legendre, 1 point': leg(1),
    'Gauss-Legendre, 2 points': leg(2),
    'Gauss-Legendre, 3 points': leg(3),
    'Gauss-Lobatto, 2 points': (np.array([-1., 1.]), np.array([1., 1.])),
    'Gauss-Lobatto, 3 points': (np.array([-1., 0., 1.]),
                                np.array([1/3, 4/3, 1/3])),
}


def H_rule(xy, points, weights):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = xy
    H = np.zeros((4, 4))
    for xi, wi in zip(points, weights):
        for eta, wj in zip(points, weights):
            detJ = ((-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16
                    - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16)
            N = np.array([(1 - xi)*(1 - eta), (1 + xi)*(1 - eta),
                          (1 + xi)*(1 + eta), (1 - xi)*(1 + eta)])/4
            H += wi*wj*detJ*np.outer(N, N)
    return H


for gname, xy in geometries.items():
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = xy
    A = ((x3 - x1)*(y4 - y2) - (x4 - x2)*(y3 - y1))/2
    print('%s, A = %g' % (gname, A))
    Hex = H_rule(xy, *leg(4))
    for rname, (points, weights) in rules.items():
        assert np.isclose(weights.sum(), 2.)
        H = H_rule(xy, points, weights)
        print('  %-26s H/A =' % rname, (H/A).round(6).tolist())
        assert np.isclose(H.sum(), A)
        if rname == 'Gauss-Legendre, 2 points':
            assert np.allclose(H, Hex)
        if rname == 'Gauss-Lobatto, 2 points':
            assert np.allclose(H, np.diag(np.diag(H)))
    if gname == 'rectangle':
        # Brockman (1987), Eq. (22), for a constant Jacobian determinant
        H22 = A/36*np.array([[4, 2, 1, 2],
                             [2, 4, 2, 1],
                             [1, 2, 4, 2],
                             [2, 1, 2, 4]])
        print('  %-26s H/A =' % 'Brockman Eq. (22)', (H22/A).round(6).tolist())
        assert np.allclose(H22, Hex)
    print()
