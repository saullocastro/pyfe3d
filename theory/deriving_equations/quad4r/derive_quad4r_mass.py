from IPython.display import display
import sympy
import numpy as np

sympy.var('offset, rho, h, z, z1, z2')

display(sympy.simplify(sympy.integrate(1, (z, -h/2+offset, h/2+offset))))

display(sympy.simplify(sympy.integrate(1, (z, z1, z2))))

display(sympy.simplify(sympy.integrate(z**2, (z, -h/2+offset, h/2+offset))))

display(sympy.simplify(sympy.integrate(z**2, (z, z1, z2))))

# distorted
x1 = 1.
x2 = 2.5
x3 = 3.5
x4 = 1.5
y1 = 0.5
y2 = 0.7
y3 = 3.1
y4 = 3.2

# regular
x1 = 1.
x2 = 2.5
x3 = 2.5
x4 = 1
y1 = 0.5
y2 = 0.5
y3 = 3.1
y4 = 3.1

A = (np.cross([x2 - x1, y2 - y1], [x4 - x1, y4 - y1])/2 +
     np.cross([x4 - x3, y4 - y3], [x2 - x3, y2 - y3])/2)

#NOTE Gauss quadratures from
#     https://keisan.casio.com/exec/system/1329114617

# 4-point Gauss-Legendre
points_weights = [
[-0.861136311594052575224, 0.3478548451374538573731],
[-0.3399810435848562648027, 0.6521451548625461426269],
[0.3399810435848562648027, 0.6521451548625461426269],
[0.861136311594052575224, 0.3478548451374538573731],
        ]

# 5-point Gauss-Legendre
points_weights = [
[-0.9061798459386639927976, 0.2369268850561890875143],
[-0.5384693101056830910363, 0.4786286704993664680413],
[0, 0.5688888888888888888889],
[0.5384693101056830910363, 0.4786286704993664680413],
[0.9061798459386639927976, 0.2369268850561890875143],
]

# 1-point Gauss-Legendre
points_weights = [
[0, 2.],
]

# 2-point Gauss-Legendre
points_weights = [
[-0.5773502691896257645092, 1],
[0.5773502691896257645092, 1],
]

# 3-point Gauss-Legendre
points_weights = [
[-0.7745966692414833770359, 0.5555555555555555555556],
[0, 0.8888888888888888888889],
[0.7745966692414833770359, 0.555555555555555555556],
]

# 3-point Gauss-Lobatto
points_weights = [
[-1, 0.3333333333333333333333],
[0, 1.333333333333333333333],
[1, 0.3333333333333333333333],
]

# 2-point Gauss-Lobatto
points_weights = [
        [-1, 1.],
        [+1, 1.],
        ]

H = 0
for xi, wi in points_weights:
    for eta, wj in points_weights:
        wij = wi*wj
        detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
        N1 = eta*xi/4 - eta/4 - xi/4 + 1/4
        N2 = -eta*xi/4 - eta/4 + xi/4 + 1/4
        N3 = eta*xi/4 + eta/4 + xi/4 + 1/4
        N4 = -eta*xi/4 + eta/4 - xi/4 + 1/4

        H += wij*detJ*np.outer([N1, N2, N3, N4], [N1, N2, N3, N4])

print('DEBUG H numeric =', H)


H = A/36.*np.array([[4, 2, 1, 2],
                    [2, 4, 2, 1],
                    [1, 2, 4, 2],
                    [2, 1, 2, 4]])
print('DEBUG H constant detJ =', H)
