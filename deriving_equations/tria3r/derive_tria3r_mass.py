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
y1 = 0.5
y2 = 0.7
y3 = 3.1
# regular
x1 = 1.
x2 = 2.5
x3 = 2.5
y1 = 0.5
y2 = 0.5
y3 = 3.1


A = np.cross([x2 - x1, y2 - y1], [x3 - x1, y3 - y1])/2
print(A)

points = [0]*3


#NOTE 3-point Gauss-Legendre quadrature for KG
#GAUSSIAN QUADRATURE FORMULAS FOR TRIANGLES
#G. R. COWPER
#https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620070316
weight = 0.333333333333333333333333333333333333333333333
points[0] = 0.66666666666666666666666666666666666666666667
points[1] = 0.16666666666666666666666666666666666666666667
points[2] = 0.16666666666666666666666666666666666666666667

#NOTE 3-point Gauss-Legendre quadrature for KG
#GAUSSIAN QUADRATURE FORMULAS FOR TRIANGLES
#G. R. COWPER
#https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620070316
weight = 1.
points[0] = 0.333333333333333333333333333333333333333333333
points[1] = 0.333333333333333333333333333333333333333333333
points[2] = 0.333333333333333333333333333333333333333333333

if False:
    #NOTE attempting a fake 3-point Gauss-Lobatto
    weight = 0.333333333333333333333333333333333333333333333
    points[0] = 1
    points[1] = 0
    points[2] = 0

H = 0
for i in range(3):
    if i == 0:
        N1 = points[0]
        N2 = points[1]
        N3 = points[2]
    elif i == 1:
        N1 = points[1]
        N2 = points[2]
        N3 = points[0]
    elif i == 2:
        N1 = points[2]
        N2 = points[0]
        N3 = points[1]
    detJ = 2*A
    H += weight*detJ*np.outer([N1, N2, N3], [N1, N2, N3])
    if np.isclose(N1, 0.33333):
        break
print('DEBUG H numeric =')
print(H)


H = A/6*np.array([[2, 1, 1],
                  [1, 2, 1],
                  [1, 1, 2]])
print('DEBUG H constant detJ =')
print(H)

A = sympy.var('A')
detJ = sympy.var('detJ')
N1 = sympy.var('N1')
N2 = sympy.var('N2')
N3 = sympy.var('N3')
wij = 1
Hmatrix = sympy.Matrix([[1, 1, 1]]).T/3
print('detJ*wij*Hmatrix*Hmatrix.T')
print(detJ*wij*Hmatrix*Hmatrix.T)
wij = sympy.var('wij')
Hmatrix = sympy.Matrix([[N1, N2, N3]]).T
print('detJ*wij*Hmatrix*Hmatrix.T')
print(detJ*wij*Hmatrix*Hmatrix.T)
N3 = 1 - N1 - N2
detJ = 2*A
H = detJ*sympy.simplify(sympy.integrate(sympy.integrate(
    Hmatrix*Hmatrix.T, (N2, 0, 1-N1)), (N1, 0, 1)))
print('DEBUG H integrated')
print(H)
