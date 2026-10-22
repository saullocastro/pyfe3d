"""
Sensitivities calculated with respect to 8 geometric variables x1, y1, x2, y2, x3, y3, x4, y4
Calculated for stiffness and mass matrix
Sensitivities from 3D global variables to be transformed in 2D element variables

"""
import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, diff
from sympy.vector import CoordSys3D, cross

r"""

    4 ____ 3
     /   /
    /   /   positive normal in CCW
   /___/
   1    2

"""

DOF = 6
num_nodes = 4

sympy.var('h', positive=True, real=True)
sympy.var('x1, y1, x2, y2, x3, y3, x4, y4', real=True, positive=True)
sympy.var('rho, xi, eta, A, alphat')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('E44, E45, E55')
#NOTE shear correction factor should be applied to E44, E45 and E55
#     in the finite element code

ONE = sympy.Integer(1)

R = CoordSys3D('R')
r1 = x1*R.i + y1*R.j
r2 = x2*R.i + y2*R.j
r3 = x3*R.i + y3*R.j
r4 = x4*R.i + y4*R.j

rbottom = r1 + (r2 - r1)*(xi + 1)/2
rtop = r4 + (r3 - r4)*(xi + 1)/2
r = rbottom + (rtop - rbottom)*(eta + 1)/2
xfunc = r.components[R.i]
yfunc = r.components[R.j]

# Jacobian
# http://kis.tu.kielce.pl/mo/COLORADO_FEM/colorado/IFEM.Ch17.pdf
#NOTE for linear element these derivatives are constant
# xi = xi(x, y)
# eta = eta(x, y)
#J = [dx/dxi  dy/dxi ]
#    [dx/deta dy/deta]
# dx   = J.T dxi
# dy         deta
#
# dxi   = Jinv.T dx
# deta           dy
#
# Jinv:
# d/dx = d/dxi*dxi/dx + d/deta*deta/dx = [dxi/dx   deta/dx] d/dxi  =  [j11  j12] d/dxi
# d/dy   d/dxi*dxi/dy + d/deta*deta/dy   [dxi/dy   deta/dy] d/deta =  [j21  j22] d/deta
#
J = Matrix([[xfunc.diff(xi),  yfunc.diff(xi)],
            [xfunc.diff(eta), yfunc.diff(eta)]])
detJ = J.det().simplify()
detJ_x1 = diff(detJ, x1).simplify()
detJ_y1 = diff(detJ, y1).simplify()
detJ_x2 = diff(detJ, x2).simplify()
detJ_y2 = diff(detJ, y2).simplify()
detJ_x3 = diff(detJ, x3).simplify()
detJ_y3 = diff(detJ, y3).simplify()
detJ_x4 = diff(detJ, x4).simplify()
detJ_y4 = diff(detJ, y4).simplify()
print('detJ =', detJ)
print('detJ_x1 =', detJ_x1)
print('detJ_y1 =', detJ_y1)
print('detJ_x2 =', detJ_x2)
print('detJ_y2 =', detJ_y2)
print('detJ_x3 =', detJ_x3)
print('detJ_y3 =', detJ_y3)
print('detJ_x4 =', detJ_x4)
print('detJ_y4 =', detJ_y4)

j = J.inv()
j11 = j[0, 0].simplify()
j11_x1 = diff(j[0, 0], x1).simplify()
j11_y1 = diff(j[0, 0], y1).simplify()
j11_x2 = diff(j[0, 0], x2).simplify()
j11_y2 = diff(j[0, 0], y2).simplify()
j11_x3 = diff(j[0, 0], x3).simplify()
j11_y3 = diff(j[0, 0], y3).simplify()
j11_x4 = diff(j[0, 0], x4).simplify()
j11_y4 = diff(j[0, 0], y4).simplify()

j12 = j[0, 1].simplify()
j12_x1 = diff(j[0, 1], x1).simplify()
j12_y1 = diff(j[0, 1], y1).simplify()
j12_x2 = diff(j[0, 1], x2).simplify()
j12_y2 = diff(j[0, 1], y2).simplify()
j12_x3 = diff(j[0, 1], x3).simplify()
j12_y3 = diff(j[0, 1], y3).simplify()
j12_x4 = diff(j[0, 1], x4).simplify()
j12_y4 = diff(j[0, 1], y4).simplify()

j21 = j[1, 0].simplify()
j21_x1 = diff(j[1, 0], x1).simplify()
j21_y1 = diff(j[1, 0], y1).simplify()
j21_x2 = diff(j[1, 0], x2).simplify()
j21_y2 = diff(j[1, 0], y2).simplify()
j21_x3 = diff(j[1, 0], x3).simplify()
j21_y3 = diff(j[1, 0], y3).simplify()
j21_x4 = diff(j[1, 0], x4).simplify()
j21_y4 = diff(j[1, 0], y4).simplify()

j22 = j[1, 1].simplify()
j22_x1 = diff(j[1, 1], x1).simplify()
j22_y1 = diff(j[1, 1], y1).simplify()
j22_x2 = diff(j[1, 1], x2).simplify()
j22_y2 = diff(j[1, 1], y2).simplify()
j22_x3 = diff(j[1, 1], x3).simplify()
j22_y3 = diff(j[1, 1], y3).simplify()
j22_x4 = diff(j[1, 1], x4).simplify()
j22_y4 = diff(j[1, 1], y4).simplify()

print('j11 =', j11)
print('j11_x1 =', j11_x1)
print('j11_y1 =', j11_y1)
print('j11_x2 =', j11_x2)
print('j11_y2 =', j11_y2)
print('j11_x3 =', j11_x3)
print('j11_y3 =', j11_y3)
print('j11_x4 =', j11_x4)
print('j11_y4 =', j11_y4)

print('j12 =', j12)
print('j12_x1 =', j12_x1)
print('j12_y1 =', j12_y1)
print('j12_x2 =', j12_x2)
print('j12_y2 =', j12_y2)
print('j12_x3 =', j12_x3)
print('j12_y3 =', j12_y3)
print('j12_x4 =', j12_x4)
print('j12_y4 =', j12_y4)

print('j21 =', j21)
print('j21_x1 =', j21_x1)
print('j21_y1 =', j21_y1)
print('j21_x2 =', j21_x2)
print('j21_y2 =', j21_y2)
print('j21_x3 =', j21_x3)
print('j21_y3 =', j21_y3)
print('j21_x4 =', j21_x4)
print('j21_y4 =', j21_y4)

print('j22 =', j22)
print('j22_x1 =', j22_x1)
print('j22_y1 =', j22_y1)
print('j22_x2 =', j22_x2)
print('j22_y2 =', j22_y2)
print('j22_x3 =', j22_x3)
print('j22_y3 =', j22_y3)
print('j22_x4 =', j22_x4)
print('j22_y4 =', j22_y4)

j11, j12, j21, j22 = sympy.var('j11, j12, j21, j22')
j11_x1, j11_y1, j11_x2, j11_y2, j11_x3, j11_y3, j11_x4, j11_y4 = sympy.var('j11_x1, j11_y1, j11_x2, j11_y2, j11_x3, j11_y3, j11_x4, j11_y4')
j12_x1, j12_y1, j12_x2, j12_y2, j12_x3, j12_y3, j12_x4, j12_y4 = sympy.var('j12_x1, j12_y1, j12_x2, j12_y2, j12_x3, j12_y3, j12_x4, j12_y4')
j21_x1, j21_y1, j21_x2, j21_y2, j21_x3, j21_y3, j21_x4, j21_y4 = sympy.var('j21_x1, j21_y1, j21_x2, j21_y2, j21_x3, j21_y3, j21_x4, j21_y4')
j22_x1, j22_y1, j22_x2, j22_y2, j22_x3, j22_y3, j22_x4, j22_y4 = sympy.var('j22_x1, j22_y1, j22_x2, j22_y2, j22_x3, j22_y3, j22_x4, j22_y4')

N1 = (eta*xi - eta - xi + 1)/4
N2 = -(eta*xi + eta - xi - 1)/4
N3 = (eta*xi + eta + xi + 1)/4
N4 = -(eta*xi - eta + xi - 1)/4

N1xi = N1.diff(xi)
N2xi = N2.diff(xi)
N3xi = N3.diff(xi)
N4xi = N4.diff(xi)

N1eta = N1.diff(eta)
N2eta = N2.diff(eta)
N3eta = N3.diff(eta)
N4eta = N4.diff(eta)

N1x = j11*N1xi + j12*N1eta
N1x_x1 = j11_x1*N1xi + j12_x1*N1eta
N1x_y1 = j11_y1*N1xi + j12_y1*N1eta
N1x_x2 = j11_x2*N1xi + j12_x2*N1eta
N1x_y2 = j11_y2*N1xi + j12_y2*N1eta
N1x_x3 = j11_x3*N1xi + j12_x3*N1eta
N1x_y3 = j11_y3*N1xi + j12_y3*N1eta
N1x_x4 = j11_x4*N1xi + j12_x4*N1eta
N1x_y4 = j11_y4*N1xi + j12_y4*N1eta

N2x = j11*N2xi + j12*N2eta
N2x_x1 = j11_x1*N2xi + j12_x1*N2eta
N2x_y1 = j11_y1*N2xi + j12_y1*N2eta
N2x_x2 = j11_x2*N2xi + j12_x2*N2eta
N2x_y2 = j11_y2*N2xi + j12_y2*N2eta
N2x_x3 = j11_x3*N2xi + j12_x3*N2eta
N2x_y3 = j11_y3*N2xi + j12_y3*N2eta
N2x_x4 = j11_x4*N2xi + j12_x4*N2eta
N2x_y4 = j11_y4*N2xi + j12_y4*N2eta

N3x = j11*N3xi + j12*N3eta
N3x_x1 = j11_x1*N3xi + j12_x1*N3eta
N3x_y1 = j11_y1*N3xi + j12_y1*N3eta
N3x_x2 = j11_x2*N3xi + j12_x2*N3eta
N3x_y2 = j11_y2*N3xi + j12_y2*N3eta
N3x_x3 = j11_x3*N3xi + j12_x3*N3eta
N3x_y3 = j11_y3*N3xi + j12_y3*N3eta
N3x_x4 = j11_x4*N3xi + j12_x4*N3eta
N3x_y4 = j11_y4*N3xi + j12_y4*N3eta

N4x = j11*N4xi + j12*N4eta
N4x_x1 = j11_x1*N4xi + j12_x1*N4eta
N4x_y1 = j11_y1*N4xi + j12_y1*N4eta
N4x_x2 = j11_x2*N4xi + j12_x2*N4eta
N4x_y2 = j11_y2*N4xi + j12_y2*N4eta
N4x_x3 = j11_x3*N4xi + j12_x3*N4eta
N4x_y3 = j11_y3*N4xi + j12_y3*N4eta
N4x_x4 = j11_x4*N4xi + j12_x4*N4eta
N4x_y4 = j11_y4*N4xi + j12_y4*N4eta

N1xxi = N1x.diff(xi)
N1xxi_x1 = N1x_x1.diff(xi)
N1xxi_y1 = N1x_y1.diff(xi)
N1xxi_x2 = N1x_x2.diff(xi)
N1xxi_y2 = N1x_y2.diff(xi)
N1xxi_x3 = N1x_x3.diff(xi)
N1xxi_y3 = N1x_y3.diff(xi)
N1xxi_x4 = N1x_x4.diff(xi)
N1xxi_y4 = N1x_y4.diff(xi)

N1xeta = N1x.diff(eta)
N1xeta_x1 = N1x_x1.diff(eta)
N1xeta_y1 = N1x_y1.diff(eta)
N1xeta_x2 = N1x_x2.diff(eta)
N1xeta_y2 = N1x_y2.diff(eta)
N1xeta_x3 = N1x_x3.diff(eta)
N1xeta_y3 = N1x_y3.diff(eta)
N1xeta_x4 = N1x_x4.diff(eta)
N1xeta_y4 = N1x_y4.diff(eta)

N2xxi = N2x.diff(xi)
N2xxi_x1 = N2x_x1.diff(xi)
N2xxi_y1 = N2x_y1.diff(xi)
N2xxi_x2 = N2x_x2.diff(xi)
N2xxi_y2 = N2x_y2.diff(xi)
N2xxi_x3 = N2x_x3.diff(xi)
N2xxi_y3 = N2x_y3.diff(xi)
N2xxi_x4 = N2x_x4.diff(xi)
N2xxi_y4 = N2x_y4.diff(xi)

N2xeta = N2x.diff(eta)
N2xeta_x1 = N2x_x1.diff(eta)
N2xeta_y1 = N2x_y1.diff(eta)
N2xeta_x2 = N2x_x2.diff(eta)
N2xeta_y2 = N2x_y2.diff(eta)
N2xeta_x3 = N2x_x3.diff(eta)
N2xeta_y3 = N2x_y3.diff(eta)
N2xeta_x4 = N2x_x4.diff(eta)
N2xeta_y4 = N2x_y4.diff(eta)

N3xxi = N3x.diff(xi)
N3xxi_x1 = N3x_x1.diff(xi)
N3xxi_y1 = N3x_y1.diff(xi)
N3xxi_x2 = N3x_x2.diff(xi)
N3xxi_y2 = N3x_y2.diff(xi)
N3xxi_x3 = N3x_x3.diff(xi)
N3xxi_y3 = N3x_y3.diff(xi)
N3xxi_x4 = N3x_x4.diff(xi)
N3xxi_y4 = N3x_y4.diff(xi)

N3xeta = N3x.diff(eta)
N3xeta_x1 = N3x_x1.diff(eta)
N3xeta_y1 = N3x_y1.diff(eta)
N3xeta_x2 = N3x_x2.diff(eta)
N3xeta_y2 = N3x_y2.diff(eta)
N3xeta_x3 = N3x_x3.diff(eta)
N3xeta_y3 = N3x_y3.diff(eta)
N3xeta_x4 = N3x_x4.diff(eta)
N3xeta_y4 = N3x_y4.diff(eta)

N4xxi = N4x.diff(xi)
N4xxi_x1 = N4x_x1.diff(xi)
N4xxi_y1 = N4x_y1.diff(xi)
N4xxi_x2 = N4x_x2.diff(xi)
N4xxi_y2 = N4x_y2.diff(xi)
N4xxi_x3 = N4x_x3.diff(xi)
N4xxi_y3 = N4x_y3.diff(xi)
N4xxi_x4 = N4x_x4.diff(xi)
N4xxi_y4 = N4x_y4.diff(xi)

N4xeta = N4x.diff(eta)
N4xeta_x1 = N4x_x1.diff(eta)
N4xeta_y1 = N4x_y1.diff(eta)
N4xeta_x2 = N4x_x2.diff(eta)
N4xeta_y2 = N4x_y2.diff(eta)
N4xeta_x3 = N4x_x3.diff(eta)
N4xeta_y3 = N4x_y3.diff(eta)
N4xeta_x4 = N4x_x4.diff(eta)
N4xeta_y4 = N4x_y4.diff(eta)


N1xy = j21*N1xxi + j22*N1xeta
N1xy_x1 = (j21_x1*N1xxi + j22_x1*N1xeta + j21*N1xxi_x1 + j22*N1xeta_x1)
N1xy_y1 = (j21_y1*N1xxi + j22_y1*N1xeta + j21*N1xxi_y1 + j22*N1xeta_y1)
N1xy_x2 = (j21_x2*N1xxi + j22_x2*N1xeta + j21*N1xxi_x2 + j22*N1xeta_x2)
N1xy_y2 = (j21_y2*N1xxi + j22_y2*N1xeta + j21*N1xxi_y2 + j22*N1xeta_y2)
N1xy_x3 = (j21_x3*N1xxi + j22_x3*N1xeta + j21*N1xxi_x3 + j22*N1xeta_x3)
N1xy_y3 = (j21_y3*N1xxi + j22_y3*N1xeta + j21*N1xxi_y3 + j22*N1xeta_y3)
N1xy_x4 = (j21_x4*N1xxi + j22_x4*N1xeta + j21*N1xxi_x4 + j22*N1xeta_x4)
N1xy_y4 = (j21_y4*N1xxi + j22_y4*N1xeta + j21*N1xxi_y4 + j22*N1xeta_y4)

N2xy = j21*N2xxi + j22*N2xeta
N2xy_x1 = (j21_x1*N2xxi + j22_x1*N2xeta + j21*N2xxi_x1 + j22*N2xeta_x1)
N2xy_y1 = (j21_y1*N2xxi + j22_y1*N2xeta + j21*N2xxi_y1 + j22*N2xeta_y1)
N2xy_x2 = (j21_x2*N2xxi + j22_x2*N2xeta + j21*N2xxi_x2 + j22*N2xeta_x2)
N2xy_y2 = (j21_y2*N2xxi + j22_y2*N2xeta + j21*N2xxi_y2 + j22*N2xeta_y2)
N2xy_x3 = (j21_x3*N2xxi + j22_x3*N2xeta + j21*N2xxi_x3 + j22*N2xeta_x3)
N2xy_y3 = (j21_y3*N2xxi + j22_y3*N2xeta + j21*N2xxi_y3 + j22*N2xeta_y3)
N2xy_x4 = (j21_x4*N2xxi + j22_x4*N2xeta + j21*N2xxi_x4 + j22*N2xeta_x4)
N2xy_y4 = (j21_y4*N2xxi + j22_y4*N2xeta + j21*N2xxi_y4 + j22*N2xeta_y4)

N3xy = j21*N3xxi + j22*N3xeta
N3xy_x1 = (j21_x1*N3xxi + j22_x1*N3xeta + j21*N3xxi_x1 + j22*N3xeta_x1)
N3xy_y1 = (j21_y1*N3xxi + j22_y1*N3xeta + j21*N3xxi_y1 + j22*N3xeta_y1)
N3xy_x2 = (j21_x2*N3xxi + j22_x2*N3xeta + j21*N3xxi_x2 + j22*N3xeta_x2)
N3xy_y2 = (j21_y2*N3xxi + j22_y2*N3xeta + j21*N3xxi_y2 + j22*N3xeta_y2)
N3xy_x3 = (j21_x3*N3xxi + j22_x3*N3xeta + j21*N3xxi_x3 + j22*N3xeta_x3)
N3xy_y3 = (j21_y3*N3xxi + j22_y3*N3xeta + j21*N3xxi_y3 + j22*N3xeta_y3)
N3xy_x4 = (j21_x4*N3xxi + j22_x4*N3xeta + j21*N3xxi_x4 + j22*N3xeta_x4)
N3xy_y4 = (j21_y4*N3xxi + j22_y4*N3xeta + j21*N3xxi_y4 + j22*N3xeta_y4)

N4xy = j21*N4xxi + j22*N4xeta
N4xy_x1 = (j21_x1*N4xxi + j22_x1*N4xeta + j21*N4xxi_x1 + j22*N4xeta_x1)
N4xy_y1 = (j21_y1*N4xxi + j22_y1*N4xeta + j21*N4xxi_y1 + j22*N4xeta_y1)
N4xy_x2 = (j21_x2*N4xxi + j22_x2*N4xeta + j21*N4xxi_x2 + j22*N4xeta_x2)
N4xy_y2 = (j21_y2*N4xxi + j22_y2*N4xeta + j21*N4xxi_y2 + j22*N4xeta_y2)
N4xy_x3 = (j21_x3*N4xxi + j22_x3*N4xeta + j21*N4xxi_x3 + j22*N4xeta_x3)
N4xy_y3 = (j21_y3*N4xxi + j22_y3*N4xeta + j21*N4xxi_y3 + j22*N4xeta_y3)
N4xy_x4 = (j21_x4*N4xxi + j22_x4*N4xeta + j21*N4xxi_x4 + j22*N4xeta_x4)
N4xy_y4 = (j21_y4*N4xxi + j22_y4*N4xeta + j21*N4xxi_y4 + j22*N4xeta_y4)

N1y = j21*N1xi + j22*N1eta
N1y_x1 = j21_x1*N1xi + j22_x1*N1eta
N1y_y1 = j21_y1*N1xi + j22_y1*N1eta
N1y_x2 = j21_x2*N1xi + j22_x2*N1eta
N1y_y2 = j21_y2*N1xi + j22_y2*N1eta
N1y_x3 = j21_x3*N1xi + j22_x3*N1eta
N1y_y3 = j21_y3*N1xi + j22_y3*N1eta
N1y_x4 = j21_x4*N1xi + j22_x4*N1eta
N1y_y4 = j21_y4*N1xi + j22_y4*N1eta

N2y = j21*N2xi + j22*N2eta
N2y_x1 = j21_x1*N2xi + j22_x1*N2eta
N2y_y1 = j21_y1*N2xi + j22_y1*N2eta
N2y_x2 = j21_x2*N2xi + j22_x2*N2eta
N2y_y2 = j21_y2*N2xi + j22_y2*N2eta
N2y_x3 = j21_x3*N2xi + j22_x3*N2eta
N2y_y3 = j21_y3*N2xi + j22_y3*N2eta
N2y_x4 = j21_x4*N2xi + j22_x4*N2eta
N2y_y4 = j21_y4*N2xi + j22_y4*N2eta

N3y = j21*N3xi + j22*N3eta
N3y_x1 = j21_x1*N3xi + j22_x1*N3eta
N3y_y1 = j21_y1*N3xi + j22_y1*N3eta
N3y_x2 = j21_x2*N3xi + j22_x2*N3eta
N3y_y2 = j21_y2*N3xi + j22_y2*N3eta
N3y_x3 = j21_x3*N3xi + j22_x3*N3eta
N3y_y3 = j21_y3*N3xi + j22_y3*N3eta
N3y_x4 = j21_x4*N3xi + j22_x4*N3eta
N3y_y4 = j21_y4*N3xi + j22_y4*N3eta

N4y = j21*N4xi + j22*N4eta
N4y_x1 = j21_x1*N4xi + j22_x1*N4eta
N4y_y1 = j21_y1*N4xi + j22_y1*N4eta
N4y_x2 = j21_x2*N4xi + j22_x2*N4eta
N4y_y2 = j21_y2*N4xi + j22_y2*N4eta
N4y_x3 = j21_x3*N4xi + j22_x3*N4eta
N4y_y3 = j21_y3*N4xi + j22_y3*N4eta
N4y_x4 = j21_x4*N4xi + j22_x4*N4eta
N4y_y4 = j21_y4*N4xi + j22_y4*N4eta

N1yxi = N1y.diff(xi)
N1yxi_x1 = N1y_x1.diff(xi)
N1yxi_y1 = N1y_y1.diff(xi)
N1yxi_x2 = N1y_x2.diff(xi)
N1yxi_y2 = N1y_y2.diff(xi)
N1yxi_x3 = N1y_x3.diff(xi)
N1yxi_y3 = N1y_y3.diff(xi)
N1yxi_x4 = N1y_x4.diff(xi)
N1yxi_y4 = N1y_y4.diff(xi)

N1yeta = N1y.diff(eta)
N1yeta_x1 = N1y_x1.diff(eta)
N1yeta_y1 = N1y_y1.diff(eta)
N1yeta_x2 = N1y_x2.diff(eta)
N1yeta_y2 = N1y_y2.diff(eta)
N1yeta_x3 = N1y_x3.diff(eta)
N1yeta_y3 = N1y_y3.diff(eta)
N1yeta_x4 = N1y_x4.diff(eta)
N1yeta_y4 = N1y_y4.diff(eta)

N2yxi = N2y.diff(xi)
N2yxi_x1 = N2y_x1.diff(xi)
N2yxi_y1 = N2y_y1.diff(xi)
N2yxi_x2 = N2y_x2.diff(xi)
N2yxi_y2 = N2y_y2.diff(xi)
N2yxi_x3 = N2y_x3.diff(xi)
N2yxi_y3 = N2y_y3.diff(xi)
N2yxi_x4 = N2y_x4.diff(xi)
N2yxi_y4 = N2y_y4.diff(xi)

N2yeta = N2y.diff(eta)
N2yeta_x1 = N2y_x1.diff(eta)
N2yeta_y1 = N2y_y1.diff(eta)
N2yeta_x2 = N2y_x2.diff(eta)
N2yeta_y2 = N2y_y2.diff(eta)
N2yeta_x3 = N2y_x3.diff(eta)
N2yeta_y3 = N2y_y3.diff(eta)
N2yeta_x4 = N2y_x4.diff(eta)
N2yeta_y4 = N2y_y4.diff(eta)

N3yxi = N3y.diff(xi)
N3yxi_x1 = N3y_x1.diff(xi)
N3yxi_y1 = N3y_y1.diff(xi)
N3yxi_x2 = N3y_x2.diff(xi)
N3yxi_y2 = N3y_y2.diff(xi)
N3yxi_x3 = N3y_x3.diff(xi)
N3yxi_y3 = N3y_y3.diff(xi)
N3yxi_x4 = N3y_x4.diff(xi)
N3yxi_y4 = N3y_y4.diff(xi)

N3yeta = N3y.diff(eta)
N3yeta_x1 = N3y_x1.diff(eta)
N3yeta_y1 = N3y_y1.diff(eta)
N3yeta_x2 = N3y_x2.diff(eta)
N3yeta_y2 = N3y_y2.diff(eta)
N3yeta_x3 = N3y_x3.diff(eta)
N3yeta_y3 = N3y_y3.diff(eta)
N3yeta_x4 = N3y_x4.diff(eta)
N3yeta_y4 = N3y_y4.diff(eta)

N4yxi = N4y.diff(xi)
N4yxi_x1 = N4y_x1.diff(xi)
N4yxi_y1 = N4y_y1.diff(xi)
N4yxi_x2 = N4y_x2.diff(xi)
N4yxi_y2 = N4y_y2.diff(xi)
N4yxi_x3 = N4y_x3.diff(xi)
N4yxi_y3 = N4y_y3.diff(xi)
N4yxi_x4 = N4y_x4.diff(xi)
N4yxi_y4 = N4y_y4.diff(xi)

N4yeta = N4y.diff(eta)
N4yeta_x1 = N4y_x1.diff(eta)
N4yeta_y1 = N4y_y1.diff(eta)
N4yeta_x2 = N4y_x2.diff(eta)
N4yeta_y2 = N4y_y2.diff(eta)
N4yeta_x3 = N4y_x3.diff(eta)
N4yeta_y3 = N4y_y3.diff(eta)
N4yeta_x4 = N4y_x4.diff(eta)
N4yeta_y4 = N4y_y4.diff(eta)

N1yx = j11*N1yxi + j12*N1yeta
N1yx_x1 = (j11_x1*N1yxi + j12_x1*N1yeta + j11*N1yxi_x1 + j12*N1yeta_x1)
N1yx_y1 = (j11_y1*N1yxi + j12_y1*N1yeta + j11*N1yxi_y1 + j12*N1yeta_y1)
N1yx_x2 = (j11_x2*N1yxi + j12_x2*N1yeta + j11*N1yxi_x2 + j12*N1yeta_x2)
N1yx_y2 = (j11_y2*N1yxi + j12_y2*N1yeta + j11*N1yxi_y2 + j12*N1yeta_y2)
N1yx_x3 = (j11_x3*N1yxi + j12_x3*N1yeta + j11*N1yxi_x3 + j12*N1yeta_x3)
N1yx_y3 = (j11_y3*N1yxi + j12_y3*N1yeta + j11*N1yxi_y3 + j12*N1yeta_y3)
N1yx_x4 = (j11_x4*N1yxi + j12_x4*N1yeta + j11*N1yxi_x4 + j12*N1yeta_x4)
N1yx_y4 = (j11_y4*N1yxi + j12_y4*N1yeta + j11*N1yxi_y4 + j12*N1yeta_y4)

N2yx = j11*N2yxi + j12*N2yeta
N2yx_x1 = (j11_x1*N2yxi + j12_x1*N2yeta + j11*N2yxi_x1 + j12*N2yeta_x1)
N2yx_y1 = (j11_y1*N2yxi + j12_y1*N2yeta + j11*N2yxi_y1 + j12*N2yeta_y1)
N2yx_x2 = (j11_x2*N2yxi + j12_x2*N2yeta + j11*N2yxi_x2 + j12*N2yeta_x2)
N2yx_y2 = (j11_y2*N2yxi + j12_y2*N2yeta + j11*N2yxi_y2 + j12*N2yeta_y2)
N2yx_x3 = (j11_x3*N2yxi + j12_x3*N2yeta + j11*N2yxi_x3 + j12*N2yeta_x3)
N2yx_y3 = (j11_y3*N2yxi + j12_y3*N2yeta + j11*N2yxi_y3 + j12*N2yeta_y3)
N2yx_x4 = (j11_x4*N2yxi + j12_x4*N2yeta + j11*N2yxi_x4 + j12*N2yeta_x4)
N2yx_y4 = (j11_y4*N2yxi + j12_y4*N2yeta + j11*N2yxi_y4 + j12*N2yeta_y4)

N3yx = j11*N3yxi + j12*N3yeta
N3yx_x1 = (j11_x1*N3yxi + j12_x1*N3yeta + j11*N3yxi_x1 + j12*N3yeta_x1)
N3yx_y1 = (j11_y1*N3yxi + j12_y1*N3yeta + j11*N3yxi_y1 + j12*N3yeta_y1)
N3yx_x2 = (j11_x2*N3yxi + j12_x2*N3yeta + j11*N3yxi_x2 + j12*N3yeta_x2)
N3yx_y2 = (j11_y2*N3yxi + j12_y2*N3yeta + j11*N3yxi_y2 + j12*N3yeta_y2)
N3yx_x3 = (j11_x3*N3yxi + j12_x3*N3yeta + j11*N3yxi_x3 + j12*N3yeta_x3)
N3yx_y3 = (j11_y3*N3yxi + j12_y3*N3yeta + j11*N3yxi_y3 + j12*N3yeta_y3)
N3yx_x4 = (j11_x4*N3yxi + j12_x4*N3yeta + j11*N3yxi_x4 + j12*N3yeta_x4)
N3yx_y4 = (j11_y4*N3yxi + j12_y4*N3yeta + j11*N3yxi_y4 + j12*N3yeta_y4)

N4yx = j11*N4yxi + j12*N4yeta
N4yx_x1 = (j11_x1*N4yxi + j12_x1*N4yeta + j11*N4yxi_x1 + j12*N4yeta_x1)
N4yx_y1 = (j11_y1*N4yxi + j12_y1*N4yeta + j11*N4yxi_y1 + j12*N4yeta_y1)
N4yx_x2 = (j11_x2*N4yxi + j12_x2*N4yeta + j11*N4yxi_x2 + j12*N4yeta_x2)
N4yx_y2 = (j11_y2*N4yxi + j12_y2*N4yeta + j11*N4yxi_y2 + j12*N4yeta_y2)
N4yx_x3 = (j11_x3*N4yxi + j12_x3*N4yeta + j11*N4yxi_x3 + j12*N4yeta_x3)
N4yx_y3 = (j11_y3*N4yxi + j12_y3*N4yeta + j11*N4yxi_y3 + j12*N4yeta_y3)
N4yx_x4 = (j11_x4*N4yxi + j12_x4*N4yeta + j11*N4yxi_x4 + j12*N4yeta_x4)
N4yx_y4 = (j11_y4*N4yxi + j12_y4*N4yeta + j11*N4yxi_y4 + j12*N4yeta_y4)

print('N1 =', N1)
print('N2 =', N2)
print('N3 =', N3)
print('N4 =', N4)

print('N1x =', N1x.simplify())
print('N1x_x1 =', N1x_x1.simplify())
print('N1x_y1 =', N1x_y1.simplify())
print('N1x_x2 =', N1x_x2.simplify())
print('N1x_y2 =', N1x_y2.simplify())
print('N1x_x3 =', N1x_x3.simplify())
print('N1x_y3 =', N1x_y3.simplify())
print('N1x_x4 =', N1x_x4.simplify())
print('N1x_y4 =', N1x_y4.simplify())

print('N2x =', N2x.simplify())
print('N2x_x1 =', N2x_x1.simplify())
print('N2x_y1 =', N2x_y1.simplify())
print('N2x_x2 =', N2x_x2.simplify())
print('N2x_y2 =', N2x_y2.simplify())
print('N2x_x3 =', N2x_x3.simplify())
print('N2x_y3 =', N2x_y3.simplify())
print('N2x_x4 =', N2x_x4.simplify())
print('N2x_y4 =', N2x_y4.simplify())

print('N3x =', N3x.simplify())
print('N3x_x1 =', N3x_x1.simplify())
print('N3x_y1 =', N3x_y1.simplify())
print('N3x_x2 =', N3x_x2.simplify())
print('N3x_y2 =', N3x_y2.simplify())
print('N3x_x3 =', N3x_x3.simplify())
print('N3x_y3 =', N3x_y3.simplify())
print('N3x_x4 =', N3x_x4.simplify())
print('N3x_y4 =', N3x_y4.simplify())

print('N4x =', N4x.simplify())
print('N4x_x1 =', N4x_x1.simplify())
print('N4x_y1 =', N4x_y1.simplify())
print('N4x_x2 =', N4x_x2.simplify())
print('N4x_y2 =', N4x_y2.simplify())
print('N4x_x3 =', N4x_x3.simplify())
print('N4x_y3 =', N4x_y3.simplify())
print('N4x_x4 =', N4x_x4.simplify())
print('N4x_y4 =', N4x_y4.simplify())

print('N1y =', N1y.simplify())
print('N1y_x1 =', N1y_x1.simplify())
print('N1y_y1 =', N1y_y1.simplify())
print('N1y_x2 =', N1y_x2.simplify())
print('N1y_y2 =', N1y_y2.simplify())
print('N1y_x3 =', N1y_x3.simplify())
print('N1y_y3 =', N1y_y3.simplify())
print('N1y_x4 =', N1y_x4.simplify())
print('N1y_y4 =', N1y_y4.simplify())

print('N2y =', N2y.simplify())
print('N2y_x1 =', N2y_x1.simplify())
print('N2y_y1 =', N2y_y1.simplify())
print('N2y_x2 =', N2y_x2.simplify())
print('N2y_y2 =', N2y_y2.simplify())
print('N2y_x3 =', N2y_x3.simplify())
print('N2y_y3 =', N2y_y3.simplify())
print('N2y_x4 =', N2y_x4.simplify())
print('N2y_y4 =', N2y_y4.simplify())

print('N3y =', N3y.simplify())
print('N3y_x1 =', N3y_x1.simplify())
print('N3y_y1 =', N3y_y1.simplify())
print('N3y_x2 =', N3y_x2.simplify())
print('N3y_y2 =', N3y_y2.simplify())
print('N3y_x3 =', N3y_x3.simplify())
print('N3y_y3 =', N3y_y3.simplify())
print('N3y_x4 =', N3y_x4.simplify())
print('N3y_y4 =', N3y_y4.simplify())

print('N4y =', N4y.simplify())
print('N4y_x1 =', N4y_x1.simplify())
print('N4y_y1 =', N4y_y1.simplify())
print('N4y_x2 =', N4y_x2.simplify())
print('N4y_y2 =', N4y_y2.simplify())
print('N4y_x3 =', N4y_x3.simplify())
print('N4y_y3 =', N4y_y3.simplify())
print('N4y_x4 =', N4y_x4.simplify())
print('N4y_y4 =', N4y_y4.simplify())
print('')
print('N1xy =', N1xy.simplify())
print('N1xy_x1 =', N1xy_x1.simplify())
print('N1xy_y1 =', N1xy_y1.simplify())
print('N1xy_x2 =', N1xy_x2.simplify())
print('N1xy_y2 =', N1xy_y2.simplify())
print('N1xy_x3 =', N1xy_x3.simplify())
print('N1xy_y3 =', N1xy_y3.simplify())
print('N1xy_x4 =', N1xy_x4.simplify())
print('N1xy_y4 =', N1xy_y4.simplify())

print('N2xy =', N2xy.simplify())
print('N2xy_x1 =', N2xy_x1.simplify())
print('N2xy_y1 =', N2xy_y1.simplify())
print('N2xy_x2 =', N2xy_x2.simplify())
print('N2xy_y2 =', N2xy_y2.simplify())
print('N2xy_x3 =', N2xy_x3.simplify())
print('N2xy_y3 =', N2xy_y3.simplify())
print('N2xy_x4 =', N2xy_x4.simplify())
print('N2xy_y4 =', N2xy_y4.simplify())

print('N3xy =', N3xy.simplify())
print('N3xy_x1 =', N3xy_x1.simplify())
print('N3xy_y1 =', N3xy_y1.simplify())
print('N3xy_x2 =', N3xy_x2.simplify())
print('N3xy_y2 =', N3xy_y2.simplify())
print('N3xy_x3 =', N3xy_x3.simplify())
print('N3xy_y3 =', N3xy_y3.simplify())
print('N3xy_x4 =', N3xy_x4.simplify())
print('N3xy_y4 =', N3xy_y4.simplify())

print('N4xy =', N4xy.simplify())
print('N4xy_x1 =', N4xy_x1.simplify())
print('N4xy_y1 =', N4xy_y1.simplify())
print('N4xy_x2 =', N4xy_x2.simplify())
print('N4xy_y2 =', N4xy_y2.simplify())
print('N4xy_x3 =', N4xy_x3.simplify())
print('N4xy_y3 =', N4xy_y3.simplify())
print('N4xy_x4 =', N4xy_x4.simplify())
print('N4xy_y4 =', N4xy_y4.simplify())

print('')
print('Niyx only for checking purposes')
print('')
print('N1yx =', N1yx.simplify())
print('N2yx =', N2yx.simplify())
print('N3yx =', N3yx.simplify())
print('N4yx =', N4yx.simplify())
print('')

detJfunc = detJ
detJ = sympy.var('detJ')
N1, N2, N3, N4 = sympy.var('N1, N2, N3, N4')
N1x, N2x, N3x, N4x = sympy.var('N1x, N2x, N3x, N4x')
N1x_x1, N2x_x1, N3x_x1, N4x_x1 = sympy.var('N1x_x1, N2x_x1, N3x_x1, N4x_x1')
N1x_y1, N2x_y1, N3x_y1, N4x_y1 = sympy.var('N1x_y1, N2x_y1, N3x_y1, N4x_y1')
N1x_x2, N2x_x2, N3x_x2, N4x_x2 = sympy.var('N1x_x2, N2x_x2, N3x_x2, N4x_x2')
N1x_y2, N2x_y2, N3x_y2, N4x_y2 = sympy.var('N1x_y2, N2x_y2, N3x_y2, N4x_y2')
N1x_x3, N2x_x3, N3x_x3, N4x_x3 = sympy.var('N1x_x3, N2x_x3, N3x_x3, N4x_x3')
N1x_y3, N2x_y3, N3x_y3, N4x_y3 = sympy.var('N1x_y3, N2x_y3, N3x_y3, N4x_y3')
N1x_x4, N2x_x4, N3x_x4, N4x_x4 = sympy.var('N1x_x4, N2x_x4, N3x_x4, N4x_x3')
N1x_y4, N2x_y4, N3x_y4, N4x_y4 = sympy.var('N1x_y4, N2x_y4, N3x_y4, N4x_y3')

N1y, N2y, N3y, N4y = sympy.var('N1y, N2y, N3y, N4y')
N1y_x1, N2y_x1, N3y_x1, N4y_x1 = sympy.var('N1y_x1, N2y_x1, N3y_x1, N4y_x1')
N1y_y1, N2y_y1, N3y_y1, N4y_y1 = sympy.var('N1y_y1, N2y_y1, N3y_y1, N4y_y1')
N1y_x2, N2y_x2, N3y_x2, N4y_x2 = sympy.var('N1y_x2, N2y_x2, N3y_x2, N4y_x2')
N1y_y2, N2y_y2, N3y_y2, N4y_y2 = sympy.var('N1y_y2, N2y_y2, N3y_y2, N4y_y2')
N1y_x3, N2y_x3, N3y_x3, N4y_x3 = sympy.var('N1y_x3, N2y_x3, N3y_x3, N4y_x3')
N1y_y3, N2y_y3, N3y_y3, N4y_y3 = sympy.var('N1y_y3, N2y_y3, N3y_y3, N4y_y3')
N1y_x4, N2y_x4, N3y_x4, N4y_x4 = sympy.var('N1y_x4, N2y_x4, N3y_x4, N4y_x4')
N1y_y4, N2y_y4, N3y_y4, N4y_y4 = sympy.var('N1y_y4, N2y_y4, N3y_y4, N4y_y4')

N1xy, N2xy, N3xy, N4xy = sympy.var('N1xy, N2xy, N3xy, N4xy')
N1xy_x1, N2xy_x1, N3xy_x1, N4xy_x1 = sympy.var('N1xy_x1, N2xy_x1, N3xy_x1, N4xy_x1')
N1xy_y1, N2xy_y1, N3xy_y1, N4xy_y1 = sympy.var('N1xy_y1, N2xy_y1, N3xy_y1, N4xy_y1')
N1xy_x2, N2xy_x2, N3xy_x2, N4xy_x2 = sympy.var('N1xy_x2, N2xy_x2, N3xy_x2, N4xy_x2')
N1xy_y2, N2xy_y2, N3xy_y2, N4xy_y2 = sympy.var('N1xy_y2, N2xy_y2, N3xy_y2, N4xy_y2')
N1xy_x3, N2xy_x3, N3xy_x3, N4xy_x3 = sympy.var('N1xy_x3, N2xy_x3, N3xy_x3, N4xy_x3')
N1xy_y3, N2xy_y3, N3xy_y3, N4xy_y3 = sympy.var('N1xy_y3, N2xy_y3, N3xy_y3, N4xy_y3')
N1xy_x4, N2xy_x4, N3xy_x4, N4xy_x4 = sympy.var('N1xy_x4, N2xy_x4, N3xy_x4, N4xy_x4')
N1xy_y4, N2xy_y4, N3xy_y4, N4xy_y4 = sympy.var('N1xy_y4, N2xy_y4, N3xy_y4, N4xy_y4')

Nu = Matrix(   [[N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0, 0]])
Nv = Matrix(   [[0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0]])
Nw = Matrix(   [[0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0]])
Nrx = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0]])
Nry = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0]])
Nrz = Matrix([[0, 0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4]])

# u v w  rx  ry  rz  (node 1, node2, node3, node4)

#exx = u,x = (dxi/dx)*u,xi + (deta/dx)*u,eta = j11 u,xi + j12 u,eta
BLexx = Matrix([[N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0, 0, 0, 0, N4x, 0, 0, 0, 0, 0]])
BLexx_x1 = Matrix([[N1x_x1, 0, 0, 0, 0, 0, N2x_x1, 0, 0, 0, 0, 0, N3x_x1, 0, 0, 0, 0, 0, N4x_x1, 0, 0, 0, 0, 0]])
BLexx_y1 = Matrix([[N1x_y1, 0, 0, 0, 0, 0, N2x_y1, 0, 0, 0, 0, 0, N3x_y1, 0, 0, 0, 0, 0, N4x_y1, 0, 0, 0, 0, 0]])
BLexx_x2 = Matrix([[N1x_x2, 0, 0, 0, 0, 0, N2x_x2, 0, 0, 0, 0, 0, N3x_x2, 0, 0, 0, 0, 0, N4x_x2, 0, 0, 0, 0, 0]])
BLexx_y2 = Matrix([[N1x_y2, 0, 0, 0, 0, 0, N2x_y2, 0, 0, 0, 0, 0, N3x_y2, 0, 0, 0, 0, 0, N4x_y2, 0, 0, 0, 0, 0]])
BLexx_x3 = Matrix([[N1x_x3, 0, 0, 0, 0, 0, N2x_x3, 0, 0, 0, 0, 0, N3x_x3, 0, 0, 0, 0, 0, N4x_x3, 0, 0, 0, 0, 0]])
BLexx_y3 = Matrix([[N1x_y3, 0, 0, 0, 0, 0, N2x_y3, 0, 0, 0, 0, 0, N3x_y3, 0, 0, 0, 0, 0, N4x_y3, 0, 0, 0, 0, 0]])
BLexx_x4 = Matrix([[N1x_x4, 0, 0, 0, 0, 0, N2x_x4, 0, 0, 0, 0, 0, N3x_x4, 0, 0, 0, 0, 0, N4x_x4, 0, 0, 0, 0, 0]])
BLexx_y4 = Matrix([[N1x_y4, 0, 0, 0, 0, 0, N2x_y4, 0, 0, 0, 0, 0, N3x_y4, 0, 0, 0, 0, 0, N4x_y4, 0, 0, 0, 0, 0]])

#eyy = v,y = (dxi/dy)*v,xi + (deta/dy)*v,eta = j21 v,xi + j22 v,eta
BLeyy = Matrix([[0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0, 0, 0, 0, 0, N4y, 0, 0, 0, 0]])
BLeyy_x1 = Matrix([[0, N1y_x1, 0, 0, 0, 0, 0, N2y_x1, 0, 0, 0, 0, 0, N3y_x1, 0, 0, 0, 0, 0, N4y_x1, 0, 0, 0, 0]])
BLeyy_y1 = Matrix([[0, N1y_y1, 0, 0, 0, 0, 0, N2y_y1, 0, 0, 0, 0, 0, N3y_y1, 0, 0, 0, 0, 0, N4y_y1, 0, 0, 0, 0]])
BLeyy_x2 = Matrix([[0, N1y_x2, 0, 0, 0, 0, 0, N2y_x2, 0, 0, 0, 0, 0, N3y_x2, 0, 0, 0, 0, 0, N4y_x2, 0, 0, 0, 0]])
BLeyy_y2 = Matrix([[0, N1y_y2, 0, 0, 0, 0, 0, N2y_y2, 0, 0, 0, 0, 0, N3y_y2, 0, 0, 0, 0, 0, N4y_y2, 0, 0, 0, 0]])
BLeyy_x3 = Matrix([[0, N1y_x3, 0, 0, 0, 0, 0, N2y_x3, 0, 0, 0, 0, 0, N3y_x3, 0, 0, 0, 0, 0, N4y_x3, 0, 0, 0, 0]])
BLeyy_y3 = Matrix([[0, N1y_y3, 0, 0, 0, 0, 0, N2y_y3, 0, 0, 0, 0, 0, N3y_y3, 0, 0, 0, 0, 0, N4y_y3, 0, 0, 0, 0]])
BLeyy_x4 = Matrix([[0, N1y_x4, 0, 0, 0, 0, 0, N2y_x4, 0, 0, 0, 0, 0, N3y_x4, 0, 0, 0, 0, 0, N4y_x4, 0, 0, 0, 0]])
BLeyy_y4 = Matrix([[0, N1y_y4, 0, 0, 0, 0, 0, N2y_y4, 0, 0, 0, 0, 0, N3y_y4, 0, 0, 0, 0, 0, N4y_y4, 0, 0, 0, 0]])

#gxy = u,y + v,x = (dxi/dy)*u,xi + (deta/dy)*u,eta + (dxi/dx)*v,xi + (deta/dy)*v,eta
BLgxy = Matrix([[N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0, 0, 0, 0, N4y, N4x, 0, 0, 0, 0]])
BLgxy_x1 = Matrix([[N1y_x1, N1x_x1, 0, 0, 0, 0, N2y_x1, N2x_x1, 0, 0, 0, 0, N3y_x1, N3x_x1, 0, 0, 0, 0, N4y_x1, N4x_x1, 0, 0, 0, 0]])
BLgxy_y1 = Matrix([[N1y_y1, N1x_y1, 0, 0, 0, 0, N2y_y1, N2x_y1, 0, 0, 0, 0, N3y_y1, N3x_y1, 0, 0, 0, 0, N4y_y1, N4x_y1, 0, 0, 0, 0]])
BLgxy_x2 = Matrix([[N1y_x2, N1x_x2, 0, 0, 0, 0, N2y_x2, N2x_x2, 0, 0, 0, 0, N3y_x2, N3x_x2, 0, 0, 0, 0, N4y_x2, N4x_x2, 0, 0, 0, 0]])
BLgxy_y2 = Matrix([[N1y_y2, N1x_y2, 0, 0, 0, 0, N2y_y2, N2x_y2, 0, 0, 0, 0, N3y_y2, N3x_y2, 0, 0, 0, 0, N4y_y2, N4x_y2, 0, 0, 0, 0]])
BLgxy_x3 = Matrix([[N1y_x3, N1x_x3, 0, 0, 0, 0, N2y_x3, N2x_x3, 0, 0, 0, 0, N3y_x3, N3x_x3, 0, 0, 0, 0, N4y_x3, N4x_x3, 0, 0, 0, 0]])
BLgxy_y3 = Matrix([[N1y_y3, N1x_y3, 0, 0, 0, 0, N2y_y3, N2x_y3, 0, 0, 0, 0, N3y_y3, N3x_y3, 0, 0, 0, 0, N4y_y3, N4x_y3, 0, 0, 0, 0]])
BLgxy_x4 = Matrix([[N1y_x4, N1x_x4, 0, 0, 0, 0, N2y_x4, N2x_x4, 0, 0, 0, 0, N3y_x4, N3x_x4, 0, 0, 0, 0, N4y_x4, N4x_x4, 0, 0, 0, 0]])
BLgxy_y4 = Matrix([[N1y_y4, N1x_y4, 0, 0, 0, 0, N2y_y4, N2x_y4, 0, 0, 0, 0, N3y_y4, N3x_y4, 0, 0, 0, 0, N4y_y4, N4x_y4, 0, 0, 0, 0]])

#kxx = phix,x = (dxi/dx)*phix,xi + (deta/dx)*phix,eta
#kxx = ry,x = (dxi/dx)*ry,xi + (deta/dx)*ry,eta
BLkxx = Matrix([[0, 0, 0, 0, N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0, 0, 0, 0, N4x, 0]])
BLkxx_x1 = Matrix([[0, 0, 0, 0, N1x_x1, 0, 0, 0, 0, 0, N2x_x1, 0, 0, 0, 0, 0, N3x_x1, 0, 0, 0, 0, 0, N4x_x1, 0]])
BLkxx_y1 = Matrix([[0, 0, 0, 0, N1x_y1, 0, 0, 0, 0, 0, N2x_y1, 0, 0, 0, 0, 0, N3x_y1, 0, 0, 0, 0, 0, N4x_y1, 0]])
BLkxx_x2 = Matrix([[0, 0, 0, 0, N1x_x2, 0, 0, 0, 0, 0, N2x_x2, 0, 0, 0, 0, 0, N3x_x2, 0, 0, 0, 0, 0, N4x_x2, 0]])
BLkxx_y2 = Matrix([[0, 0, 0, 0, N1x_y2, 0, 0, 0, 0, 0, N2x_y2, 0, 0, 0, 0, 0, N3x_y2, 0, 0, 0, 0, 0, N4x_y2, 0]])
BLkxx_x3 = Matrix([[0, 0, 0, 0, N1x_x3, 0, 0, 0, 0, 0, N2x_x3, 0, 0, 0, 0, 0, N3x_x3, 0, 0, 0, 0, 0, N4x_x3, 0]])
BLkxx_y3 = Matrix([[0, 0, 0, 0, N1x_y3, 0, 0, 0, 0, 0, N2x_y3, 0, 0, 0, 0, 0, N3x_y3, 0, 0, 0, 0, 0, N4x_y3, 0]])
BLkxx_x4 = Matrix([[0, 0, 0, 0, N1x_x4, 0, 0, 0, 0, 0, N2x_x4, 0, 0, 0, 0, 0, N3x_x4, 0, 0, 0, 0, 0, N4x_x4, 0]])
BLkxx_y4 = Matrix([[0, 0, 0, 0, N1x_y4, 0, 0, 0, 0, 0, N2x_y4, 0, 0, 0, 0, 0, N3x_y4, 0, 0, 0, 0, 0, N4x_y4, 0]])

#kyy = phiy,y = (dxi/dy)*phiy,xi + (deta/dy)*phiy,eta
#kyy = -rx,y = (dxi/dy)*(-rx),xi + (deta/dy)*(-rx),eta
BLkyy = Matrix([[0, 0, 0, -N1y, 0, 0, 0, 0, 0, -N2y, 0, 0, 0, 0, 0, -N3y, 0, 0, 0, 0, 0, -N4y, 0, 0]])
BLkyy_x1 = Matrix([[0, 0, 0, -N1y_x1, 0, 0, 0, 0, 0, -N2y_x1, 0, 0, 0, 0, 0, -N3y_x1, 0, 0, 0, 0, 0, -N4y_x1, 0, 0]])
BLkyy_y1 = Matrix([[0, 0, 0, -N1y_y1, 0, 0, 0, 0, 0, -N2y_y1, 0, 0, 0, 0, 0, -N3y_y1, 0, 0, 0, 0, 0, -N4y_y1, 0, 0]])
BLkyy_x2 = Matrix([[0, 0, 0, -N1y_x2, 0, 0, 0, 0, 0, -N2y_x2, 0, 0, 0, 0, 0, -N3y_x2, 0, 0, 0, 0, 0, -N4y_x2, 0, 0]])
BLkyy_y2 = Matrix([[0, 0, 0, -N1y_y2, 0, 0, 0, 0, 0, -N2y_y2, 0, 0, 0, 0, 0, -N3y_y2, 0, 0, 0, 0, 0, -N4y_y2, 0, 0]])
BLkyy_x3 = Matrix([[0, 0, 0, -N1y_x3, 0, 0, 0, 0, 0, -N2y_x3, 0, 0, 0, 0, 0, -N3y_x3, 0, 0, 0, 0, 0, -N4y_x3, 0, 0]])
BLkyy_y3 = Matrix([[0, 0, 0, -N1y_y3, 0, 0, 0, 0, 0, -N2y_y3, 0, 0, 0, 0, 0, -N3y_y3, 0, 0, 0, 0, 0, -N4y_y3, 0, 0]])
BLkyy_x4 = Matrix([[0, 0, 0, -N1y_x4, 0, 0, 0, 0, 0, -N2y_x4, 0, 0, 0, 0, 0, -N3y_x4, 0, 0, 0, 0, 0, -N4y_x4, 0, 0]])
BLkyy_y4 = Matrix([[0, 0, 0, -N1y_y4, 0, 0, 0, 0, 0, -N2y_y4, 0, 0, 0, 0, 0, -N3y_y4, 0, 0, 0, 0, 0, -N4y_y4, 0, 0]])

#kxy = phix,y + phiy,x = (dxi/dy)*phix,xi + (deta/dy)*phix,eta
#                       +(dxi/dx)*phiy,xi + (deta/dx)*phiy,eta
#kxy = ry,y + (-rx),x = (dxi/dy)*ry,xi + (deta/dy)*ry,eta
#                       +(dxi/dx)*(-rx),xi + (deta/dx)*(-rx),eta
BLkxy = Matrix([[0, 0, 0, -N1x, N1y, 0, 0, 0, 0, -N2x, N2y, 0, 0, 0, 0, -N3x, N3y, 0, 0, 0, 0, -N4x, N4y, 0]])
BLkxy_x1 = Matrix([[0, 0, 0, -N1x_x1, N1y_x1, 0, 0, 0, 0, -N2x_x1, N2y_x1, 0, 0, 0, 0, -N3x_x1, N3y_x1, 0, 0, 0, 0, -N4x_x1, N4y_x1, 0]])
BLkxy_y1 = Matrix([[0, 0, 0, -N1x_y1, N1y_y1, 0, 0, 0, 0, -N2x_y1, N2y_y1, 0, 0, 0, 0, -N3x_y1, N3y_y1, 0, 0, 0, 0, -N4x_y1, N4y_y1, 0]])
BLkxy_x2 = Matrix([[0, 0, 0, -N1x_x2, N1y_x2, 0, 0, 0, 0, -N2x_x2, N2y_x2, 0, 0, 0, 0, -N3x_x2, N3y_x2, 0, 0, 0, 0, -N4x_x2, N4y_x2, 0]])
BLkxy_y2 = Matrix([[0, 0, 0, -N1x_y2, N1y_y2, 0, 0, 0, 0, -N2x_y2, N2y_y2, 0, 0, 0, 0, -N3x_y2, N3y_y2, 0, 0, 0, 0, -N4x_y2, N4y_y2, 0]])
BLkxy_x3 = Matrix([[0, 0, 0, -N1x_x3, N1y_x3, 0, 0, 0, 0, -N2x_x3, N2y_x3, 0, 0, 0, 0, -N3x_x3, N3y_x3, 0, 0, 0, 0, -N4x_x3, N4y_x3, 0]])
BLkxy_y3 = Matrix([[0, 0, 0, -N1x_y3, N1y_y3, 0, 0, 0, 0, -N2x_y3, N2y_y3, 0, 0, 0, 0, -N3x_y3, N3y_y3, 0, 0, 0, 0, -N4x_y3, N4y_y3, 0]])
BLkxy_x4 = Matrix([[0, 0, 0, -N1x_x4, N1y_x4, 0, 0, 0, 0, -N2x_x4, N2y_x4, 0, 0, 0, 0, -N3x_x4, N3y_x4, 0, 0, 0, 0, -N4x_x4, N4y_x4, 0]])
BLkxy_y4 = Matrix([[0, 0, 0, -N1x_y4, N1y_y4, 0, 0, 0, 0, -N2x_y4, N2y_y4, 0, 0, 0, 0, -N3x_y4, N3y_y4, 0, 0, 0, 0, -N4x_y4, N4y_y4, 0]])

BLgyz = Matrix([[0, 0, N1y, -N1, 0, 0, 0, 0, N2y, -N2, 0, 0, 0, 0, N3y, -N3, 0, 0, 0, 0, N4y, -N4, 0, 0]])
BLgyz_x1 = Matrix([[0, 0, N1y_x1, 0, 0, 0, 0, 0, N2y_x1, 0, 0, 0, 0, 0, N3y_x1, 0, 0, 0, 0, 0, N4y_x1, 0, 0, 0]])
BLgyz_y1 = Matrix([[0, 0, N1y_y1, 0, 0, 0, 0, 0, N2y_y1, 0, 0, 0, 0, 0, N3y_y1, 0, 0, 0, 0, 0, N4y_y1, 0, 0, 0]])
BLgyz_x2 = Matrix([[0, 0, N1y_x2, 0, 0, 0, 0, 0, N2y_x2, 0, 0, 0, 0, 0, N3y_x2, 0, 0, 0, 0, 0, N4y_x2, 0, 0, 0]])
BLgyz_y2 = Matrix([[0, 0, N1y_y2, 0, 0, 0, 0, 0, N2y_y2, 0, 0, 0, 0, 0, N3y_y2, 0, 0, 0, 0, 0, N4y_y2, 0, 0, 0]])
BLgyz_x3 = Matrix([[0, 0, N1y_x3, 0, 0, 0, 0, 0, N2y_x3, 0, 0, 0, 0, 0, N3y_x3, 0, 0, 0, 0, 0, N4y_x3, 0, 0, 0]])
BLgyz_y3 = Matrix([[0, 0, N1y_y3, 0, 0, 0, 0, 0, N2y_y3, 0, 0, 0, 0, 0, N3y_y3, 0, 0, 0, 0, 0, N4y_y3, 0, 0, 0]])
BLgyz_x4 = Matrix([[0, 0, N1y_x4, 0, 0, 0, 0, 0, N2y_x4, 0, 0, 0, 0, 0, N3y_x4, 0, 0, 0, 0, 0, N4y_x4, 0, 0, 0]])
BLgyz_y4 = Matrix([[0, 0, N1y_y4, 0, 0, 0, 0, 0, N2y_y4, 0, 0, 0, 0, 0, N3y_y4, 0, 0, 0, 0, 0, N4y_y4, 0, 0, 0]])

BLgxz = Matrix([[0, 0, N1x, 0, N1, 0, 0, 0, N2x, 0, N2, 0, 0, 0, N3x, 0, N3, 0, 0, 0, N4x, 0, N4, 0]])
BLgxz_x1 = Matrix([[0, 0, N1x_x1, 0, 0, 0, 0, 0, N2x_x1, 0, 0, 0, 0, 0, N3x_x1, 0, 0, 0, 0, 0, N4x_x1, 0, 0, 0]])
BLgxz_y1 = Matrix([[0, 0, N1x_y1, 0, 0, 0, 0, 0, N2x_y1, 0, 0, 0, 0, 0, N3x_y1, 0, 0, 0, 0, 0, N4x_y1, 0, 0, 0]])
BLgxz_x2 = Matrix([[0, 0, N1x_x2, 0, 0, 0, 0, 0, N2x_x2, 0, 0, 0, 0, 0, N3x_x2, 0, 0, 0, 0, 0, N4x_x2, 0, 0, 0]])
BLgxz_y2 = Matrix([[0, 0, N1x_y2, 0, 0, 0, 0, 0, N2x_y2, 0, 0, 0, 0, 0, N3x_y2, 0, 0, 0, 0, 0, N4x_y2, 0, 0, 0]])
BLgxz_x3 = Matrix([[0, 0, N1x_x3, 0, 0, 0, 0, 0, N2x_x3, 0, 0, 0, 0, 0, N3x_x3, 0, 0, 0, 0, 0, N4x_x3, 0, 0, 0]])
BLgxz_y3 = Matrix([[0, 0, N1x_y3, 0, 0, 0, 0, 0, N2x_y3, 0, 0, 0, 0, 0, N3x_y3, 0, 0, 0, 0, 0, N4x_y3, 0, 0, 0]])
BLgxz_x4 = Matrix([[0, 0, N1x_x4, 0, 0, 0, 0, 0, N2x_x4, 0, 0, 0, 0, 0, N3x_x4, 0, 0, 0, 0, 0, N4x_x3, 0, 0, 0]])
BLgxz_y4 = Matrix([[0, 0, N1x_y4, 0, 0, 0, 0, 0, N2x_y4, 0, 0, 0, 0, 0, N3x_y4, 0, 0, 0, 0, 0, N4x_y3, 0, 0, 0]])

#drilling stiffness
#[1] F.M. Adam, A.E. Mohamed, A.E. Hassaballa, Degenerated Four Nodes Shell Element with Drilling Degree of Freedom, IOSR J. Eng. 3 (2013) 10–20. www.iosrjen.org (accessed April 20, 2020).
BLdrilling = Matrix([[N1y/2, -N1x/2, 0, 0, 0, N1, N2y/2, -N2x/2, 0, 0, 0, N2, N3y/2, -N3x/2, 0, 0, 0, N3, N4y/2, -N4x/2, 0, 0, 0, N4]])
BLdrilling_x1 = Matrix([[N1y_x1/2, -N1x_x1/2, 0, 0, 0, 0, N2y_x1/2, -N2x_x1/2, 0, 0, 0, 0, N3y_x1/2, -N3x_x1/2, 0, 0, 0, 0, N4y_x1/2, -N4x_x1/2, 0, 0, 0, 0]])
BLdrilling_y1 = Matrix([[N1y_y1/2, -N1x_y1/2, 0, 0, 0, 0, N2y_y1/2, -N2x_y1/2, 0, 0, 0, 0, N3y_y1/2, -N3x_y1/2, 0, 0, 0, 0, N4y_y1/2, -N4x_y1/2, 0, 0, 0, 0]])
BLdrilling_x2 = Matrix([[N1y_x2/2, -N1x_x2/2, 0, 0, 0, 0, N2y_x2/2, -N2x_x2/2, 0, 0, 0, 0, N3y_x2/2, -N3x_x2/2, 0, 0, 0, 0, N4y_x2/2, -N4x_x2/2, 0, 0, 0, 0]])
BLdrilling_y2 = Matrix([[N1y_y2/2, -N1x_y2/2, 0, 0, 0, 0, N2y_y2/2, -N2x_y2/2, 0, 0, 0, 0, N3y_y2/2, -N3x_y2/2, 0, 0, 0, 0, N4y_y2/2, -N4x_y2/2, 0, 0, 0, 0]])
BLdrilling_x3 = Matrix([[N1y_x3/2, -N1x_x3/2, 0, 0, 0, 0, N2y_x3/2, -N2x_x3/2, 0, 0, 0, 0, N3y_x3/2, -N3x_x3/2, 0, 0, 0, 0, N4y_x3/2, -N4x_x3/2, 0, 0, 0, 0]])
BLdrilling_y3 = Matrix([[N1y_y3/2, -N1x_y3/2, 0, 0, 0, 0, N2y_y3/2, -N2x_y3/2, 0, 0, 0, 0, N3y_y3/2, -N3x_y3/2, 0, 0, 0, 0, N4y_y3/2, -N4x_y3/2, 0, 0, 0, 0]])
BLdrilling_x4 = Matrix([[N1y_x4/2, -N1x_x4/2, 0, 0, 0, 0, N2y_x4/2, -N2x_x4/2, 0, 0, 0, 0, N3y_x4/2, -N3x_x4/2, 0, 0, 0, 0, N4y_x4/2, -N4x_x4/2, 0, 0, 0, 0]])
BLdrilling_y4 = Matrix([[N1y_y4/2, -N1x_y4/2, 0, 0, 0, 0, N2y_y4/2, -N2x_y4/2, 0, 0, 0, 0, N3y_y4/2, -N3x_y4/2, 0, 0, 0, 0, N4y_y4/2, -N4x_y4/2, 0, 0, 0, 0]])

BL = Matrix([BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy, BLgyz, BLgxz])
BL_x1 = Matrix([BLexx_x1, BLeyy_x1, BLgxy_x1, BLkxx_x1, BLkyy_x1, BLkxy_x1, BLgyz_x1, BLgxz_x1])
BL_y1 = Matrix([BLexx_y1, BLeyy_y1, BLgxy_y1, BLkxx_y1, BLkyy_y1, BLkxy_y1, BLgyz_y1, BLgxz_y1])
BL_x2 = Matrix([BLexx_x2, BLeyy_x2, BLgxy_x2, BLkxx_x2, BLkyy_x2, BLkxy_x2, BLgyz_x2, BLgxz_x2])
BL_y2 = Matrix([BLexx_y2, BLeyy_y2, BLgxy_y2, BLkxx_y2, BLkyy_y2, BLkxy_y2, BLgyz_y2, BLgxz_y2])
BL_x3 = Matrix([BLexx_x3, BLeyy_x3, BLgxy_x3, BLkxx_x3, BLkyy_x3, BLkxy_x3, BLgyz_x3, BLgxz_x3])
BL_y3 = Matrix([BLexx_y3, BLeyy_y3, BLgxy_y3, BLkxx_y3, BLkyy_y3, BLkxy_y3, BLgyz_y3, BLgxz_y3])
BL_x4 = Matrix([BLexx_x4, BLeyy_x4, BLgxy_x4, BLkxx_x4, BLkyy_x4, BLkxy_x4, BLgyz_x4, BLgxz_x4])
BL_y4 = Matrix([BLexx_y4, BLeyy_y4, BLgxy_y4, BLkxx_y4, BLkyy_y4, BLkxy_y4, BLgyz_y4, BLgxz_y4])

# hourglass control as per Brockman 1987
# adapted to composites replacing E*h by A11 and E*h**3 by 12*D11
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208
print('gamma1 = N1xy')
print('gamma1_x1 = N1xy_x1')
print('gamma1_y1 = N1xy_y1')
print('gamma1_x2 = N1xy_x2')
print('gamma1_y2 = N1xy_y2')
print('gamma1_x3 = N1xy_x3')
print('gamma1_y3 = N1xy_y3')
print('gamma1_x4 = N1xy_x4')
print('gamma1_y4 = N1xy_y4')

print('gamma2 = N2xy')
print('gamma2_x1 = N2xy_x1')
print('gamma2_y1 = N2xy_y1')
print('gamma2_x2 = N2xy_x2')
print('gamma2_y2 = N2xy_y2')
print('gamma2_x3 = N2xy_x3')
print('gamma2_y3 = N2xy_y3')
print('gamma2_x4 = N2xy_x4')
print('gamma2_y4 = N2xy_y4')

print('gamma3 = N3xy')
print('gamma3_x1 = N3xy_x1')
print('gamma3_y1 = N3xy_y1')
print('gamma3_x2 = N3xy_x2')
print('gamma3_y2 = N3xy_y2')
print('gamma3_x3 = N3xy_x3')
print('gamma3_y3 = N3xy_y3')
print('gamma3_x4 = N3xy_x4')
print('gamma3_y4 = N3xy_y4')

print('gamma4 = N4xy')
print('gamma4_x1 = N4xy_x1')
print('gamma4_y1 = N4xy_y1')
print('gamma4_x2 = N4xy_x2')
print('gamma4_y2 = N4xy_y2')
print('gamma4_x3 = N4xy_x3')
print('gamma4_y3 = N4xy_y3')
print('gamma4_x4 = N4xy_x4')
print('gamma4_y4 = N4xy_y4')

#NOTE ignoring sensitivities of Eu, Erx, Erx, Ery, Erz with respect to the area "A"
Eu = Ev = 0.10*A11/(1 + 1/A)
Erx = 12*0.10*D22/(1 + 1/A)
Ery = 12*0.10*D11/(1 + 1/A)
Ew = (Erx + Ery)/2
Erz = 0
print('Eu =', Eu)
print('Ev =', Ev)
print('Ew =', Ew)
print('Erx =', Erx)
print('Ery =', Ery)
print('Erz =', Erz)
gamma1, gamma2, gamma3, gamma4 = sympy.var('gamma1, gamma2, gamma3, gamma4')
gamma1_x1, gamma2_x1, gamma3_x1, gamma4_x1 = sympy.var('gamma1_x1, gamma2_x1, gamma3_x1, gamma4_x1')
gamma1_y1, gamma2_y1, gamma3_y1, gamma4_y1 = sympy.var('gamma1_y1, gamma2_y1, gamma3_y1, gamma4_y1')
gamma1_x2, gamma2_x2, gamma3_x2, gamma4_x2 = sympy.var('gamma1_x2, gamma2_x2, gamma3_x2, gamma4_x2')
gamma1_y2, gamma2_y2, gamma3_y2, gamma4_y2 = sympy.var('gamma1_y2, gamma2_y2, gamma3_y2, gamma4_y2')
gamma1_x3, gamma2_x3, gamma3_x3, gamma4_x3 = sympy.var('gamma1_x3, gamma2_x3, gamma3_x3, gamma4_x3')
gamma1_y3, gamma2_y3, gamma3_y3, gamma4_y3 = sympy.var('gamma1_y3, gamma2_y3, gamma3_y3, gamma4_y3')
gamma1_x4, gamma2_x4, gamma3_x4, gamma4_x4 = sympy.var('gamma1_x4, gamma2_x4, gamma3_x4, gamma4_x4')
gamma1_y4, gamma2_y4, gamma3_y4, gamma4_y4 = sympy.var('gamma1_y4, gamma2_y4, gamma3_y4, gamma4_y4')

Eu, Ev, Ew, Erx, Ery, Erz = sympy.var('Eu, Ev, Ew, Erx, Ery, Erz')
Bhourglass = Matrix([[gamma1, 0, 0, 0, 0, 0, gamma2, 0, 0, 0, 0, 0, gamma3, 0, 0, 0, 0, 0, gamma4, 0, 0, 0, 0, 0],
                     [0, gamma1, 0, 0, 0, 0, 0, gamma2, 0, 0, 0, 0, 0, gamma3, 0, 0, 0, 0, 0, gamma4, 0, 0, 0, 0],
                     [0, 0, gamma1, 0, 0, 0, 0, 0, gamma2, 0, 0, 0, 0, 0, gamma3, 0, 0, 0, 0, 0, gamma4, 0, 0, 0],
                     [0, 0, 0, gamma1, 0, 0, 0, 0, 0, gamma2, 0, 0, 0, 0, 0, gamma3, 0, 0, 0, 0, 0, gamma4, 0, 0],
                     [0, 0, 0, 0, gamma1, 0, 0, 0, 0, 0, gamma2, 0, 0, 0, 0, 0, gamma3, 0, 0, 0, 0, 0, gamma4, 0],
                     [0, 0, 0, 0, 0, gamma1, 0, 0, 0, 0, 0, gamma2, 0, 0, 0, 0, 0, gamma3, 0, 0, 0, 0, 0, gamma4]])
Bhourglass_x1 = Matrix([[gamma1_x1, 0, 0, 0, 0, 0, gamma2_x1, 0, 0, 0, 0, 0, gamma3_x1, 0, 0, 0, 0, 0, gamma4_x1, 0, 0, 0, 0, 0],
                        [0, gamma1_x1, 0, 0, 0, 0, 0, gamma2_x1, 0, 0, 0, 0, 0, gamma3_x1, 0, 0, 0, 0, 0, gamma4_x1, 0, 0, 0, 0],
                        [0, 0, gamma1_x1, 0, 0, 0, 0, 0, gamma2_x1, 0, 0, 0, 0, 0, gamma3_x1, 0, 0, 0, 0, 0, gamma4_x1, 0, 0, 0],
                        [0, 0, 0, gamma1_x1, 0, 0, 0, 0, 0, gamma2_x1, 0, 0, 0, 0, 0, gamma3_x1, 0, 0, 0, 0, 0, gamma4_x1, 0, 0],
                        [0, 0, 0, 0, gamma1_x1, 0, 0, 0, 0, 0, gamma2_x1, 0, 0, 0, 0, 0, gamma3_x1, 0, 0, 0, 0, 0, gamma4_x1, 0],
                        [0, 0, 0, 0, 0, gamma1_x1, 0, 0, 0, 0, 0, gamma2_x1, 0, 0, 0, 0, 0, gamma3_x1, 0, 0, 0, 0, 0, gamma4_x1]])
Bhourglass_y1 = Matrix([[gamma1_y1, 0, 0, 0, 0, 0, gamma2_y1, 0, 0, 0, 0, 0, gamma3_y1, 0, 0, 0, 0, 0, gamma4_y1, 0, 0, 0, 0, 0],
                        [0, gamma1_y1, 0, 0, 0, 0, 0, gamma2_y1, 0, 0, 0, 0, 0, gamma3_y1, 0, 0, 0, 0, 0, gamma4_y1, 0, 0, 0, 0],
                        [0, 0, gamma1_y1, 0, 0, 0, 0, 0, gamma2_y1, 0, 0, 0, 0, 0, gamma3_y1, 0, 0, 0, 0, 0, gamma4_y1, 0, 0, 0],
                        [0, 0, 0, gamma1_y1, 0, 0, 0, 0, 0, gamma2_y1, 0, 0, 0, 0, 0, gamma3_y1, 0, 0, 0, 0, 0, gamma4_y1, 0, 0],
                        [0, 0, 0, 0, gamma1_y1, 0, 0, 0, 0, 0, gamma2_y1, 0, 0, 0, 0, 0, gamma3_y1, 0, 0, 0, 0, 0, gamma4_y1, 0],
                        [0, 0, 0, 0, 0, gamma1_y1, 0, 0, 0, 0, 0, gamma2_y1, 0, 0, 0, 0, 0, gamma3_y1, 0, 0, 0, 0, 0, gamma4_y1]])
Bhourglass_x2 = Matrix([[gamma1_x2, 0, 0, 0, 0, 0, gamma2_x2, 0, 0, 0, 0, 0, gamma3_x2, 0, 0, 0, 0, 0, gamma4_x2, 0, 0, 0, 0, 0],
                        [0, gamma1_x2, 0, 0, 0, 0, 0, gamma2_x2, 0, 0, 0, 0, 0, gamma3_x2, 0, 0, 0, 0, 0, gamma4_x2, 0, 0, 0, 0],
                        [0, 0, gamma1_x2, 0, 0, 0, 0, 0, gamma2_x2, 0, 0, 0, 0, 0, gamma3_x2, 0, 0, 0, 0, 0, gamma4_x2, 0, 0, 0],
                        [0, 0, 0, gamma1_x2, 0, 0, 0, 0, 0, gamma2_x2, 0, 0, 0, 0, 0, gamma3_x2, 0, 0, 0, 0, 0, gamma4_x2, 0, 0],
                        [0, 0, 0, 0, gamma1_x2, 0, 0, 0, 0, 0, gamma2_x2, 0, 0, 0, 0, 0, gamma3_x2, 0, 0, 0, 0, 0, gamma4_x2, 0],
                        [0, 0, 0, 0, 0, gamma1_x2, 0, 0, 0, 0, 0, gamma2_x2, 0, 0, 0, 0, 0, gamma3_x2, 0, 0, 0, 0, 0, gamma4_x2]])
Bhourglass_y2 = Matrix([[gamma1_y2, 0, 0, 0, 0, 0, gamma2_y2, 0, 0, 0, 0, 0, gamma3_y2, 0, 0, 0, 0, 0, gamma4_y2, 0, 0, 0, 0, 0],
                        [0, gamma1_y2, 0, 0, 0, 0, 0, gamma2_y2, 0, 0, 0, 0, 0, gamma3_y2, 0, 0, 0, 0, 0, gamma4_y2, 0, 0, 0, 0],
                        [0, 0, gamma1_y2, 0, 0, 0, 0, 0, gamma2_y2, 0, 0, 0, 0, 0, gamma3_y2, 0, 0, 0, 0, 0, gamma4_y2, 0, 0, 0],
                        [0, 0, 0, gamma1_y2, 0, 0, 0, 0, 0, gamma2_y2, 0, 0, 0, 0, 0, gamma3_y2, 0, 0, 0, 0, 0, gamma4_y2, 0, 0],
                        [0, 0, 0, 0, gamma1_y2, 0, 0, 0, 0, 0, gamma2_y2, 0, 0, 0, 0, 0, gamma3_y2, 0, 0, 0, 0, 0, gamma4_y2, 0],
                        [0, 0, 0, 0, 0, gamma1_y2, 0, 0, 0, 0, 0, gamma2_y2, 0, 0, 0, 0, 0, gamma3_y2, 0, 0, 0, 0, 0, gamma4_y2]])
Bhourglass_x3 = Matrix([[gamma1_x3, 0, 0, 0, 0, 0, gamma2_x3, 0, 0, 0, 0, 0, gamma3_x3, 0, 0, 0, 0, 0, gamma4_x3, 0, 0, 0, 0, 0],
                        [0, gamma1_x3, 0, 0, 0, 0, 0, gamma2_x3, 0, 0, 0, 0, 0, gamma3_x3, 0, 0, 0, 0, 0, gamma4_x3, 0, 0, 0, 0],
                        [0, 0, gamma1_x3, 0, 0, 0, 0, 0, gamma2_x3, 0, 0, 0, 0, 0, gamma3_x3, 0, 0, 0, 0, 0, gamma4_x3, 0, 0, 0],
                        [0, 0, 0, gamma1_x3, 0, 0, 0, 0, 0, gamma2_x3, 0, 0, 0, 0, 0, gamma3_x3, 0, 0, 0, 0, 0, gamma4_x3, 0, 0],
                        [0, 0, 0, 0, gamma1_x3, 0, 0, 0, 0, 0, gamma2_x3, 0, 0, 0, 0, 0, gamma3_x3, 0, 0, 0, 0, 0, gamma4_x3, 0],
                        [0, 0, 0, 0, 0, gamma1_x3, 0, 0, 0, 0, 0, gamma2_x3, 0, 0, 0, 0, 0, gamma3_x3, 0, 0, 0, 0, 0, gamma4_x3]])
Bhourglass_y3 = Matrix([[gamma1_y3, 0, 0, 0, 0, 0, gamma2_y3, 0, 0, 0, 0, 0, gamma3_y3, 0, 0, 0, 0, 0, gamma4_y3, 0, 0, 0, 0, 0],
                        [0, gamma1_y3, 0, 0, 0, 0, 0, gamma2_y3, 0, 0, 0, 0, 0, gamma3_y3, 0, 0, 0, 0, 0, gamma4_y3, 0, 0, 0, 0],
                        [0, 0, gamma1_y3, 0, 0, 0, 0, 0, gamma2_y3, 0, 0, 0, 0, 0, gamma3_y3, 0, 0, 0, 0, 0, gamma4_y3, 0, 0, 0],
                        [0, 0, 0, gamma1_y3, 0, 0, 0, 0, 0, gamma2_y3, 0, 0, 0, 0, 0, gamma3_y3, 0, 0, 0, 0, 0, gamma4_y3, 0, 0],
                        [0, 0, 0, 0, gamma1_y3, 0, 0, 0, 0, 0, gamma2_y3, 0, 0, 0, 0, 0, gamma3_y3, 0, 0, 0, 0, 0, gamma4_y3, 0],
                        [0, 0, 0, 0, 0, gamma1_y3, 0, 0, 0, 0, 0, gamma2_y3, 0, 0, 0, 0, 0, gamma3_y3, 0, 0, 0, 0, 0, gamma4_y3]])
Bhourglass_x4 = Matrix([[gamma1_x4, 0, 0, 0, 0, 0, gamma2_x4, 0, 0, 0, 0, 0, gamma3_x4, 0, 0, 0, 0, 0, gamma4_x4, 0, 0, 0, 0, 0],
                        [0, gamma1_x4, 0, 0, 0, 0, 0, gamma2_x4, 0, 0, 0, 0, 0, gamma3_x4, 0, 0, 0, 0, 0, gamma4_x4, 0, 0, 0, 0],
                        [0, 0, gamma1_x4, 0, 0, 0, 0, 0, gamma2_x4, 0, 0, 0, 0, 0, gamma3_x4, 0, 0, 0, 0, 0, gamma4_x4, 0, 0, 0],
                        [0, 0, 0, gamma1_x4, 0, 0, 0, 0, 0, gamma2_x4, 0, 0, 0, 0, 0, gamma3_x4, 0, 0, 0, 0, 0, gamma4_x4, 0, 0],
                        [0, 0, 0, 0, gamma1_x4, 0, 0, 0, 0, 0, gamma2_x4, 0, 0, 0, 0, 0, gamma3_x4, 0, 0, 0, 0, 0, gamma4_x4, 0],
                        [0, 0, 0, 0, 0, gamma1_x4, 0, 0, 0, 0, 0, gamma2_x4, 0, 0, 0, 0, 0, gamma3_x4, 0, 0, 0, 0, 0, gamma4_x4]])
Bhourglass_y4 = Matrix([[gamma1_y4, 0, 0, 0, 0, 0, gamma2_y4, 0, 0, 0, 0, 0, gamma3_y4, 0, 0, 0, 0, 0, gamma4_y4, 0, 0, 0, 0, 0],
                        [0, gamma1_y4, 0, 0, 0, 0, 0, gamma2_y4, 0, 0, 0, 0, 0, gamma3_y4, 0, 0, 0, 0, 0, gamma4_y4, 0, 0, 0, 0],
                        [0, 0, gamma1_y4, 0, 0, 0, 0, 0, gamma2_y4, 0, 0, 0, 0, 0, gamma3_y4, 0, 0, 0, 0, 0, gamma4_y4, 0, 0, 0],
                        [0, 0, 0, gamma1_y4, 0, 0, 0, 0, 0, gamma2_y4, 0, 0, 0, 0, 0, gamma3_y4, 0, 0, 0, 0, 0, gamma4_y4, 0, 0],
                        [0, 0, 0, 0, gamma1_y4, 0, 0, 0, 0, 0, gamma2_y4, 0, 0, 0, 0, 0, gamma3_y4, 0, 0, 0, 0, 0, gamma4_y4, 0],
                        [0, 0, 0, 0, 0, gamma1_y4, 0, 0, 0, 0, 0, gamma2_y4, 0, 0, 0, 0, 0, gamma3_y4, 0, 0, 0, 0, 0, gamma4_y4]])

#NOTE assuming Egamma constant with respect to x1, y1, ... x4, y4
Egamma = Matrix([[Eu, 0, 0, 0, 0, 0],
                 [0, Ev, 0, 0, 0, 0],
                 [0, 0, Ew, 0, 0, 0],
                 [0, 0, 0, Erx, 0, 0],
                 [0, 0, 0, 0, Ery, 0],
                 [0, 0, 0, 0, 0, Erz]
                 ])

# Constitutive linear stiffness matrix
Ke = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_x1 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_y1 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_x2 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_y2 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_x3 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_y3 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_x4 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Ke_y4 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

Me = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_x1 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_y1 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_x2 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_y2 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_x3 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_y3 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_x4 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_y4 = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

Me_lumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

sympy.var('wij, offset, intrho, intrhoz2')

#NOTE lumping mass contribution due to rotation about z
Izz1 = simplify(integrate(((xfunc-x1)**2 + (yfunc-y1)**2)*intrho*detJfunc, (xi, -1, 0), (eta, -1, 0)))
Izz1_x1 = diff(Izz1, x1)
Izz1_y1 = diff(Izz1, y1)
Izz1_x2 = diff(Izz1, x2)
Izz1_y2 = diff(Izz1, y2)
Izz1_x3 = diff(Izz1, x3)
Izz1_y3 = diff(Izz1, y3)
Izz1_x4 = diff(Izz1, x4)
Izz1_y4 = diff(Izz1, y4)

Izz2 = simplify(integrate(((xfunc-x2)**2 + (yfunc-y2)**2)*intrho*detJfunc, (xi, 0, +1), (eta, -1, 0)))
Izz2_x1 = diff(Izz2, x1)
Izz2_y1 = diff(Izz2, y1)
Izz2_x2 = diff(Izz2, x2)
Izz2_y2 = diff(Izz2, y2)
Izz2_x3 = diff(Izz2, x3)
Izz2_y3 = diff(Izz2, y3)
Izz2_x4 = diff(Izz2, x4)
Izz2_y4 = diff(Izz2, y4)

Izz3 = simplify(integrate(((xfunc-x3)**2 + (yfunc-y3)**2)*intrho*detJfunc, (xi, 0, +1), (eta, 0, +1)))
Izz3_x1 = diff(Izz3, x1)
Izz3_y1 = diff(Izz3, y1)
Izz3_x2 = diff(Izz3, x2)
Izz3_y2 = diff(Izz3, y2)
Izz3_x3 = diff(Izz3, x3)
Izz3_y3 = diff(Izz3, y3)
Izz3_x4 = diff(Izz3, x4)
Izz3_y4 = diff(Izz3, y4)

Izz4 = simplify(integrate(((xfunc-x4)**2 + (yfunc-y4)**2)*intrho*detJfunc, (xi, -1, 0), (eta, 0, +1)))
Izz4_x1 = diff(Izz4, x1)
Izz4_y1 = diff(Izz4, y1)
Izz4_x2 = diff(Izz4, x2)
Izz4_y2 = diff(Izz4, y2)
Izz4_x3 = diff(Izz4, x3)
Izz4_y3 = diff(Izz4, y3)
Izz4_x4 = diff(Izz4, x4)
Izz4_y4 = diff(Izz4, y4)

ABDE = Matrix(
        [[A11, A12, A16, B11, B12, B16, 0, 0],
         [A12, A22, A26, B12, B22, B26, 0, 0],
         [A16, A26, A66, B16, B26, B66, 0, 0],
         [B11, B12, B16, D11, D12, D16, 0, 0],
         [B12, B22, B26, D12, D22, D26, 0, 0],
         [B16, B26, B66, D16, D26, D66, 0, 0],
         [0, 0, 0, 0, 0, 0, E44, E45],
         [0, 0, 0, 0, 0, 0, E45, E55]])

Ke_x1[:, :] = wij*(
        detJ_x1*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_x1.T*ABDE*BL + Bhourglass_x1.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_x1.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_x1 + Bhourglass.T*Egamma*Bhourglass_x1 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_x1)
        )
Ke_y1[:, :] = wij*(
        detJ_y1*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_y1.T*ABDE*BL + Bhourglass_y1.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_y1.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_y1 + Bhourglass.T*Egamma*Bhourglass_y1 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_y1)
        )
Ke_x2[:, :] = wij*(
        detJ_x2*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_x2.T*ABDE*BL + Bhourglass_x2.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_x2.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_x2 + Bhourglass.T*Egamma*Bhourglass_x2 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_x2)
        )
Ke_y2[:, :] = wij*(
        detJ_y2*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_y2.T*ABDE*BL + Bhourglass_y2.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_y2.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_y2 + Bhourglass.T*Egamma*Bhourglass_y2 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_y2)
        )
Ke_x3[:, :] = wij*(
        detJ_x3*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_x3.T*ABDE*BL + Bhourglass_x3.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_x3.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_x3 + Bhourglass.T*Egamma*Bhourglass_x3 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_x3)
        )
Ke_y3[:, :] = wij*(
        detJ_y3*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_y3.T*ABDE*BL + Bhourglass_y3.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_y3.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_y3 + Bhourglass.T*Egamma*Bhourglass_y3 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_y3)
        )
Ke_x4[:, :] = wij*(
        detJ_x4*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_x4.T*ABDE*BL + Bhourglass_x4.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_x4.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_x4 + Bhourglass.T*Egamma*Bhourglass_x4 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_x4)
        )
Ke_y4[:, :] = wij*(
        detJ_y4*(BL.T*ABDE*BL + Bhourglass.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling.T*BLdrilling)
        +detJ*(BL_y4.T*ABDE*BL + Bhourglass_y4.T*Egamma*Bhourglass + 2*alphat*A12/h*BLdrilling_y4.T*BLdrilling)
        +detJ*(BL.T*ABDE*BL_y4 + Bhourglass.T*Egamma*Bhourglass_y4 + 2*alphat*A12/h*BLdrilling.T*BLdrilling_y4)
        )

Me_x1[:, :] = wij*(
        detJ_x1*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_x1.T*Nu + intrho*Nv_x1.T*Nv + intrho*Nw_x1.T*Nw + intrhoz2*Nrx_x1.T*Nrx + intrhoz2*Nry_x1.T*Nry)
        + detJ*(intrho*Nu.T*Nu_x1 + intrho*Nv.T*Nv_x1 + intrho*Nw.T*Nw_x1 + intrhoz2*Nrx.T*Nrx_x1 + intrhoz2*Nry.T*Nry_x1)
        )
Me_y1[:, :] = wij*(
        detJ_y1*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_y1.T*Nu + intrho*Nv_y1.T*Nv + intrho*Nw_y1.T*Nw + intrhoz2*Nrx_y1.T*Nrx + intrhoz2*Nry_y1.T*Nry)
        + detJ*(intrho*Nu.T*Nu_y1 + intrho*Nv.T*Nv_y1 + intrho*Nw.T*Nw_y1 + intrhoz2*Nrx.T*Nrx_y1 + intrhoz2*Nry.T*Nry_y1)
        )
Me_x2[:, :] = wij*(
        detJ_x2*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_x2.T*Nu + intrho*Nv_x2.T*Nv + intrho*Nw_x2.T*Nw + intrhoz2*Nrx_x2.T*Nrx + intrhoz2*Nry_x2.T*Nry)
        + detJ*(intrho*Nu.T*Nu_x2 + intrho*Nv.T*Nv_x2 + intrho*Nw.T*Nw_x2 + intrhoz2*Nrx.T*Nrx_x2 + intrhoz2*Nry.T*Nry_x2)
        )
Me_y2[:, :] = wij*(
        detJ_y2*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_y2.T*Nu + intrho*Nv_y2.T*Nv + intrho*Nw_y2.T*Nw + intrhoz2*Nrx_y2.T*Nrx + intrhoz2*Nry_y2.T*Nry)
        + detJ*(intrho*Nu.T*Nu_y2 + intrho*Nv.T*Nv_y2 + intrho*Nw.T*Nw_y2 + intrhoz2*Nrx.T*Nrx_y2 + intrhoz2*Nry.T*Nry_y2)
        )
Me_x3[:, :] = wij*(
        detJ_x3*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_x3.T*Nu + intrho*Nv_x3.T*Nv + intrho*Nw_x3.T*Nw + intrhoz2*Nrx_x3.T*Nrx + intrhoz2*Nry_x3.T*Nry)
        + detJ*(intrho*Nu.T*Nu_x3 + intrho*Nv.T*Nv_x3 + intrho*Nw.T*Nw_x3 + intrhoz2*Nrx.T*Nrx_x3 + intrhoz2*Nry.T*Nry_x3)
        )
Me_y3[:, :] = wij*(
        detJ_y3*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_y3.T*Nu + intrho*Nv_y3.T*Nv + intrho*Nw_y3.T*Nw + intrhoz2*Nrx_y3.T*Nrx + intrhoz2*Nry_y3.T*Nry)
        + detJ*(intrho*Nu.T*Nu_y3 + intrho*Nv.T*Nv_y3 + intrho*Nw.T*Nw_y3 + intrhoz2*Nrx.T*Nrx_y3 + intrhoz2*Nry.T*Nry_y3)
        )
Me_x4[:, :] = wij*(
        detJ_x4*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_x4.T*Nu + intrho*Nv_x4.T*Nv + intrho*Nw_x4.T*Nw + intrhoz2*Nrx_x4.T*Nrx + intrhoz2*Nry_x4.T*Nry)
        + detJ*(intrho*Nu.T*Nu_x4 + intrho*Nv.T*Nv_x4 + intrho*Nw.T*Nw_x4 + intrhoz2*Nrx.T*Nrx_x4 + intrhoz2*Nry.T*Nry_x4)
        )
Me_y4[:, :] = wij*(
        detJ_y4*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry)
        + detJ*(intrho*Nu_y4.T*Nu + intrho*Nv_y4.T*Nv + intrho*Nw_y4.T*Nw + intrhoz2*Nrx_y4.T*Nrx + intrhoz2*Nry_y4.T*Nry)
        + detJ*(intrho*Nu.T*Nu_y4 + intrho*Nv.T*Nv_y4 + intrho*Nw.T*Nw_y4 + intrhoz2*Nrx.T*Nrx_y4 + intrhoz2*Nry.T*Nry_y4)
        )
#TODO check this 1/wij factor
Me_x1[5, 5] =   Izz1_x1/wij
Me_y1[5, 5] =   Izz1_y1/wij
Me_x1[11, 11] = Izz2_x1/wij
Me_y1[11, 11] = Izz2_y1/wij
Me_x1[17, 17] = Izz3_x1/wij
Me_y1[17, 17] = Izz3_y1/wij
Me_x1[23, 23] = Izz4_x1/wij
Me_y1[23, 23] = Izz4_y1/wij

Me_x2[5, 5] =   Izz1_x2/wij
Me_y2[5, 5] =   Izz1_y2/wij
Me_x2[11, 11] = Izz2_x2/wij
Me_y2[11, 11] = Izz2_y2/wij
Me_x2[17, 17] = Izz3_x2/wij
Me_y2[17, 17] = Izz3_y2/wij
Me_x2[23, 23] = Izz4_x2/wij
Me_y2[23, 23] = Izz4_y2/wij

Me_x3[5, 5] =   Izz1_x3/wij
Me_y3[5, 5] =   Izz1_y3/wij
Me_x3[11, 11] = Izz2_x3/wij
Me_y3[11, 11] = Izz2_y3/wij
Me_x3[17, 17] = Izz3_x3/wij
Me_y3[17, 17] = Izz3_y3/wij
Me_x3[23, 23] = Izz4_x3/wij
Me_y3[23, 23] = Izz4_y3/wij

Me_x4[5, 5] =   Izz1_x4/wij
Me_y4[5, 5] =   Izz1_y4/wij
Me_x4[11, 11] = Izz2_x4/wij
Me_y4[11, 11] = Izz2_y4/wij
Me_x4[17, 17] = Izz3_x4/wij
Me_y4[17, 17] = Izz3_y4/wij
Me_x4[23, 23] = Izz4_x4/wij
Me_y4[23, 23] = Izz4_y4/wij

calc_lumped = False

if calc_lumped:
    m1 = simplify(integrate(intrho*detJfunc, (xi, -1, 0), (eta, -1, 0)))
    m2 = simplify(integrate(intrho*detJfunc, (xi, 0, +1), (eta, -1, 0)))
    m3 = simplify(integrate(intrho*detJfunc, (xi, 0, +1), (eta, 0, +1)))
    m4 = simplify(integrate(intrho*detJfunc, (xi, -1, 0), (eta, 0, +1)))

    Ixx1 = simplify(integrate((yfunc-y1)**2*intrho*detJfunc, (xi, -1, 0), (eta, -1, 0)))
    Ixx2 = simplify(integrate((yfunc-y2)**2*intrho*detJfunc, (xi, 0, +1), (eta, -1, 0)))
    Ixx3 = simplify(integrate((yfunc-y3)**2*intrho*detJfunc, (xi, 0, +1), (eta, 0, +1)))
    Ixx4 = simplify(integrate((yfunc-y4)**2*intrho*detJfunc, (xi, -1, 0), (eta, 0, +1)))

    Iyy1 = simplify(integrate((xfunc-x1)**2*intrho*detJfunc, (xi, -1, 0), (eta, -1, 0)))
    Iyy2 = simplify(integrate((xfunc-x2)**2*intrho*detJfunc, (xi, 0, +1), (eta, -1, 0)))
    Iyy3 = simplify(integrate((xfunc-x3)**2*intrho*detJfunc, (xi, 0, +1), (eta, 0, +1)))
    Iyy4 = simplify(integrate((xfunc-x4)**2*intrho*detJfunc, (xi, -1, 0), (eta, 0, +1)))

    diag = (
            m1, m1, m1, Ixx1, Iyy1, Izz1,
            m2, m2, m2, Ixx2, Iyy2, Izz2,
            m3, m3, m3, Ixx3, Iyy3, Izz3,
            m4, m4, m4, Ixx4, Iyy4, Izz4
            )

    for i in range(Me.shape[0]):
        Me_lumped[i, i] = diag[i]

# K represents the global stiffness matrix
# see mapy https://github.com/saullocastro/mapy/blob/master/mapy/model/coords.py#L284
sympy.var('cosa, cosb, cosg, sina, sinb, sing')
R2local = Matrix([
           [ cosb*cosg               ,  cosb*sing ,                  -sinb ],
           [-cosa*sing+cosg*sina*sinb,  cosa*cosg+sina*sinb*sing, cosb*sina],
           [ sina*sing+cosa*cosg*sinb, -cosg*sina+cosa*sinb*sing, cosa*cosb]]])
R2global = R2local.T
K = Ke
M = Me
M_lumped = Me_lumped

def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    elif i >= 3*DOF and i < 4*DOF:
        return 'c4'
    else:
        raise

print()
for ind, val in np.ndenumerate(K):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', K[ind])


print()
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])

if calc_lumped:
    print()
    print('M_lumped')
    print()
    for ind, val in np.ndenumerate(M_lumped):
        if val == 0:
            continue
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=',
                M_lumped[ind])

