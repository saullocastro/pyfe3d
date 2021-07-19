"""
Constitutive linear stiffness matrix for BFS cylinder
"""
import numpy as np
import sympy
from sympy import var, symbols, simplify, integrate, Matrix
from sympy.vector import CoordSys3D, cross

r"""

    4 ____ 3
     /   /
    /   /   positive normal in CCW
   /___/
   1    2

"""

num_nodes = 4
DOF = 6

var('xi, eta, lex, ley, rho, weight')
var('R')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')

#ley calculated from nodal positions and radius

ONE = sympy.Integer(1)

# shape functions
# - from Reference:
#     OCHOA, O. O.; REDDY, J. N. Finite Element Analysis of Composite Laminates. Dordrecht: Springer, 1992.
# bi-linear
Li = lambda xii, etai: ONE/4.*(1 + xi*xii)*(1 + eta*etai)
# cubic
Hw_i = lambda xii, etai: ONE/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
Hwx_i = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)
Hwy_i = lambda xii, etai: -ley/32.*(xi + xii)**2*(xi*xii - 2)*etai*(eta + etai)**2*(eta*etai - 1)
Hwxy_i = lambda xii, etai: lex*ley/64.*xii*(xi + xii)**2*(xi*xii - 1)*etai*(eta + etai)**2*(eta*etai - 1)

# node 1 (-1, -1)
# node 2 (+1, -1)
# node 3 (+1, +1)
# node 4 (-1, +1)

Nu = sympy.Matrix([[
   #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
    Li(-1, -1), 0, 0, 0, 0, 0,
    Li(+1, -1), 0, 0, 0, 0, 0,
    Li(+1, +1), 0, 0, 0, 0, 0,
    Li(-1, +1), 0, 0, 0, 0, 0,
    ]])
Nv = sympy.Matrix([[
   #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
    0, Li(-1, -1), 0, 0, 0, 0,
    0, Li(+1, -1), 0, 0, 0, 0,
    0, Li(+1, +1), 0, 0, 0, 0,
    0, Li(-1, +1), 0, 0, 0, 0,
    ]])
Nw = sympy.Matrix([[
   #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
    0, 0, Hw_i(-1, -1), Hwx_i(-1, -1), Hwy_i(-1, -1), Hwxy_i(-1, -1),
    0, 0, Hw_i(+1, -1), Hwx_i(+1, -1), Hwy_i(+1, -1), Hwxy_i(+1, -1),
    0, 0, Hw_i(+1, +1), Hwx_i(+1, +1), Hwy_i(+1, +1), Hwxy_i(+1, +1),
    0, 0, Hw_i(-1, +1), Hwx_i(-1, +1), Hwy_i(-1, +1), Hwxy_i(-1, +1),
    ]])
A = sympy.Matrix([
    [A11, A12, A16],
    [A12, A22, A26],
    [A16, A26, A66]])
B = sympy.Matrix([
    [B11, B12, B16],
    [B12, B22, B26],
    [B16, B26, B66]])
D = sympy.Matrix([
    [D11, D12, D16],
    [D12, D22, D26],
    [D16, D26, D66]])

# membrane
Nu_x = (2/lex)*Nu.diff(xi)
Nu_y = (2/ley)*Nu.diff(eta)
Nv_x = (2/lex)*Nv.diff(xi)
Nv_y = (2/ley)*Nv.diff(eta)

Bm = sympy.Matrix([
    Nu_x, # epsilon_xx
    Nv_y + 1/R*Nw, # epsilon_yy
    Nu_y + Nv_x # gamma_xy
    ])

Bms = []
for i in range(Bm.shape[0]):
    Bmis = []
    for j in range(Bm.shape[1]):
        Bmij = Bm[i, j]
        if Bmij != 0:
            print('                Bm%d_%02d = %s' % ((i+1), (j+1),
                str(simplify(Bmij))))
            Bmis.append(symbols('Bm%d_%02d' % (i+1, j+1)))
        else:
            Bmis.append(0)
    Bms.append(Bmis)
Bm = sympy.Matrix(Bms)

# bending
Nphix = -(2/lex)*Nw.diff(xi)
Nphiy = -(2/ley)*Nw.diff(eta)
Nphix_x = (2/lex)*Nphix.diff(xi)
Nphix_y = (2/ley)*Nphix.diff(eta)
Nphiy_x = (2/lex)*Nphiy.diff(xi)
Nphiy_y = (2/ley)*Nphiy.diff(eta)
Bb = sympy.Matrix([
    Nphix_x,
    Nphiy_y,
    Nphix_y + Nphiy_x
    ])

Bbs = []
for i in range(Bb.shape[0]):
    Bbis = []
    for j in range(Bb.shape[1]):
        Bbij = Bb[i, j]
        if Bbij != 0:
            print('                Bb%d_%02d = %s' % ((i+1), (j+1),
                str(simplify(Bbij))))
            Bbis.append(symbols('Bb%d_%02d' % (i+1, j+1)))
        else:
            Bbis.append(0)
    Bbs.append(Bbis)
Bb = sympy.Matrix(Bbs)

print()
print()
print()

# Constitutive linear stiffness matrix
KC0e = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
KC0e[:, :] = weight*(lex*ley)/4.*(Bm.T*A*Bm + Bm.T*B*Bb + Bb.T*B*Bm + Bb.T*D*Bb)

# KC0 represents the global constitutive linear stiffness matrix
# in case we want to apply coordinate transformations in the future
KC0 = KC0e

def name_ind(i):
    if i >=0 and i < DOF:
        return 'c1'
    elif i >= DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    elif i >= 3*DOF and i < 4*DOF:
        return 'c4'
    else:
        raise

print('printing code for sparse implementation')
for ind, val in np.ndenumerate(KC0):
    if val == 0:
        continue
    print('                k += 1')
    print('                KC0v[k] +=', KC0[ind])

print()
print()
print()
KC0_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KC0):
    if val == 0:
        continue
    KC0_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        KC0r[k] = %d+%s' % (i%DOF, si))
    print('        KC0c[k] = %d+%s' % (j%DOF, sj))
print('KC0_SPARSE_SIZE', KC0_SPARSE_SIZE)

