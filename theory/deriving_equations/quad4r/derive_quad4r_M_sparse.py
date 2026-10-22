r"""
Mass matrix for Quad4 element

    4 ____ 3
     /   /
    /   /   positive normal in CCW
   /___/
   1    2

"""
import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var
from sympy.vector import CoordSys3D, cross


DOF = 6
num_nodes = 4

var('h', positive=True, real=True)
var('x1, y1, x2, y2, x3, y3, x4, y4', real=True, positive=True)
var('rho, xi, eta, A')
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
detJfunc = detJ
print('detJ =', detJ.simplify())
print('xi=0, eta=0')
print('detJ =', detJ.subs(dict(xi=0, eta=0)).simplify())

j = J.inv()
j11 = j[0, 0].simplify()
j12 = j[0, 1].simplify()
j21 = j[1, 0].simplify()
j22 = j[1, 1].simplify()

print('j11 =', j11.simplify())
print('j12 =', j12.simplify())
print('j21 =', j21.simplify())
print('j22 =', j22.simplify())

print('xi=0, eta=0')
print('j11 =', j11.subs(dict(xi=0, eta=0)).simplify())
print('j12 =', j12.subs(dict(xi=0, eta=0)).simplify())
print('j21 =', j21.subs(dict(xi=0, eta=0)).simplify())
print('j22 =', j22.subs(dict(xi=0, eta=0)).simplify())

j11, j12, j21, j22 = var('j11, j12, j21, j22')

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
N2x = j11*N2xi + j12*N2eta
N3x = j11*N3xi + j12*N3eta
N4x = j11*N4xi + j12*N4eta

N1xxi = N1x.diff(xi)
N1xeta = N1x.diff(eta)
N2xxi = N2x.diff(xi)
N2xeta = N2x.diff(eta)
N3xxi = N3x.diff(xi)
N3xeta = N3x.diff(eta)
N4xxi = N4x.diff(xi)
N4xeta = N4x.diff(eta)

N1xy = j21*N1xxi + j22*N1xeta
N2xy = j21*N2xxi + j22*N2xeta
N3xy = j21*N3xxi + j22*N3xeta
N4xy = j21*N4xxi + j22*N4xeta

N1y = j21*N1xi + j22*N1eta
N2y = j21*N2xi + j22*N2eta
N3y = j21*N3xi + j22*N3eta
N4y = j21*N4xi + j22*N4eta

N1yxi = N1y.diff(xi)
N1yeta = N1y.diff(eta)
N2yxi = N2y.diff(xi)
N2yeta = N2y.diff(eta)
N3yxi = N3y.diff(xi)
N3yeta = N3y.diff(eta)
N4yxi = N4y.diff(xi)
N4yeta = N4y.diff(eta)

N1yx = j11*N1yxi + j12*N1yeta
N2yx = j11*N2yxi + j12*N2yeta
N3yx = j11*N3yxi + j12*N3yeta
N4yx = j11*N4yxi + j12*N4yeta

print('N1 =', N1.simplify())
print('N2 =', N2.simplify())
print('N3 =', N3.simplify())
print('N4 =', N4.simplify())

print('N1x =', N1x.simplify())
print('N2x =', N2x.simplify())
print('N3x =', N3x.simplify())
print('N4x =', N4x.simplify())

print('N1y =', N1y.simplify())
print('N2y =', N2y.simplify())
print('N3y =', N3y.simplify())
print('N4y =', N4y.simplify())
print('')
print('N1xy =', N1xy.simplify())
print('N2xy =', N2xy.simplify())
print('N3xy =', N3xy.simplify())
print('N4xy =', N4xy.simplify())
print('')
print('Niyx only for checking purposes')
print('')
print('N1yx =', N1yx.simplify())
print('N2yx =', N2yx.simplify())
print('N3yx =', N3yx.simplify())
print('N4yx =', N4yx.simplify())
print('')
print('')
print('xi=0, eta=0')
print('N1yx =', N1yx.subs(dict(xi=0, eta=0)).simplify())
print('N2yx =', N2yx.subs(dict(xi=0, eta=0)).simplify())
print('N3yx =', N3yx.subs(dict(xi=0, eta=0)).simplify())
print('N4yx =', N4yx.subs(dict(xi=0, eta=0)).simplify())

print('N1 =', N1.subs(dict(xi=0, eta=0)).simplify())
print('N2 =', N2.subs(dict(xi=0, eta=0)).simplify())
print('N3 =', N3.subs(dict(xi=0, eta=0)).simplify())
print('N4 =', N4.subs(dict(xi=0, eta=0)).simplify())

print('N1x =', N1x.subs(dict(xi=0, eta=0)).simplify())
print('N2x =', N2x.subs(dict(xi=0, eta=0)).simplify())
print('N3x =', N3x.subs(dict(xi=0, eta=0)).simplify())
print('N4x =', N4x.subs(dict(xi=0, eta=0)).simplify())

print('N1y =', N1y.subs(dict(xi=0, eta=0)).simplify())
print('N2y =', N2y.subs(dict(xi=0, eta=0)).simplify())
print('N3y =', N3y.subs(dict(xi=0, eta=0)).simplify())
print('N4y =', N4y.subs(dict(xi=0, eta=0)).simplify())
print('')
print('N1xy =', N1xy.subs(dict(xi=0, eta=0)).simplify())
print('N2xy =', N2xy.subs(dict(xi=0, eta=0)).simplify())
print('N3xy =', N3xy.subs(dict(xi=0, eta=0)).simplify())
print('N4xy =', N4xy.subs(dict(xi=0, eta=0)).simplify())
print('')

detJ = var('detJ')
N1, N2, N3, N4 = var('N1, N2, N3, N4')
N1x, N2x, N3x, N4x = var('N1x, N2x, N3x, N4x')
N1y, N2y, N3y, N4y = var('N1y, N2y, N3y, N4y')

Nu =  Matrix([[N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0]])
Nw =  Matrix([[0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0]])
Nrx = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0]])
Nry = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0]])
Nrz = Matrix([[0, 0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4]])

var('wij, intrho, intrhoz, intrhoz2')

#NOTE reduced integration of stiffness to remove shear locking
#subs(xi=0, eta=0) in many places above was used

#NOTE intrho and intrhoz2 already consider offset
# https://saullocastro.github.io/composites/#composites.core.Laminate
#intrho     integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z) dz`,
#           used in equivalent single layer finite element mass
#           matrices
#intrhoz    integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z dz`,
#           used in equivalent single layer finite element mass
#           matrices
#intrhoz2   integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z^2 dz`,
#           used in equivalent single layer finite element mass
#           matrices
#
# For drilling contribution in the mass matrix, see Eq. 3.34 in https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/nv935321z
alpha=0 # drilling penalty factor for mass matrix, I found that only alpha=0 works
#
# Fully integrated mass matrix according to Brockman 1987
# See Eq. 20 for kinetic energy, with R1=intrho, R2=intrhoz, R3=intrhoz2, as per Eq. 18
Me = wij*detJ*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw
    + intrhoz*Nu.T*Nry + intrhoz*Nry.T*Nu
    - intrhoz*Nv.T*Nrx - intrhoz*Nrx.T*Nv
    + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry + intrho*Nrz.T*Nrz)

# Mass matrix by a mix of reduced integration and projection method by Brockman 1987 Eq. 21
# as suggested by Brockman 1987, only the out-of-plane and rotations rx and ry
# are projected
# whereas u, v and here rz will be calculated using single-point quadrature

# Brockman 1987, Eq. 8a
x = Matrix([[x1, x2, x3, x4]]).T
# Brockman 1987, Eq. 8b
y = Matrix([[y1, y2, y3, y4]]).T

# Brockman 1987, Eq. 5a
s = Matrix([[1, 1, 1, 1]]).T

N = Matrix([[N1, N2, N3, N4]]).T
print()
print('H matrix exactly integrated')
# Brockman 1987, Eq. 19
H = wij*detJ*N*N.T
for i in range(4):
    for j in range(4):
        print('h%d%d +=' % (i+1, j+1), H[i, j])
print()
print('verifying symmetry of H matrix')
print((H[1, 0] - H[0, 1]).expand())
print((H[2, 0] - H[0, 2]).expand())
print((H[2, 1] - H[1, 2]).expand())
print((H[3, 0] - H[0, 3]).expand())
print((H[3, 1] - H[1, 3]).expand())
print((H[3, 2] - H[2, 3]).expand())
print()
var('h11, h12, h13, h14, h22, h23, h24, h33, h34, h44')
H = Matrix([[h11, h12, h13, h14],
            [h12, h22, h23, h24],
            [h13, h23, h33, h34],
            [h14, h24, h34, h44]])

# Brockman 1987, Eq. 22, assumed a constant Jacobian determinant
#NOTE distorted elements do not have a constant Jacobian determinant
#H = A/36.*Matrix([[4, 2, 1, 2],
                  #[2, 4, 2, 1],
                  #[1, 2, 4, 2],
                  #[2, 1, 2, 4]])
#h11 = A/36.*4
#h12 = A/36.*2
#h13 = A/36.*1
#h14 = A/36.*2
#h22 = A/36.*4
#h23 = A/36.*2
#h24 = A/36.*1
#h33 = A/36.*4
#h34 = A/36.*2
#h44 = A/36.*4

# Brockman 1987, Eq. 28a-c
cxx = simplify(x.T*H*x)
cyy = simplify(y.T*H*y)
cxy = simplify(x.T*H*y)
assert cxx.shape == (1, 1)
assert cyy.shape == (1, 1)
assert cxy.shape == (1, 1)

print('cxx =', cxx[0, 0])
print('cyy =', cyy[0, 0])
print('cxy =', cxy[0, 0])
print()

cxx, cyy, cxy = var('cxx, cyy, cxy')

# Brockman 1987, Eq. 6a and 6b
#NOTE both definitions below produce the same result, if Nx and Ny are evaluated at xi=eta=0
#b1 = 1/(2.*A)*Matrix([[y2-y4, y3-y1, y4-y2, y1-y3]]).T
#b2 = 1/(2.*A)*Matrix([[x4-x2, x1-x3, x2-x4, x3-x1]]).T
b1 = Matrix([[N1x, N2x, N3x, N4x]]).T
b2 = Matrix([[N1y, N2y, N3y, N4y]]).T
print('b1, b2 at center point xi=eta=0')

# Brockman 1987, Eq. 31, H matrix with 1-point reduced integration
valH1 = A/16.
print('valH1 =', valH1)

valH1 = var('valH1')
H1 = sympy.ones(4, 4)*valH1

# Consistent mass matrix assuming constant Jacobian determinant by Brockman 1987 Eq. 21 using H per Eq. 22
#NOTE distorted elements do not have a constant Jacobian determinant
Z = sympy.zeros(4, 4)
# Me_cons is originally in the order u1,u2,u3,u4,v1,v2,v3,v4, ... , rz1,rz2,rz3,rz4
Me_cons = Matrix([
    [ intrho*H,          Z,        Z,          Z,  intrhoz*H,        Z],
    [        Z,   intrho*H,        Z, -intrhoz*H,          Z,        Z],
    [        Z,          Z, intrho*H,          Z,          Z,        Z],
    [        Z, -intrhoz*H,        Z, intrhoz2*H,          Z,        Z],
    [intrhoz*H,          Z,        Z,          Z, intrhoz2*H,        Z],
    [        Z,          Z,        Z,          Z,          Z, alpha*intrho*H]])
# Making Me_cons in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_cons, dtype=object)
order = [0, 4, 8, 12, 16, 20,   1, 5, 9, 13, 17, 21,    2, 6, 10, 14, 18, 22,   3, 7, 11, 15, 19, 23]
tmp = tmp[order]
tmp = tmp[:, order]
Me_cons = Matrix(tmp)

# Mass matrix by reduced integration by Brockman 1987 Eq. 21 using H1 per Eq. 31
# Me_red is originally in the order u1,u2,u3,u4,v1,v2,v3,v4, ... , rz1,rz2,rz3,rz4
Me_red = Matrix([
    [ intrho*H1,           Z,         Z,           Z,  intrhoz*H1,         Z],
    [         Z,   intrho*H1,         Z, -intrhoz*H1,           Z,         Z],
    [         Z,           Z, intrho*H1,           Z,           Z,         Z],
    [         Z, -intrhoz*H1,         Z, intrhoz2*H1,           Z,         Z],
    [intrhoz*H1,           Z,         Z,           Z, intrhoz2*H1,         Z],
    [         Z,           Z,         Z,           Z,           Z, alpha*intrho*H1]])
# Making Me_red in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_red, dtype=object)
tmp = tmp[order]
tmp = tmp[:, order]
Me_red = Matrix(tmp)

# Lumped mass matrix using Lobatto integration, where integration points are placed at nodes
# Brockman 1987 as reference, and A. Ralston, A First Course in Nurnericul Analysis, McGraw-Hill. Ncw York, 196
# Forcing intrhoz=0 to end up with a diagonal matrix
#NOTE Using H = A/4. only works for regular meshes
# Me_lump is originally in the order u1,u2,u3,u4,v1,v2,v3,v4, ... , rz1,rz2,rz3,rz4
Me_lump = Matrix([
    [ intrho*H,          Z,        Z,          Z, 0*intrhoz*H,        Z],
    [        Z,   intrho*H,        Z, -0*intrhoz*H,          Z,        Z],
    [        Z,          Z, intrho*H,          Z,          Z,        Z],
    [        Z, -0*intrhoz*H,        Z, intrhoz2*H,          Z,        Z],
    [0*intrhoz*H,          Z,        Z,          Z, intrhoz2*H,        Z],
    [        Z,          Z,        Z,          Z,          Z, alpha*intrho*H]])
# Making Me_lump in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_lump, dtype=object)
tmp = tmp[order]
tmp = tmp[:, order]
Me_lump = Matrix(tmp)

# Projection approach by Brockman 1987, Eq. 30
print('Htilde')
#NOTE
# - zeroing cxx, cyy, cxy makes this equal to the reduced integration approach
Htilde = valH1*s*s.T + cxx*b1*b1.T + cyy*b2*b2.T + cxy*(b1*b2.T + b2*b1.T)
print()
print('checking symmetry of Htilde', sympy.expand(Htilde.T - Htilde))
print()
for i in range(4):
    for j in range(4):
        print('h%d%d =' % (i+1, j+1), Htilde[i, j])

h11, h12, h13, h14, h22, h23, h24, h33, h34, h44 = var('h11, h12, h13, h14, h22, h23, h24, h33, h34, h44')
Htilde = Matrix([[h11, h12, h13, h14],
                 [h12, h22, h23, h24],
                 [h13, h23, h33, h34],
                 [h14, h24, h34, h44]])

# Me_brock is originally in the order u1,u2,u3,u4,v1,v2,v3,v4, ... , rz1,rz2,rz3,rz4
Me_brock = Matrix([
    [     intrho*H1,               Z,              Z,               Z,  intrhoz*Htilde,         Z],
    [             Z,       intrho*H1,              Z, -intrhoz*Htilde,               Z,         Z],
    [             Z,               Z,  intrho*Htilde,               Z,               Z,         Z],
    [             Z, -intrhoz*Htilde,              Z, intrhoz2*Htilde,               Z,         Z],
    [intrhoz*Htilde,               Z,              Z,               Z, intrhoz2*Htilde,         Z],
    [             Z,               Z,              Z,               Z,               Z, alpha*intrho*Htilde]])
# Making Me_brock in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_brock, dtype=object)
tmp = tmp[order]
tmp = tmp[:, order]
Me_brock = Matrix(tmp)

# Mass matrix by the projection method by Brockman 1987 Eq. 21 using Htilde per Eq. 30
# Me_proj is originally in the order u1,u2,u3,u4,v1,v2,v3,v4, ... , rz1,rz2,rz3,rz4
Me_proj = Matrix([
    [ intrho*Htilde,               Z,              Z,               Z,  intrhoz*Htilde,         Z],
    [             Z,   intrho*Htilde,              Z, -intrhoz*Htilde,               Z,         Z],
    [             Z,               Z,  intrho*Htilde,               Z,               Z,         Z],
    [             Z, -intrhoz*Htilde,              Z, intrhoz2*Htilde,               Z,         Z],
    [intrhoz*Htilde,               Z,              Z,               Z, intrhoz2*Htilde,         Z],
    [             Z,               Z,              Z,               Z,               Z, alpha*intrho*Htilde]])
# Making Me_proj in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_proj, dtype=object)
tmp = tmp[order]
tmp = tmp[:, order]
Me_proj = Matrix(tmp)


if False:
    # to illustrate the reordering procedure
    Z = sympy.zeros(2, 2)
    test = Matrix([
        [1*sympy.ones(2, 2), Z, Z, Z, Z, Z],
        [Z, 2*sympy.ones(2, 2), Z, Z, Z, Z],
        [Z, Z, 3*sympy.ones(2, 2), Z, Z, Z],
        [Z, Z, Z, 4*sympy.ones(2, 2), Z, Z],
        [Z, Z, Z, Z, 5*sympy.ones(2, 2), Z],
        [Z, Z, Z, Z, Z, 6*sympy.ones(2, 2)]])
    test2 = np.array(test, dtype=object)
    order_test = [0, 2, 4, 6, 8, 10,   1, 3, 5, 7, 9, 11]
    test2 = test2[order_test]
    test2 = test2[:, order_test]


calc_lumped = False

if calc_lumped:
    # Lumped mass matrix
    Me_lumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

    analytical = True
    if analytical:
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

        #NOTE lumping mass contribution due to rotation about z
        Izz1 = simplify(integrate(((xfunc-x1)**2 + (yfunc-y1)**2)*intrho*detJfunc, (xi, -1, 0), (eta, -1, 0)))
        Izz2 = simplify(integrate(((xfunc-x2)**2 + (yfunc-y2)**2)*intrho*detJfunc, (xi, 0, +1), (eta, -1, 0)))
        Izz3 = simplify(integrate(((xfunc-x3)**2 + (yfunc-y3)**2)*intrho*detJfunc, (xi, 0, +1), (eta, 0, +1)))
        Izz4 = simplify(integrate(((xfunc-x4)**2 + (yfunc-y4)**2)*intrho*detJfunc, (xi, -1, 0), (eta, 0, +1)))

        diag = (
                m1, m1, m1, Ixx1, Iyy1, Izz1,
                m2, m2, m2, Ixx2, Iyy2, Izz2,
                m3, m3, m3, Ixx3, Iyy3, Izz3,
                m4, m4, m4, Ixx4, Iyy4, Izz4
                )

        for i in range(Me.shape[0]):
            Me_lumped[i, i] = diag[i]

print()
print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

M = R*Me*R.T
M_brock = R*Me_brock*R.T
M_cons = R*Me_cons*R.T
M_proj = R*Me_proj*R.T
M_red = R*Me_red*R.T
M_lump = R*Me_lump*R.T

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
print()
print('_______________________________________')
print('')
print('printing code for sparse implementation')
print('_______________________________________')
print()
print('consistent mass matrix according to Saullo G. P. Castro, arbitrary integration points')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M[ind])
print()
print()
print()
print('consistent mass matrix according to Brockman')
print('M_cons')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_cons):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M_cons):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_cons[ind])
print()
print()
print()
print('fully produced by reduced integration mass matrix')
print('M_red')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_red):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M_red):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_red[ind])
print()
print()
print()
print('lumped mass matrix using Lobatto method, integration points at four nodes')
print('M_lump')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_lump):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
for ind, val in np.ndenumerate(M_lump):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_lump[ind])
print()
print()
print()
print('mass matrix with reduced integration for u,v,rz + projected w,rx,ry, recommended approach by Brockman 1987')
print('M_brock')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_brock):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M_brock):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_brock[ind])
print()
print()
print('fully projected mass matrix')
print('M_proj')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_proj):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M_proj):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_proj[ind])
print()
print()
