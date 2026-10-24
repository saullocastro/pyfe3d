"""
Nonlinear constitutive stiffness matrix KCNL for the Quad4R element

KCNL = KC0L + KCL0 + KCLL + KGLL, built from the von Karman membrane strains

    exx = u,x + w,x**2/2
    eyy = v,y + w,y**2/2
    gxy = u,y + v,x + w,x*w,y

whose nonlinear part is epsNL = {w,x**2/2, w,y**2/2, w,x*w,y}:

- KC0L = Bm.T*A*BmL + Bb.T*B*BmL
- KCL0 = BmL.T*A*Bm + BmL.T*B*Bb = KC0L.T
- KCLL = BmL.T*A*BmL
- KGLL = G.T*[NNL]*G, with {NNL} = A*epsNL

where BmL = d(epsNL)/d(ue), such that epsNL = BmL*ue/2 and not BmL*ue, and
G = {w,x, w,y} is the gradient of w.

KGLL is the geometric stiffness carried by the stress of the nonlinear membrane
strain, called KGNL in pyfe3d/quad4r.pyx. update_KG builds the geometric
stiffness from the stress of the linear strains, A*Bm*ue + B*Bb*ue, alone, which
keeps KG homogeneous of degree one in ue, as a linear buckling analysis needs.
The missing piece is collected in KCNL instead, such that

    KT = KC0 + KCNL(ue) + KG(ue)

is the exact Hessian of the strain energy, i.e. the exact Jacobian of fint,
which is asserted below. The terms of KC0 that do not interact with the
membrane strains, transverse shear and drilling, are left out of this check.

All terms are given at one integration point, in element coordinates, with the
same two-point Gauss-Legendre quadrature of update_KG.

"""
import numpy as np
import sympy
from sympy import expand, Matrix, var, symbols

DOF = 6
NUM_NODES = 4

var('wij, detJ')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
N1x, N2x, N3x, N4x = var('N1x, N2x, N3x, N4x')
N1y, N2y, N3y, N4y = var('N1y, N2y, N3y, N4y')

A = Matrix([
    [A11, A12, A16],
    [A12, A22, A26],
    [A16, A26, A66]])
B = Matrix([
    [B11, B12, B16],
    [B12, B22, B26],
    [B16, B26, B66]])
D = Matrix([
    [D11, D12, D16],
    [D12, D22, D26],
    [D16, D26, D66]])

# u v w  rx  ry  rz  (node 1, node2, node3, node4)

#exx = u,x
BLexx = Matrix([[N1x, 0, 0, 0, 0, 0,
                 N2x, 0, 0, 0, 0, 0,
                 N3x, 0, 0, 0, 0, 0,
                 N4x, 0, 0, 0, 0, 0]])
#eyy = v,y
BLeyy = Matrix([[0, N1y, 0, 0, 0, 0,
                 0, N2y, 0, 0, 0, 0,
                 0, N3y, 0, 0, 0, 0,
                 0, N4y, 0, 0, 0, 0]])
#gxy = u,y + v,x
BLgxy = Matrix([[N1y, N1x, 0, 0, 0, 0,
                 N2y, N2x, 0, 0, 0, 0,
                 N3y, N3x, 0, 0, 0, 0,
                 N4y, N4x, 0, 0, 0, 0]])
#kxx = ry,x
BLkxx = Matrix([[0, 0, 0, 0, N1x, 0,
                 0, 0, 0, 0, N2x, 0,
                 0, 0, 0, 0, N3x, 0,
                 0, 0, 0, 0, N4x, 0]])
#kyy = -rx,y
BLkyy = Matrix([[0, 0, 0, -N1y, 0, 0,
                 0, 0, 0, -N2y, 0, 0,
                 0, 0, 0, -N3y, 0, 0,
                 0, 0, 0, -N4y, 0, 0]])
#kxy = ry,y - rx,x
BLkxy = Matrix([[0, 0, 0, -N1x, N1y, 0,
                 0, 0, 0, -N2x, N2y, 0,
                 0, 0, 0, -N3x, N3y, 0,
                 0, 0, 0, -N4x, N4y, 0]])
#w,x
Gwx = Matrix([[0, 0, N1x, 0, 0, 0,
               0, 0, N2x, 0, 0, 0,
               0, 0, N3x, 0, 0, 0,
               0, 0, N4x, 0, 0, 0]])
#w,y
Gwy = Matrix([[0, 0, N1y, 0, 0, 0,
               0, 0, N2y, 0, 0, 0,
               0, 0, N3y, 0, 0, 0,
               0, 0, N4y, 0, 0, 0]])

# membrane
Bm = Matrix([BLexx, BLeyy, BLgxy])
# bending
Bb = Matrix([BLkxx, BLkyy, BLkxy])
# gradient of w (see Eq. A.10, for Donnell's equations, in https://www.sciencedirect.com/science/article/pii/S0263822314003602)
G = Matrix([Gwx, Gwy])

ue = Matrix([symbols(r'ue[%d]' % i) for i in range(NUM_NODES*DOF)])
w_x = (Gwx*ue)[0]
w_y = (Gwy*ue)[0]

# nonlinear part of the membrane strains and its variation
epsNL = Matrix([w_x**2/2, w_y**2/2, w_x*w_y])
BmL = epsNL.jacobian(ue)
assert expand(BmL - Matrix([w_x*Gwx, w_y*Gwy, w_x*Gwy + w_y*Gwx])) == sympy.zeros(3, NUM_NODES*DOF)
# NOTE epsNL is quadratic in ue, such that BmL*ue counts it twice
assert expand(BmL*ue/2 - epsNL) == sympy.zeros(3, 1)
NNL = A*epsNL
NNLmatrix = Matrix([[NNL[0], NNL[2]],
                    [NNL[2], NNL[1]]])

# KCNL in element coordinates at one integration point
KC0Le = Bm.T*A*BmL + Bb.T*B*BmL
KCL0e = BmL.T*A*Bm + BmL.T*B*Bb
KCLLe = BmL.T*A*BmL
KGLLe = G.T*NNLmatrix*G
KCNLe = wij*detJ*(KC0Le + KCL0e + KCLLe + KGLLe)

print('checking that KC0 + KCNL + KG is the Hessian of the strain energy', flush=True)
eps = Bm*ue + epsNL
kappa = Bb*ue
U = wij*detJ*((eps.T*A*eps)[0]/2 + (eps.T*B*kappa)[0] + (kappa.T*D*kappa)[0]/2)
KC0e = wij*detJ*(Bm.T*A*Bm + Bm.T*B*Bb + Bb.T*B*Bm + Bb.T*D*Bb)
N = A*Bm*ue + B*Bb*ue
Nmatrix = Matrix([[N[0], N[2]],
                  [N[2], N[1]]])
KGe = wij*detJ*G.T*Nmatrix*G
grad = [U.diff(ui) for ui in ue]
for i in range(NUM_NODES*DOF):
    for j in range(i, NUM_NODES*DOF):
        assert expand(grad[i].diff(ue[j]) - (KC0e[i, j] + KCNLe[i, j] + KGe[i, j])) == 0
print('    OK', flush=True)

print()
print()
print('_______________________________________')
print()
print('printing code for _update_probe_KCNLve')
print('_______________________________________')
print()
print()
# same terms in terms of w_x, w_y and NNL, as implemented in quad4r.pyx
var('w_x, w_y')
print('                # NNL = A*{w_x**2/2, w_y**2/2, w_x*w_y}')
for i, val in enumerate(A*Matrix([w_x**2/2, w_y**2/2, w_x*w_y])):
    print('                NNL[%d] = %s' % (i, val))
print()

sNNL = Matrix(symbols('NNL[0], NNL[1], NNL[2]'))
sBmL = Matrix([w_x*Gwx, w_y*Gwy, w_x*Gwy + w_y*Gwx])
sNNLmatrix = Matrix([[sNNL[0], sNNL[2]],
                     [sNNL[2], sNNL[1]]])
sKCNLe = wij*detJ*(Bm.T*A*sBmL + Bb.T*B*sBmL
                   + sBmL.T*A*Bm + sBmL.T*B*Bb
                   + sBmL.T*A*sBmL
                   + G.T*sNNLmatrix*G)
for ind, val in np.ndenumerate(sKCNLe):
    if val == 0:
        continue
    i, j = ind
    print('                KCNLve[%d] += %s' % (NUM_NODES*DOF*i + j, val))

print()
print()
print('_______________________________________')
print()
print('printing code for sparse implementation')
print('_______________________________________')
print()
print()
# NOTE KCNL = R*KCNLe*R.T, with R made of 3x3 blocks for the translations
#     and the rotations of each node, such that any non-zero term fills its
#     whole 3x3 block in global coordinates
nonzero = np.array([[expand(KCNLe[i, j]) != 0
                     for j in range(NUM_NODES*DOF)]
                    for i in range(NUM_NODES*DOF)])
blocks = nonzero.reshape(2*NUM_NODES, 3, 2*NUM_NODES, 3).any(axis=(1, 3))
nonzero = np.kron(blocks, np.ones((3, 3), dtype=bool))

def name_ind(i):
    node = i//DOF
    if node >= 0 and node < NUM_NODES:
        return 'c%d' % (node + 1)
    else:
        raise

# NOTE printing only the non-zero terms, such that KCNL_SPARSE_SIZE can be
#     smaller than the (NUM_NODES*DOF)**2 terms stored by update_KCNL in
#     quad4r.pyx
KCNL_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(nonzero):
    if not val:
        continue
    KCNL_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            KCNLr[k] = %d+%s' % (i%DOF, si))
    print('            KCNLc[k] = %d+%s' % (j%DOF, sj))
print('KCNL_SPARSE_SIZE', KCNL_SPARSE_SIZE)
