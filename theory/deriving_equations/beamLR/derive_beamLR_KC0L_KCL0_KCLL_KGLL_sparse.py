"""
Nonlinear stiffness matrix KCNL, K_CNL + K_GNL, for the BeamLR element

KCNL = KC0L + KCL0 + KCLL + KGLL, built from the von Kármán axial strain

    exx = u,x + eNL,    eNL = (v,x**2 + w,x**2)/2

with the slopes v,x and w,x of the linear interpolations of v and w, as in
derive_beamLR_KG_sparse.py; up to pyfe3d 0.11 they were taken from the
rotations, v,x = rz and w,x = -ry, which neglects the shear deformation

- KC0L = BL.T*D*BNL
- KCL0 = BNL.T*D*BL = KC0L.T
- KCLL = BNL.T*D*BNL
- KGLL = NNL*(Sv_x.T*Sv_x + Sw_x.T*Sw_x), with NNL = C00*eNL

where BNL = d(eNL)/d(ue) = v,x*Sv_x + w,x*Sw_x acts on the axial strain only,
such that eNL = BNL*ue/2 and not BNL*ue.

KGLL is the geometric stiffness carried by the axial force of the nonlinear
axial strain, called KGNL in pyfe3d/beamlr.pyx. update_KG builds the geometric
stiffness from the linear axial force N = E*(A*u,x + Ay*ky + Az*kz) alone, which
keeps KG homogeneous of degree one in ue, as a linear buckling analysis needs.
The missing piece is collected in KCNL instead, such that

    KT = KC0 + KCNL(ue) + KG(ue)

is the exact Hessian of the strain energy, i.e. the exact Jacobian of fint,
which is asserted below.

"""
import numpy as np
import sympy
from sympy import simplify, expand, Matrix, var, symbols, Poly

DOF = 6
NUM_NODES = 2
# NOTE number of Gauss-Legendre points used in beamlr.pyx: the integrands are
#      constant along the element and one point at the center is exact
NUM_GAUSS_POINTS = 1

var('xi', real=True)
var('L, E, Iyy, Izz, Iyz, G12, G13, GJ, A, Ay, Az', real=True)

# NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
# NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy

S1 = (1-xi)/2
S2 = (1+xi)/2

#              u, v, w, rx, ry, rz (for each node)
# linear interpolation for all field variables
Nu =  Matrix([[S1, 0, 0, 0, 0, 0,
               S2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, S1, 0, 0, 0, 0,
               0, S2, 0, 0, 0, 0]])
Nw =  Matrix([[0, 0, S1, 0, 0, 0,
               0, 0, S2, 0, 0, 0]])
Nrx = Matrix([[0, 0, 0, S1, 0, 0,
               0, 0, 0, S2, 0, 0]])
Nry = Matrix([[0, 0, 0, 0, S1, 0,
               0, 0, 0, 0, S2, 0]])
Nrz = Matrix([[0, 0, 0, 0, 0, S1,
               0, 0, 0, 0, 0, S2]])

# From Eqs. 12 in Luo, Y. 2008
# NOTE two transverse shear moduli, as the attributes of
#      pyfe3d.beamprop.BeamProp: G12 for gammay of the plane xy (v, rz) and
#      G13 for gammaz of the plane xz (w, ry), and the torsional stiffness
#      GJ given directly; the shear block of D is checked in
#      derive_beamLR_KC0_sparse.py
D11 = Matrix([
    [ E*A,  E*Ay,  E*Az, 0, 0, 0],
    [E*Ay,  E*Iy, E*Iyz, 0, 0, 0],
    [E*Az, E*Iyz,  E*Iz, 0, 0, 0],
    [   0,     0,     0,   G12*A,      0, -G12*Az],
    [   0,     0,     0,       0,  G13*A,  G13*Ay],
    [   0,     0,     0, -G12*Az, G13*Ay,      GJ]])

# NOTE sectional stiffness matrix C of pyfe3d.beamprop.BeamProp, symmetric,
#      with the 21 independent entries Cij, i <= j, in the order of the
#      strains {eps, gammay, gammaz, kx, ky, kz} of pyfe3d, see
#      pyfe3d.vabs.yu_to_pyfe3d. D is C in the order {exx, ky, kz, gammay,
#      gammaz, kx} of the rows of BL below
C = Matrix(6, 6, lambda i, j: var('C%d%d' % (min(i, j), max(i, j)), real=True))
PERM = [0, 4, 5, 1, 2, 3]
D = C.extract(PERM, PERM)
# NOTE without the attribute C, BeamProp builds C from its 11 stiffness
#      attributes, such that D becomes D11 above
C_FROM_11 = {D[i, j]: D11[i, j] for i in range(6) for j in range(i, 6)}
assert D.subs(C_FROM_11) == D11

# From Eq. 8 in Luo, Y. 2008
# {exx, ky, kz, gammay, gammaz, kx}
BLexx = 2/L*Nu.diff(xi)
BLky = -2/L*Nrz.diff(xi)
BLkz = 2/L*Nry.diff(xi)
BL = Matrix([
    BLexx,
    BLky,
    BLkz,
    2/L*Nv.diff(xi) - Nrz,
    2/L*Nw.diff(xi) + Nry,
    2/L*Nrx.diff(xi)])

# slopes of the interpolations of v and w
Sv_x = 2/L*Nv.diff(xi)
Sw_x = 2/L*Nw.diff(xi)

ue = Matrix([symbols(r'ue[%d]' % i) for i in range(NUM_NODES*DOF)])
v_x = (Sv_x*ue)[0]
w_x = (Sw_x*ue)[0]

# nonlinear part of the axial strain and its variation
eNL = (v_x**2 + w_x**2)/2
BNL = Matrix([[eNL.diff(ui) for ui in ue]])
assert expand(BNL - (v_x*Sv_x + w_x*Sw_x)) == sympy.zeros(1, NUM_NODES*DOF)
# NOTE eNL is quadratic in ue, such that BNL*ue counts it twice
assert expand((BNL*ue)[0]/2 - eNL) == 0
# BNL acts on the axial strain only
BNLfull = Matrix([BNL, sympy.zeros(5, NUM_NODES*DOF)])
epsNL = Matrix([eNL, 0, 0, 0, 0, 0])
NNL = (D*epsNL)[0]

# integrands of KCNL in element coordinates
KC0Le = BL.T*D*BNLfull
KCL0e = BNLfull.T*D*BL
KCLLe = BNLfull.T*D*BNLfull
KGLLe = NNL*(Sv_x.T*Sv_x + Sw_x.T*Sw_x)
KCNLe = KC0Le + KCL0e + KCLLe + KGLLe

# NOTE the identity below holds at every xi, in particular at the centre,
#      xi = 0, where BeamLR evaluates the strain energy with its one-point
#      integration, see derive_beamLR_KC0_sparse.py
print('checking that KC0 + KCNL + KG is the Hessian of the strain energy', flush=True)
N = (D*BL*ue)[0]
KC0e = BL.T*D*BL
KGe = N*(Sv_x.T*Sv_x + Sw_x.T*Sw_x)
eps = BL*ue + epsNL
U = (eps.T*D*eps)[0]/2
grad = [U.diff(ui) for ui in ue]
for i in range(NUM_NODES*DOF):
    for j in range(i, NUM_NODES*DOF):
        assert expand(grad[i].diff(ue[j]) - (KC0e[i, j] + KCNLe[i, j] + KGe[i, j])) == 0
print('    OK', flush=True)

print('checking that %d Gauss-Legendre points integrate KCNL exactly' % NUM_GAUSS_POINTS, flush=True)
max_degree = 0
for val in KCNLe:
    val = expand(val)
    if val == 0:
        continue
    max_degree = max(max_degree, Poly(val, xi).degree())
print('    polynomial degree in xi:', max_degree)
assert max_degree <= 2*NUM_GAUSS_POINTS - 1
print('    OK', flush=True)

print()
print()
print('_______________________________________')
print()
print('printing code for _update_probe_BL_G')
print('_______________________________________')
print()
print()
BLgammay = 2/L*Nv.diff(xi) - Nrz
BLgammaz = 2/L*Nw.diff(xi) + Nry
BLkx = 2/L*Nrx.diff(xi)
for name, row in [('BLexx', BLexx), ('BLky', BLky), ('BLkz', BLkz),
                  ('BLgammay', BLgammay), ('BLgammaz', BLgammaz),
                  ('BLkx', BLkx), ('Sv_x', Sv_x), ('Sw_x', Sw_x)]:
    for i, val in enumerate(row):
        if val == 0:
            continue
        print('        %s[%d] = %s' % (name, i, simplify(val)))
    print()

print()
print('_______________________________________')
print()
print('printing code for _update_probe_KCNLve')
print('_______________________________________')
print()
print()
# row of D*BL giving the axial force, N = EBL*ue
EBL = (D*BL)[0, :]
assert expand(EBL - (C00*BLexx + C04*BLky + C05*BLkz + C01*BLgammay
                     + C02*BLgammaz + C03*BLkx)) == sympy.zeros(1, NUM_NODES*DOF)
# NOTE without the attribute C of BeamProp, EBL = E*(A*BLexx + Ay*BLky + Az*BLkz)
assert expand(EBL.subs(C_FROM_11) - E*(A*BLexx + Ay*BLky + Az*BLkz)) == sympy.zeros(1, NUM_NODES*DOF)
print('            # EBL = C00*BLexx + C04*BLky + C05*BLkz + C01*BLgammay + C02*BLgammaz + C03*BLkx')
print('            # BNL = v_x*Sv_x + w_x*Sw_x')
print('            # NNL =', NNL.subs(eNL, var('eNL')), 'with eNL = (v_x**2 + w_x**2)/2')
print()

# same integrand in terms of the probe rows, as implemented in beamlr.pyx
var('v_x, w_x, NNL, weight')
def row(name, vals):
    return Matrix([[symbols('%s[%d]' % (name, i)) if vals[i] != 0 else 0
                    for i in range(NUM_NODES*DOF)]])
sBLexx = row('BLexx', BLexx)
sBLky = row('BLky', BLky)
sBLkz = row('BLkz', BLkz)
sSv_x = row('Sv_x', Sv_x)
sSw_x = row('Sw_x', Sw_x)
sBLgammay = row('BLgammay', BLgammay)
sBLgammaz = row('BLgammaz', BLgammaz)
sBLkx = row('BLkx', BLkx)
sEBL = (C00*sBLexx + C04*sBLky + C05*sBLkz + C01*sBLgammay + C02*sBLgammaz
        + C03*sBLkx)
sBNL = v_x*sSv_x + w_x*sSw_x
sKCNLe = weight*(sEBL.T*sBNL + sBNL.T*sEBL + C00*sBNL.T*sBNL
                 + NNL*(sSv_x.T*sSv_x + sSw_x.T*sSw_x))
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
#     beamlr.pyx
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
