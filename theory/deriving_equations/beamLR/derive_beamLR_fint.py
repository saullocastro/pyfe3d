import numpy as np
import sympy
from sympy import simplify, Matrix, var, symbols

r"""

   ^ y axis
   |
   |
   ______   --> x axis
   1    2
   Reference 2D plane for all derivations

   Timoshenko 3D beam element with consistent shape functions from:
   Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

"""

DOF = 6
num_nodes = 2

var('xi', real=True)
var('L, E, Iyy, Izz, Iyz, G12, G13, GJ, A, Ay, Az', real=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
# NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
# NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy

S1 = (1-xi)/2
S2 = (1+xi)/2

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
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

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

#From Eqs. 8 and 9 in Luo, Y. 2008
#exx = u,x + (-rz,x)*y + (ry,x)*z
#exy = (v.diff(x) - rz) - (rx)*z
#exz = (w.diff(x) + ry) + (rx)*y
#BL = Matrix([
    #Nu.diff(x) + (-Nrz.diff(x))*y + Nry.diff(x)*z,
    #(Nv.diff(x) - Nrz) - Nrx*z,
    #(Nw.diff(x) + Nry) + Nrx*y,
    #])
#dy = dz = 0
#BL = integrate(BL, (y, -hy/2+dy, +hy/2+dy))
#BL = simplify(integrate(BL, (z, -hz/2+dz, +hz/2+dz)))

#From Eqs. 11 and 12 in Luo, Y. 2008
# p = D rho
# p = [N Mz My Qy Qz Mx]
# NOTE Mz = int(sigma_xx*y) and My = int(sigma_xx*z), Eq. 11 of Luo, Y. 2008,
#      whose Eq. 12 lists My before Mz
# p = [e ky kz gammay gammaz kx]
# NOTE two transverse shear moduli, as the attributes of
#      pyfe3d.beamprop.BeamProp: G12 for gammay of the plane xy (v, rz) and
#      G13 for gammaz of the plane xz (w, ry), and the torsional stiffness
#      GJ given directly; the shear block of D is checked in
#      derive_beamLR_KC0_sparse.py
D11 = Matrix([
    [ E*A, E*Ay, E*Az, 0, 0, 0],
    [E*Ay, E*Iy,  E*Iyz, 0, 0, 0],
    [E*Az,  E*Iyz, E*Iz, 0, 0, 0],
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
#From Eq. 8 in Luo, Y. 2008
#epsilon = u,x
#kappay = -theta,x = -rz,x
#kappaz = psi,x = ry,x
#gammay = v,x - theta = v,x - rz
#gammaz = w,x + psi = w,x + ry
#kappax = phi,x
# putting in a BL matrix
BL = Matrix([
    2/L*Nu.diff(xi),
    -2/L*Nrz.diff(xi),
    2/L*Nry.diff(xi),
    2/L*Nv.diff(xi) - Nrz,
    2/L*Nw.diff(xi) + Nry,
    2/L*Nrx.diff(xi)])

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, BL.shape[1])])

# NOTE one integration point at the centre, xi = 0, with weight 2, such
#      that KC0e = L*BL0.T*D*BL0 with BL0 = BL at xi = 0: exact for the
#      axial, bending and torsion strains, constant along the element, and
#      reduced for the transverse shear strains, linear in xi through the
#      rotations, which avoids shear locking
BL0 = BL.subs(xi, 0)
KC0e = L*BL0.T*D*BL0

print('checking that KC0e reduces to that of the 11 attributes', flush=True)
KC0e11 = L*BL0.T*D11*BL0
assert sympy.expand(KC0e.subs(C_FROM_11) - KC0e11) == sympy.zeros(num_nodes*DOF, num_nodes*DOF)
print('    OK', flush=True)

nonzero = set()
for ind, val in np.ndenumerate(KC0e):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    if i > j:
        continue # NOTE ignoring symmetric part
    name = 'KC0e%02d%02d' % (i, j)
    nonzero.add(name)
    print('%s = %s' % (name, simplify(val)))

rows = []
for i in range(num_nodes*DOF):
    cols = []
    for j in range(num_nodes*DOF):
        if j >= i:
            name = 'KC0e%02d%02d' % (i, j)
        else:
            name = 'KC0e%02d%02d' % (j, i)
        if name in nonzero:
            cols.append(var(name))
        else:
            cols.append(0)
    rows.append(cols)
KC0e = Matrix(rows)

finte = KC0e*ue

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

nonzero = set()
for ind, val in np.ndenumerate(finte):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    name = 'finte[%d]' % (i)
    nonzero.add(name)
    print('%s = %s' % (name, simplify(val)))

rows = []
for i in range(num_nodes*DOF):
    name = 'finte[%d]' % (i)
    if name in nonzero:
        rows.append(var(name))
    else:
        rows.append(0)
finte = Matrix(rows)
fint = R*finte


def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'

        raise

print()
print()
print('_______________________________________')
print()
print('printing fint')
print('_______________________________________')
print()
print()
for ind, val in np.ndenumerate(fint):
    i, j = ind
    si = name_ind(i)
    if sympy.expand(val) == 0:
        continue
    print('            fint[%d+%s] +=' % (i%DOF, si), val)
print()
print()
