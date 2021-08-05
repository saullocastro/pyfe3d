import sympy
from sympy import cos, sin, Matrix

sympy.var('xi, xj, xk, yi, yj, yk, zi, zj, zk', real=True)
sympy.var('r11, r12, r13', real=True)
sympy.var('r21, r22, r23', real=True)
sympy.var('r31, r32, r33', real=True)

x_old = Matrix([[xi, xj, xk]]).T
y_old = Matrix([[yi, yj, yk]]).T
z_old = Matrix([[zi, zj, zk]]).T
x_global = Matrix([[1, 0, 0]]).T
y_global = Matrix([[0, 1, 0]]).T
z_global = Matrix([[0, 0, 1]]).T

R2global = Matrix([
   [r11, r12, r13],
   [r21, r22, r23],
   [r31, r32, r33]])

tmp1 = -x_global + R2global*x_old
tmp2 = -y_global + R2global*y_old
tmp3 = -z_global + R2global*z_old

eqs = (tmp1[0], tmp1[1], tmp1[2],
       tmp2[0], tmp2[1], tmp2[2],
       tmp3[0], tmp3[1], tmp3[2])

res = sympy.solve(eqs, r11, r12, r13, r21, r22, r23, r31, r32, r33)
for k, v in res.items():
    print('%s = %s' % (str(k), str(v)))
