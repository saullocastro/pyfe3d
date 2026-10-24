import numpy as np
import sympy
from sympy import cos, sin, Matrix

sympy.var('A11mat, A12mat, A16mat, A22mat, A26mat, A66mat', real=True)
sympy.var('B11mat, B12mat, B16mat, B22mat, B26mat, B66mat', real=True)
sympy.var('D11mat, D12mat, D16mat, D22mat, D26mat, D66mat', real=True)
sympy.var('m11, m12', real=True)
sympy.var('m21, m22', real=True)
sympy.var('theta', real=True)


R112 = Matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 2]])
T = Matrix([
   [cos(theta)**2, sin(theta)**2, 2*sin(theta)*cos(theta)],
   [sin(theta)**2, cos(theta)**2, -2*sin(theta)*cos(theta)],
   [-sin(theta)*cos(theta), sin(theta)*cos(theta), cos(theta)**2 - sin(theta)**2]])
Tinv = Matrix([
   [cos(theta)**2, sin(theta)**2, -2*sin(theta)*cos(theta)],
   [sin(theta)**2, cos(theta)**2, 2*sin(theta)*cos(theta)],
   [sin(theta)*cos(theta), -sin(theta)*cos(theta), cos(theta)**2 - sin(theta)**2]])

#NOTE from https://scicomp.stackexchange.com/questions/35600/4th-order-tensor-rotation-sources-to-refer
Mstress = Matrix([
    [m11**2, m12**2, 2*m11*m12],
    [m21**2, m22**2, 2*m21*m22],
    [m11*m21, m12*m22, m11*m22+m12*m21]])
Nstrain = Matrix([
    [m11**2, m12**2, m11*m12],
    [m21**2, m22**2, m21*m22],
    [2*m11*m21, m12*m22, m11*m22+m12*m21]])

A = Matrix([
   [A11mat, A12mat, A16mat],
   [A12mat, A22mat, A26mat],
   [A16mat, A26mat, A66mat]])
B = Matrix([
   [B11mat, B12mat, B16mat],
   [B12mat, B22mat, B26mat],
   [B16mat, B26mat, B66mat]])
D = Matrix([
   [D11mat, D12mat, D16mat],
   [D12mat, D22mat, D26mat],
   [D16mat, D26mat, D66mat]])

Anew = Mstress*A*Mstress.T
for ind, val in np.ndenumerate(Anew):
    ind_num = {0:1, 1:2, 2:6}
    print('        A%d%d = %s' % (ind_num[ind[0]], ind_num[ind[1]], str(val)))

print()
Bnew = Mstress*B*Mstress.T
for ind, val in np.ndenumerate(Bnew):
    print('        B%d%d = %s' % (ind_num[ind[0]], ind_num[ind[1]], str(val)))

print()
Dnew = Mstress*D*Mstress.T
for ind, val in np.ndenumerate(Dnew):
    print('        D%d%d = %s' % (ind_num[ind[0]], ind_num[ind[1]], str(val)))
