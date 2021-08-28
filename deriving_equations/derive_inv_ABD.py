import numpy as np
import sympy
from sympy import cos, sin, Matrix

sympy.var('A11, A12, A16, A22, A26, A66', real=True)
sympy.var('B11, B12, B16, B22, B26, B66', real=True)
sympy.var('D11, D12, D16, D22, D26, D66', real=True)
ABD = sympy.zeros(6, 6)
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
ABD[0:3, 0:3] = A
ABD[0:3, 3:6] = B
ABD[3:6, 0:3] = B
ABD[3:6, 3:6] = D

print(A.inv())
