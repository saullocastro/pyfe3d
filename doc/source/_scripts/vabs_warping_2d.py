r"""Warping of a thin-walled channel under torsion and bending

Run by the plot directive of cross_sectional_analysis.rst at every build of
the documentation; the page shows the figure only.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pyfe3d.vabs import CrossSection, isotropic

# channel: web of height h along x3, flanges of width b along x2,
# thickness t, meshed with bilinear quadrilaterals of side t/2
h, b, t = 1., 0.5, 0.05
d = t/2
xs = np.linspace(0., b, round(b/d) + 1)
ys = np.linspace(-h/2, h/2, round(h/d) + 1)
X2, X3 = np.meshgrid(xs, ys)
nx = xs.size
elements = []
for j in range(ys.size - 1):
    for i in range(nx - 1):
        xc, yc = (xs[i] + xs[i + 1])/2, (ys[j] + ys[j + 1])/2
        if xc < t or abs(yc) > h/2 - t:
            n = j*nx + i
            elements.append([n, n + 1, n + 1 + nx, n + nx])
used = np.unique(elements)
new = -np.ones(X2.size, dtype=int)
new[used] = np.arange(used.size)
x2, x3 = X2.ravel()[used], X3.ravel()[used]
elements = new[np.array(elements)]

sec = CrossSection(x2, x3, elements, materials=[isotropic(70e9, 0.3)],
                   mat_ids=[0]*len(elements))
sec.calc_stiffness()
tri = Triangulation(sec.x2, sec.x3, sec.triangles())
# origin at the shear centre, same nodes and triangles
sc = sec.translated(*sec.shear_centre)
sc.calc_stiffness()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
w = sc.warping([0., 1., 0., 0.])  # unit twist rate kappa1
cs = ax1.tricontourf(tri, w[:, 0], levels=21, cmap='RdBu_r')
fig.colorbar(cs, ax=ax1, label=r'$w_1 = -\omega$ per unit $\kappa_1$')
ax1.plot(*sec.tension_centre, 'ko', mfc='none')
ax1.plot(*sec.shear_centre, 'kx')
ax1.set_title('torsion, about the shear centre')

w = sec.warping([0., 0., 1., 0.])  # unit curvature kappa2
ax2.triplot(tri, color='0.8', lw=0.5)
ax2.triplot(sec.x2 + w[:, 1], sec.x3 + w[:, 2], sec.triangles(),
            color='C0', lw=0.5)
ax2.set_title('bending, in-plane warping')
for ax in (ax1, ax2):
    ax.set_aspect('equal')
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
fig.tight_layout()
