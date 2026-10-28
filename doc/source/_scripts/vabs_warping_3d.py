r"""Interactive 3D view of the torsion of a thin-walled channel

Shown, up to the marked line, in cross_sectional_analysis.rst; the last
lines write the figure for the documentation and are run by conf.py at the
start of every build.
"""
import numpy as np
import plotly.graph_objects as go

from pyfe3d.vabs import CrossSection, isotropic

# channel: web of height h along x3, flanges of width b along x2, thickness t
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
w1 = sec.warping([0., 1., 0., 0.])[:, 0]  # warping per unit twist rate
tri = sec.triangles()

# beam of length L under the uniform twist rate k1: the sections rotate by
# k1*x1 about the reference axis and warp by k1*w1 along x1, exaggerated
# by scale
L, k1, scale = 3., np.pi/(3*3.), 3.
stations = np.linspace(0., L, 31)
# NOTE edges used by one triangle only are on the boundary of the section
edges, counts = np.unique(np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]],
                                             tri[:, [2, 0]])), axis=1),
                          axis=0, return_counts=True)
edges = edges[counts == 1]
nn = x2.size
X, Y, Z, C = [], [], [], []
for s in stations:
    c, si = np.cos(k1*s), np.sin(k1*s)
    X.append(s + scale*k1*w1)
    Y.append(c*x2 - si*x3)
    Z.append(si*x2 + c*x3)
    C.append(w1)
X, Y, Z, C = map(np.concatenate, (X, Y, Z, C))
faces = [tri, tri + (stations.size - 1)*nn]  # end sections
for k in range(stations.size - 1):
    a, e = edges[:, 0] + k*nn, edges[:, 1] + k*nn
    faces += [np.column_stack((a, e, e + nn)), np.column_stack((a, e + nn, a + nn))]
I, J, K = np.vstack(faces).T

fig = go.Figure(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, intensity=C,
                          colorscale='RdBu_r', colorbar=dict(title='w1'),
                          lighting=dict(ambient=0.8, diffuse=0.3,
                                        specular=0., fresnel=0.)))
fig.update_layout(scene=dict(aspectmode='data', xaxis_title='x1',
                             yaxis_title='x2', zaxis_title='x3'),
                  scene_camera=dict(eye=dict(x=-1.3, y=1.5, z=0.9)),
                  height=500, margin=dict(l=0, r=0, t=0, b=0))
# fig.show()
# NOTE written for the documentation
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                   '_generated')
os.makedirs(out, exist_ok=True)
fig.write_html(os.path.join(out, 'vabs_warping_3d.html'), full_html=False,
               include_plotlyjs='cdn')
