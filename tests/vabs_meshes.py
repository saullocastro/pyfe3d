r"""Meshes of cross sections for the tests of :mod:`pyfe3d.vabs`

Structured patches of linear or quadratic elements, mapped from the unit
square, are merged into the meshes of rectangles, circles, tubes, boxes and
other sections. Every function returns the node coordinates ``x2``, ``x3``,
the connectivity ``elements``, with the nodes numbered counter-clockwise as
required by :class:`pyfe3d.vabs.CrossSection`, and, for the walls, the
information needed to orient the plies.
"""
import numpy as np


def patch(P, nu, nv, etype):
    r"""Structured mesh of the image of the unit square by ``P``

    ``P(u, v)`` returns ``(x2, x3)`` for arrays ``u``, ``v``; the mapping
    must keep the orientation, i.e. ``u`` to the right and ``v`` upwards
    give counter-clockwise elements. ``etype`` is the number of nodes per
    element: 3 and 6 (each cell split into two triangles), 4, 8 or 9.
    Returns ``x2``, ``x3``, ``elements`` and, per element, the indices
    ``(i, j)`` of its cell along ``u`` and ``v`` and the parameters ``(u,
    v)`` of the centre of the cell.
    """
    quad = etype in (6, 8, 9)
    k = 2 if quad else 1
    us = np.linspace(0, 1, k*nu + 1)
    vs = np.linspace(0, 1, k*nv + 1)
    U, V = np.meshgrid(us, vs)
    x2, x3 = P(U.ravel(), V.ravel())
    nrow = k*nu + 1

    def idx(i, j):
        return j*nrow + i

    elements, cells, centres = [], [], []
    for j in range(nv):
        for i in range(nu):
            I, J = k*i, k*j
            uc = 0.5*(us[I] + us[I + k])
            vc = 0.5*(vs[J] + vs[J + k])
            if not quad:
                c = [idx(I, J), idx(I + 1, J), idx(I + 1, J + 1), idx(I, J + 1)]
                if etype == 4:
                    els = [c]
                else:
                    els = [[c[0], c[1], c[2]], [c[0], c[2], c[3]]]
            else:
                c = [idx(I, J), idx(I + 2, J), idx(I + 2, J + 2), idx(I, J + 2)]
                m = [idx(I + 1, J), idx(I + 2, J + 1), idx(I + 1, J + 2),
                     idx(I, J + 1)]
                ctr = idx(I + 1, J + 1)
                if etype == 8:
                    els = [c + m]
                elif etype == 9:
                    els = [c + m + [ctr]]
                else:
                    els = [[c[0], c[1], c[2], m[0], m[1], ctr],
                           [c[0], c[2], c[3], ctr, m[2], m[3]]]
            for el in els:
                elements.append(el)
                cells.append((i, j))
                centres.append((uc, vc))
    return (np.asarray(x2, dtype=float), np.asarray(x3, dtype=float),
            elements, np.array(cells), np.array(centres))


def merge(parts, decimals=9):
    r"""Merge meshes, given as ``(x2, x3, elements)``, and their unused or
    coincident nodes; returns ``x2``, ``x3``, ``elements``"""
    X2, X3, ELS = [], [], []
    offset = 0
    for x2, x3, els in parts:
        X2.append(x2)
        X3.append(x3)
        ELS.extend([[n + offset for n in el] for el in els])
        offset += len(x2)
    X2 = np.concatenate(X2)
    X3 = np.concatenate(X3)
    used = np.unique([n for el in ELS for n in el])
    keys = np.round(np.column_stack((X2[used], X3[used])), decimals)
    # NOTE -0. and 0. must give the same key
    keys += 0.
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    new = -np.ones(len(X2), dtype=int)
    new[used] = inv
    x2 = np.zeros(len(uniq))
    x3 = np.zeros(len(uniq))
    x2[inv] = X2[used]
    x3[inv] = X3[used]
    elements = [[int(new[n]) for n in el] for el in ELS]
    return x2, x3, elements


def rectangle(a, b, nx, ny, etype, x0=0., y0=0.):
    r"""Rectangle of width ``a`` along `x_2` and height ``b`` along `x_3`
    centred at ``(x0, y0)``"""
    def P(u, v):
        return x0 + a*(u - 0.5), y0 + b*(v - 0.5)
    x2, x3, els, _, _ = patch(P, nx, ny, etype)
    return merge([(x2, x3, els)])


def annulus(Ri, Ro, nr, nt, etype, x0=0., y0=0.):
    r"""Annulus of radii ``Ri`` and ``Ro`` centred at ``(x0, y0)``, with
    ``nr`` elements across and ``nt`` around"""
    def P(u, v):
        r = Ri + (Ro - Ri)*u
        t = 2*np.pi*v
        return x0 + r*np.cos(t), y0 + r*np.sin(t)
    x2, x3, els, _, _ = patch(P, nr, nt, etype)
    return merge([(x2, x3, els)])


def circle(R, n, etype):
    r"""Circle of radius ``R`` meshed with a central square of half side
    ``R/2`` and four mapped patches, ``n`` elements along each side of the
    square and ``n`` across each patch"""
    s = R/2
    parts = [rectangle(2*s, 2*s, n, n, etype)]
    for q in range(4):
        rot = q*np.pi/2

        def P(u, v, rot=rot):
            # NOTE patch on the side x2 = s of the square, rotated by rot
            phi = -np.pi/4 + v*np.pi/2
            sx, sy = s*np.ones_like(v), -s + 2*s*v
            ax, ay = R*np.cos(phi), R*np.sin(phi)
            px = (1 - u)*sx + u*ax
            py = (1 - u)*sy + u*ay
            c, sn = np.cos(rot), np.sin(rot)
            return c*px - sn*py, sn*px + c*py
        x2, x3, els, _, _ = patch(P, n, n, etype)
        parts.append((x2, x3, els))
    return merge(parts)


def walls(wall_patches, nt, etype):
    r"""Merge thin walls, each given as ``(P, nu)``, where ``P(u, v)`` maps
    ``u`` along the wall, counter-clockwise around the section, and ``v``
    across it, from the outer to the inner surface, and ``nu`` is the number
    of elements along the wall; ``nt`` elements across each wall. Returns
    ``x2``, ``x3``, ``elements`` and, per element, the index of its wall,
    the index of its layer across the wall, from the outer surface, and
    the angle ``theta1`` of the tangent to the wall at its centre, in
    degrees."""
    parts, wall_id, layer, theta1 = [], [], [], []
    for w, (P, nu) in enumerate(wall_patches):
        x2, x3, els, cells, centres = patch(P, nu, nt, etype)
        parts.append((x2, x3, els))
        du = 1.e-7
        for (i, j), (uc, vc) in zip(cells, centres):
            xa, ya = P(np.array([uc - du]), np.array([vc]))
            xb, yb = P(np.array([uc + du]), np.array([vc]))
            theta1.append(np.rad2deg(np.arctan2(yb[0] - ya[0],
                                                xb[0] - xa[0])))
            wall_id.append(w)
            layer.append(j)
    x2, x3, elements = merge(parts)
    return x2, x3, elements, np.array(wall_id), np.array(layer), \
        np.array(theta1)


def box(a2, a3, t, n2, n3, nt, etype):
    r"""Thin-walled rectangular box of outer width ``a2`` and height ``a3``
    and wall thickness ``t``, with mitred corners; walls numbered bottom,
    right, top, left, see :func:`walls`"""
    def straight(p0, p1, q0, q1):
        # NOTE outer edge from p0 to p1, inner edge from q0 to q1
        def P(u, v):
            ox = p0[0] + (p1[0] - p0[0])*u
            oy = p0[1] + (p1[1] - p0[1])*u
            ix = q0[0] + (q1[0] - q0[0])*u
            iy = q0[1] + (q1[1] - q0[1])*u
            return (1 - v)*ox + v*ix, (1 - v)*oy + v*iy
        return P
    h2, h3 = a2/2, a3/2
    o = [(-h2, -h3), (h2, -h3), (h2, h3), (-h2, h3)]
    i = [(-h2 + t, -h3 + t), (h2 - t, -h3 + t), (h2 - t, h3 - t),
         (-h2 + t, h3 - t)]
    patches = [(straight(o[0], o[1], i[0], i[1]), n2),
               (straight(o[1], o[2], i[1], i[2]), n3),
               (straight(o[2], o[3], i[2], i[3]), n2),
               (straight(o[3], o[0], i[3], i[0]), n3)]
    return walls(patches, nt, etype)


def ibeam(bf, H, t, nf, nt, nw, etype):
    r"""Doubly symmetric I-section with flanges of width ``bf`` and outer
    height ``H``, walls of thickness ``t``, centred at the origin, web
    along `x_3`; ``nf`` elements along each half flange outside the web,
    ``nt`` across the walls and ``nw`` along the web between the flanges"""
    w = (bf - t)/2
    parts = []
    for yc in (-H/2 + t/2, H/2 - t/2):
        parts.append(rectangle(w, t, nf, nt, etype, -t/2 - w/2, yc))
        parts.append(rectangle(t, t, nt, nt, etype, 0., yc))
        parts.append(rectangle(w, t, nf, nt, etype, t/2 + w/2, yc))
    parts.append(rectangle(t, H - 2*t, nt, nw, etype, 0., 0.))
    return merge(parts)


def channel(b, h, t, nf, nt, nw, etype):
    r"""Channel with the mid-line of the web on `x_2 = 0`, from `x_3 =
    -h/2` to `h/2`, and flanges along `+x_2` up to `x_2 = b`, walls of
    thickness ``t``; ``nf`` elements along each flange outside the web,
    ``nt`` across the walls and ``nw`` along the web between the flanges"""
    lf = b - t/2
    parts = []
    for yc in (-h/2, h/2):
        parts.append(rectangle(t, t, nt, nt, etype, 0., yc))
        parts.append(rectangle(lf, t, nf, nt, etype, t/2 + lf/2, yc))
    parts.append(rectangle(t, h - t, nt, nw, etype, 0., 0.))
    return merge(parts)
