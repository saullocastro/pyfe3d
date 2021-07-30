r"""
Coordinate systems (:mod:`pyfe3d.coord`)
========================================

.. currentmodule:: pyfe3d.coord

Useful classes and methods to define coordinate systems ("coordsys") and
calculate transformation matrices bewteen coordinate systems.

"""
import numpy as np
from alg3dpy.vector import Vec
from alg3dpy.angles import cosplanevec, sinplanevec, cos2vecs, sin2vecs
from alg3dpy.plane import Plane
from alg3dpy.point import Point


def __common__str__(text, csys):
    return '%s ID %d:\n\
    \tO %2.3f i + %2.3f j + %2.3f k\n\
    \tX %2.3f i + %2.3f j + %2.3f k\n\
    \tY %2.3f i + %2.3f j + %2.3f k\n\
    \tZ %2.3f i + %2.3f j + %2.3f k' \
    % (text, csys.id,\
       csys.o[0], csys.o[1], csys.o[2],\
       csys.x[0], csys.x[1], csys.x[2],\
       csys.y[0], csys.y[1], csys.y[2],\
       csys.z[0], csys.z[1], csys.z[2])


class Coord(object):
    """
    Each coordinate system is defined by three points:
        o:  the origin
        z:  the Z axis
        vecxz: a point laying in the azimuthal origin
    Basically the point are defined using reference GRIDs or POINT coordinates.

    Attributes
    ----------
    id : int
        Coordinate system id
    o : :class:`.Point` object
        The origin of the coordsys alg3dpy.Point
    x : :class:`.Vec` object
        The x vector of the coordsys
    y : :class:`.Vec` object
        The y vector of the coordsys
    z : :class:`.Vec` object
        The z vector of the coordsys
    xy : :class:`.Plane` object
        The xy plane of the coordsys
    xz : :class:`.Plane` object
        The xz plane of the coordsys
    yz : :class:`.Plane` object
        The yz plane of the coordsys
    vecxz : :class:`.Vec` object
        A vector laying in the xz plane of the coordsys

    ____________________________________________________________________________

    Note: not all attributes are used simultaneously, but the Coord class is
          already prepared for many ways for defining a coordsys
    ____________________________________________________________________________
    """
    __slots__ = [ 'id','o','x','y','z',
                  'xy','xz','yz','vecxz' ]

    def __init__(self, id, o, z, vecxz):
        """
        Parameters
        ----------
        o : :class:`.Point` object
            The origin of the coordsys alg3dpy.Point
        z : :class:`.Vec` object
            The z vector of the coordsys
        vecxz : :class:`.Vec` object
            A vector laying in the xz plane of the coordsys
        """
        # given
        self.id = id
        self.o = np.asarray(o).view(Point)
        self.z = np.asarray(z).view(Vec)/np.linalg.norm(z)
        self.vecxz = np.asarray(vecxz).view(Vec)
        # calculated
        self.y = self.z.cross(self.vecxz)
        self.x = self.y.cross(self.z)
        self.xy = Plane(self.z[0],  self.z[1],  self.z[2], np.linalg.norm(self.o))
        self.xz = Plane(-self.y[0], -self.y[1], -self.y[2], np.linalg.norm(self.o))
        self.yz = Plane(self.x[0],  self.x[1],  self.x[2], np.linalg.norm(self.o))

    def transform(self, vec, new_csys):
        r"""
        The transformation will go as follows:
            - transform to cartesian in the local coordsys;
            - rotate to the new_csys (which is cartesian);
            - translate to the new_csys.
        All systems: cartesian, cylindrical or spherical; have
        the method vec2cr which will automatically transform vec into
        cartesian coordinates in the local coordsys.
        The two other steps will rotate and translate vec to new_csys.
        The last step will transform again from the new_csys cartesian
        coordinates to its cylindrical or spherical coordinates.
        All coordinate systems have the method cr2me to transform from
        local cartesian to local something.

        """
        from pyfe3d import CSYSGLOBAL
        if new_csys is None:
            new_csys = CSYSGLOBAL
        vec_cr = self.vec2cr(vec)
        R = self.Rmatrix(new_csys)
        vec_rot = np.dot(R, vec_cr)
        vec_t = self.translate(vec_rot, new_csys)
        vec_final = new_csys.cr2me(vec_t)
        return vec_final

    def translate(self, vec, newcr):
        """
        Calculate the translation matrix to a new cartesian system (newcr)
        """
        vec = vec.view(Vec)
        vec_t = vec + newcr.o + self.o
        return vec_t

    def cosines_to_new_coord(self, newcr):
        """
        Calculate the rotation cosines to a new cartesian system (newcr)
        """
        cosb = cosplanevec(newcr.xy, self.x)
        cosg = cosplanevec(newcr.xz, self.x)
        cosa = cosplanevec(newcr.xy, self.y)
        return cosa, cosb, cosg

    def cosines_to_global(self):
        """
        Calculate the rotation cosines to the global coordinate system
        """
        from pyfe3d import CSYSGLOBAL
        return self.cosines_to_new_coord(CSYSGLOBAL)

    def Rmatrix(self, newcr):
        """
        Calculate the rotation matrix to a new cartesian system (newcr)
        """
        cosb = cosplanevec(newcr.xy, self.x)
        sinb = sinplanevec(newcr.xy, self.x)
        cosg = cosplanevec(newcr.xz, self.x)
        sing = sinplanevec(newcr.xz, self.x)
        tmpT =  np.array([\
            [ -sing,  0, 0 ],
            [   0, cosg, 0 ],
            [   0,  0, 0 ]])
        Y2 = (np.dot(tmpT, newcr.y.array)).view(Vec)
        cosa = cos2vecs(Y2, self.y)
        sina = sin2vecs(Y2, self.y)
        Rself = np.array([\
           [ cosb*cosg               ,  cosb*sing ,                  -sinb ], \
           [-cosa*sing+cosg*sina*sinb,  cosa*cosg+sina*sinb*sing, cosb*sina], \
           [ sina*sing+cosa*cosg*sinb, -cosg*sina+cosa*sinb*sing, cosa*cosb]])
        R2new = Rself.transpose()
        return R2new

    def R2basic(self):
        from pyfe3d import CSYSGLOBAL
        return self.Rmatrix(CSYSGLOBAL)

class CoordR(Coord):
    __slots__ = Coord.__slots__
    def __init__(self, id, o, z, vecxz):
        super(CoordR, self).__init__(id, o, z, vecxz)

    def __str__(self):
        return __common__str__('Cartesian Coord Sys', self)

    def vec2cr(self, vec):
        return vec

    def cr2me(self, vec):
        return vec

class CoordC(Coord):
    __slots__ = Coord.__slots__
    def __init__(self, id=None, o=None, z=None, vecxz=None):
        super(CoordC, self).__init__(id, o, z, vecxz)

    def vec2cr(self, vec):
        """
        Transformation from cylindrical to cartesian
        vec must be in cylindrical cordinates: [r, theta, z]
        """
        T =  np.array([\
            [ np.cos(vec[1]), 0,   0 ],
            [ 0, np.sin(vec[1]),   0 ],
            [ 0,                0, 1 ]])
        tmp = np.array([ vec[0], vec[0], vec[2] ])
        vec_cr = np.dot(T, tmp)
        return vec_cr

    def cr2me(self, vec):
        """
        Transformation from cartesian to cylindrical
        vec must be in cartesian coordinates: [x, y, z]
        """
        T = np.array([\
            [ np.sqrt(vec[0] ** 2 + vec[1] ** 2), 0,   0 ],
            [ 0,           np.arctan(vec[1] / vec[0]),   0 ],
            [ 0,                                    0, 1 ]])
        tmp = np.array([ 1, 1, vec[2] ])
        return np.dot(T, tmp)

    def __str__(self):
        return __common__str__('Cylindrical Coord Sys', self)

class CoordS(Coord):
    __slots__ = Coord.__slots__
    def __init__(self, id=None, o=None, z=None, vecxz=None):
        super(CoordS, self).__init__(id, o, z, vecxz)

    def vec2cr(self, vec):
        """
        Transformation from spherical to cartesian
        vec must be in spherical coordinates: [r, theta, phi]
        """
        T =  np.array([\
            [ np.sin(vec[1])*np.cos(vec[2]), 0, 0 ],
            [ 0, np.sin(vec[1])*np.sin(vec[2]), 0 ],
            [ 0, 0,                  np.cos(vec[1]) ]])
        tmp = np.array([ vec[0], vec[0], vec[0] ])
        return np.dot(T, tmp)

    def cr2me(self, vec):
        """
        Transformation from cartesian to spherical
        vec must be in cartesian coordinates: [x, y, z]
        """
        h = vec[0] ** 2 + vec[1] ** 2
        T = np.array([\
            [ np.sqrt(h + vec[2] ** 2),     0,   0 ],
            [ 0, np.arctan(np.sqrt(h) / vec[2]),   0 ],
            [ 0,       0, np.arctan(vec[1] / vec[0]) ]])
        tmp = np.array([ 1, 1, 1])
        return np.dot(T, tmp)

    def __str__(self):
        return __common__str__('Spherical Coord Sys', self)

# Weisstein, Eric W. "Rotation Matrix." From MathWorld--A Wolfram Web Resource.
#   http://mathworld.wolfram.com/RotationMatrix.html
