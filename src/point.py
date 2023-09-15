from typing import Self

import numpy as np


class Point(np.ndarray):
    def __new__(cls, x, y, z):
        """
        :param cls:
        """
        obj = np.asarray([x, y, z]).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @x.setter
    def x(self, value):
        self[0] = value

    @y.setter
    def y(self, value):
        self[1] = value

    @z.setter
    def z(self, value):
        self[2] = value

    def normalized(self) -> "Point":
        """Normalize the point in place and return it"""
        return self / np.linalg.norm(self, 2)

    def norm2(self) -> float:
        """Compute the norm2 of a point"""
        return np.linalg.norm([self], 2)

    def get_local_coord(self, origin: Self, Ox: Self, Oy: Self, Oz: Self) -> "Point":
        """Convert the point from absolute coordinate to local from the perspective of the
        camera. Used for containment assessment. The point coordinates stay
        absolute i.e. they are not modified in place"""
        translated = self - origin

        return Point(translated.dot(Ox), translated.dot(Oy), translated.dot(Oz))
