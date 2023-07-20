from typing import Self

import numpy as np


class Point(object):
    """Class defining a 3D point with overloaded operators and norm for
    convenience"""

    def __init__(self, x: float, y: float, z: float):
        super(Point, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other) -> "Point":
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return Point(self.x + other, self.y + other, self.z + other)
        else:
            return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other) -> "Point":
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return Point(self.x - other, self.y - other, self.z - other)
        else:
            return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other) -> "Point":
        # Divide by constant
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return Point(self.x / other, self.y / other, self.z / other)
        else:
            # Divide coord wise
            return Point(self.x / other.x, self.y / other.y, self.z / other.z)

    def __mul__(self, other) -> "Point":
        # Multiply by constant
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return Point(self.x * other, self.y * other, self.z * other)
        else:
            # Multiply coord wise
            return Point(self.x * other.x, self.y * other.y, self.z * other.z)

    def __abs__(self) -> "Point":
        return Point(abs(self.x), abs(self.y), abs(self.z))

    def __str__(self) -> str:
        return str(vars(self))

    def __repr__(self) -> str:
        return str(vars(self))

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def dot(self, other) -> float:
        return sum([self.x * other.x, self.y * other.y, self.z * other.z])

    def normalized(self) -> "Point":
        """Normalize the point in place and return it"""
        self.norm = float(np.linalg.norm([self.x, self.y, self.z], 2))
        self.x = self.x / self.norm
        self.y = self.y / self.norm
        self.z = self.z / self.norm
        return self

    def norm2(self) -> float:
        """Compute the norm2 of a point"""
        return float(np.linalg.norm([self.x, self.y, self.z], 2))

    def get_local_coord(self, origin: Self, Ox: Self, Oy: Self, Oz: Self) -> "Point":
        """Convert the point from absolute coordinate to local from the perspective of the
        camera. Used for containment assessment. The point coordinates stay
        absolute i.e. they are not modified in place"""
        translated = self - origin

        return Point(translated.dot(Ox), translated.dot(Oy), translated.dot(Oz))
