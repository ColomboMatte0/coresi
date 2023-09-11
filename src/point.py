from typing import Self

import numpy as np


class Point(np.ndarray):
    def __new__(cls, x: float, y: float, z: float) -> "Point":
        """
        :param cls:
        """
        return np.asarray([x, y, z]).view(cls)

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, value: float) -> None:
        self[0] = value

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, value: float) -> None:
        self[1] = value

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, value: float) -> None:
        self[2] = value

    def __str__(self) -> str:
        return f"x: {self.x}, y: {self.y}, z: {self.z}"

    def normalized(self) -> "Point":
        """Normalize the point in place and return it"""
        return self / np.linalg.norm(self, 2)

    def norm2(self) -> float:
        """Compute the norm2 of a point"""
        return float(np.linalg.norm([self], 2))

    def get_local_coord(self, origin: Self, Ox: Self, Oy: Self, Oz: Self) -> "Point":
        """Convert the point from absolute coordinate to local from the perspective of the
        camera. Used for containment assessment. The point coordinates stay
        absolute i.e. they are not modified in place"""
        translated = self - origin

        return Point(translated.dot(Ox), translated.dot(Oy), translated.dot(Oz))
