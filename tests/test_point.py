import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from point import Point


class TestPoint(unittest.TestCase):
    def test_norm(self):
        point = Point(1, 2, 3).normalized()
        self.assertEqual(point.x, 0.2672612419124244, "Wrong x normalized")
        self.assertEqual(point.y, 0.5345224838248488, "Wrong y normalized")
        self.assertEqual(point.z, 0.8017837257372732, "Wrong z normalized")

    def test_abs(self):
        point = Point(1, 2, -3)
        np.testing.assert_array_equal(abs(point), Point(1, 2, 3)),

    def test_mul(self):
        a = Point(1, 2, 3)
        print(a)
        b = Point(3, 4, 5)
        np.testing.assert_array_equal(a * 2, Point(2, 4, 6)),
        np.testing.assert_array_equal(a**2, Point(1, 4, 9)),
        np.testing.assert_array_equal(a * b, Point(3, 8, 15)),


if __name__ == "__main__":
    unittest.main()
