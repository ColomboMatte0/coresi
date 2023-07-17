import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from point import Point


class TestPoint(unittest.TestCase):
    def test_norm(self):
        point = Point(1, 2, 3).normalized()
        self.assertEqual(point.x, 0.2672612419124244, "Wrong x normalized")
        self.assertEqual(point.y, 0.5345224838248488, "Wrong y normalized")
        self.assertEqual(point.z, 0.8017837257372732, "Wrong z normalized")
        self.assertEqual(point.norm, 3.7416573867739413, "Wrong norm")

    def test_abs(self):
        point = Point(1, 2, -3)
        self.assertEqual(abs(point), Point(1, 2, 3), "Wrong point absolute value")


if __name__ == "__main__":
    unittest.main()
