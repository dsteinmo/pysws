import unittest
import sys
from os import path
from numpy.linalg import norm

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.abspath(__file__)))
import spatial_discretizations.FourierChebyshevSpatialDiscretization as FCSD
import numpy as np

EPS = 2.e-8

class FourierChebyshevSpatialDiscretizationTests(unittest.TestCase):
    def test_init(self):
        config = dict()
        config['length_x'] = 1000.
        config['length_y'] = 2000.
        config['num_points_x'] = 6
        config['num_points_y'] = 6

        fcsd = FCSD.FourierChebyshevSpatialDiscretization(config)

        self.assertEqual(config['length_x'], fcsd.length_x)
        self.assertEqual(config['length_y'], fcsd.length_y)
        self.assertEqual(config['num_points_x'], fcsd.num_points_x)
        self.assertEqual(config['num_points_y'], fcsd.num_points_x)

if __name__ == '__main__':
    unittest.main()