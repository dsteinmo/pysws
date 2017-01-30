from __future__ import absolute_import
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


    def test_x_derivative(self):
        config = dict()
        config['length_x'] = 1000.
        config['length_y'] = 2000.
        config['num_points_x'] = 128
        config['num_points_y'] = 256

        fcsd = FCSD.FourierChebyshevSpatialDiscretization(config)
        x = fcsd.x

        f = np.tanh(x/(0.25*fcsd.length_x))

        fx = fcsd.differentiate_x(f)

        fx_expected = 1/(np.cosh(x/(0.25*fcsd.length_x))**2)/(0.25*fcsd.length_x)

        self.assertLess(norm(fx-fx_expected, ord=2), EPS, "x derivative test")

    def test_y_derivative(self):
        config = dict()
        config['length_x'] = 1000.
        config['length_y'] = 2000.
        config['num_points_x'] = 128
        config['num_points_y'] = 256

        fcsd = FCSD.FourierChebyshevSpatialDiscretization(config)

        y = fcsd.y

        f = np.sin(2*np.pi*y/fcsd.length_y)
        fy = fcsd.differentiate_y(f)

        fy_expected = 2*np.pi/fcsd.length_y*np.cos(2*np.pi*y/fcsd.length_y)

        self.assertLess(norm(fy - fy_expected, ord=2), EPS, "y derivative test")

    def test_filter(self):
        config = dict()
        config['length_x'] = 1000.
        config['length_y'] = 2000.
        config['num_points_x'] = 32
        config['num_points_y'] = 32

        fcsd = FCSD.FourierChebyshevSpatialDiscretization(config)

        f = (fcsd.x-0.5*fcsd.length_x)/fcsd.length_x * np.sin(4*np.pi*fcsd.y/fcsd.length_y)
        f_filtered = fcsd.filter_field(f)

        self.assertLess(norm(f-f_filtered, ord=2), EPS, "filter test")


if __name__ == '__main__':
    unittest.main()