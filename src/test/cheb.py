from __future__ import absolute_import
import unittest
import sys
from os import path
from numpy.linalg import norm

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import spatial_discretizations.cheb.cheb as cheb
import numpy as np

EPS = 2.e-8

class ChebTests(unittest.TestCase):
    def test_cheb_dif(self):
        x, DM = cheb.cheb_dif(4, 2)

        x_expected = np.array([1., 0.5, -0.5, -1.])

        self.assertLess(norm(x-x_expected, ord=2), EPS, "cheb grid")

        D = DM[:, :, 0]
        D_expected = np.array([[ 3.16666667, -4.,          1.33333333, -0.5       ],
                               [ 1.,         -0.33333333, -1.,          0.33333333],
                               [-0.33333333,  1.,          0.33333333, -1.        ],
                               [ 0.5,        -1.33333333,  4.,         -3.16666667]])

        self.assertLess(norm(D-D_expected, ord=2), EPS, "cheb first derivative matrix")

        D2 = DM[:, :, 1]

        D2_expected = np.array([[5.33333333, -9.33333333,  6.66666667, -2.66666667],
                                [3.33333333, -5.33333333,  2.66666667, -0.66666667],
                                [-0.66666667, 2.66666667, -5.33333333, 3.33333333],
                                [-2.66666667,  6.66666667, -9.33333333,  5.33333333]])

        self.assertLess(norm(D2-D2_expected, ord=2), EPS, "cheb second derivative matrix")


    def test_cosine_transform(self):
        f = np.array([[8], [8], [8], [8], [8], [8]])
        fhat = cheb.cosine_transform(f)

        fhat_expected = np.array([[80.], [0.], [0.], [0.], [0.], [0.]])

        self.assertLess(norm(fhat-fhat_expected, ord=2), EPS, "Cosine transform test 1")

        f2 = np.array([[1],[2],[3],[4],[5]])
        f2hat = cheb.cosine_transform(f2)
        f2hat_expected = np.array([[24.00000], [-6.828427124746190], [0.00000], [-1.171572875253810], [0.00000]])

        self.assertLess(norm(f2hat-f2hat_expected, ord=2), EPS, "Cosine transform test 2")


    def test_cosine_transform_withaxis(self):
        f = np.array([[8], [8], [8], [8], [8], [8]])
        fhat = cheb.cosine_transform(f, axis=0)

        fhat_expected = np.array([[80.], [0.], [0.], [0.], [0.], [0.]])
        self.assertLess(norm(fhat - fhat_expected, ord=2), EPS, "Cosine transform with axis test 1")

        f2 = np.array([[8, 8, 8, 8, 8, 8]])
        f2hat = cheb.cosine_transform(f2, axis=1)
        f2hat_expected = np.array([[80., 0., 0., 0., 0., 0.]])

        self.assertLess(norm(f2hat-f2hat_expected, ord=2), EPS, "Cosine transform with axis test 2")


    def test_cheb_derivative(self):
        f = np.array([[8], [8], [8], [8], [8], [8]])
        dfdx = cheb.cheb_derivative(f)

        dfdx_expected = np.array([[0], [0], [0], [0], [0], [0]])

        self.assertLess(norm(dfdx-dfdx_expected, ord=2), EPS, "Cheb derivative of constant function")

        f2 = np.array([[1], [2], [3], [4]])
        df2dx = cheb.cheb_derivative(f2)

        df2dx_expected = np.array([[-4.33333333333333], [-1.33333333333333],  [-1.33333333333333],  [-4.33333333333333]])

        self.assertLess(norm(df2dx-df2dx_expected, ord=2), EPS, "Cheb derivative of [1,2,3,4]")


    def test_cheb_derivative_withaxis(self):
        f = np.array([[8], [8], [8], [8], [8], [8]])
        dfdx = cheb.cheb_derivative(f, axis=0)

        dfdx_expected = np.array([[0], [0], [0], [0], [0], [0]])

        self.assertLess(norm(dfdx-dfdx_expected, ord=2), EPS, "Cheb derivative of constant down axis=0")

        f2 = np.array([[8, 8, 8, 8, 8, 8]])
        df2dx = cheb.cheb_derivative(f2, axis=1)

        df2dx_expected = np.array([[0, 0, 0, 0, 0, 0]])
        self.assertLess(norm(df2dx-df2dx_expected, ord=2), EPS, "Cheb derivative of constant down axis=1")


    # 2D tests:
    def test_cosine_transform_2D(self):
        f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        fhat = cheb.cosine_transform(f, axis=0)

        fhat_expected = np.array([[ 16.,  20.,  24.], [ -6.,  -6.,  -6.], [  0.,   0.,   0.]])

        self.assertLess(norm(fhat-fhat_expected, ord=2), EPS, "2D cosine transform down axis=0")

        fhat2 = cheb.cosine_transform(f, axis=1)

        fhat2_expected = np.array([[8., -2., 0.], [20., -2., 0.], [32., -2., 0.]])

        self.assertLess(norm(fhat2-fhat2_expected, ord=2), EPS, "2D cosine transform down axis=1")


    def test_cheb_derivative_2D(self):
        f = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        fy = cheb.cheb_derivative(f, axis=0)
        fy_expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        self.assertLess(norm(fy-fy_expected, ord=2), EPS, "Cheb derivative down axis=0")

        fx = cheb.cheb_derivative(f, axis=1)
        fx_expected = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])

        self.assertLess(norm(fx-fx_expected, ord=2), EPS, "Cheb derivative down axis=1")


if __name__ == '__main__':
    unittest.main()
