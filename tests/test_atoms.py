import numpy as np
import amnet

import sys
import unittest
import itertools

from numpy.linalg import norm
from itertools import product


class TestAtoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print 'Setting up test floats.'
        cls.floatvals = np.concatenate(
            (np.linspace(-5., 5., 11),
             np.linspace(-5., 5., 10)),
            axis=0
        )
        cls.floatvals2 = np.concatenate(
            (np.linspace(-5., 5., 3),
             np.linspace(-.5, .5, 2)),
            axis=0
        )

    def test_select(self):
        x = amnet.Variable(2, name='x')
        x0 = amnet.atoms.select(x, 0)
        x1 = amnet.atoms.select(x, 1)

        for (xv0, xv1) in product(self.floatvals, repeat=2):
            xinp = np.array([xv0, xv1])
            self.assertEqual(x0.eval(xinp), xv0)
            self.assertEqual(x1.eval(xinp), xv1)

    def test_to_from_list(self):
        x = amnet.Variable(2, name='x')

        w = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([7, 8, 9])
        y = amnet.Affine(
            w,
            x,
            b
        )

        self.assertEqual(y.outdim, 3)

        ylist = amnet.atoms.to_list(y)
        self.assertEqual(len(ylist), y.outdim)

        ytilde = amnet.atoms.from_list(ylist)
        self.assertEqual(ytilde.outdim, y.outdim)

        for (xv0, xv1) in product(self.floatvals, repeat=2):
            xinp = np.array([xv0, xv1])
            yv = y.eval(xinp)
            ylv = np.array([yi.eval(xinp) for yi in ylist]).flatten()  # note the flatten!
            ytv = ytilde.eval(xinp)
            self.assertAlmostEqual(norm(yv - ylv), 0)
            self.assertAlmostEqual(norm(yv - ytv), 0)

    def test_identity(self):
        x = amnet.Variable(2, name='x')

        w = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([7, 8, 9])

        y = amnet.Affine(
            w,
            x,
            b
        )
        self.assertEqual(y.outdim, 3)

        z = amnet.atoms.neg(y)
        self.assertEqual(z.outdim, y.outdim)

        def aff(x): return np.dot(w, x) + b
        for (xv0, xv1) in product(self.floatvals, repeat=2):
            xinp = np.array([xv0, xv1])
            yv = y.eval(xinp)
            zv = z.eval(xinp)
            self.assertAlmostEqual(norm(yv - aff(xinp)), 0)
            self.assertAlmostEqual(norm(zv + aff(xinp)), 0)

    def test_stack(self):
        x = amnet.Variable(3, name='x')

        w1 = np.array([[1, -2, 3]])
        b1 = np.zeros(1)

        w2 = np.array([[-5, 6, -7], [-8, 9, 10]])
        b2 = np.array([11, -12])

        w3 = np.array([[7, 8, 9]])
        b3 = np.array([-1])

        y1 = amnet.Linear(w1, x)
        y2 = amnet.Affine(w2, x, b2)
        y3 = amnet.Affine(w3, x, b3)

        y4 = x

        ystack = amnet.atoms.from_list([y1, y2, y3, y4])
        self.assertEqual(ystack.outdim, 4+3)
        self.assertEqual(ystack.indim, 3)

        def ystack_true(xinp):
            y1v = np.dot(w1, xinp) + b1
            y2v = np.dot(w2, xinp) + b2
            y3v = np.dot(w3, xinp) + b3
            y4v = xinp
            return np.concatenate((y1v, y2v, y3v, y4v), axis=0)

        for (xv0, xv1, xv2) in product(self.floatvals2, repeat=3):
            xinp = np.array([xv0, xv1, xv2])
            ysv = ystack.eval(xinp)
            ytv = ystack_true(xinp)
            self.assertAlmostEqual(norm(ysv - ytv), 0)

    def test_relu(self):
        x = amnet.Variable(4, name='x')
        y = amnet.Variable(1, name='y')
        phi_relu = amnet.atoms.relu(x)
        phi_reluy = amnet.atoms.relu(y)

        # true relu
        def relu(x): return np.maximum(x, 0)

        for xv,yv,zv,wv in itertools.product(self.floatvals2, repeat=4):
            xyzwv = np.array([xv, yv, zv, wv])
            r = phi_relu.eval(xyzwv) # 4-d relu of x
            s = relu(xyzwv)
            self.assertTrue(len(r) == 4)
            self.assertTrue(len(s) == 4)
            self.assertTrue(all(r == s))

        for yv in self.floatvals:
            r = phi_reluy.eval(np.array([yv])) # 1-d relu of y
            s = relu(np.array([yv]))
            self.assertEqual(r, s)

    def test_max2_min2(self):
        xy = amnet.Variable(6, name='xy')
        x = amnet.Linear(
            np.eye(3,6,0),
            xy
        )
        y = amnet.Linear(
            np.eye(3,6,3),
            xy
        )
        max_xy = amnet.atoms.max2(x, y)
        min_xy = amnet.atoms.min2(x, y)

        def true_max2(xinp, yinp):
            return np.maximum(xinp, yinp)

        def true_min2(xinp, yinp):
            return np.minimum(xinp, yinp)

        for xyv in itertools.product(self.floatvals2, repeat=6):
            xyinp = np.array(xyv)
            xinp = xyinp[:3]
            yinp = xyinp[3:]

            max_xy_tv = true_max2(xinp, yinp)
            max_xy_v  = max_xy.eval(xyinp)

            min_xy_tv = true_min2(xinp, yinp)
            min_xy_v = min_xy.eval(xyinp)

            self.assertEqual(len(max_xy_tv), 3)
            self.assertEqual(len(max_xy_v), 3)
            self.assertAlmostEqual(norm(max_xy_v - max_xy_tv), 0)

            self.assertEqual(len(min_xy_tv), 3)
            self.assertEqual(len(min_xy_v), 3)
            self.assertAlmostEqual(norm(min_xy_v - min_xy_tv), 0)

    def test_max_all_max_aff(self):
        x = amnet.Variable(4, name='x')

        w = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        b = np.array([1.1, 1.2, 1.3])

        y = amnet.Affine(
            w,
            x,
            b
        )
        max_y = amnet.atoms.max_all(y)
        maff = amnet.atoms.max_aff(w, b, x)

        def true_max_aff(xinp):
            return np.max(np.dot(w, xinp) + b)

        for xv in itertools.product(self.floatvals2, repeat=4):
            xinp = np.array(xv)

            tv = true_max_aff(xinp)
            max_y_v = max_y.eval(xinp)
            maff_v  = maff.eval(xinp)

            self.assertAlmostEqual(norm(tv - max_y_v), 0)
            self.assertAlmostEqual(norm(tv - maff_v), 0)

    def test_triplexer(self):
        x = amnet.Variable(1, name='xv')

        np.random.seed(1)

        for _ in range(10):
            a = 3 * (2 * np.random.rand(4) - 1)
            b = 3 * (2 * np.random.rand(4) - 1)
            c = 3 * (2 * np.random.rand(4) - 1)
            d = 3 * (2 * np.random.rand(4) - 1)
            e = 3 * (2 * np.random.rand(4) - 1)
            f = 3 * (2 * np.random.rand(4) - 1)
            phi_tri = amnet.atoms.triplexer(x, a, b, c, d, e, f)

            xvals = 100 * (2 * np.random.rand(1000) - 1)
            for xv in xvals:
                self.assertEqual(phi_tri.eval(np.array([xv])),
                                 amnet.atoms.fp_triplexer(xv, a, b, c, d, e, f))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAtoms)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
