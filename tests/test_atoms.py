import numpy as np
import amnet

import unittest
from itertools import chain, product


class TestAtoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print 'Setting up test floats.'
        cls.floatvals = np.concatenate(
            (np.linspace(-5., 5., 11), np.linspace(-5., 5., 10)),
            axis=0
        )
        cls.floatvals2 = np.concatenate(
            (np.linspace(-5., 5., 3), np.linspace(-.5, .5, 2)),
            axis=0
        )

    def test_make_relu1(self):
        # define one-dimensional input variable
        x = amnet.Variable(1, name='x')
        phi_relu = amnet.atoms.make_relu1(x)

        # true relu
        def relu(x): return x if x > 0 else 0

        # implemented relu
        for xv in self.floatvals:
            inp = np.array([xv])
            self.assertEqual(phi_relu.eval(inp), relu(inp))

    def test_make_max2(self):
        x = amnet.Variable(2, name='x')
        phi_max2 = amnet.atoms.make_max2(x)

        # true max2
        def max2(x): return x[0] if x[0] > x[1] else x[1]

        # implemented max2
        for xv in self.floatvals:
            for yv in self.floatvals:
                xyv = np.array([xv, yv])
                self.assertEqual(phi_max2.eval(xyv), max2(xyv))

    def test_make_max3(self):
        x = amnet.Variable(3, name='x')
        phi_max3 = amnet.atoms.make_max3(x)

        # true max3
        def max3(x): return np.max(x)

        # implemented max3
        for xv in self.floatvals:
            for yv in self.floatvals:
                for zv in self.floatvals:
                    xyzv = np.array([xv, yv, zv])
                    self.assertEqual(phi_max3.eval(xyzv), max3(xyzv))

    def test_make_max4(self):
        x = amnet.Variable(4, name='x')
        phi_max4 = amnet.atoms.make_max4(x)

        # true max4
        def max4(x): return np.max(x)

        # implemented max4
        for xv, yv, zv, wv in product(self.floatvals2, repeat=4):
            xyzwv = np.array([xv, yv, zv, wv])
            self.assertEqual(phi_max4.eval(xyzwv), max4(xyzwv))

    def test_make_max(self):
        x = amnet.Variable(4, name='x')
        phi_max = amnet.atoms.make_max(x)

        # true max4
        def max_true(x): return np.max(x)

        # implemented max4
        for xv,yv,zv,wv in product(self.floatvals2, repeat=4):
            xyzwv = np.array([xv, yv, zv, wv])
            self.assertEqual(phi_max.eval(xyzwv), max_true(xyzwv))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAtoms)
    unittest.TextTestRunner(verbosity=2).run(suite)
