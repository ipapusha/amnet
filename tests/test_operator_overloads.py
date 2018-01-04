import numpy as np
import amnet

import sys
import unittest
import itertools

from numpy.linalg import norm
from itertools import product

class TestOperatorOverloads(unittest.TestCase):
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
        cls.floatvals3 = np.linspace(-5., 5., 3)
        cls.FPTOL = 1e-8

        # set up global z3 parameters
        # parameters from https://stackoverflow.com/a/12516269
        #z3.set_param('auto_config', False)
        #z3.set_param('smt.case_split', 5)
        #z3.set_param('smt.relevancy', 2)

    def test_indexing(self):
        x = amnet.Variable(3, name='x')
        y = amnet.atoms.relu(x)

        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)

        x0    = x[0]
        x1    = x[1]
        x2    = x[2]
        x12   = x[1:3]
        x12_2 = x[1:]
        x210  = x[::-1]

        y0    = y[0]
        y1    = y[1]
        y2    = y[2]
        y12   = y[1:3]
        y12_2 = y[1:]
        y210  = y[::-1]

        self.assertEqual(len(x0), 1)
        self.assertEqual(len(x1), 1)
        self.assertEqual(len(x2), 1)
        self.assertEqual(len(x12), 2)
        self.assertEqual(len(x12_2), 2)
        self.assertEqual(len(x210), 3)

        self.assertEqual(len(y0), 1)
        self.assertEqual(len(y1), 1)
        self.assertEqual(len(y2), 1)
        self.assertEqual(len(y12), 2)
        self.assertEqual(len(y12_2), 2)
        self.assertEqual(len(y210), 3)

        # true relu
        def relu(inp): return np.maximum(inp, 0)

        for xv in product(self.floatvals2, repeat=3):
            xinp = np.array(xv)

            self.assertAlmostEqual(norm(x0.eval(xinp) - np.array([xv[0]])), 0)
            self.assertAlmostEqual(norm(x1.eval(xinp) - np.array([xv[1]])), 0)
            self.assertAlmostEqual(norm(x2.eval(xinp) - np.array([xv[2]])), 0)
            self.assertAlmostEqual(norm(x12.eval(xinp) - np.array([xv[1], xv[2]])), 0)
            self.assertAlmostEqual(norm(x12_2.eval(xinp) - np.array([xv[1], xv[2]])), 0)
            self.assertAlmostEqual(norm(x210.eval(xinp) - np.array([xv[2], xv[1], xv[0]])), 0)

            self.assertAlmostEqual(norm(y0.eval(xinp) - relu(np.array([xv[0]]))), 0)
            self.assertAlmostEqual(norm(y1.eval(xinp) - relu(np.array([xv[1]]))), 0)
            self.assertAlmostEqual(norm(y2.eval(xinp) - relu(np.array([xv[2]]))), 0)
            self.assertAlmostEqual(norm(y12.eval(xinp) - relu(np.array([xv[1], xv[2]]))), 0)
            self.assertAlmostEqual(norm(y12_2.eval(xinp) - relu(np.array([xv[1], xv[2]]))), 0)
            self.assertAlmostEqual(norm(y210.eval(xinp) - relu(np.array([xv[2], xv[1], xv[0]]))), 0)

        z = amnet.Variable(1, name='z')
        self.assertTrue(z[0] is z)
        self.assertTrue(z[0] is amnet.atoms.select(z, 0))

    def test_negate(self):
        x = amnet.Variable(3, name='x')
        y = amnet.atoms.relu(x)
        z = -y

        def negrelu(inp):
            return -np.maximum(inp, 0)

        self.assertEqual(len(z), 3)
        self.assertEqual(z.outdim, 3)

        for xv in product(self.floatvals2, repeat=3):
            xinp = np.array(xv)
            self.assertAlmostEqual(norm(z.eval(xinp) - negrelu(xinp)), 0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOperatorOverloads)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
