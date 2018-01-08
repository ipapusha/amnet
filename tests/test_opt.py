import numpy as np
import amnet

import sys
import unittest
import itertools

from numpy.linalg import norm
from itertools import product

class TestOpt(unittest.TestCase):
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

    def test_minimize(self):
        #f = amnet.atoms.norm1(2*x - 2)

        # A = np.array([
        #     [1, 2, -3],
        #     [4, -5, 6],
        #     [7, 8, -9],
        #     [-1, 0, 2]])
        # b = np.array([1, 2, 3, 4])
        # f = amnet.atoms.norm1(A*x + b)

        x = amnet.Variable(3, name='x')
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        b = np.array([7, 8])
        f = amnet.atoms.norm1(amnet.Affine(A, x, b))

        one3 = amnet.Constant(x, np.ones(3))
        none3 = amnet.Constant(x, -np.ones(3))

        obj = amnet.opt.Minimize(f)
        cons = [amnet.opt.Constraint(x, one3, amnet.opt.Relation.LE),
                amnet.opt.Constraint(x, none3, amnet.opt.Relation.GE),]
        prob = amnet.opt.Problem(obj, cons)
        result = prob.solve()
        print result


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOpt)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
