import numpy as np
import amnet
import amnet.util

import z3

import sys
import unittest


class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_maxN_z3(self):
        x = z3.Real('x')
        y = z3.Real('y')
        z = z3.Real('z')
        w1 = amnet.util.maxN_z3([x])
        w2 = amnet.util.maxN_z3([x, y])
        w3 = amnet.util.maxN_z3([x, y, z])

        s = z3.Solver()
        s.push()
        s.add(x == -3)
        s.add(y == 2)
        s.add(z == 12)
        self.assertTrue(s.check() == z3.sat)

        # extract the output
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, x) == -3)
        self.assertTrue(amnet.util.mfp(model, y) == 2)
        self.assertTrue(amnet.util.mfp(model, z) == 12)
        self.assertTrue(amnet.util.mfp(model, w1) == -3)
        self.assertTrue(amnet.util.mfp(model, w2) == 2)
        self.assertTrue(amnet.util.mfp(model, w3) == 12)
        s.pop()

    def test_minN_z3(self):
        x = z3.Real('x')
        y = z3.Real('y')
        z = z3.Real('z')
        w1 = amnet.util.minN_z3([x])
        w2 = amnet.util.minN_z3([x, y])
        w3 = amnet.util.minN_z3([x, y, z])

        s = z3.Solver()
        s.push()
        s.add(x == -3)
        s.add(y == 2)
        s.add(z == 12)
        self.assertTrue(s.check() == z3.sat)

        # extract the output
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, x) == -3)
        self.assertTrue(amnet.util.mfp(model, y) == 2)
        self.assertTrue(amnet.util.mfp(model, z) == 12)
        self.assertTrue(amnet.util.mfp(model, w1) == -3)
        self.assertTrue(amnet.util.mfp(model, w2) == -3)
        self.assertTrue(amnet.util.mfp(model, w3) == -3)
        s.pop()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtil)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
