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

    def test_abs_z3(self):
        x = z3.Real('x')
        y = z3.Real('y')

        s = z3.Solver()

        s.push()
        s.add(x == -3)
        s.add(y == amnet.util.abs_z3(x))
        self.assertTrue(s.check() == z3.sat)

        # extract the output
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, x) == -3)
        self.assertTrue(amnet.util.mfp(model, y) == 3)
        s.pop()

        s.push()
        s.add(x == 4)
        s.add(y == amnet.util.abs_z3(x))
        self.assertTrue(s.check() == z3.sat)

        # extract the output
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, x) == 4)
        self.assertTrue(amnet.util.mfp(model, y) == 4)
        s.pop()

    def test_norm1_z3(self):
        x = z3.RealVector(prefix='x', sz=3)
        y = z3.Real('y')

        s = z3.Solver()
        s.add(y == amnet.util.norm1_z3(x))

        s.push()
        s.add([x[0] == 1, x[1] == 12, x[2] == -2])
        self.assertTrue(s.check() == z3.sat)
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, y) == abs(1) + abs(12) + abs(-2))
        s.pop()

        s.push()
        s.add([x[0] == -1, x[1] == 0, x[2] == 0])
        self.assertTrue(s.check() == z3.sat)
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, y) == abs(-1) + abs(0) + abs(0))
        s.pop()

    def test_norminf_z3(self):
        x = z3.RealVector(prefix='x', sz=3)
        y = z3.Real('y')

        s = z3.Solver()
        s.add(y == amnet.util.norminf_z3(x))

        s.push()
        s.add([x[0] == 1, x[1] == 12, x[2] == -2])
        self.assertTrue(s.check() == z3.sat)
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, y) == 12)
        s.pop()

        s.push()
        s.add([x[0] == -1, x[1] == 0, x[2] == 0])
        self.assertTrue(s.check() == z3.sat)
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, y) == 1)
        s.pop()

        s.push()
        s.add([x[0] == -1, x[1] == -11, x[2] == 0])
        self.assertTrue(s.check() == z3.sat)
        model = s.model()
        self.assertTrue(amnet.util.mfp(model, y) == 11)
        s.pop()

    def test_gaxpy_z3(self):
        m = 2
        n = 3
        A = np.array([[1, 2, -3], [4, -5, 6]])
        x = z3.RealVector(prefix='x', sz=n)
        y = np.array([7, -8])
        w0 = z3.RealVector(prefix='w0', sz=m)
        w1 = z3.RealVector(prefix='w1', sz=m)
        w0v = amnet.util.gaxpy_z3(A, x)
        w1v = amnet.util.gaxpy_z3(A, x, y)

        self.assertEqual(len(w0), m)
        self.assertEqual(len(w1), m)
        self.assertEqual(len(w0v), m)
        self.assertEqual(len(w1v), m)

        s = z3.Solver()
        s.add([w0[i] == w0v[i]
               for i in range(m)])
        s.add([w1[i] == w1v[i]
               for i in range(m)])

        s.push()
        xc = np.array([1, 2, 3])
        s.add([x[i] == xc[i] for i in range(n)])
        w0_true = np.dot(A, xc)
        w1_true = np.dot(A, xc) + y
        self.assertTrue(s.check() == z3.sat)

        model = s.model()
        for i in range(m):
            self.assertEqual(amnet.util.mfp(model, w0[i]), w0_true[i])
            self.assertEqual(amnet.util.mfp(model, w1[i]), w1_true[i])

        s.pop()

        s.push()
        xc = np.array([1, 0, -3])
        s.add([x[i] == xc[i] for i in range(n)])
        w0_true = np.dot(A, xc)
        w1_true = np.dot(A, xc) + y
        self.assertTrue(s.check() == z3.sat)

        model = s.model()
        for i in range(m):
            self.assertEqual(amnet.util.mfp(model, w0[i]), w0_true[i])
            self.assertEqual(amnet.util.mfp(model, w1[i]), w1_true[i])

        s.pop()



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtil)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
