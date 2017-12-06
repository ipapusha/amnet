from __future__ import division

import numpy as np
import scipy as sp
from scipy.linalg import expm
from numpy.linalg import eigvals, norm

import amnet

import sys
import unittest

import z3


class TestLyap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # generate a known stable linear system
        n = 2
        zeta = 0.2  # damping factor
        wn = 2.0 * np.pi * 10.0  # natural frequency
        h = 0.01  # sampling rate
        assert 2.0 * np.pi / wn >= 2.0 * h, 'Nyquist is unhappy'

        # damped oscillator
        cls.Ac_osc = np.array([[0, 1], [-wn*wn, -2*zeta*wn]])
        cls.Ad_osc = expm(h * cls.Ac_osc)

        # known Metzler system
        cls.Ac_met = np.array([[-15, 12], [4, -25]])
        cls.Ad_met = expm(h * cls.Ac_met)

        # diagonal system
        cls.Ac_diag = np.array([[-20, 0], [0, -25]])
        cls.Ad_diag = expm(h * cls.Ac_diag)

        # check that generated examples are already stable
        for Ad in [cls.Ad_osc, cls.Ad_met, cls.Ad_diag]:
            assert all([abs(ev) < 0.9 for ev in eigvals(Ad)])

        # set up global z3 parameters
        # parameters from https://stackoverflow.com/a/12516269
        z3.set_param('auto_config', False)
        z3.set_param('smt.case_split', 5)
        z3.set_param('smt.relevancy', 2)

    def test_verify_forward_invariance(self):
        # generate a known positive system
        n = 2
        A = np.array([[1, 2], [3, 4]])

        # dynamics Amn
        x = amnet.Variable(2, name='x')
        f = amnet.Linear(A, x)

        # nonnegative orthant Amn
        # (V(x) <= 0) iff (-min_i x_i <= 0)
        V = amnet.atoms.negate(
            amnet.atoms.min_all(
                x
            )
        )

        # should be forward invariant
        result = amnet.lyap.verify_forward_invariance(f, V)
        self.assertEqual(result.code, amnet.lyap.VerificationResult.SUCCESS)

        # generate known *not* forward invariant system
        A2 = np.array([[-1, 2], [3, 4]])
        f2 = amnet.Linear(A2, x)

        # should get counterexample
        result = amnet.lyap.verify_forward_invariance(f2, V)
        self.assertEqual(result.code, amnet.lyap.VerificationResult.FAIL_WITH_COUNTEREXAMPLE)
        xv = result.x
        xvp = f2.eval(result.x)

        self.assertAlmostEqual(norm(xvp - result.xp), 0)
        self.assertLessEqual(V.eval(xv)[0], 0) # start in S
        self.assertGreater(V.eval(xvp)[0], 0)  # go outside S


    def donot_test_stability_search1(self):
        #Ad = self.Ad_osc
        Ad = self.Ad_met
        (n, _) = Ad.shape
        assert n == 2

        xsys = amnet.Variable(n, name='xsys')
        phi = amnet.Affine(
            Ad,
            xsys,
            np.zeros(n)
        )

        # look for a Lyapunov function
        #amnet.lyap.stability_search1(phi, xsys, 10)
        amnet.lyap.stability_search1(phi, xsys, 4)  # enough for Metzler sys


    def donot_test_disprove_maxaff_local_lyapunov(self):
        # a simple system
        Ad = self.Ad_diag
        (n, _) = Ad.shape
        assert n == 2

        xsys = amnet.Variable(n, name='xsys')
        phi = amnet.Affine(
            Ad,
            xsys,
            np.zeros(n)
        )

        # a simple Lyapunov function
        Astar = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        bstar = np.zeros(4)
        xc = amnet.lyap.find_local_counterexample(phi, xsys, Astar, bstar)

        # simple Lyapunov function works
        self.assertTrue(xc is None)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLyap)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
