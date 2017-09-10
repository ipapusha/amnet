from __future__ import division

import numpy as np
import scipy as sp
from scipy.linalg import expm
from numpy.linalg import eigvals

import amnet

import sys
import unittest


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

    def stability_search1(self):
        Ad = self.Ad_osc
        (n, _) = Ad.shape
        assert n == 2

        xsys = amnet.Variable(n, name='xsys')
        phi = amnet.AffineTransformation(
            Ad,
            xsys,
            np.zeros(n)
        )

        # look for a Lyapunov function
        #amnet.lyap.stability_search1(phi, xsys, 10)
        amnet.lyap.stability_search1(phi, xsys, 4)  # enough for Metzler sys

    def cvxpy(self):
        pass

    def test_disprove_maxaff_local_lyapunov(self):
        # a simple system
        Ad = self.Ad_diag
        (n, _) = Ad.shape
        assert n == 2

        xsys = amnet.Variable(n, name='xsys')
        phi = amnet.AffineTransformation(
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