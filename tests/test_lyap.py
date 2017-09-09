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
        pass

    def test_stability_search1(self):
        # generate a known stable linear system
        n = 2
        zeta = 0.2             # damping factor
        wn = 2.0*np.pi*10.0    # natural frequency
        h = 0.01               # sampling rate
        assert 2.0*np.pi/wn >= 2.0 * h, 'Nyquist is unhappy'

        # damped oscillator
        #Ac = np.array([[0, 1], [-wn*wn, -2*zeta*wn]])
        #Ad = expm(h * Ac)

        # known Metzler system
        Ac = np.array([[-15, 12], [4, -25]])
        Ad = expm(h * Ac)

        assert all([abs(ev) < 0.9 for ev in eigvals(Ad)])

        xsys = amnet.Variable(2, name='xsys')
        phi = amnet.AffineTransformation(
            Ad,
            xsys,
            np.zeros(n)
        )

        # look for a Lyapunov function
        #amnet.lyap.stability_search1(phi, xsys, 10)
        amnet.lyap.stability_search1(phi, xsys, 4)  # enough for Metzler sys

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLyap)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())