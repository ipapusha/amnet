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
        # pick a known stable linear system
        zeta = 0.2             # damping factor
        wn = 2.0*np.pi*10.0    # natural frequency
        h = 0.01               # sampling rate
        assert 2.0*np.pi/wn >= 2.0 * h, 'Nyquist is unhappy'

        Ac = np.array([[0, 1], [-wn*wn, -2*zeta*wn]])
        Ad = expm(h * Ac)
        assert all([abs(ev) < 1.0 for ev in eigvals(Ad)])

        print

        x = amnet.Variable(2, name='x')
        # phi = amnet.AffineTransformation(
        #
        # )

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLyap)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())