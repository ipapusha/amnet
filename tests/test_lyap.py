import numpy as np
import amnet

import sys
import unittest


class TestLyap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_stability_search1(self):
        print 'Testing stability_search1'

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLyap)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())