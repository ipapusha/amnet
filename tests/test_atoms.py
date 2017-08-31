import numpy as np
import amnet

import unittest
from itertools import chain


class TestAtoms(unittest.TestCase):
    def test_make_relu1(self):
        # define one-dimensional input variable
        x = amnet.Variable(1, name='x')
        phi_relu = amnet.atoms.make_relu1(x)

        # true relu
        def relu(x): return x if x > 0 else 0

        # implemented relu
        for xv in chain(np.linspace(-5.0, 5.0, 11), np.linspace(-5.0, 5.0, 10)):
            inp = np.array([xv])
            self.assertEqual(phi_relu.eval(inp), relu(inp))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAtoms)
    unittest.TextTestRunner(verbosity=2).run(suite)
