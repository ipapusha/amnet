import numpy as np
import amnet
import amnet.vis
import amnet.tree

from numpy.linalg import norm

import sys
import unittest
import itertools

VISUALIZE = True

class TestTree(unittest.TestCase):
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

    def test_tree_descendants_dag(self):
        xyz = amnet.Variable(3, name='xyz')
        x = amnet.atoms.select(xyz, 0)
        yz = amnet.Linear(
            np.array([[0, 1, 0], [0, 0, 1]]),
            xyz
        )

        maxyz = amnet.atoms.max_all(yz)
        twoxp1 = amnet.Affine(
            np.array([[2]]),
            x,
            np.array([1])
        )
        twox = amnet.atoms.add2(x, x)
        threex = amnet.atoms.add2(x, twox)
        fivexp1 = amnet.atoms.add2(twoxp1, threex)
        phi = amnet.atoms.add2(fivexp1, maxyz)

        # visualize dag
        if VISUALIZE: amnet.vis.quick_vis(phi, title='tree_dag')

        # simple membership tests
        # NOTE: May need to convert to any(x is d for d in desc)
        desc = amnet.tree.descendants(phi)
        # number of descendents may change
        # depending on implementation of max
        # self.assertEqual(len(desc), 17)
        self.assertGreaterEqual(len(desc), 9)
        self.assertTrue(xyz in desc)
        self.assertTrue(x in desc)
        self.assertTrue(yz in desc)
        self.assertTrue(maxyz in desc)
        self.assertTrue(twoxp1 in desc)
        self.assertTrue(twox in desc)
        self.assertTrue(threex in desc)
        self.assertTrue(fivexp1 in desc)
        self.assertTrue(phi in desc)

        # descendants list must contain unique pointers
        self.assertEqual(
            len(desc),
            len(set(id(d) for d in desc))
        )

        # descendants list has exactly one variable
        self.assertEqual(
            len([d for d in desc if isinstance(d, amnet.Variable)]),
            1
        )

        # determine that variable
        v = amnet.tree.unique_leaf_of(phi)
        self.assertTrue(v is xyz)
        self.assertTrue(isinstance(v, amnet.Variable))
        self.assertEqual(v.outdim, 3)
        self.assertEqual(v.indim, 3)

        # determine the other variables
        for psi in [xyz, x, yz, maxyz, twoxp1, twox, threex, fivexp1, phi]:
            v = amnet.tree.unique_leaf_of(psi)
            self.assertTrue(v is xyz)
            self.assertTrue(isinstance(v, amnet.Variable))
            self.assertEqual(v.outdim, 3)
            self.assertEqual(v.indim, 3)

        # this dag has only one mu
        self.assertEqual(
            len([d for d in desc if isinstance(d, amnet.Mu)]),
            1
        )

        # true dag represents 5*x + 1 + max(y, z)
        # so it should have at least 3 + 1 = 4 (x, y, z + outside add)
        # affine transformations
        self.assertGreaterEqual(
            len([d for d in desc if isinstance(d, amnet.Affine)]),
            4
        )

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTree)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
