import numpy as np
import amnet
from amnet.util import r2f

import z3

from numpy.linalg import norm

import sys
import unittest
import itertools

class TestSmt(unittest.TestCase):
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
        cls.FPTOL = 1e-8

    def validate_outputs(self, phi, onvals, true_f=None):
        # encode phi using default context and solver
        enc = amnet.smt.SmtEncoder(phi=phi, solver=None)

        # tap the input and output vars
        invar = enc.var_of_input()
        outvar = enc.var_of(phi)

        # check dimensions
        self.assertEqual(phi.indim, len(invar))
        self.assertEqual(phi.outdim, len(outvar))

        # go through inputs
        for val in onvals:
            # get a new value
            fpval = np.array(val)
            self.assertEqual(len(fpval), phi.indim)

            # evaluate using the Amn tree
            fpeval = phi.eval(fpval)
            self.assertEqual(len(fpeval), phi.outdim)

            # compare to true floating point function, if it's provided
            if true_f is not None:
                true_eval = true_f(fpval)
                self.assertAlmostEqual(norm(true_eval - fpeval), 0)

            # set the z3 input
            enc.solver.push()
            for i in range(len(invar)):
                enc.solver.add(invar[i] == fpval[i])

            # run z3 to check for satisfiability
            result = enc.solver.check()
            self.assertTrue(result == z3.sat)

            # extract the output
            model = enc.solver.model()
            smteval = np.zeros(len(outvar))
            for i in range(len(outvar)):
                smteval[i] = amnet.util.mfp(model, outvar[i])

            # check that the outputs match
            self.assertAlmostEqual(norm(smteval - fpeval), 0)

            enc.solver.pop()

    def donot_test_SmtEncoder_mu_big(self):
        xyz = amnet.Variable(3, name='xyz')

        x = amnet.atoms.select(xyz, 0)
        y = amnet.atoms.select(xyz, 1)
        z = amnet.atoms.select(xyz, 2)
        w = amnet.Mu(x, y, z)

        def true_mu(fpin):
            x, y, z = fpin
            return x if z <= 0 else y

        self.validate_outputs(
            phi=w,
            onvals=itertools.product(self.floatvals, repeat=w.indim),
            true_f=true_mu
        )

    def test_SmtEncoder_mu_small(self):
        xyz = amnet.Variable(3, name='xyz')

        x = amnet.atoms.select(xyz, 0)
        y = amnet.atoms.select(xyz, 1)
        z = amnet.atoms.select(xyz, 2)
        w = amnet.Mu(x, y, z)

        def true_mu(fpin):
            x, y, z = fpin
            return x if z <= 0 else y

        self.validate_outputs(
            phi=w,
            onvals=itertools.product(self.floatvals2, repeat=w.indim),
            true_f=true_mu
        )

    def test_SmtEncoder_triplexer(self):
        np.random.seed(1)

        TOTAL_RUNS=10

        for iter in range(TOTAL_RUNS):
            print "Testing random triplexer [%d/%d]..." % (iter+1, TOTAL_RUNS),
            # create a random triplexer
            x = amnet.Variable(1, name='x')

            a = 3 * (2 * np.random.rand(4) - 1)
            b = 3 * (2 * np.random.rand(4) - 1)
            c = 3 * (2 * np.random.rand(4) - 1)
            d = 3 * (2 * np.random.rand(4) - 1)
            e = 3 * (2 * np.random.rand(4) - 1)
            f = 3 * (2 * np.random.rand(4) - 1)
            phi_tri = amnet.atoms.triplexer(x, a, b, c, d, e, f)

            def true_tri(fpin):
                return amnet.atoms.fp_triplexer(fpin, a, b, c, d, e, f)

            xvals = 50 * (2 * np.random.rand(100) - 1)
            onvals = itertools.product(xvals, repeat=1)

            self.validate_outputs(
                phi=phi_tri,
                onvals=onvals,
                true_f=true_tri
            )
            print "done!"


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmt)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
