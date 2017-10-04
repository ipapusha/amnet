import numpy as np
import amnet
import amnet.vis

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

    def test_SmtEncoder_max_all_2(self):
        xy = amnet.Variable(2, name='xy')
        phi_max2 = amnet.atoms.max_all(xy)
        self.assertEqual(phi_max2.indim, 2)

        def true_max2(fpin):
            x, y = fpin
            return max(x, y)

        self.validate_outputs(
            phi=phi_max2,
            onvals=itertools.product(self.floatvals, repeat=phi_max2.indim),
            true_f=true_max2
        )

    def test_SmtEncoder_min_all_2(self):
        xy = amnet.Variable(2, name='xy')
        phi_min2 = amnet.atoms.min_all(xy)
        self.assertEqual(phi_min2.indim, 2)

        def true_min2(fpin):
            x, y = fpin
            return min(x, y)

        self.validate_outputs(
            phi=phi_min2,
            onvals=itertools.product(self.floatvals, repeat=phi_min2.indim),
            true_f=true_min2
        )

    def test_SmtEncoder_max_all_3_small(self):
        xyz = amnet.Variable(3, name='xy')
        phi_max3 = amnet.atoms.max_all(xyz)
        self.assertEqual(phi_max3.indim, 3)

        def true_max3(fpin):
            x, y, z = fpin
            return max(x, y, z)

        self.validate_outputs(
            phi=phi_max3,
            onvals=itertools.product(self.floatvals2, repeat=phi_max3.indim),
            true_f=true_max3
        )

    def test_SmtEncoder_min_all_3_small(self):
        xyz = amnet.Variable(3, name='xy')
        phi_min3 = amnet.atoms.min_all(xyz)
        self.assertEqual(phi_min3.indim, 3)

        def true_min3(fpin):
            x, y, z = fpin
            return min(x, y, z)

        self.validate_outputs(
            phi=phi_min3,
            onvals=itertools.product(self.floatvals2, repeat=phi_min3.indim),
            true_f=true_min3
        )

    def test_SmtEncoder_triplexer(self):
        np.random.seed(1)

        TOTAL_RUNS=5

        #print ""
        for iter in range(TOTAL_RUNS):
            #print "Testing random triplexer [%d/%d]..." % (iter+1, TOTAL_RUNS),

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
            #print "done!"

    def test_SmtEncoder_dag(self):
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

        def true_dag(fpin):
            x, y, z = fpin
            return 5*x + 1 + max(y, z)

        self.validate_outputs(
            phi=phi,
            onvals=itertools.product(self.floatvals2, repeat=3),
            true_f=true_dag
        )

        # visualize dag
        #dot = amnet.vis.amn2gv(phi, ctx=None, title='dag')
        #dot.render(filename='dag.gv', directory='vis')

    def test_SmtEncoder_relu_1(self):
        x = amnet.Variable(1, name='x')
        y = amnet.atoms.relu(x)

        def true_relu(fpin):
            return max(fpin[0], 0)

        self.validate_outputs(
            phi=y,
            onvals=itertools.product(self.floatvals, repeat=y.indim),
            true_f=true_relu
        )

    def test_SmtEncoder_relu_2(self):
        x = amnet.Variable(3, name='x')
        y = amnet.atoms.relu(x)

        def true_relu(fpin):
            return np.maximum(fpin, 0)

        self.validate_outputs(
            phi=y,
            onvals=itertools.product(self.floatvals2, repeat=y.indim),
            true_f=true_relu
        )

    def test_SmtEncoder_gates(self):
        xy_z1z2 = amnet.Variable(2+2+1+1, name='xyz1z2')
        x = amnet.Linear(
            np.eye(2, 6, 0),
            xy_z1z2
        )
        y = amnet.Linear(
            np.eye(2, 6, 2),
            xy_z1z2
        )
        z1 = amnet.atoms.select(xy_z1z2, 4)
        z2 = amnet.atoms.select(xy_z1z2, 5)

        phi_and = amnet.atoms.gate_and(x, y, z1, z2)
        phi_or = amnet.atoms.gate_or(x, y, z1, z2)
        phi_xor = amnet.atoms.gate_xor(x, y, z1, z2)
        phi_not = amnet.atoms.gate_not(x, y, z1)

        # check dimensions
        self.assertEqual(xy_z1z2.outdim, 6)
        self.assertEqual(x.outdim, 2)
        self.assertEqual(y.outdim, 2)
        self.assertEqual(z1.outdim, 1)
        self.assertEqual(z2.outdim, 1)

        # true gate functions
        def true_and(fpin):
            return fpin[0:2] if (fpin[4] <= 0 and fpin[5] <= 0) else fpin[2:4]

        def true_or(fpin):
            return fpin[0:2] if (fpin[4] <= 0 or fpin[5] <= 0) else fpin[2:4]

        def true_xor(fpin):
            return fpin[0:2] if ((fpin[4] <= 0) != (fpin[5] <= 0)) else fpin[2:4]

        def true_not(fpin): # ignores last input
            return fpin[2:4] if (fpin[4] <= 0) else fpin[0:2]

        # evaluate
        vals = np.array([1, -2, -3, 4])
        sels = itertools.product([-1, 0, 1], repeat=2)
        onvals = [np.concatenate((vals, sel), axis=0) for sel in sels]

        self.validate_outputs(phi=phi_and, onvals=onvals, true_f=true_and)
        self.validate_outputs(phi=phi_or, onvals=onvals, true_f=true_or)
        self.validate_outputs(phi=phi_xor, onvals=onvals, true_f=true_xor)
        self.validate_outputs(phi=phi_not, onvals=onvals, true_f=true_not)



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmt)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
