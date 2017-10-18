import numpy as np
import amnet
import amnet.vis
from amnet import tf_utils

import tensorflow as tf

import z3

from numpy.linalg import norm

import sys
import unittest
import itertools

VISUALIZE = True

class TestSmt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print 'Setting up test floats.'
        cls.floatvals = np.linspace(-5., 5., 2)
        cls.FPTOL = 1e-8

        # set up global z3 parameters
        # parameters from https://stackoverflow.com/a/12516269
        #z3.set_param('auto_config', False)
        #z3.set_param('smt.case_split', 5)
        #z3.set_param('smt.relevancy', 2)

    def validate_outputs(self, phi, onvals, true_f=None, verbose=False):
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

            if verbose:
                print 'inp:', fpval
                print 'fpeval: ', fpeval

            # compare to true floating point function, if it's provided
            if true_f is not None:
                true_eval = true_f(np.array(fpval).reshape(1, fpval.shape[0]))
                if verbose: print 'true_eval: ', true_eval
                self.assertAlmostEqual(norm(true_eval - fpeval), 0, delta=1e-6)

            # set the z3 input
            enc.solver.push()
            for i in range(len(invar)):
                enc.solver.add(invar[i] == fpval[i])

            # run z3 to check for satisfiability
            result = enc.solver.check()
            #if verbose: print enc.solver
            self.assertTrue(result == z3.sat)

            # extract the output
            model = enc.solver.model()
            smteval = np.zeros(len(outvar))
            for i in range(len(outvar)):
                smteval[i] = amnet.util.mfp(model, outvar[i])

            # check that the outputs match
            if verbose: print 'smteval: ', smteval
            self.assertAlmostEqual(norm(smteval - fpeval), 0, delta=1e-5)

            enc.solver.pop()


    def test_relu_nn(self):

        # create input placeholder
        x = tf.placeholder(tf.float32, [1, 2])

        w1 = tf.Variable(tf.random_normal([2, 3], stddev=0.03), name='w1')
        b1 = tf.Variable(tf.random_normal([3]), name='b1')
        # weights connecting the hidden layer to the output layer
        w2 = tf.Variable(tf.random_normal([3, 1], stddev=0.03), name='w2')
        b2 = tf.Variable(tf.random_normal([1]), name='b2')

        # variable initialization call
        init_op = tf.global_variables_initializer()

        # calculate the output of the hidden layer
        hidden_out = tf.add(tf.matmul(x, w1), b1)
        hidden_out = tf.nn.relu(hidden_out)

        # full network output
        y = tf.add(tf.matmul(hidden_out, w2), b2)

        with tf.Session() as sess:

            # tensorflow output
            def real_relu_nn(arg):
                return sess.run(y, feed_dict={x: arg})

            # initialize variables
            sess.run(init_op)
            # grab weights and biases
            weights, biases = tf_utils.get_vars(tf.trainable_variables(), sess)
            # construct amnet
            relu_nn = tf_utils.relu_amn(weights, biases)

            self.validate_outputs(
                phi=relu_nn,
                onvals=itertools.product(self.floatvals, repeat=relu_nn.indim),
                true_f=real_relu_nn,
            )
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmt)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
