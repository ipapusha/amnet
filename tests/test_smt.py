import numpy as np
import amnet

import z3

import unittest


class TestSmt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_SmtEncoder(self):
        xyz = amnet.Variable(3, name='xyz')

        x = amnet.select(xyz, 0)
        y = amnet.select(xyz, 1)
        z = amnet.select(xyz, 2)
        w = amnet.Mu(x, y, z)

        enc = amnet.smt.SmtEncoder(w)

        # check naming
        name0 = enc.get_unique_varname(prefix='x')
        self.assertEqual(name0, 'x0')
        enc.add_new_symbol(name0)

        name1 = enc.get_unique_varname(prefix='x')
        self.assertEqual(name1, 'x1')
        enc.add_new_symbol(name1)

        name2 = enc.get_unique_varname(prefix='y')
        self.assertEqual(name2, 'y0')
        enc.add_new_symbol(name2)

        name3 = enc.get_unique_varname(prefix='y')
        self.assertEqual(name3, 'y1')
        enc.add_new_symbol(name3)

        # clear symbols
        enc.symbols = dict()
        enc.solver = z3.Solver()

        # now add some variables
        name = enc.get_unique_varname(prefix='x')
        enc.add_new_var(name)

        name = enc.get_unique_varname(prefix='x')
        enc.add_new_var(name, 2)

        self.assertTrue('x0' in enc.symbols)
        self.assertEqual(len(enc.symbols['x0']), 1)

        self.assertTrue('x1' in enc.symbols)
        self.assertEqual(len(enc.symbols['x1']), 2)

    def test_SmtEncoder2(self):
        xyz = amnet.Variable(3, name='xyz')

        x = amnet.select(xyz, 0)
        y = amnet.select(xyz, 1)
        z = amnet.select(xyz, 2)
        w = amnet.Mu(x, y, z)

        # do the smt encoding
        enc = amnet.smt.SmtEncoder(w)
        enc.init_tree()

        self.assertTrue(enc.solver.check())

        # bind the variables to an input
        enc.set_const(xyz, np.array([1, 2, 3]))
        self.assertTrue(enc.solver.check())
        model = enc.solver.model()

        #print model

        out_z3_list = enc.symbols[w.outvar]
        out_z3_val = model[out_z3_list[0]]
        self.assertEqual(out_z3_val, 2)

    def test_SmtEncoder3(self):
        xyz = amnet.Variable(3, name='xyz')

        x = amnet.select(xyz, 0)
        y = amnet.select(xyz, 1)
        z = amnet.select(xyz, 2)
        w = amnet.Mu(x, y, z)

        # do the smt encoding
        enc = amnet.smt.SmtEncoder(w)
        enc.init_tree()

        self.assertTrue(enc.solver.check())

        # bind the variables to an input
        enc.set_const(xyz, np.array([1, 2, -3.1]))
        self.assertTrue(enc.solver.check())
        model = enc.solver.model()

        # print model

        out_z3_list = enc.symbols[w.outvar]
        out_z3_val = model[out_z3_list[0]]
        self.assertEqual(out_z3_val, 1)

    def test_SmtEncoder3(self):
        xyz = amnet.Variable(3, name='xyz')

        x = amnet.select(xyz, 0)
        y = amnet.select(xyz, 1)
        z = amnet.select(xyz, 2)
        w = amnet.Mu(x, y, z)

        # do the smt encoding
        enc = amnet.smt.SmtEncoder(w)
        enc.init_tree()

        self.assertTrue(enc.solver.check())

        # bind the variables to an input
        enc.set_const(xyz, np.array([1.2, 2, -3.1]))
        self.assertTrue(enc.solver.check())
        model = enc.solver.model()

        # print model

        out_z3_list = enc.symbols[w.outvar]
        out_z3_val = model[out_z3_list[0]]
        self.assertEqual(out_z3_val, 1.2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmt)
    unittest.TextTestRunner(verbosity=2).run(suite)
