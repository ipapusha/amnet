import amnet
import z3
from itertools import izip, chain


class SmtEncoder(object):
    def __init__(self, phi, solver=None):
        self.symbols = dict()  # name str -> z3 variable
        self.phi = phi
        if solver is None:
            self.solver = z3.Solver()
        else:
            self.solver = solver

    def get_unique_varname(self, prefix='x'):
        assert len(prefix) >= 1
        assert prefix[0].isalpha()

        # all variables already used, with the given prefix
        existing = filter(lambda x: x.startswith(prefix),
                          self.symbols.keys())

        # find a unique suffix
        if len(existing) == 0:
            return prefix + '0'

        # TODO: can be more efficient by keeping track of max suffix state
        max_suffix = max(int(varname[len(prefix):]) for varname in existing)
        return prefix + str(max_suffix + 1)

    def add_new_symbol(self, name, target=None):
        assert name not in self.symbols
        self.symbols[name] = target

    def add_new_var(self, name, dim=1):
        assert dim >= 1
        self.add_new_symbol(name, z3.RealVector(name, dim))

    # helper methods for adding variables to each element type
    def _get_unique_outvarname(self, psi):
        if isinstance(psi, amnet.Variable):
            return self.get_unique_varname(prefix='xv')
        elif isinstance(psi, amnet.AffineTransformation):
            return self.get_unique_varname(prefix='ya')
        elif isinstance(psi, amnet.Mu):
            return self.get_unique_varname(prefix='wm')
        elif isinstance(psi, amnet.Constant):
            return self.get_unique_varname(prefix='c')
        elif isinstance(psi, amnet.Stack):
            return self.get_unique_varname(prefix='xys'),
        else:
            return self.get_unique_varname(prefix='aa')

    def _init_outvar(self, psi):
        """ only touches the specific node in the tree """
        name = self._get_unique_outvarname(psi)
        psi.outvar = name
        self.add_new_var(name, psi.outdim)
        psi.enc = self

        assert len(self.symbols[name]) == psi.outdim

    def _init_tree(self, psi):
        """ touches the whole tree """
        # 1. initialize the output variable
        self._init_outvar(psi)

        # 2. bind to the inputs by adding a constraint
        if isinstance(psi, amnet.Variable):
            # do not bind the input of a variable
            pass
        elif isinstance(psi, amnet.AffineTransformation):
            self._init_tree(psi.x)

            xv = self.symbols[psi.x.outvar]
            yv = self.symbols[psi.outvar]

            assert len(yv) == psi.outdim
            assert len(xv) >= 1

            for i in range(psi.outdim):
                rowi = psi.w[i,:]
                bi = psi.b[i]

                rowsum = z3.Sum([wij * xj for wij, xj in izip(rowi, xv) if wij != 0])
                if bi == 0:
                    self.solver.add(yv[i] == rowsum)
                else:
                    self.solver.add(yv[i] == rowsum + bi)

        elif isinstance(psi, amnet.Mu):
            self._init_tree(psi.x)
            self._init_tree(psi.y)
            self._init_tree(psi.z)

            xv = self.symbols[psi.x.outvar]
            yv = self.symbols[psi.y.outvar]
            zv = self.symbols[psi.z.outvar]
            wv = self.symbols[psi.outvar]

            assert len(zv) == 1
            assert len(xv) == len(yv)
            assert len(xv) == psi.outdim
            assert len(wv) == len(xv)

            z = wv[0]
            for i in range(len(wv)):
                x, y, w = xv[i], yv[i], wv[i]
                self.solver.add(w == z3.If(z <= 0, x, y))

        elif isinstance(psi, amnet.Constant):
            # convert z3 variable ('RealVec') to a z3 constant (list of 'RealVal')
            # and bind it to the constant value
            self.symbols[psi.outvar] = [z3.RealVal(ci) for ci in psi.c]
        elif isinstance(psi, amnet.Stack):
            self._init_tree(psi.x)
            self._init_tree(psi.y)

            xv = self.symbols[psi.x.outvar]
            yv = self.symbols[psi.y.outvar]
            xyv = self.symbols[psi.outvar]

            assert len(xv) + len(yv) == len(xyv)

            for left, right in izip(xyv, chain(xv, yv)):
                self.solver.add(left == right)
        else:
            return NotImplemented

    def init_tree(self):
        self._init_tree(self.phi)
