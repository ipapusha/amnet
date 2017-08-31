import amnet
import z3


class SmtEncoder(object):
    def __init__(self, phi):
        self.symbols = dict()  # name str -> z3 variable
        self.phi = phi

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


