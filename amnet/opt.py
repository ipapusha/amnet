import numpy as np
import amnet
from enum import Enum

################################################################################
# Classes and routines related to Multiplexing Programming with AMNs
################################################################################


############
# Objective
############

class Objective(object):
    """
    Class that keeps track of:
        flag to minimize or maximize
        real-valued Amn instance (ensures 1-dimensional output)
        unique variable input
    """
    def __init__(self, phi, minimize=True):
        assert phi.outdim == 1
        self.minimize = minimize
        self.phi = phi
        self.variable = amnet.tree.unique_leaf_of(phi)


class Minimize(Objective):
    def __init__(self, phi):
        super(Minimize, self).__init__(phi, minimize=True)


class Maximize(Objective):
    def __init__(self, phi):
        super(Maximize, self).__init__(phi, minimize=False)


##############
# Constraints
##############

class Relation(Enum):
    LT = 1  # <
    LE = 2  # <=
    GT = 3  # >
    GE = 4  # >=
    EQ = 5  # ==


class Constraint(object):
    """
    Class that keeps track of the lhs, rhs, the type of relation,
    and ensures dimensionality coherence between the lhs and rhs
    """
    def __init__(self, lhs, rhs, rel):
        assert lhs.outdim == rhs.outdim
        assert lhs.outdim >= 1
        assert rel in [Relation.LT, Relation.LE, Relation.GT, Relation.GE, Relation.EQ]
        self.lhs = lhs
        self.rhs = rhs
        self.rel = rel

        # cache input variable reference
        # XXX: possibly move this check into the problem creation routines
        lhs_variable = amnet.tree.unique_leaf_of(lhs)
        rhs_variable = amnet.tree.unique_leaf_of(rhs)
        assert lhs_variable is rhs_variable, 'LHS and RHS must depend on the same Variable'
        self.variable = lhs_variable


##########
# Problem
##########

class Result(object):
    def __init__(self, objval, optpoint):
        self.objval = objval
        self.optpoint = optpoint


class OptOptions(object):
    def __init__(self):
        # default options
        self.obj_lo = -float(2**20)
        self.obj_hi = float(2**20)
        self.fptol = float(2**-9)


class Problem(object):
    # Objective (or constant)
    # list of Constraints
    # solve()
    # single variable
    def __init__(self, objective, constraints=None, options=None):
        assert objective is not None
        self.objective = objective
        self.constraints = [] if constraints is None else constraints
        self.options = OptOptions() if options is None else options  # default options
