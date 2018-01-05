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
        assert rel in [Relation.LT, Relation.LE, Relation.GT, Relation.GE, Relation.EQ]
        self.lhs = lhs
        self.rhs = rhs
        self.rel = rel



##########
# Problem
##########

class Problem(object):
    # Objective (or constant)
    # list of Constraints
    # solve()
    # single variable
    pass