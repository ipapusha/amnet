import numpy as np
import amnet
from amnet.tree import descendants

################################################################################
# Classes and routines related to Multiplexing Programming with AMNs
################################################################################

############
# Objective
############

class SingleVariableAmn(amnet.Amn):
    """ An expression is an Amn that keeps track of its input variable """
    pass


class Objective(SingleVariableAmn):
    # flag to minimize or maximize
    # real-valued Amn instance (ensures 1-dimensional output)
    pass


class Minimize(Objective):
    pass


class Maximize(Objective):
    pass


##############
# Constraints
##############
class Constraint(object):
    # lhs, rhs, type of relation
    # (ensures dimensionality coherence between lhs and rhs)
    pass


##########
# Problem
##########
class Problem(object):
    # Objective (or constant)
    # list of Constraints
    # solve()
    # single variable
    pass