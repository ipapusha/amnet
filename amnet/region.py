import numpy as np
import amnet
from amnet.atoms import norm1, norminf, sub2, min_list, max_list


################################################################################
# Classes for dealing with regions as sublevels of AMNs
################################################################################

class AmnRegion(object):
    """
    Keeps track of an AMN phi,
    which corresponds to the 0-sublevel region
    {x | phi(x) <= 0}
    """
    def __init__(self, phi):
        assert phi.outdim == 1
        self.phi = phi

    def __str__(self):
        return 'Region(phi=%s)' % (str(self.phi))

    def contains_point(self, inp):
        """returns True if phi(inp) <= 0"""
        assert len(inp) == self.phi.indim
        xv = self.phi.eval(inp)
        assert len(xv) == 1
        return xv[0] <= 0


################################################################################
# Convenience methods
################################################################################

def ball_norm1(x, alpha=1.0):
    """
    Returns the region {x | norm1(x) <= alpha}.
    Must pass in a variable ref x.
    """
    con = amnet.Constant(x, np.array([alpha]))
    phi = sub2(
        norm1(x),
        con
    )
    return AmnRegion(phi)


def ball_norminf(x, alpha=1.0):
    """
    Returns the region {x | norminf(x) <= alpha}.
    Must pass in a variable ref x.
    """
    con = amnet.Constant(x, np.array([alpha]))
    phi = sub2(
        norminf(x),
        con
    )
    return AmnRegion(phi)


def global_space(x):
    """
    Returns the region {x | -1 <= 0}, i.e., all of R^n.
    Must pass in a variable ref x.
    """
    phi = amnet.Constant(x, np.array([-1]))
    return AmnRegion(phi)


def union_from(reg_list):
    assert len(reg_list) >= 1
    phi = min_list([reg.phi for reg in reg_list])
    return AmnRegion(phi)


def intersection_from(reg_list):
    assert len(reg_list) >= 1
    phi = max_list([reg.phi for reg in reg_list])
    return AmnRegion(phi)
