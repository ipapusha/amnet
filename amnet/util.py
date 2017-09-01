from __future__ import division
import z3


def rat2float(r):
    """ converts z3 rational to a python float,
        consider calling r = model[var].approx(20) to be within 1e-20
    """
    return float(r.numerator_as_long())/float(r.denominator_as_long())