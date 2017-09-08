from __future__ import division
import z3


def rat2float(r):
    """ converts z3 rational to a python float,
        consider calling r = model[var].approx(20) to be within 1e-20
    """
    return float(r.numerator_as_long())/float(r.denominator_as_long())


def foldl(f, z, xs):
    """
    Left fold (not lazy), similar to to Haskell's
    foldl f z []     = z
    foldl f z (x:xs) = foldl f (f z x) xs
    """
    return z if len(xs) == 0 else foldl(f, f(z, xs[0]), xs[1:])