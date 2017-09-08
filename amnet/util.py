from __future__ import division
import z3


def r2f(r):
    """ converts z3 rational to a python float,
        consider calling r = model[var].approx(20) to be within 1e-20
    """
    return float(r.numerator_as_long())/float(r.denominator_as_long())


def mfp(model, var):
    """ returns floating point representation of z3 variable var in model,
        or 0.0 if var is not in model
    """
    return r2f(model.eval(var, model_completion=True))


def foldl(f, z, xs):
    """
    Left fold (not lazy), similar to to Haskell's
    foldl f z []     = z
    foldl f z (x:xs) = foldl f (f z x) xs
    """
    return z if len(xs) == 0 else foldl(f, f(z, xs[0]), xs[1:])