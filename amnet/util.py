from __future__ import division
import z3
from fractions import Fraction



def r2f(r):
    """ converts z3 rational to a python float,
        consider calling r = model[var].approx(20) to be within 1e-20
    """
    # return float(r.numerator_as_long())/float(r.denominator_as_long())

    #r2 = r.approx(10)
    #return float(r2.numerator_as_long())/float(r2.denominator_as_long())

    r2 = r.as_fraction().limit_denominator()
    return r2.numerator/r2.denominator


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
    
    Example:
        if arr == [1,2,3], then 
        foldl(add2, arr[0], arr[1:]) == add2(add2(1, 2), 3)
    """
    return z if len(xs) == 0 else foldl(f, f(z, xs[0]), xs[1:])


def foldr(f, z, xs):
    """
    Right fold, similar to Haskell's
    foldr f z []     = z 
    foldr f z (x:xs) = f x (foldr f z xs)
     
    Example:
        if arr == [1,2,3], then 
        foldr(add2, arr[-1], arr[:-1]) == add2(1, add2(2, 3))
    """
    return z if len(xs) == 0 else f(xs[0], foldr(f, z, xs[1:]))
