from __future__ import division
import z3
from itertools import izip
from fractions import Fraction


################################################################################
# general utilities
################################################################################

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


def allsame(xs):
    """
    True if xs is the empty list or list with one element
    True if all the components of xs are the same
    False otherwise
    """
    if len(xs) <= 1:
        return True

    return all([x == xs[0] for x in xs[1:]])


################################################################################
# useful functions for z3 variables
################################################################################

def is_nonempty_vector_z3(xs):
    return (len(xs) >= 1) and all([z3.is_expr(x) for x in xs])


def max2_z3(x, y):
    assert z3.is_expr(x)
    assert z3.is_expr(y)
    return z3.If(x <= y, y, x)


def min2_z3(x, y):
    assert z3.is_expr(x)
    assert z3.is_expr(y)
    return z3.If(x <= y, x, y)


def maxN_z3(xs):
    assert is_nonempty_vector_z3(xs)
    return foldl(max2_z3, xs[0], xs[1:])


def minN_z3(xs):
    assert is_nonempty_vector_z3(xs)
    return foldl(min2_z3, xs[0], xs[1:])


def abs_z3(x):
    assert z3.is_expr(x)
    return max2_z3(x, -x)


def normL1_z3(xs):
    assert is_nonempty_vector_z3(xs)
    return z3.Sum([abs_z3(x) for x in xs])


def normLinf_z3(xs):
    assert is_nonempty_vector_z3(xs)
    return maxN_z3([abs_z3(x) for x in xs])


# matrix-vector operations
def gaxpy_z3(A, xs, ys=None, skip_zeros=True):
    assert is_nonempty_vector_z3(xs)

    # extract dimensions
    n = len(xs)
    m = len(A)

    # do dimension-checking
    assert m >= 1
    assert n >= 1
    if ys:
        assert len(ys) == m
    assert all([len(row) == n for row in A])

    # encode matrix-multiply
    output = [None] * m

    for i in range(m):
        assert len(A[i]) == n
        if skip_zeros:
            rowsum = z3.Sum([Aij * xj
                             for Aij, xj in izip(A[i], xs)
                             if Aij != 0])
        else:
            rowsum = z3.Sum([Aij * xj
                             for Aij, xj in izip(A[i], xs)])

        output[i] = rowsum + ys[i] if ys else rowsum