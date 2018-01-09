from __future__ import division

from itertools import izip
from enum import Enum
from fractions import Fraction

import numpy as np
import z3


################################################################################
# useful Enums
################################################################################
class Relation(Enum):
    LT = 1  # <
    LE = 2  # <=
    GT = 3  # >
    GE = 4  # >=
    EQ = 5  # ==
    NE = 6  # !=


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


def mfpv(model, var_vector):
    """ vector version of mfp """
    return np.array([mfp(model, v) for v in var_vector])


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

def eqv_z3(solver, var, arr):
    """adds the constraints {var[i] == arr[i], i = 1..n} to solver """
    assert len(var) == len(arr)
    for v_i, arr_i in izip(var, arr):
        solver.add(v_i == arr_i)


def neqv_z3(solver, var, arr):
    """adds the constraints {var[i] != arr[i], i = 1..n} to solver """
    assert len(var) == len(arr)
    for v_i, arr_i in izip(var, arr):
        solver.add(v_i != arr_i)

def leqv_z3(solver, var, arr):
    """adds the constraints {var[i] <= arr[i], i = 1..n} to solver """
    assert len(var) == len(arr)
    for v_i, arr_i in izip(var, arr):
        solver.add(v_i <= arr_i)


def ltv_z3(solver, var, arr):
    """adds the constraints {var[i] < arr[i], i = 1..n} to solver """
    assert len(var) == len(arr)
    for v_i, arr_i in izip(var, arr):
        solver.add(v_i < arr_i)


def geqv_z3(solver, var, arr):
    """adds the constraints {var[i] >= arr[i], i = 1..n} to solver """
    assert len(var) == len(arr)
    for v_i, arr_i in izip(var, arr):
        solver.add(v_i >= arr_i)


def gtv_z3(solver, var, arr):
    """adds the constraints {var[i] > arr[i], i = 1..n} to solver """
    assert len(var) == len(arr)
    for v_i, arr_i in izip(var, arr):
        solver.add(v_i > arr_i)


def relv_z3(solver, var, arr, rel):
    """
    adds the constraints {var[i] rel arr[i], i = 1..n} to solver,
    where rel is one of the instances of Relation
    """
    assert len(var) == len(arr)
    assert rel in [Relation.LT, Relation.LE, Relation.GT, Relation.GE, Relation.EQ, Relation.NE]

    if rel == Relation.LT:
        ltv_z3(solver, var, arr)
    elif rel == Relation.LE:
        leqv_z3(solver, var, arr)
    elif rel == Relation.GT:
        gtv_z3(solver, var, arr)
    elif rel == Relation.GE:
        geqv_z3(solver, var, arr)
    elif rel == Relation.EQ:
        eqv_z3(solver, var, arr)
    elif rel == Relation.NE:
        neqv_z3(solver, var, arr)
    else:
        assert False, 'Impossible relation'


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


def norm1_z3(xs):
    assert is_nonempty_vector_z3(xs)
    return z3.Sum([abs_z3(x) for x in xs])


def norminf_z3(xs):
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
    if ys is not None:
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

        output[i] = rowsum + ys[i] if ys is not None else rowsum

    return output