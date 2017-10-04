import numpy as np
from scipy.linalg import norm
import amnet

import copy

"""
Contains routines for manipulating and simplifying Amn trees
"""

FPTOL=1e-8

def simplify(phi):
    """
    Returns a new Amn that is equivalent to phi from the
    perspective of phi.eval(..), but potentially has
    * fewer nodes (e.g., fewer Mu's)
    * affine simplifications

    The affine simplifications are greedy, and may not be performed
    if the result is a higher-dimensional
    """

    # 1. only manipulate the copy
    phic = copy.deepcopy(phi)
    return phic

def eval_ones(phi):
    """
    evaluates phi on the all ones vector
    and returns the floating point answer
    """
    return phi.eval(np.ones(phi.indim))

def _simp_aff_aff(aff, force=False):
    """
    TODO: this does not work if the child of aff
    is the child of someone else
    """
    assert isinstance(aff, amnet.Affine)

    # whether an operation was performed
    simp = False

    # ensure we can do a simplification
    if not(isinstance(aff.x, amnet.Affine)):
        return simp
    assert isinstance(aff.x, amnet.Affine)

    # simplify if dimensions are reduced, or if forced to
    m1, n1 = aff.w.shape
    m2, n2 = aff.x.w.shape
    assert n1 == m2

    if force:
        simp = True
    elif m1*n2 <= ((m1 + n1) + (m2 + n2)):
        simp = True

    # before manipulation
    val_a = eval_ones(aff)

    if simp:
        w1 = aff.w
        b1 = aff.b
        w2 = aff.x.w
        b2 = aff.x.b2

        # compute new affine
        w3 = np.dot(w1, w2)
        b3 = np.dot(w1, b2) + b1

        # save grandchild
        assert isinstance(aff.x, amnet.Affine)
        x2 = aff.x.x

        # rewrite node
        aff.w = w3
        aff.b = b3
        aff.x = x2

    # after manipulation
    val_b = eval_ones(aff)
    assert norm(val_a - val_b) <= FPTOL

    return simp
