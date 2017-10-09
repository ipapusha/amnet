import numpy as np
from scipy.linalg import norm
import amnet

from collections import deque

"""
Contains routines for manipulating and simplifying Amn trees
"""


def compose_rewire(phi1, phi2):
    """
    Given two AMNs and pointers to their input variables,
    creates a new AMN that is a composition of the two.

    Given:
        phi1(x1)
        phi2(x2)
    Returns:
        phi(x2) = phi1(phi2(x2))

    Note: the variable of the new AMN is x2
    """
    assert phi1.indim == phi2.outdim

    # base case: variable node
    # throw away node
    if isinstance(phi1, amnet.Variable):
        return

    # base case: already did the rewire
    if phi1 == phi2:
        return

    # depth-first: somewhere in the middle
    # 1. recurse down
    if hasattr(phi1, 'x'):
        compose_rewire(phi1.x, phi2)
    if hasattr(phi1, 'y'):
        compose_rewire(phi1.y, phi2)
    if hasattr(phi1, 'z'):
        compose_rewire(phi1.z, phi2)

    # 2. change the indim
    phi1.indim = phi2.outdim

    # 3. if child child is a variable, rewire it
    # rewire phi1's variable to point to phi2
    if hasattr(phi1, 'x') and isinstance(phi1.x, amnet.Variable):
        phi1.x = phi2
    if hasattr(phi1, 'y') and isinstance(phi1.y, amnet.Variable):
        assert isinstance(phi1, amnet.Mu) or \
               isinstance(phi1, amnet.Stack)
        phi1.y = phi2
    if hasattr(phi1, 'z') and isinstance(phi1.z, amnet.Variable):
        assert isinstance(phi1, amnet.Mu)
        phi1.z = phi2


def children(phi):
    """
    returns a (possibly empty) list of all direct children
    of the node phi
    """
    ret = []
    if hasattr(phi, 'x'):
        ret.append(phi.x)
    if hasattr(phi, 'y'):
        assert isinstance(phi, amnet.Mu) or \
               isinstance(phi, amnet.Stack)
        ret.append(phi.y)
    if hasattr(phi, 'z'):
        assert isinstance(phi, amnet.Mu)
        ret.append(phi.z)

    assert len(ret) <= 3
    return ret


def descendants(phi):
    """
    returns a list of all descendants of phi,
    including phi itself
    """
    q = deque([phi])  # queue of nodes to check
    d = list()        # list of descendants

    while len(q) > 0:
        node = q.popleft()
        # cannot use not(node in d) because Python's `in`
        # checks for `==` or `is` equality;
        # we want only `is` equality
        if not(any(node is e for e in d)):
            # node is new
            d.append(node)
            # add its children to check for reachability
            q.extend(children(node))

    # done finding reachable nodes
    return d


################################################################################
# ABANDONED METHODS
################################################################################

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
    #phic = copy.deepcopy(phi)
    #return phic
    pass

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
