import numpy as np
from scipy.linalg import norm
import amnet

from collections import deque

import copy

"""
Contains routines for manipulating and simplifying Amn trees
"""


def compose_rewire(phi1, phi2):
    """
    Given two AMNs and pointers to their input variables,
    rewires the first AMN's variable to point to the output of the second AMN.

    Given:
        phi1(x1)
        phi2(x2)
    Side effects:
        phi1 is now rewired to
        phi(x) = phi1(phi2(x)),
        where x = x2 (the variable of phi2)
    Note: x1 is thrown away!
    Note: this routine modifies phi1!
    """
    # cannot compose when dimensions are wrong
    assert phi1.indim == phi2.outdim

    # it does not make sense to compose with phi1 a variable
    assert not (isinstance(phi1, amnet.Variable))

    # compute the list of descendants of phi1 and phi2
    desc1 = descendants(phi1)
    desc2 = descendants(phi2)

    # the trees should have no overlaps
    nodeids1 = set([id(d) for d in desc1])
    nodeids2 = set([id(d) for d in desc2])
    assert len(nodeids1) == len(desc1)
    assert len(nodeids2) == len(desc2)
    assert len(nodeids1 & nodeids2) == 0

    # determine the variables x1, x2 associated with phi1, phi2
    vars1 = [d for d in desc1 if isinstance(d, amnet.Variable)]
    vars2 = [d for d in desc2 if isinstance(d, amnet.Variable)]
    assert len(vars1) == 1
    assert len(vars2) == 1
    x1 = vars1[0]
    x2 = vars2[0]

    # TODO: rewire here


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
            q.extend([c for c in children(node) if c not in d])

    # done finding reachable nodes
    return d


def valid_tree(phi):
    """
    goes through the tree of phi and ensures that
    1. the dimensions work out
    2. there is only one variable
    3. there are no directed cycles
    """
    q = deque([phi])  # queue of nodes to check
    visited = list()  # already checked

    # save the indim of the root node, and make sure all the indims
    # of the children are the same
    indim = phi.indim
    retval = True
    varsfound = 0

    while len(q) > 0:
        # node to check
        node = q.popleft()

        # check outdim
        if isinstance(node, amnet.Variable):
            retval &= (node.outdim == node.indim)
            varsfound += 1
        elif isinstance(node, amnet.Linear):
            m, n = node.w.shape
            retval &= (node.outdim == m)
            retval &= (node.x.outdim == n)
            retval &= (all([bi == 0 for bi in node.b]))  # check value
        elif isinstance(node, amnet.Constant):
            retval &= (node.outdim == len(node.b))
            retval &= (all([wij == 0 for wij in np.nditer(node.w)]))  # check value
        elif isinstance(node, amnet.Affine):
            m, n = node.w.shape
            retval &= (node.outdim == m)
            retval &= (node.x.outdim == n)
            retval &= (m == len(node.b))
        elif isinstance(node, amnet.Mu):
            retval &= (node.outdim == node.x.outdim)
            retval &= (node.outdim == node.y.outdim)
            retval &= (node.z.outdim == 1)
        elif isinstance(node, amnet.Stack):
            retval &= (node.outdim == node.x.outdim + node.y.outdim)
        else:
            retval = False  # unknown node type

        # check indim
        retval &= (node.indim == indim)

        # short-circuit if an inconsistency has been found
        if not retval:
            return False

        # add children to queue
        if not(any(node is e for e in visited)):
            visited.append(node)
            #q.extend(children(node))
            q.extend([c for c in children(node) if c not in visited])

    # finished iterating
    # TODO: also check if graph is cyclic
    return (varsfound == 1)


def is_cyclic(phi):
    # 1. walk through the graph to determine available nodes
    white = deque(descendants(phi))  # all reachable nodes
    stk = deque([])  # stack for dfs
    gray = list()    # exploring set
    black = list()   # explored set

    # 2. walk through the graph in DFS order
    while len(white) > 0:
        # get a new white vertex
        stk.append(white.popleft())

        #
        while len(stk) > 0:
            pass
            # TODO



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
