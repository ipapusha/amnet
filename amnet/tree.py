import numpy as np
import amnet

import copy

"""
Contains routines for manipulating and simplifying Amn trees
"""


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
