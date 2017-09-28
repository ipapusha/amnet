import amnet
import graphviz

import copy

class NamingContext(object):
    def __init__(self):
        self.symbols = dict()

    def next_unique_name(self, prefix):
        # all names beginning with the given prefix
        pnames = [name
                  for name in self.symbols.keys()
                  if name.startswith(prefix)]

        # if there are no existing variables
        if len(pnames) == 0:
            return prefix + '0'

        # otherwise, return 1 + largest value in the names




def _walk_amn(dot, phi):
    pass

def amn2gv(phi):
    # first do a deep copy of the amn to ensure we don't touch it
    phi_copy = copy.deepcopy(phi)

    # walk the copy to generate a graph
    dot = graphviz.Digraph()
    _walk_amn(dot, phi_copy)

    # return the dot object
    return dot
