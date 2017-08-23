import numpy as np
import copy

class Amn(object):
    def __init__(self, outdim=0, indim=0):
        self.outdim = outdim
        self.indim = indim

    def __add__(self, other):
        return NotImplemented

    def eval(self, inp):
        return NotImplemented


# # even though Weight acts in many ways like an Amn, it is not
# # (by itself) a function, so it is *not* a subclass of Amn
# class Weight(object):
#     def __init__(self, outdim=0, indim=0, w=None):
#         self.outdim = outdim
#         self.indim = indim
#         self.w = copy.deepcopy(w)
#         if hasattr(w, 'shape'):
#             (self.outdim, self.indim) = w.shape
#
#     def __str__(self):
#         if self.w is None:
#             return 'w(%d,%d)' % (self.outdim, self.indim)
#         else:
#             return str(self.w)
#
#     def eval(self, inp):
#         return np.dot(self.w, inp)
#
#
# # aka constant vector
# class Bias(Amn):
#     def __init__(self, b):
#         self.b = np.copy(b).flatten()
#         super(Bias, self).__init__(outdim=len(self.b), indim=0)
#
#     def __str__(self):
#         return str(self.b)
#
#     def eval(self, inp=None):
#         return self.b


class Variable(Amn):
    def __init__(self, outdim=1, name='Variable'):
        # a variable has the same input and output sizes
        super(Variable, self).__init__(outdim=outdim, indim=outdim)
        self.name = name

    def __str__(self):
        return self.name + '(' + str(self.outdim) + ')'

    def eval(self, inp):
        assert self.indim == self.outdim
        assert len(inp) == self.indim
        return inp


class AffineTransformation(Amn):
    def __init__(self, w, x, b):
        assert w.ndim == 2
        super(AffineTransformation, self).__init__(outdim=w.shape[0], indim=w.shape[1])
        assert len(b) == self.outdim
        assert x.outdim == self.indim

        self.w = np.copy(w)
        self.x = x
        self.b = np.copy(b).flatten()

    def __str__(self):
        return str(self.w) + ' * ' + str(self.x) + ' + ' + str(self.b)

    def eval(self, inp):
        xv = self.x.eval(inp)
        assert len(xv) == self.indim
        return np.dot(self.w, xv) + self.b
