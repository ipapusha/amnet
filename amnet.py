import numpy as np

# Affine Multiplexing Network classes
class Mu:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'Mu(' + str(self.x) + ', ' + \
               str(self.y) + ', ' + \
               str(self.z) + ')'

    def eval(self, inp):
        zv = self.z.eval(inp)
        assert zv.size == 1
        if zv <= 0:
            return self.x.eval(inp)
        else:
            return self.y.eval(inp)


class Variable:
    def __init__(self, n=1, name='Variable'):
        self.n = n
        self.name = name

    def __str__(self):
        return self.name + '(' + str(self.n) + ')'

    def eval(self, inp):
        assert inp.size == self.n
        return inp


class Constant:
    def __init__(self, c):
        self.c = c.flatten()
        self.n = self.c.size

    def __str__(self):
        return str(self.c)

    def eval(self, inp=None):
        return self.c


class AffineTransformation:
    def __init__(self, w, x, b):
        if w.ndim == 1:
            m = 1
            n = w.size
            self.w = np.reshape(w, (m, n))
        elif w.ndim == 2:
            m, n = w.shape
            self.w = np.copy(w)
        else:
            assert False

        self.x = x

        self.b = b.flatten()
        assert self.b.size == m

        self.m = m
        self.n = n

    def __str__(self):
        return str(self.w) + ' * ' + str(self.x) + ' + ' + str(self.b)

    def eval(self, inp):
        xv = self.x.eval(inp)
        assert xv.size == self.n
        return np.dot(self.w, xv) + self.b
