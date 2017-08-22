import numpy as np

# Affine Multiplexing Network classes
class Constant:
    def __init__(self, c):
        self.c = c.flatten()
        self.n = self.c.size

    def __str__(self):
        return str(self.c)

    def eval(self, inp=None):
        return self.c


class Variable:
    def __init__(self, n=1, name='Variable'):
        self.n = n
        self.name = name

    def __str__(self):
        return self.name + '(' + str(self.n) + ')'

    def eval(self, inp):
        assert inp.size == self.n
        return inp

    def __add__(self, other):
        if isinstance(other, Mu):
            return NotImplemented
        elif isinstance(other, Variable):
            assert other.n == self.n
            w = np.concatenate((np.eye(self.n), np.eye(other.n)), axis=1)
            xy = Vcat(self, other)
            b = np.zeros((self.n,1))
            return AffineTransformation(w, xy, b)
        elif isinstance(other, Constant):
            assert other.n == self.n
            id = IdentityTransformation(self)
            id.b = np.copy(other.c)
            return id
        elif isinstance(other, AffineTransformation):
            assert other.m == self.n
            w = np.concatenate((np.eye(self.n), other.w), axis=1)
            xy = Vcat(self, other)
            b = other.b
            return AffineTransformation(w, xy, b)
        elif isinstance(other, Vcat):
            assert other.n == self.n
            w = np.concatenate((np.eye(self.n), np.eye(other.n)), axis=1)
            xy = Vcat(self, other)
            b = np.zeros((self.n, 1))
            return AffineTransformation(w, xy, b)
        else:
            return NotImplemented


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


class IdentityTransformation(AffineTransformation):
    def __init__(self, x):
        w = np.eye(x.n)
        b = np.zeros((x.n, 1))
        AffineTransformation.__init__(self, w, x, b)


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


# concatenates two affine networks vertically
class Vcat:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = x.n + y.n

    def __str__(self):
        return '[' + str(self.x) + ' ; ' + str(self.y) + ']'

    def eval(self, inp1, inp2=None):
        if inp2:
            xv = self.x.eval(inp1)
            yv = self.y.eval(inp2)
        else:
            n1 = self.x.n
            xv = self.x.eval(inp1[0:n1])
            yv = self.y.eval(inp1[n1:])
        assert xv.size + yv.size == self.n
        return np.concatenate((xv, yv), axis=0)