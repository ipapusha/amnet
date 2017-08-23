import numpy as np


class Amn(object):
    def __init__(self, outdim=0, indim=0):
        self.outdim = outdim
        self.indim = indim

    def __add__(self, other):
        return NotImplemented

    def eval(self, inp):
        return NotImplemented


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


class Mu(Amn):
    def __init__(self, x, y, z):
        assert x.outdim == y.outdim
        assert z.outdim == 1

        assert (x.indim == y.indim) and (y.indim == z.indim)

        super(Mu, self).__init__(outdim=x.outdim, indim=x.indim)

        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'Mu(%s, %s, %s)' % (str(self.x), str(self.y), str(self.z))

    def eval(self, inp):
        zv = self.z.eval(inp)
        assert len(zv) == 1
        if zv <= 0:
            return self.x.eval(inp)
        else:
            return self.y.eval(inp)

################################################################################

# convenience methods
def compose_affine(aff, phi):
    """ returns aff(phi) """
    assert isinstance(aff, AffineTransformation)
    assert phi.outdim == aff.indim
    return AffineTransformation(
        aff.w,
        phi,
        aff.b
    )


def compose_affine_simp(aff1, aff2):
    """ returns aff1(aff2), simplified into one affine expression """
    assert isinstance(aff1, AffineTransformation)
    assert isinstance(aff2, AffineTransformation)
    assert aff2.outdim == aff1.indim
    return AffineTransformation(
        np.dot(aff1.w, aff2.w),
        aff2.x,
        np.dot(aff1.w, aff2.b) + aff1.b
    )


def simplify(phi):
    assert isinstance(phi, Variable) or \
           isinstance(phi, Mu) or \
           isinstance(phi, AffineTransformation)

    if isinstance(phi, Variable):
        return phi

    if isinstance(phi, Mu):
        return Mu(
            simplify(phi.x),
            simplify(phi.y),
            simplify(phi.z)
        )

    assert isinstance(phi, AffineTransformation)
    alpha = phi

    if isinstance(alpha.x, Variable):
        return phi

    if isinstance(alpha.x, Mu):
        return Mu(
            simplify(compose_affine(alpha, (alpha.x).x)),
            simplify(compose_affine(alpha, (alpha.x).y)),
            simplify((alpha.x).z)
        )

    assert isinstance(alpha.x, AffineTransformation)
    return simplify(compose_affine_simp(alpha, alpha.x))


def select(phi, k):
    """ returns kth component of phi """
    assert (0 <= phi.outdim) and (k < phi.outdim)
    return AffineTransformation(
        np.eye(1, phi.outdim, k),
        phi,
        np.zeros(1)
    )

