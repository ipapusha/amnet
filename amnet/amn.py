import numpy as np


################################################################################
# main AMN classes
################################################################################

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
# various methods for composing
################################################################################

def _compose_aff_phi(aff, phi):
    """ returns aff(phi) """
    assert isinstance(aff, AffineTransformation)
    assert phi.outdim == aff.indim
    return AffineTransformation(
        aff.w,
        phi,
        aff.b
    )


def _compose_aff_aff_simp(aff1, aff2):
    """ returns aff1(aff2), simplified into one affine expression """
    assert isinstance(aff1, AffineTransformation)
    assert isinstance(aff2, AffineTransformation)
    assert aff2.outdim == aff1.indim
    return AffineTransformation(
        np.dot(aff1.w, aff2.w),
        aff2.x,
        np.dot(aff1.w, aff2.b) + aff1.b
    )

def compose(phi1, phi2):
    """ returns phi1(phi2) """
    if isinstance(phi1, Variable):
        return phi2
    elif isinstance(phi1, AffineTransformation):
        return _compose_aff_phi(phi1, phi2)
    elif isinstance(phi1, Mu):
        return Mu(
            compose(phi2.x, phi2),
            compose(phi2.y, phi2),
            compose(phi2.z, phi2),
        )

    assert False


################################################################################
# convenience methods for simplification
################################################################################

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
            simplify(_compose_aff_phi(alpha, (alpha.x).x)),
            simplify(_compose_aff_phi(alpha, (alpha.x).y)),
            simplify((alpha.x).z)
        )

    assert isinstance(alpha.x, AffineTransformation)
    return simplify(_compose_aff_aff_simp(alpha, alpha.x))


def select(phi, k):
    """ returns kth component of phi """
    assert (0 <= phi.outdim) and (k < phi.outdim)
    return AffineTransformation(
        np.eye(1, phi.outdim, k),
        phi,
        np.zeros(1)
    )

def wrap_identity(phi):
    """returns phi, wrapped in an identity affine transformation"""
    return AffineTransformation(
        np.eye(phi.outdim, phi.outdim),
        phi,
        np.zeros(phi.outdim)
    )

################################################################################
# various methods for stacking
################################################################################

def _stack_var_var(v1, v2):
    assert isinstance(v1, Variable) and \
           isinstance(v2, Variable)

    # if variable is the same, replicate it with an affine transformation
    if v1.name == v2.name:
        assert v1.outdim == v2.outdim
        return AffineTransformation(
            np.concatenate((np.eye(v1.outdim), np.eye(v2.outdim)), axis=0),
            v1,
            np.zeros(v1.outdim + v2.outdim)
        )

    # if variable is not the same, create a new variable
    return Variable(
        outdim=v1.outdim + v2.outdim,
        name='<' + v1.name + ':' + v2.name + '>'
    )


def _stack_aff_aff(aff1, aff2):
    assert isinstance(aff1, AffineTransformation) and \
           isinstance(aff2, AffineTransformation)

    (m1, n1) = (aff1.outdim, aff1.indim)
    (m2, n2) = (aff2.outdim, aff2.indim)
    w = np.bmat([[aff1.w, np.zeros((m1, n2))],
                 [np.zeros((m2, n1)), aff2.w]])
    b = np.concatenate((aff1.b, aff2.b), axis=0)
    return AffineTransformation(
        w,
        stack(aff1.x, aff2.x),
        b
    )


def _stack_aff_mu(aff, mu):
    assert isinstance(aff, AffineTransformation) and \
           isinstance(mu, Mu)
    return Mu(
        stack(aff, mu.x),
        stack(aff, mu.y),
        mu.z
    )


def _stack_mu_aff(mu, aff):
    assert isinstance(aff, AffineTransformation) and \
           isinstance(mu, Mu)
    return Mu(
        stack(mu.x, aff),
        stack(mu.y, aff),
        mu.z
    )


def _stack_mu_mu(mu1, mu2):
    assert isinstance(mu1, Mu) and \
           isinstance(mu2, Mu)

    return Mu(
        Mu(
            stack(mu1.x, mu2.x),
            stack(mu1.y, mu2.x),
            mu1.z
        ),
        Mu(
            stack(mu1.x, mu2.y),
            stack(mu1.y, mu2.y),
            mu1.z
        ),
        mu2.z
    )


def stack(phi1, phi2):
    # TODO: need to be able to stack non-variables
    if not(isinstance(phi1, Variable) and isinstance(phi2, Variable)):
        return NotImplemented

    # code below is wrong
    if isinstance(phi1, Variable):
        if isinstance(phi2, Variable):
            # base case
            return _stack_var_var(phi1, phi2)
        elif isinstance(phi2, AffineTransformation):
            return _stack_aff_aff(wrap_identity(phi1), phi2)
        elif isinstance(phi2, Mu):
            return _stack_aff_mu(wrap_identity(phi1), phi2)
    elif isinstance(phi1, AffineTransformation):
        if isinstance(phi2, Variable):
            return _stack_aff_aff(phi1, wrap_identity(phi2))
        elif isinstance(phi2, AffineTransformation):
            return _stack_aff_aff(phi1, phi2)
        elif isinstance(phi2, Mu):
            return _stack_aff_mu(phi1, phi2)
    elif isinstance(phi1, Mu):
        if isinstance(phi2, Variable):
            return _stack_mu_aff(phi1, wrap_identity(phi2))
        elif isinstance(phi2, AffineTransformation):
            return _stack_mu_aff(phi1, phi2)
        elif isinstance(phi2, Mu):
            return _stack_mu_mu(phi1, phi2)

    assert False
