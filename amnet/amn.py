import numpy as np


################################################################################
# main AMN classes
################################################################################

class Amn(object):
    """
    Abstract class that every AMN node must implement.
    """
    def __init__(self, outdim=0, indim=0):
        self.outdim = outdim
        self.indim = indim

    def eval(self, inp):
        return NotImplemented


class Variable(Amn):
    """
    A Variable instance is the sole leaf of an AMN tree.
    
    It has a name, evaluates like the identity function, 
    and has the same input and output dimensions.
    """
    def __init__(self, outdim=1, name='Variable'):
        super(Variable, self).__init__(outdim=outdim, indim=outdim)
        self.name = name

    def __str__(self):
        return '%s(%d)' % (self.name, self.outdim)

    def eval(self, inp):
        assert self.indim == self.outdim
        assert len(inp) == self.indim
        return inp


class Affine(Amn):
    """
    Affine encodes an alpha node, and  
    evaluates to alpha.eval(inp) = W * x(inp) + b.
        
    w: an m-by-n numpy array
    x: a pointer to an AMN with outdim = n
    b: an m-dimensional numpy array
    """
    def __init__(self, w, x, b):
        assert w.ndim == 2
        assert x.outdim == w.shape[1]
        assert len(b) == w.shape[0]

        super(Affine, self).__init__(outdim=w.shape[0], indim=x.indim)

        self.w = np.copy(w)
        self.x = x
        self.b = np.copy(b).flatten()

    def __str__(self):
        return 'Affine(w=%s, x=%s, b=%s)' % \
               (str(self.w), str(self.x), str(self.b))

    def eval(self, inp):
        assert len(inp) == self.indim
        xv = self.x.eval(inp)
        return np.dot(self.w, xv) + self.b


class Mu(Amn):
    """
    Mu encodes a multiplexing (or if-then-else) node, and
    evaluates to mu.eval(inp) = if z(inp) <= 0 then x(inp) else y(inp).
    
    x: (select-true) a pointer to an AMN
    y: (select-false) a pointer to an AMN  
    z: (enable input) a pointer to an AMN
    
    The dimension rules are:
    1) x, y, and z must all have the same indim.
    2) x and y must have the same outdim.
    3) z has outdim 1.
    """
    def __init__(self, x, y, z):
        assert x.outdim == y.outdim
        assert z.outdim == 1
        assert x.indim == y.indim and y.indim == z.indim

        super(Mu, self).__init__(outdim=x.outdim, indim=x.indim)

        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'Mu(%s, %s, %s)' % \
               (str(self.x), str(self.y), str(self.z))

    def eval(self, inp):
        assert len(inp) == self.indim
        zv = self.z.eval(inp)
        assert len(zv) == 1
        if zv <= 0:
            return self.x.eval(inp)
        else:
            return self.y.eval(inp)


class Stack(Amn):
    """
    A Stack is an AMN node that evaluates to an ordered pair 
    of its inputs. Care must be taken when evaluating,  
    because stack.eval(inp) is semantically (x(inp), y(inp)). 
    
    In particular x and y must have the same indim. 
    
    Avoid stacking variables-- instead create a new variable of 
    appropriate dimension.
    
    x: pointer to first component
    y: pointer to second component
    """
    def __init__(self, x, y):
        assert x.indim == y.indim
        assert not(isinstance(x, Variable) and isinstance(y, Variable)), \
            'stacking variables not supported'
        super(Stack, self).__init__(outdim=x.outdim + y.outdim, indim=x.indim)

        self.x = x
        self.y = y

    def __str__(self):
        return '[%s; %s]' % (str(self.x), str(self.y))

    def eval(self, inp):
        assert len(inp) == self.indim
        xv = self.x.eval(inp)
        yv = self.y.eval(inp)

        assert len(xv) + len(yv) == self.outdim
        outv = np.concatenate((xv, yv), axis=0)
        return outv


################################################################################
# Convenience classes
################################################################################

class Constant(Affine):
    def __init__(self, x, b):
        m = len(b)
        super(Constant, self).__init__(
            np.zeros((m, x.outdim)),
            x,
            b
        )

    def __str__(self):
        return 'Constant(x=%s, b=%s)' % (str(self.x), str(self.b))

    def eval(self, inp):
        # short-circuit evaluation
        return self.b


class Linear(Affine):
    def __init__(self, w, x):
        assert w.ndim == 2
        assert x.outdim == w.shape[1]

        super(Linear, self).__init__(
            w,
            x,
            np.zeros(w.shape[0])
        )

    def __str__(self):
        return 'Linear(w=%s, x=%s)' % (str(self.w), str(self.x))