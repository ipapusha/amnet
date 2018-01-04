import numpy as np
import amnet.atoms as atoms
import numbers


################################################################################
# main AMN classes
################################################################################

class Amn(object):
    """
    Every affine multiplexing network node descends from Amn.
    """
    # allows operators defined in this class to override
    # numpy's operators when interacting with numpy objects
    __array_priority__ = 200

    def __init__(self, outdim=0, indim=0):
        self.outdim = outdim
        self.indim = indim

    def eval(self, inp):
        return NotImplemented

    def _is_derived_instance(self):
        """
        returns True if the object is an instance of one of the
        recognized direct subclasses of Amn
        """
        return any([
            isinstance(self, cls) for cls in [
                Variable, Affine, Mu, Stack
            ]
        ])

    # array operations
    def __len__(self):
        return self.outdim

    def __getitem__(self, item):
        # we should not be slicing into the abstract class
        assert self._is_derived_instance()

        if isinstance(item, slice):
            return atoms.select_slice(self, item)

        elif isinstance(item, int):
            if not(0 <= item < len(self)):
                raise IndexError("Invalid index.")
            return atoms.select(self, item)

        else:
            raise TypeError("Invalid slice type.")

    # unary operations
    def __neg__(self):
        assert self._is_derived_instance()
        return atoms.negate(self)

    # binary operations
    def __add__(self, other):
        assert self._is_derived_instance()

        if isinstance(other, numbers.Real):
            # up-dimension and add
            c = Constant(
                self,
                np.repeat(other, self.outdim)
            )
            return atoms.add2(self, c)

        elif isinstance(other, np.ndarray):
            # check dimensions
            if not(other.shape == (self.outdim,)):
                return ValueError("Dimension mismatch adding an array.")

            # perform add
            c = Constant(
                self,
                other
            )
            return atoms.add2(self, c)

        elif isinstance(other, Amn):
            # check dimensions
            if not(self.outdim == other.outdim):
                return ValueError("Dimension mismatch adding an Amn.")

            # perform add
            return atoms.add2(self, other)

        else:
            raise TypeError("Invalid overload while calling %s.__add__(%s)" % (repr(self), repr(other)))

    __radd__ = __add__

    def __sub__(self, other):
        assert self._is_derived_instance()

        if isinstance(other, numbers.Real):
            # up-dimension and add
            c = Constant(
                self,
                np.repeat(other, self.outdim)
            )
            return atoms.sub2(self, c)

        elif isinstance(other, np.ndarray):
            # check dimensions
            if not(other.shape == (self.outdim,)):
                return ValueError("Dimension mismatch subtracting an array.")

            # perform add
            c = Constant(
                self,
                other
            )
            return atoms.sub2(self, c)

        elif isinstance(other, Amn):
            # check dimensions
            if not(self.outdim == other.outdim):
                return ValueError("Dimension mismatch subtracting an Amn.")

            # perform add
            return atoms.sub2(self, other)

        else:
            raise TypeError("Invalid overload while calling %s.__sub__(%s)" % (repr(self), repr(other)))

    def __rsub__(self, other):
        return (self.__neg__()).__add__(other)

    def __mul__(self, other):
        """
        since each node is thought of as a column vector,
        right-multiplication is only supported by a scalar or 1x1 array
        """
        if isinstance(other, numbers.Real):
            # perform multiply
            return atoms.scale(other, self)

        elif isinstance(other, np.ndarray):
            # check dimensions
            if not (other.shape == (1,)):
                return ValueError("Dimension mismatch while multiplying on the right.")

            # perform multiply
            assert isinstance(other[0], numbers.Real)
            return atoms.scale(other[0], self)

        else:
            raise TypeError("Invalid overload while calling %s.__mul__(%s)" % (repr(self), repr(other)))

    def __rmul__(self, other):
        """
        since each node is thought of as a column vector,
        left-multiplication is only supported by a scalar, 1x1 array, or a matrix of appropriate size
        """
        if isinstance(other, numbers.Real):
            # perform multiply
            return atoms.scale(other, self)

        elif isinstance(other, np.ndarray) and other.shape == (1,):
            # perform multiply
            assert isinstance(other[0], numbers.Real)
            return atoms.scale(other[0], self)

        elif isinstance(other, np.ndarray) and other.ndim == 2:
            # check dimensions
            (m, n) = other.shape
            assert m >= 1 and n >= 1
            if not (n == self.outdim):
                return ValueError("Dimension mismatch while multiplying on the left.")
            return Linear(
                other,
                self
            )

        else:
            raise TypeError("Invalid overload while calling %s.__rmul__(%s)" % (repr(self), repr(other)))


class Variable(Amn):
    """
    A Variable instance is the sole leaf of an AMN tree.
    
    It has a name, evaluates like the identity function, 
    and has the same input and output dimensions.
    """
    def __init__(self, outdim=1, name='Variable'):
        super(Variable, self).__init__(outdim=outdim, indim=outdim)
        self.name = name

    def __repr__(self):
        return "Variable(outdim=%s, name='%s')" % (self.outdim, self.name)

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

    def __repr__(self):
        return 'Affine(w=%s, x=%s, b=%s)' % \
               (repr(self.w), repr(self.x), repr(self.b))

    def __str__(self):
        return 'Affine(w=%s, x=%s, b=%s)' % \
               (str(self.w.tolist()), str(self.x), str(self.b.tolist()))

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

    def __repr__(self):
        return 'Mu(%s, %s, %s)' % \
               (repr(self.x), repr(self.y), repr(self.z))

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
        # TODO: assertion commented out to allow operations like x + x,
        #       however, we need a way to bring this back, disallowing variable stacks
        #assert not(isinstance(x, Variable) and isinstance(y, Variable)), \
        #    'stacking variables not supported'
        super(Stack, self).__init__(outdim=x.outdim + y.outdim, indim=x.indim)

        self.x = x
        self.y = y

    def __repr__(self):
        return 'Stack(%s, %s)' % (repr(self.x), repr(self.y))

    def __str__(self):
        return 'Stack(%s, %s)' % (str(self.x), str(self.y))

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

    def __repr__(self):
        return 'Constant(x=%s, b=%s)' % (repr(self.x), repr(self.b))

    def __str__(self):
        return 'Constant(x=%s, b=%s)' % (str(self.x), str(self.b.tolist()))

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

    def __repr__(self):
        return 'Linear(w=%s, x=%s)' % (repr(self.w), repr(self.x))

    def __str__(self):
        return 'Linear(w=%s, x=%s)' % (str(self.w.tolist()), str(self.x))
