import numpy as np
import amnet


# following ideas from:
# https://stackoverflow.com/questions/14619449/how-can-i-override-comparisons-between-numpys-ndarray-and-my-type
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.classes.html

class Expr(object):
    # allows operators defined in this class to override
    # numpy's operators when interacting with numpy objects
    __array_priority__ = 200

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return self.data

    def __add__(self, other):
        print "Calling %s.__add__(%s)" % (repr(self), repr(other))
        return 0

    def __radd__(self, other):
        print "Calling %s.__radd__(%s)" % (repr(self), repr(other))
        return 1

    def __mul__(self, other):
        print "Calling %s.__mul__(%s)" % (repr(self), repr(other))
        return 2

    def __rmul__(self, other):
        print "Calling %s.__rmul__(%s)" % (repr(self), repr(other))
        return 3


A = np.arange(6).reshape((2, 3))
x = Expr("x")

print "x + A"
print x + A

print "A + x"
print A + x

print "x + 2"
print x + 2

print "2 + x"
print 2 + x

print "x * 2"
print x * 2

print "2 * x"
print 2 * x

print "A * x"
print A * x

print "x * A"
print x * A