import numpy as np
import amnet

# following ideas from:
# https://stackoverflow.com/questions/14619449/how-can-i-override-comparisons-between-numpys-ndarray-and-my-type
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.classes.html

# TODO: implement slicing and array access
# TODO: add divide and true divide
# TODO: add negative indexing
# TODO: implement __setitem__ if necessary

class Expr(object):
    # allows operators defined in this class to override
    # numpy's operators when interacting with numpy objects
    __array_priority__ = 200

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return self.data

    # array operations
    def __len__(self):
        return 10

    def __getitem__(self, item):
        if isinstance(item, slice):
            # create a matrix corresponding to the slice
            n = len(self)
            m = len(range(*item.indices(n)))
            W = np.concatenate(
                [np.eye(1, n, i) for i in xrange(*item.indices(n))],
                axis=0
            )
            assert W.shape == (m, n)
            print 'slice:', W
            return 1
        elif isinstance(item, int):
            n = len(self)
            assert 0 <= item < n
            W = np.eye(1, n, item)
            assert W.shape == (1, n)
            print 'item:', W
            return 2
        else:
            raise TypeError("Invalid slice type.")

    # unary expressions
    def __neg__(self):
        print "Calling %s.__neg__()" % (repr(self))
        return -1

    # binary expressions
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

    # comparisons
    # def __cmp__(self, other):
    #    print "Calling %s.__cmp__(%s)" % (repr(self), repr(other))
    #    return 4

    def __lt__(self, other):
        print "Calling %s.__lt__(%s)" % (repr(self), repr(other))
        return 5

    def __le__(self, other):
        print "Calling %s.__le__(%s)" % (repr(self), repr(other))
        return 6

    def __eq__(self, other):
        print "Calling %s.__eq__(%s)" % (repr(self), repr(other))
        return 7

    def __ne__(self, other):
        print "Calling %s.__ne__(%s)" % (repr(self), repr(other))
        return 7

    def __ge__(self, other):
        print "Calling %s.__ge__(%s)" % (repr(self), repr(other))
        return 8

    def __gt__(self, other):
        print "Calling %s.__gt__(%s)" % (repr(self), repr(other))
        return 9


# from: https://stackoverflow.com/a/14636997
# def _override(name):
#     print "OVERRIDE"
#     def ufunc(x, y):
#         print "UFUNC:", y
#         if isinstance(y, Expr):
#             return NotImplemented
#         return np.getattr(name)(x, y)
#     return ufunc
#
#
# np.set_numeric_ops(
#     ** {
#         ufunc: _override(ufunc) for ufunc in (
#             "less", "less_equal", "equal", "not_equal", "greater_equal", "greater"
#         )
#     }
# )


A = np.arange(6).reshape((2, 3))
c = np.arange(3)
M = np.matrix(A)
x = Expr("x")

print "===================="
print "-x"
print -x

print "===================="
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

print "===================="
print "x < A"
print x < A

print "A < x"
print A < x

print "x < c"
print x < c

print "c < x"
print c < x

print "c == x"
print c == x

print "A == x"
print A == x

print "x == c"
print x == c

print "x == A"
print x == A

print "x != A"
print x != A

print "M != x"
print M != x

print "===================="
print "x <= 5"
print x <= 5

print "16 <= x"
print 16 <= x

print "===================="
print "The length of x is:", len(x)
print "Slice: x[0] = ", x[0]
print "Slice: x[1] = ", x[1]
print "Slice: x[0:] = ", x[0:]
print "Slice: x[0::2] = ", x[0::2]
# print "End: x[-1] = ", x[-1]
# print "Setitem: x[0] = 2 begin..."
# x[0] = 2
# print "end."