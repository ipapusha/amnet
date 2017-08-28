import numpy as np
import amnet


def test_max_atom():
    x = amnet.Variable(2, name='x')
    phimax = amnet.atoms.make_max2_s(x)

    print phimax
    print phimax.eval([1, -2])


def test_max_stack():
    x = amnet.Variable(1, name='x')
    y = amnet.Variable(1, name='y')

    phimax_xy = amnet.atoms.make_max2(x, y)

    print phimax_xy
    print phimax_xy.eval([1, -2])


def test_max3():
    x = amnet.Variable(1, name='x')
    y = amnet.Variable(1, name='y')
    z = amnet.Variable(1, name='z')

    phimax_xyz = amnet.atoms.make_max2(x, amnet.atoms.make_max2(y, z))

    print phimax_xyz
    print phimax_xyz([1, 2, -3])

def main():
    test_max_atom()
    test_max_stack()
    test_max3()



if __name__ == '__main__':
    main()