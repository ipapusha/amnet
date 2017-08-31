import numpy as np
import amnet


def test_max_atom():
    x = amnet.Variable(2, name='x')
    phimax = amnet.atoms.make_max2(x)

    print phimax
    print phimax.eval([1, -2])


def main():
    test_max_atom()


if __name__ == '__main__':
    main()