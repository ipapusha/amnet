import numpy as np
import amnet


def main():
    x = amnet.Variable(2, name='x')
    phimax = amnet.atoms.max2(x)

    print phimax

    print phimax.eval([1,-2])

if __name__ == '__main__':
    main()