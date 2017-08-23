import numpy as np
import amnet


def main():
    x = amnet.Variable(2, name='x')
    a1 = amnet.AffineTransformation(
        np.array([[1, 0]]),
        x,
        np.array([0])
    )
    a2 = amnet.AffineTransformation(
        np.array([[0, 1]]),
        x,
        np.array([0])
    )
    a3 = amnet.AffineTransformation(
        np.array([[-1, 1]]),
        x,
        np.array([0])
    )

    print x
    print a1
    print a2
    print a3

if __name__ == '__main__':
    main()