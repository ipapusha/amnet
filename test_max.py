import numpy as np
import amnet


def test1():
    # test the affine transformations
    w1 = np.array([[1, 0]])
    b1 = np.array([0])
    w2 = np.array([[0, 1]])
    b2 = np.array([0])
    w3 = np.array([[-1, 1]])
    b3 = np.array([0])

    x = amnet.Variable(2, name='x')
    a1 = amnet.AffineTransformation(w1, x, b1)
    a2 = amnet.AffineTransformation(w2, x, b2)
    a3 = amnet.AffineTransformation(w3, x, b3)

    phimax = amnet.Mu(a1, a2, a3)
    print phimax
    print phimax.eval(np.array([1,2]))

def test2():
    x1 = amnet.Variable(1, name='x1')
    x2 = amnet.Variable(1, name='x2')

    a1 = x1
    a2 = x2

    w3 = np.array([[-1, 1]])
    b3 = np.array([0])
    a3 = amnet.AffineTransformation(w3, amnet.Vcat(x1, x2), b3)

    phimax = amnet.Mu(a1, a2, a3)
    print phimax
    print phimax.eval(np.array([1, 2]))

def main():
    test1()
    test2()

if __name__ == '__main__':
    main()