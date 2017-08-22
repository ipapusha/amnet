import z3


def main():
    x = z3.Real('x')
    y = z3.Real('y')
    s = z3.Solver()
    s.add(x + y > 5, x > 1, y > 1)
    print s.check()
    print s.model()

if __name__ == '__main__':
    main()