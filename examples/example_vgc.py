import numpy as np
import amnet


def make_vgc(alpha):
    # inputs
    e_var = amnet.Variable(1, name='e')
    edot_var = amnet.Variable(1, name='edot')
    x = amnet.stack(e_var, edot_var)

    # affine transformations
    zero1 = amnet.atoms.make_const(np.zeros(1), x)
    ae    = amnet.Affine(np.array([[alpha, 0]]), x, np.zeros(1))
    e     = amnet.Affine(np.array([[1, 0]]), x, np.zeros(1))
    neg_e = amnet.Affine(np.array([[-1, 0]]), x, np.zeros(1))
    edot  = amnet.Affine(np.array([[0, 1]]), x, np.zeros(1))
    neg_edot = amnet.Affine(np.array([[0, -1]]), x, np.zeros(1))

    return amnet.atoms.gate_or(
        amnet.atoms.gate_or(
            zero1,
            ae,
            neg_e,
            neg_edot
        ),
        ae,
        e,
        edot
    )


def true_vgc(e, edot, alpha):
    if e * edot > 0:
        return alpha * e
    else:
        return 0


def test_vgc():
    alpha = 1.1
    phi_vgc = make_vgc(alpha)

    # test that true_vgc and test_vgc return the same evaluation
    for e in np.linspace(-2,2,12):
        for edot in np.linspace(-2,2,12):
            val1 = phi_vgc.eval(np.array([e, edot]))
            val2 = true_vgc(e, edot, alpha)
            if abs(val1 - val2) > 1e-10:
                print 'FAIL: (e,edot) = (%e, %e), val1=%e, val2=%e' % (e, edot, val1, val2)


def main():
    phi_vgc = make_vgc(1.0)
    print phi_vgc
    test_vgc()


if __name__ == '__main__':
    main()