import numpy as np
import amnet
import amnet.vis

def make_vgc(alpha):
    # inputs
    x = amnet.Variable(2, name='eedot')

    # affine transformations
    zero1 = amnet.Constant(x, np.zeros(1))
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

# visualize VGC nonlinearity
phi_vgc = make_vgc(alpha=1.1)
dot = amnet.vis.amn2gv(phi_vgc, title='phi_vgc(var0)')
dot.render(filename='phi_vgc.gv', directory='vis')

print dot