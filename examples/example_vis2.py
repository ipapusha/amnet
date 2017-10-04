import numpy as np
import amnet
import amnet.vis

# visualze the VGC nonlinearity with graph names

def make_vgc(alpha):
    # inputs
    x = amnet.Variable(2, name='x')

    # affine transformations
    zero1 = amnet.Constant(x, np.zeros(1))
    e     = amnet.atoms.select(x, 0)
    edot  = amnet.atoms.select(x, 1)
    ae    = amnet.Linear(np.array([[alpha, 0]]), x)
    neg_e = amnet.Linear(np.array([[-1, 0]]), x)
    neg_edot = amnet.Linear(np.array([[0, -1]]), x)

    or0 = amnet.atoms.gate_or(
        zero1,
        ae,
        neg_e,
        neg_edot
    )

    or1 = amnet.atoms.gate_or(
        or0,
        ae,
        e,
        edot
    )

    # create naming context of the retval
    ctx = amnet.vis.NamingContext(or1)
    ctx.rename(x,        'x')
    ctx.rename(zero1,    'zero')
    ctx.rename(e,        'e')
    ctx.rename(edot,     'edot')
    ctx.rename(ae,       'ae')
    ctx.rename(neg_e,    'neg_e')
    ctx.rename(neg_edot, 'neg_edot')
    ctx.rename(or0,      'or0')
    ctx.rename(or1,      'or1')

    return or1, ctx

# visualize VGC nonlinearity
phi_vgc, ctx = make_vgc(alpha=1.1)
dot = amnet.vis.amn2gv(phi_vgc, ctx=ctx, title='phi_vgc(x)')
dot.render(filename='phi_vgc.gv', directory='vis')

print dot