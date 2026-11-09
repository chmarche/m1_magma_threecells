import numpy as np


def vanderls(x, q, nout):
    """Risolutore del sistema di Vandermonde V w = q, con V[i, j] = x[j]**i"""
    w = np.zeros(nout)

    if nout == 1:
        w[0] = q[0]
        return w

    c = np.zeros(nout)
    c[nout - 1] = -x[0]
    for i in range(1, nout):
        xx = -x[i]
        for j in range(nout - i - 1, nout - 1):
            c[j] = c[j] + xx * c[j + 1]
        c[nout - 1] = c[nout - 1] + xx

    for i in range(nout):
        xx = x[i]
        t = 1.0
        b = 1.0
        s = q[nout - 1]
        for k in range(nout - 1, 0, -1):
            b = c[k] + xx * b
            s = s + q[k - 1] * b
            t = xx * t + b
        w[i] = s / t

    return w
