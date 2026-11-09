import numpy as np


def normalized(y0, f, Ns, norms):
    """ vettore: [momenti (num) | specie disciolte (Ns) | specie gassose (Ns-1)]"""
    y0 = np.asarray(y0, dtype=float)
    num = len(f)

    momn = np.array([y0[p] / (norms.w_0 * norms.v_0 ** f[p, 0] * norms.m_0 ** f[p, 1])
                     for p in range(num)])
    yl_part = y0[num:num + Ns] / norms.yl
    yg_part = y0[num + Ns:num + 2 * Ns - 1] / norms.yg

    return np.concatenate([momn, yl_part, yg_part])


def renormalize(y0, f, Ns, norms):
    """Riporta i momenti normalizzati alle grandezze fisiche"""
    y0 = np.asarray(y0, dtype=float)
    num = len(f)

    momns = np.array([y0[p] * (norms.w_0 * norms.v_0 ** f[p, 0] * norms.m_0 ** f[p, 1])
                      for p in range(num)])
    yl_part = y0[num:num + Ns] * norms.yl
    yg_part = y0[num + Ns:num + 2 * Ns - 1] * norms.yg

    return np.concatenate([momns, yl_part, yg_part])
