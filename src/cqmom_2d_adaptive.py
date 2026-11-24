import numpy as np
from .vandermatrix import vandermatrix
from .vandermone import vanderls
from .wheeler_adaptive import wheeler_adaptive


def cqmom_2d_adaptive(mom, N1, N2):
    if N1 == 1 and N2 == 1:
        if len(mom) < 3:
            raise ValueError('Per il caso monodisperso 2D servono almeno 3 momenti.')

        M00, M10, M01 = mom[0], mom[1], mom[2]

        if M00 <= 0:
            raise ValueError('M00 deve essere > 0 nel caso monodisperso.')

        w = np.array([M00])
        x1 = np.array([M10 / M00])   # volume della bolla  (k = 1)
        x2 = np.array([M01 / M00])   # massa della bolla   (l = 1)
        return w, x1, x2, 1

    mom = np.asarray(mom, dtype=float)
    mcall = vandermatrix(mom, N1, N2)                     # (N1, 2*N2-1)

    for n_try in range(N1, 0, -1):
        # Prima coordinata interna (v)
        w1, u1, nout1 = wheeler_adaptive(mom[:2 * n_try], n_try)
        w1, u1 = np.asarray(w1)[:nout1], np.asarray(u1)[:nout1]

        # Momenti condizionati:  V (R mc) = m2,  V[i, j] = u1[j]**i
        V = np.vander(u1, nout1, increasing=True).T
        m2 = mcall[:nout1, :]
        momc1 = np.zeros((nout1, 2 * N2 - 1))
        for i in range(2 * N2 - 1):
            q = m2[:, i]
            momc1[:, i] = vanderls(u1, q, nout1)
            err = V @ momc1[:, i] - q
            momc1[:, i] -= vanderls(u1, err, nout1)       # raffinamento iterativo
        mc = momc1 / w1[:, None]                          # R \ momc1

        # Test: momenti condizionati positivi e non troppo distanti
        flag = bool(np.any(mc[:, 0] < 0.0))
        if not flag and nout1 > 1:
            iu = np.triu_indices(nout1, 1)
            c = mc[iu[0], 0] / mc[iu[1], 0]
            flag = bool(np.any((c < 1e-6) | (c > 1e6)))

        # Seconda coordinata interna (m), per ogni nodo in v
        w, x1, x2 = [], [], []
        for i in range(nout1):
            momc2 = np.concatenate([[1.0], mc[i, :]])
            w2, u2, nout2 = wheeler_adaptive(momc2, N2)
            for j in range(nout2):
                w.append(w1[i] * w2[j])
                x1.append(u1[i])
                x2.append(u2[j])

        if not flag:
            break

    return np.array(w), np.array(x1), np.array(x2), len(w)
