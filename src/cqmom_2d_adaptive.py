import numpy as np


def cqmom_2d_adaptive(mom, N1, N2):
    """Inversione dei momenti 2d: restituisce (w, x1, x2, nout): pesi, ascisse della 
    prima coordinata (v), ascisse della seconda coordinata (m) e numero di nodi.

    Caso monodisperso (N1 = N2 = 1): un solo nodo, con
        w = M00,  x1 = v = M10 / M00,  x2 = m = M01 / M00
    """
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

    raise NotImplementedError('solo monodisperso')
