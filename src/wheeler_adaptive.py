import numpy as np


def wheeler_adaptive(m, n):
    """Versione adattiva dell'algoritmo di Wheeler (porting di wheeler_adaptive.m).
    m: momenti (almeno 2*n elementi), n: numero massimo di nodi.
    Restituisce (w, xi, nout): pesi, ascisse e numero di nodi effettivo.
    """
    m = np.asarray(m, dtype=float)

    for n1 in range(n, 0, -1):

        if n1 == 1:                                            # distribuzione monodispersa
            return np.array([m[0]]), np.array([m[1] / m[0]]), 1

        # Matrice Sigma per i coefficienti della matrice di Jacobi (indici MATLAB - 1)
        sig = np.zeros((2 * n1 + 1, 2 * n1 + 1))
        sig[1, 1:2 * n1 + 1] = m[:2 * n1]

        a = np.zeros(n1)
        b = np.zeros(n1)
        a[0] = m[1] / m[0]

        for i in range(2, n1 + 1):
            for j in range(i, 2 * n1 - i + 2):
                sig[i, j] = sig[i - 1, j + 1] - a[i - 2] * sig[i - 1, j] - b[i - 2] * sig[i - 2, j]
            a[i - 1] = sig[i, i + 1] / sig[i, i] - sig[i - 1, i] / sig[i - 1, i - 1]
            b[i - 1] = sig[i, i] / sig[i - 1, i - 1]

        # Matrice di Jacobi
        off = -np.sqrt(np.abs(b[1:n1]))
        Ja = np.diag(a) + np.diag(off, 1) + np.diag(off, -1)

        if not np.all(np.isfinite(Ja)):                        # (aggiunta rispetto al MATLAB)
            continue                                           # momenti degeneri -> meno nodi

        # Pesi e ascisse
        xi, D = np.linalg.eigh(Ja)
        w = D[0, :] ** 2 * m[0]

        # Controlli: pesi non troppo diversi, momenti e ascisse non negativi
        iu = np.triu_indices(n1, 1)
        c = w[iu[0]] / w[iu[1]]
        flag = (np.any((c < 1e-6) | (c > 1e6))
                or np.any(m[:2 * n] < 0)
                or np.any(xi < 0))

        if not flag:
            return w, xi, n1
