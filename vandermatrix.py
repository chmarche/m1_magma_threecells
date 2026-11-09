import numpy as np

# Indici dei momenti per il termine noto del sistema di Vandermonde

_INDICI = {
    '11': [[2]],
    '21': [[4],
           [5]],
    '22': [[4, 5, 6],
           [7, 8, 9]],
    '31': [[6],
           [7],
           [8]],
    '32': [[6, 7, 8],
           [9, 10, 11],
           [12, 13, 14]],
}


def vandermatrix(mom, N1, N2):
    y = str(N1) + str(N2)
    if y not in _INDICI:
        raise ValueError('Errore: ordine dei momenti non definito')
    mom = np.array(mom, dtype=float)
    return mom[np.array(_INDICI[y])]
