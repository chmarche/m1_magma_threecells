"""Stato di ingresso per il caso a due specie: fuso saturo alla pressione di ingresso.

Dati p_in, T, la frazione di gas alpha0 e i volatili totali (guess), la tabella SOLWCAD da' la solubilita'
di H2O e CO2 (yl) e la composizione del gas (yg). I totali veri dipendono pero' da alpha0 e dalla densita' del
gas, quindi si itera fino a che i totali usati per interrogare la tabella coincidono con quelli della miscela.
"""
import numpy as np

from .constants import Constants


def saturated_initial(table, p_in, T, alpha0, rho_l, PM, ytot_guess, tol=1e-12, itmax=100):
    PM = np.asarray(PM, dtype=float)
    ytot = np.asarray(ytot_guess, dtype=float).copy()
    for _ in range(itmax):
        yl = np.array([table.H2Om(p_in, T, *ytot), table.CO2m(p_in, T, *ytot)])
        yg_h2o = table.H2Og(p_in, T, *ytot)
        yg = np.array([yg_h2o, 1.0 - yg_h2o])
        rho_g = p_in / (Constants.R * np.sum(yg / PM) * T)       # gas perfetto (sigma = 0, come nell'init 0-D)
        M01 = rho_g * alpha0
        c = yl * rho_l * (1 - alpha0)
        rho_mix = M01 + (1 - alpha0) * rho_l
        ytot_new = (c + yg * M01) / rho_mix
        if np.max(np.abs(ytot_new - ytot)) < tol:
            ytot = ytot_new
            break
        ytot = ytot_new
    return yl, yg, ytot
