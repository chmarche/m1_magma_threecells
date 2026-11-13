import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat

# --- Cartella con solwcad_jax.py (quella dello zip SOLWCAD_python) -------------------------------
SOLWCAD_DIR = Path(r'C:\Users\chiar\Desktop\Tesi\SOLWCAD_python')      # <-- MODIFICA se serve
sys.path.insert(0, str(SOLWCAD_DIR))
import jax
import solwcad_jax as S

# --- Griglie (le stesse di run_monodisperse_mGrowth.m) --------------------------------------------
n = 10                                              # punti per variabile: n**4 punti in totale
T = np.linspace(1100 + 273.15, 1100 + 273.15 + 300, n)         # K
p = np.linspace(80e6, 220e6, n)                                 # Pa
H2Otot = np.linspace(0.03, 0.09, n)                             # frazione in massa
CO2tot = np.linspace(0.003, 0.01, n)                            # frazione in massa
# SiO2 TiO2 Al2O3 Fe2O3 FeO MnO MgO CaO Na2O K2O (frazioni in massa, conta la proporzione)
Composition = np.array([68.65, 0.55, 14.30, 0, 3.49, 0, 1.46, 3.95, 3.61, 3.09]) * 0.01

# --- Calcolo vettorializzato ---------------------------------------------------------------------
P_, T_, H_, C_ = np.meshgrid(p, T, H2Otot, CO2tot, indexing='ij')      # ordine (p, T, H2Otot, CO2tot)
flat = [a.ravel() for a in (P_, T_, H_, C_)]
equilibrium = jax.jit(jax.vmap(S.equilibrium, in_axes=(0, 0, 0, 0, None)))

print(f'Tabella {n}x{n}x{n}x{n} = {n**4} punti ...')
t0 = time.time()
out = np.asarray(equilibrium(*flat, Composition))                      # (N, 4)
print(f'fatto in {time.time() - t0:.1f} s (compilazione inclusa)')

shape = P_.shape
H2Oeq, CO2eq, H2Oeq_g, xgas_eq = (out[:, i].reshape(shape) for i in range(4))
for name, a in (('H2Oeq', H2Oeq), ('CO2eq', CO2eq), ('H2Oeq_g', H2Oeq_g), ('xgas_eq', xgas_eq)):
    bad = ~np.isfinite(a)
    print(f'{name:8s} min {np.nanmin(a):.5g}  max {np.nanmax(a):.5g}  valori non finiti: {bad.sum()}')

Ltable = dict(p=p, T=T, H2Otot=H2Otot, CO2tot=CO2tot, Composition=Composition,
              H2Oeq=H2Oeq, CO2eq=CO2eq, H2Oeq_g=H2Oeq_g, xgas_eq=xgas_eq)
np.savez('lookupTable.npz', **Ltable)
savemat('lookupTable.mat', {'Ltable': Ltable})
print('scritti lookupTable.npz e lookupTable.mat')
