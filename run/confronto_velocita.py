import numpy as np

import matplotlib.pyplot as plt
from src.conduit import magma_conduit

# Tempo di integrazione
class Time:
    tout = 10.0 # s  (almeno ~3 tempi di transito N*dz/u)
time = Time()

# Numerical methods
class CQMOM:
    N1 = 1
    N2 = 1
    ODEsolver = '15s'
cqmom = CQMOM()

# Condizioni iniziali
class Initial:
    method = 'discrete'
    diam = 1e-3 * 2          # Initial bubble diameters (m)
    Nb = 2363660             # number of bubbles per unit volume of melt (m^-3)
    
    # derived quantities
    meltvolume = 1 / Nb
    gasvolume = np.pi * diam**3 / 6
    alpha = gasvolume / (gasvolume + meltvolume)
    wt = Nb * (1 - alpha)    # initial number of bubbles for each group per unit total volume
    yl = 0.02                # concentration dissolved species [mass fraction]
    yg = 1.0                 # concentration of gas species

initial = Initial()


# Griglia del condotto (z verso l'alto, cella 0 = la piu' profonda)
class Grid:
    N = 3             # numero di celle
    dz = 1.0          # altezza di ogni cella (m)
    u = 1.0           # velocita' alla base del condotto (m/s); con 'constant' vale in tutto il condotto
    CFL = 0.5         # dt = CFL * dz / u  (usato come max_step del solver)
    z0 = 0.0          # quota della base del condotto (m)
    p_top = 101325.0  # pressione del liquido alla superficie libera (Pa)
    velocityModel = 'continuity'   # 'constant' = u fissa | 'continuity' = u(z) dalla continuita'
grid = Grid()

# Proprietà del magma
class Liquid:
    mu = 1e4          # Melt viscosity (kg / m s)
    rho_l = 2500      # Melt density (kg / m^3)

class Species:
    name = ['H_2O']
    v_mol = 25e-6     # Molar volume (m^3/mol)
    PM = 18.02e-3     # Molecular weight of gas species (kg/mol)

class Bubble:
    vs = (1/initial.alpha - 1) * np.pi/6 * initial.diam**3  # shell volume (m^3)

class Magma:
    liquid = Liquid()
    species = Species()
    bubble = Bubble()

magma = Magma()

# Modelli di P e T
# In questo run la pressione dipende da z (vedi Grid): dpdt e p0 NON vengono usati,
# p0 viene sovrascritto dentro magma_conduit con la pressione della cella di ingresso.
class Pressure:
    name = 'linear'   # serve solo a get_thermo_pt
    dpdt = 0.0
    p0 = 0.0
    pMin = 1e3

class Temperature:
    name = 'linear'
    dTdt = 0.0        # Cooling rate (K / s)
    T0 = 1100.0       # Initial temperature (K)

class Models:
    pressure = Pressure()
    temperature = Temperature()

# Modello di solubilità
class Saturation:
    name = 'Henry'
    hcp = 4e-6        # Henry constant (Pa^(-1/2))

Models.saturation = Saturation()

# Proprietà
class SurfaceTension:
    name = 'constant'
    sigma = 0.06      # (N/m)

class Diffusivity:
    name = ['constant']
    D = 0.0

Models.surfaceTension = SurfaceTension()
Models.diffusivity = Diffusivity()

# Leggi di chiusura
class VolumeGrowth:
    name = 'Rayleigh-Plesset'

class MassGrowth:
    name = 'zero'

class Nucleation:
    name = 'zero'

class Coalescence:
    name = 'constant'    # 'growthControlled'
    const = 0.0

Models.volumeGrowth = VolumeGrowth()
Models.massGrowth = MassGrowth()
Models.nucleation = Nucleation()
Models.coalescence = Coalescence()



# ---------------------------------------------------------------------------
# Confronto velocita' fissa vs u(z) dalla continuita': stesso caso, due modelli.
# Per vedere un effetto netto servono gas fraction non piccola: condotto piu' lungo,
# piu' celle e pressione in cima bassa (modifica Grid e time qui sotto).
# ---------------------------------------------------------------------------
grid.N = 20
grid.dz = 5.0
grid.u = 1.0
grid.p_top = 1e6      # pressione in cima: abbastanza alta da risolvere bene lo strato superficiale
time.tout = 300.0       # >= 3 tempi di transito N*dz/u
# Per vedere differenze anche nello stato (M10, raggio) serve un fuso lento a rilassarsi:
# prova Liquid.mu = 1e6 in alto nel file (con mu = 1e4 le bolle seguono la pressione quasi in equilibrio).

risultati = {}
for modello in ('constant', 'continuity'):
    grid.velocityModel = modello
    risultati[modello] = magma_conduit(time, grid, cqmom, initial, magma, Models,
                                       save_out_figure=False, show=False)

rc, rv = risultati['constant'], risultati['continuity']
alpha_in = rc.M10[0, 0]                    # stato di ingresso = stato iniziale di ogni cella

def flusso_liquido(res):
    """u*(1-alpha) a fine run, normalizzato al valore all'ingresso: deve valere 1 in ogni cella"""
    u_c = 0.5 * (res.u_face[:-1, -1] + res.u_face[1:, -1])
    return u_c * (1 - res.M10[:, -1]) / (grid.u * (1 - alpha_in))

print('\n=== Controllo di conservazione: u(1-alpha)/(u_b(1-alpha_in)), cella piu\' alta ===')
print(f"  u costante : {flusso_liquido(rc)[-1]:.4f}")
print(f"  u(z)       : {flusso_liquido(rv)[-1]:.4f}")

print('\n=== Differenza relativa u(z) rispetto a u costante, cella piu\' alta, a fine run ===')
for nome, a, b in (('M10 (gas fraction)', rv.M10, rc.M10),
                   ('M00 (bolle/m3)    ', rv.M00, rc.M00),
                   ('raggio medio      ', rv.R_mean, rc.R_mean)):
    print(f"  {nome}: {100 * (a[-1, -1] / b[-1, -1] - 1):+.2f} %")
print(f"  velocita' in uscita (faccia superiore): {rv.u_face[-1, -1]:.3f} m/s  (costante: {rc.u_face[-1, -1]:.3f})")

fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
for res, lab, st in ((rc, 'u costante', 'o--'), (rv, 'u(z)', 'o-')):
    axs[0].plot(res.z, res.M10[:, -1], st, label=lab)
    axs[1].plot(res.z, res.M00[:, -1], st, label=lab)
    axs[2].plot(res.z_face, res.u_face[:, -1], st, label=lab)
    u_c = 0.5 * (res.u_face[:-1, -1] + res.u_face[1:, -1])
    axs[3].plot(res.z, u_c * (1 - res.M10[:, -1]), st, label=lab)
axs[0].set_ylabel('Gas volume fraction (M$_{10}$)')
axs[1].set_ylabel('Bubbles per m$^3$ (M$_{00}$)')
axs[2].set_ylabel('Velocity at faces [m/s]')
axs[3].set_ylabel('u (1 - M$_{10}$)  [m/s]  (deve essere costante)')
for a in axs:
    a.set_xlabel('z [m]')
axs[0].legend(loc='best')
fig.tight_layout()
fig.savefig('Confronto_velocita.jpg', dpi=300)
plt.show()
