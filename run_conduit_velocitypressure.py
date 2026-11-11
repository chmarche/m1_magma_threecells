import numpy as np

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


# Chiamata del solver
magma_conduit(time, grid, cqmom, initial, magma, Models, save_out_figure=True)
