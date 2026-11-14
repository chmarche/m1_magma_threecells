"""Caso 3: condotto a N celle, monodisperso, due specie volatili (H2O, CO2), crescita di volume (RP) e
crescita di massa per diffusione, solubilita' dalla tabella SOLWCAD (lookupTable.npz).

Parametri come run_monodisperse_mGrowth.m: p_in = 200 MPa, T = 1373.15 K, alpha0 = 0.0594, d = 50 um,
D = [1e-11, 1e-12] m^2/s, mu = 1e4 Pa s, rho_l = 2400, sigma = 0.15 N/m.
La velocita' alla base e' quella che da' la stessa decompressione del caso 0-D (dp/dt = -1e6 Pa/s):
u = 1e6 / (rho_l g) = 42.5 m/s.
"""
import numpy as np

from src.constants import Constants
from src.conduit import magma_conduit
from src.init_saturated import saturated_initial
from src.lookup import LookupTable


class Time:
    tout = 400.0      # s  (qualche tempo di transito: N*dz/u ~ 100 s)
time = Time()

class CQMOM:
    N1 = 1
    N2 = 1
    ODEsolver = '15s'
cqmom = CQMOM()

# Griglia: p_l all'ingresso = 200 MPa; la tabella copre 80-220 MPa, quindi tutte le celle devono
# restare sopra 80 MPa (controllo sotto).
class Grid:
    N = 3
    dz = 1500.0
    u = 42.5                       # velocita' alla base (m/s)
    CFL = 0.5
    z0 = 0.0
    p_top = None                   # calcolata sotto dalla pressione di ingresso
    velocityModel = 'continuity'   # 'constant' | 'continuity'
grid = Grid()

class Liquid:
    mu = 1e4
    rho_l = 2400.0

class Species:
    name = ['H_2O', 'CO_2']
    v_mol = [25e-6, 25e-6]
    PM = [18.02e-3, 44.01e-3]

class Magma:
    liquid = Liquid()
    species = Species()
magma = Magma()

class Pressure:
    name = 'linear'; dpdt = 0.0; p0 = 0.0; pMin = 1e3     # non usati: p dipende da z
class Temperature:
    name = 'linear'; dTdt = 0.0; T0 = 1373.15
class Models:
    pressure = Pressure()
    temperature = Temperature()

p_in = 200e6
z_top = grid.z0 + grid.N * grid.dz
z_in = grid.z0 - 0.5 * grid.dz
grid.p_top = p_in - magma.liquid.rho_l * Constants.g * (z_top - z_in)
p_cells = grid.p_top + magma.liquid.rho_l * Constants.g * (z_top - (grid.z0 + (np.arange(grid.N) + .5) * grid.dz))
print(f'p_top = {grid.p_top*1e-6:.1f} MPa, p celle = {np.round(p_cells*1e-6, 1)} MPa')
assert p_cells.min() > 80e6, 'celle fuori dalla griglia della tabella (80-220 MPa)'

# Saturazione: tabella SOLWCAD
class Saturation:
    name = 'Lookuptable'
    table = LookupTable('lookupTable.npz')
Models.saturation = Saturation()

class SurfaceTension:
    name = 'constant'; sigma = 0.15
class Diffusivity:
    name = ['constant', 'constant']
    D = [1e-11, 1e-12]
Models.surfaceTension = SurfaceTension()
Models.diffusivity = Diffusivity()

class VolumeGrowth:
    name = 'Rayleigh-Plesset'
class MassGrowth:
    name = 'diffusion'
class Nucleation:
    name = 'zero'
class Coalescence:
    name = 'constant'; const = 0.0
Models.volumeGrowth = VolumeGrowth()
Models.massGrowth = MassGrowth()
Models.nucleation = Nucleation()
Models.coalescence = Coalescence()

# Condizioni iniziali: bolle uguali (d, alpha0) in fuso saturo a p_in
class Initial:
    method = 'discrete'
    alpha0 = 0.0594
    diam = 50e-6
    Nb = 6 / np.pi * alpha0 / (1 - alpha0) / diam**3      # bolle per m3 di fuso
    meltvolume = 1 / Nb
    gasvolume = np.pi * diam**3 / 6
    alpha = gasvolume / (gasvolume + meltvolume)
    wt = Nb * (1 - alpha)
initial = Initial()

initial.yl, initial.yg, ytot_in = saturated_initial(
    Models.saturation.table, p_in, Models.temperature.T0, initial.alpha0, magma.liquid.rho_l,
    magma.species.PM, ytot_guess=[0.053, 0.0067])
print(f'Nb = {initial.Nb:.3e} 1/m3;  yl_in = {initial.yl};  yg_in = {initial.yg};  ytot_in = {ytot_in}')

class Bubble:
    vs = (1 / initial.alpha - 1) * np.pi / 6 * initial.diam**3
magma.bubble = Bubble()

if __name__ == '__main__':
    res = magma_conduit(time, grid, cqmom, initial, magma, Models, save_out_figure=True)
    np.savez('risultati_case3.npz', t=res.t, z=res.z, pl=res.pl, M00=res.M00, M10=res.M10, M01=res.M01,
             cs=res.cs, cg=res.cg, p_g=res.p_g, u_face=res.u_face)
