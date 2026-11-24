"""Closure laws: equivalenti delle function handle get di matlab"""
from types import SimpleNamespace
import numpy as np
from .constants import Constants


# Inizializzazione dei momenti
def get_init_method(method_name):
    if method_name != 'discrete':
        raise ValueError(f"Metodo di inizializzazione '{method_name}' non implementato")
    return discrete_initialize


def discrete_initialize(f, initial, magma, funH, Models):
    """Funzione dei momenti di una distribuzione discreta."""
    diam = np.atleast_1d(np.asarray(initial.diam, dtype=float))
    R0 = diam / 2
    Rs = Constants.R * np.sum(np.atleast_1d(initial.yg) / np.atleast_1d(magma.species.PM))
    sigma = 0.0  # come nel MATLAB: la tensione superficiale non entra nella pressione iniziale del gas
    rho_g0 = (Models.pressure.p0 + sigma * 2.0 / R0) / (Rs * Models.temperature.T0)
    w = np.atleast_1d(np.asarray(initial.wt, dtype=float))

    v = np.pi / 6 * diam ** 3        # volume di ogni bolla
    m = rho_g0 * v                   # massa di gas di ogni bolla

    mom = np.zeros(len(f))
    for p in range(len(f)):
        k, l = f[p, 0], f[p, 1]
        mom[p] = np.sum(w * v ** k * m ** l)
    return mom

# Pressione, temperatura, saturazione
def get_thermo_pt(model_pressure, model_temperature, model_saturation):
    if model_pressure != 'linear':
        raise ValueError(f"Modello di pressione '{model_pressure}' non implementato")
    if model_temperature != 'linear':
        raise ValueError(f"Modello di temperatura '{model_temperature}' non implementato")

    def p(t, par):
        pmin = getattr(par, 'pMin', Constants.p_atm)   # default: 101325 Pa
        return np.maximum(par.p0 + par.dpdt * t, pmin)

    def T(t, par):
        return np.maximum(par.T0 + par.dTdt * t, 273.15)

    # Solubilita' dell'H2O nel fuso (frazione in massa).
    if model_saturation == 'Henry':
        def yl(p, T, ytot, par):
            return par.hcp * np.sqrt(p)
    elif model_saturation == 'constant':
        def yl(p, T, ytot, par):
            return par.yl_const
    else:
        raise ValueError(f"Modello di saturazione '{model_saturation}' non implementato")

    def yg(p, T, ytot, par):
        return 1.0

    return SimpleNamespace(p=p, T=T, yl=yl, yg=yg)

# Proprieta': tensione superficiale e diffusivita
def get_properties(model_surface_tension, model_diffusivity):
    if model_surface_tension != 'constant':
        raise ValueError(f"Modello di tensione superficiale '{model_surface_tension}' non implementato")

    def sigma(T, yl, ytot, funH, Models):
        return Models.surfaceTension.sigma

    def D_constant(yl, p, T, Is, Models):
        return np.atleast_1d(Models.diffusivity.D)[Is]

    def D_nowak_behrens(yl, p, T, Is, Models):
        """Diffusivita' dell'acqua in fusi granitici (Nowak e Behrens, 1997; Eq. 6 del paper).
        Calibrata per 1073-1473 K, 50-500 MPa, 0.5-6 wt% di H2O.   Ritorna D in m^2/s."""
        Cw = 100.0 * np.atleast_1d(yl)[Is]      # contenuto d'acqua [wt%]
        P = p / 1e8                              # pressione [kbar]
        logD = (-4.81 - 0.045 * Cw + 0.027 * Cw ** 2) \
               - (3378 - 483 * Cw + 46.9 * Cw ** 2 + 47.5 * P) / T     # log10(D [cm^2/s])
        return 10.0 ** logD * 1e-4               # cm^2/s -> m^2/s

    funs = []
    for name in model_diffusivity:
        if name == 'constant':
            funs.append(D_constant)
        elif name == 'NowakBehrens':
            funs.append(D_nowak_behrens)
        else:
            raise ValueError(f"Modello di diffusivita' '{name}' non implementato")

    return SimpleNamespace(sigma=sigma, D=funs)


# Crescita di massa
def get_mass_growth(model_mass_growth):
    if model_mass_growth == 'zero':
        def Psi(v, m, yl, yg, ytot, p, T, Is, magma, funH, Models):
            return 0.0

    elif model_mass_growth == 'diffusion':
        # Flusso di Fick verso una bolla isolata con profilo di concentrazione rilassato
        # (Eq. 17 del paper):   Psi = 4 pi R rho_l D (y_l - y_i)   [kg/s per bolla]
        #   y_l = frazione in massa della specie disciolta nel fuso (media)
        #   y_i = solubilita' all'interfaccia, in equilibrio con il gas della bolla (pressione p_g)
        def Psi(v, m, yl, yg, ytot, p, T, Is, magma, funH, Models):
            beta = Constants.beta
            R = (v * beta) ** (1 / 3)        # beta = 3/(4 pi): R = (3 v / 4 pi)^(1/3)
            Rs = Constants.R * np.sum(np.atleast_1d(yg) / np.atleast_1d(magma.species.PM))
            p_g = m * Rs * T / v                                      # pressione del gas nella bolla
            y_i = funH.thermoPT.yl(p_g, T, ytot, Models.saturation)   # solubilita' alla p_g
            D = funH.properties.D[Is](yl, p, T, Is, Models)      # p = pressione del liquido
            return 4 * np.pi * R * magma.liquid.rho_l * D * (np.atleast_1d(yl)[Is] - y_i)

    else:
        raise ValueError(f"Modello di crescita di massa '{model_mass_growth}' non implementato")

    return SimpleNamespace(Psi=Psi)


# Crescita di volume: Rayleigh-Plesset
def get_volume_growth(model_volume_growth):
    if model_volume_growth != 'Rayleigh-Plesset':
        raise ValueError(f"Modello di crescita di volume '{model_volume_growth}' non implementato")

    def Gamma(v, m, yl, yg, ytot, p, T, magma, funH, Models):
        Rs = Constants.R * np.sum(np.atleast_1d(yg) / np.atleast_1d(magma.species.PM))
        beta = Constants.beta
        Scube = 3 / (4 * np.pi) * (magma.bubble.vs + v)
        mu_eff = magma.liquid.mu * (1 - (beta * v) / Scube)   # viscosita' efficace, mu liquido costante
        pGv = m * Rs * T                                      # gas perfetto: p_g * v
        sigma = funH.properties.sigma(T, yl, ytot, funH, Models)
        return 3 / (4 * mu_eff) * (pGv - p * v - 2 * sigma * v ** (2 / 3) / beta ** (1 / 3))

    return SimpleNamespace(Gamma=Gamma)


# Nucleazione
def get_nucleation(model_nucleation):
    if model_nucleation != 'zero':
        raise ValueError(f"Modello di nucleazione '{model_nucleation}' non implementato")

    def J(yl, yg, ytot, p, T, magma, funH, Models):
        return 0.0, 0.0, 0.0     # J, v_c, m_c

    return SimpleNamespace(J=J)


# Coalescenza
def get_coalescence(model_coalescence):
    if model_coalescence != 'constant':
        raise ValueError(f"Modello di coalescenza '{model_coalescence}' non implementato")

    def H(vi, vj, mi, mj, yl, yg, ytot, p_l, T, magma, E, funH, Models):
        return Models.coalescence.const

    return SimpleNamespace(H=H)
