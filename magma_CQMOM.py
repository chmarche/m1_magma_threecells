import re
import time as _time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from . import models
from .constants import Constants
from .momord import momord
from .momsources import momsources
from .normalization import normalized, renormalize

# Dizionario per confronto nomi dei solver di matlab e quelli di solve_ivp
_ODE_SOLVERS = {
    '15s': 'BDF',        # ode15s (stiff, formule BDF/NDF)  -> BDF
    '113': 'LSODA',      # ode113 (non stiff, Adams)         -> LSODA
    'BDF': 'BDF', 'Radau': 'Radau', 'LSODA': 'LSODA',
}


def magma_cqmom(time, cqmom, initial, magma, Models, save_out_figure=False, show=True):
    
    Ns = len(magma.species.name)
    if Ns != 1:
        raise NotImplementedError('Implementata una sola specie volatile (H2O).')
    if (cqmom.N1, cqmom.N2) != (1, 1):
        raise NotImplementedError('Implementato solo il caso monodisperso (N1 = N2 = 1).')

    # Get functions
    funH = SimpleNamespace(
        initMethod=models.get_init_method(initial.method),
        thermoPT=models.get_thermo_pt(Models.pressure.name, Models.temperature.name,
                                      Models.saturation.name),
        properties=models.get_properties(Models.surfaceTension.name, Models.diffusivity.name, Ns),
        massGrowth=models.get_mass_growth(Models.massGrowth.name),
        volumeGrowth=models.get_volume_growth(Models.volumeGrowth.name),
        nucleation=models.get_nucleation(Models.nucleation.name),
        coalescence=models.get_coalescence(Models.coalescence.name),
    )

    # Inizializzazione
    f = momord(cqmom.N1, cqmom.N2)
    num = len(f)
    i_M10 = 1
    i_M01 = 2 * cqmom.N1
    rho_l = magma.liquid.rho_l
    PM = np.atleast_1d(magma.species.PM)

    y0 = funH.initMethod(f, initial, magma, funH, Models)
    yl_rho_alpha0 = initial.yl * rho_l * (1 - y0[i_M10])     # massa disciolta / volume totale
    Ytot0 = np.concatenate([y0, [yl_rho_alpha0]])

    class Norm:
        pass

    norm = Norm()
    norm.w_0 = y0[0]
    norm.v_0 = y0[i_M10] / y0[0]
    norm.m_0 = y0[i_M01] / y0[0]
    norm.yl = initial.yl
    norm.yg = initial.yg

    # aggiungo f e norm agli oggetti passati
    cqmom.f = f
    initial.norm = norm
    Ytot0_norm = normalized(Ytot0, f, Ns, norm)

    # soluzione ode
    print('\n--- Starting simulation ---')
    method = _ODE_SOLVERS.get(cqmom.ODEsolver)
    if method is None:
        raise ValueError(f"ODEsolver '{cqmom.ODEsolver}' non riconosciuto: {list(_ODE_SOLVERS)}")

    def funsolve(t, Y):
        return momsources(t, Y, funH, cqmom, magma, Models, initial)

    t_start = _time.perf_counter()
    sol = solve_ivp(funsolve, (0.0, time.tout), Ytot0_norm, method=method,
                    rtol=1e-10, atol=1e-10)
    tcf = _time.perf_counter() - t_start
    if not sol.success:
        raise RuntimeError(f'Integrazione fallita: {sol.message}')
    print(f'--- Simulation completed in {tcf:.4f} s ({sol.t.size} steps) ---')

    # Post-processing
    t = sol.t
    Ytot = np.array([renormalize(sol.y[:, i], f, Ns, norm) for i in range(t.size)])
    Y = Ytot[:, :num]
    c = Ytot[:, num]                                         # massa disciolta / volume totale

    M00, M10, M01 = Y[:, 0], Y[:, i_M10], Y[:, i_M01]
    yl = c / rho_l / (1 - M10)                               # frazione in massa disciolta
    yg = np.ones(t.size)                                     # una sola specie: yg = 1
    rho_mix = M01 + (1 - M10) * rho_l
    ytot = (c + yg * M01) / rho_mix                          # frazione in massa totale di volatili
    Rs = Constants.R * (yg / PM[0])
    T = funH.thermoPT.T(t, Models.temperature)
    p_l = funH.thermoPT.p(t, Models.pressure)
    p_g = M01 / M10 * Rs * T                                 # pressione media del gas

    res = SimpleNamespace(t=t, Y=Y, f=f, yl=yl, yg=yg, ytot=ytot, p_g=p_g, p_l=p_l, T=T,
                          rho_mix=rho_mix, cpu_time=tcf, sol=sol)

    # Grafici
    print('\n--- Plot figures ---')
    figs = _make_figures(res, funH, Models, magma)

    if save_out_figure:
        print('\n--- Save figures ---')
        for i, fig in enumerate(figs, start=1):
            fig.savefig(f'Fig{i}.jpg', dpi=600)

    if show:
        plt.show()

    return res


def _make_figures(res, funH, Models, magma):
    t, Y = res.t, res.Y
    M00, M10, M01 = Y[:, 0], Y[:, 1], Y[:, 2]
    col = '-b'
    col2 = '--b'
    sp_name = re.sub(r'_(\d+|\w)', r'$_{\1}$', magma.species.name[0])   # H_2O -> H$_2$O

    # Fig. 1: pressione liquido/gas e temperatura
    fig1, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(t, res.p_l * 1e-6, col, label='Liquid')
    ax.plot(t, res.p_g * 1e-6, col2, label='Gas')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Liquid/Gas pressure [MPa]')
    ax2 = ax.twinx()
    ax2.plot(t, res.T, color='tab:orange')
    ax2.set_ylabel('Temperature [K]')
    ax.legend(loc='best')
    fig1.tight_layout()

    # Fig. 2: momenti M00, M10, M01
    fig2, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(t, M00, col); axs[0].set_ylabel('N bubbles per unit volume (M$_{00}$) [m$^{-3}$]')
    axs[1].plot(t, M10, col); axs[1].set_ylabel('Gas volume fraction (M$_{10}$)')
    axs[2].plot(t, M01, col); axs[2].set_ylabel('Gas mass per unit volume (M$_{01}$) [kg m$^{-3}$]')
    for a in axs:
        a.set_xlabel('Time [s]')
    fig2.tight_layout()

    # Fig. 3: raggio medio, massa media, densita' media del gas
    fig3, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(t, (3 / 4 / np.pi * M10 / M00) ** (1 / 3), col); axs[0].set_ylabel('Average radius [m]')
    axs[1].plot(t, M01 / M00, col); axs[1].set_ylabel('Average mass [kg]')
    axs[2].plot(t, M01 / M10, col); axs[2].set_ylabel('Average gas density [kg m$^{-3}$]')
    for a in axs:
        a.set_xlabel('Time [s]')
    fig3.tight_layout()

    # Fig. 4: H2O disciolta vs solubilita' di equilibrio
    cs = funH.thermoPT.yl(res.p_g, res.T, res.ytot, Models.saturation.parameters[0])
    fig4, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(t, res.yl * 100, col, label='Dissolved')
    ax.plot(t, np.broadcast_to(cs, t.shape) * 100, col2, label='Equilibrium')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'Dissolved {sp_name} (wt%)')
    ax.legend(loc='best')
    fig4.tight_layout()

    # Fig. 5: H2O nel gas vs composizione di equilibrio (una sola specie: 100 wt%)
    cgs = funH.thermoPT.yg(res.p_g, res.T, res.ytot, Models.saturation.parameters[0])
    fig5, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(t, res.yg * 100, col, label='Gas')
    ax.plot(t, np.broadcast_to(cgs, t.shape) * 100, col2, label='Equilibrium')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'Gas {sp_name} (wt%)')
    ax.legend(loc='best')
    fig5.tight_layout()

    return [fig1, fig2, fig3, fig4, fig5]
