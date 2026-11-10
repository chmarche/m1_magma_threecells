# NB rimodificare z_0 con z_top perchè fa riferimento alla superficie del condotto, mentre 
# z0 fa riferimento alla base

""" 
    Modello con 3 celle caratteristiche nel condotto. metodo delle linee
    Convenzione: asse z verso l'alto, z_0 superficie libera e la prima cella (0) è la più profonda
    e riceve in ingresso lo stato fisso Yin
"""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from . import models
from .constants import Constants
from .momord import momord
from .momsources import momsources
from .normalization import normalized, renormalize


def pressure_z(z, z_top, p_top, rho_l):
    """Pressione idrostatica del liquido alla quota z: aumenta andando in profondita'."""
    return p_top + rho_l * Constants.g * (z_top - z)


def build_funH(Models, initial):
    return SimpleNamespace(
        thermoPT=models.get_thermo_pt(Models.pressure.name, Models.temperature.name,
                                      Models.saturation.name),
        properties=models.get_properties(Models.surfaceTension.name, Models.diffusivity.name),
        massGrowth=models.get_mass_growth(Models.massGrowth.name),
        volumeGrowth=models.get_volume_growth(Models.volumeGrowth.name),
        nucleation=models.get_nucleation(Models.nucleation.name),
        coalescence=models.get_coalescence(Models.coalescence.name),
        initMethod=models.get_init_method(initial.method),
    )


def magma_conduit(time, grid, cqmom, initial, magma, Models, save_out_figure=False, show=True):

    Ns = len(magma.species.name)
    if Ns != 1:
        raise NotImplementedError('Implementata una sola specie volatile (H2O).')
    if (cqmom.N1, cqmom.N2) != (1, 1):
        raise NotImplementedError('Implementato solo il caso monodisperso (N1 = N2 = 1).')

    funH = build_funH(Models, initial)
    rho_l = magma.liquid.rho_l
    PM = np.atleast_1d(magma.species.PM)

    # Geometria
    N = grid.N
    z_top = grid.z0 + N * grid.dz
    z = grid.z0 + (np.arange(N) + 0.5) * grid.dz       # centri cella, 0 = piu' profonda
    z_in = grid.z0 - 0.5 * grid.dz                     # cella fantasma di ingresso

    # DIPENDENZA DALLA QUOTA: pressione del liquido in ogni cella (costante nel tempo)
    pl = pressure_z(z, z_top, grid.p_top, rho_l)
    p_in = pressure_z(z_in, z_0, grid.p_top, rho_l)

    # Stato di ingresso: bolle in equilibrio meccanico con il liquido alla pressione di ingresso
    Models.pressure.p0 = p_in
    f = momord(cqmom.N1, cqmom.N2)
    num = len(f)
    i_M10 = 1
    i_M01 = 2 * cqmom.N1

    y0 = funH.initMethod(f, initial, magma, funH, Models)
    c0 = initial.yl * rho_l * (1 - y0[i_M10])
    Ytot0 = np.concatenate([y0, [c0]])

    class Norm:
        pass

    norm = Norm()
    norm.w_0 = y0[0]
    norm.v_0 = y0[i_M10] / y0[0]
    norm.m_0 = y0[i_M01] / y0[0]
    norm.yl = initial.yl
    norm.yg = initial.yg
    cqmom.f = f
    initial.norm = norm

    Yin = normalized(Ytot0, f, Ns, norm)
    nvar = Yin.size
    y_init = np.tile(Yin, N)                           # tutte le celle partono dallo stato di ingresso

    def rhs(t, yflat):
        Y = yflat.reshape(N, nvar)
        Yup = np.vstack([Yin, Y[:-1]])                 # upwind: cella a monte (sotto)
        adv = -grid.u * (Y - Yup) / grid.dz
        S = np.empty_like(Y)
        for i in range(N):
            S[i] = momsources(t, Y[i], funH, cqmom, magma, Models, initial, pl=pl[i])
        return (adv + S).ravel()

    dt = grid.CFL * grid.dz / grid.u
    print(f'\n--- Conduit: N={N}, dz={grid.dz} m, u={grid.u} m/s, CFL={grid.CFL} -> dt={dt} s ---')
    print(f'p_l celle [Pa] = {pl},  p_in = {p_in:.1f} Pa')

    sol = solve_ivp(rhs, (0.0, time.tout), y_init, method='BDF', max_step=dt,
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f'Integrazione fallita: {sol.message}')
    print(f'--- Done ({sol.t.size} steps) ---')

    # Post-processing: Yfis[i, k, t]
    t = sol.t
    Yn = sol.y.reshape(N, nvar, t.size)
    Yfis = np.empty_like(Yn)
    for i in range(N):
        for k in range(t.size):
            Yfis[i, :, k] = renormalize(Yn[i, :, k], f, Ns, norm)

    M00, M10, M01 = Yfis[:, 0, :], Yfis[:, i_M10, :], Yfis[:, i_M01, :]
    Rs = Constants.R / PM[0]
    T = funH.thermoPT.T(t, Models.temperature)
    p_g = M01 / M10 * Rs * T
    R_mean = (3 / 4 / np.pi * M10 / M00) ** (1 / 3)

    res = SimpleNamespace(t=t, z=z, pl=pl, Y=Yfis, M00=M00, M10=M10, M01=M01,
                          p_g=p_g, R_mean=R_mean, sol=sol)

    figs = _make_figures(res)
    if save_out_figure:
        for i, fig in enumerate(figs, start=1):
            fig.savefig(f'Conduit_Fig{i}.jpg', dpi=300)
    if show:
        plt.show()
    return res


def _make_figures(res):
    t, N = res.t, res.z.size
    labels = [f'cella {i} (z={res.z[i]:.1f} m)' for i in range(N)]

    fig1, ax = plt.subplots(figsize=(6.5, 5))
    for i in range(N):
        l, = ax.plot(t, res.p_g[i] * 1e-6, label=labels[i])
        ax.axhline(res.pl[i] * 1e-6, color=l.get_color(), ls='--', lw=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Gas pressure (solid) / liquid pressure (dashed) [MPa]')
    ax.legend(loc='best')
    fig1.tight_layout()

    fig2, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for i in range(N):
        axs[0].plot(t, res.M10[i], label=labels[i])
        axs[1].plot(t, res.R_mean[i], label=labels[i])
    axs[0].set_ylabel('Gas volume fraction (M$_{10}$)')
    axs[1].set_ylabel('Average radius [m]')
    for a in axs:
        a.set_xlabel('Time [s]')
    axs[0].legend(loc='best')
    fig2.tight_layout()

    fig3, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    axs[0].plot(res.z, res.pl * 1e-6, 'o--', label='Liquid')
    axs[0].plot(res.z, res.p_g[:, -1] * 1e-6, 'o-', label='Gas')
    axs[0].set_ylabel('Pressure [MPa]')
    axs[0].legend(loc='best')
    axs[1].plot(res.z, res.M10[:, -1], 'o-')
    axs[1].set_ylabel('Gas volume fraction (M$_{10}$)')
    axs[2].plot(res.z, res.R_mean[:, -1], 'o-')
    axs[2].set_ylabel('Average radius [m]')
    for a in axs:
        a.set_xlabel('z [m]')
    fig3.suptitle('Profili a fine simulazione')
    fig3.tight_layout()

    return [fig1, fig2, fig3]
