"""Modello a N celle (default 3) lungo il condotto: metodo delle linee.

Per ogni cella lo stato e' quello del codice 0-D (momenti M00, M10, M01 + massa disciolta c),
gli stessi termini sorgente di momsources, piu' un termine di avvezione upwind del primo ordine:

    dY_i/dt = -u_i (Y_i - Y_{i-1}) / dz + S(Y_i, p_l(z_i))

Convenzione: z cresce verso l'alto, z_top e' la superficie libera, la cella 0 e' la piu' profonda
e riceve in ingresso lo stato fisso Yin (cella 'fantasma' a monte).

Modello di velocita' (grid.velocityModel):
  'constant'   u = grid.u in tutto il condotto
  'continuity' u(z) dalla conservazione del volume: la velocita' alla base e' grid.u e cresce
               salendo per l'espansione delle bolle:  u_{i+1/2} = u_{i-1/2} + dz * div_u,i
               (div_u,i = divergenza della velocita' nella cella i, da momsources)
u_i qui e' la velocita' alla faccia inferiore della cella i (la faccia a monte per l'upwind).
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

    vel_model = getattr(grid, 'velocityModel', 'constant')
    if vel_model not in ('constant', 'continuity'):
        raise ValueError(f"velocityModel '{vel_model}' non riconosciuto: 'constant' o 'continuity'")
    var_u = (vel_model == 'continuity')

    funH = build_funH(Models, initial)
    rho_l = magma.liquid.rho_l
    PM = np.atleast_1d(magma.species.PM)

    # Geometria
    N = grid.N
    z_top = grid.z0 + N * grid.dz
    z = grid.z0 + (np.arange(N) + 0.5) * grid.dz       # centri cella, 0 = piu' profonda
    z_in = grid.z0 - 0.5 * grid.dz                     # cella fantasma di ingresso
    z_face = grid.z0 + np.arange(N + 1) * grid.dz      # facce delle celle (N+1)

    # DIPENDENZA DALLA QUOTA: pressione del liquido in ogni cella (costante nel tempo)
    pl = pressure_z(z, z_top, grid.p_top, rho_l)
    p_in = pressure_z(z_in, z_top, grid.p_top, rho_l)

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

    def sources_and_velocity(t, Y):
        """Sorgenti (N, nvar) e velocita' alle N+1 facce. Si procede dal basso verso l'alto:
        la velocita' della faccia superiore della cella i dipende solo dalle celle sotto."""
        S = np.empty_like(Y)
        u_face = np.empty(N + 1)
        u_face[0] = grid.u                             # velocita' alla base (dato in ingresso)
        for i in range(N):
            if var_u:
                S[i], div_u = momsources(t, Y[i], funH, cqmom, magma, Models, initial,
                                         pl=pl[i], return_div=True)
                u_face[i + 1] = u_face[i] + grid.dz * div_u
            else:
                S[i] = momsources(t, Y[i], funH, cqmom, magma, Models, initial, pl=pl[i])
                u_face[i + 1] = grid.u
        return S, u_face

    def rhs(t, yflat):
        Y = yflat.reshape(N, nvar)
        Yup = np.vstack([Yin, Y[:-1]])                 # upwind: cella a monte (sotto)
        S, u_face = sources_and_velocity(t, Y)
        adv = -u_face[:-1, None] * (Y - Yup) / grid.dz
        return (adv + S).ravel()

    dt = grid.CFL * grid.dz / grid.u
    print(f'\n--- Conduit ({vel_model}): N={N}, dz={grid.dz} m, u_base={grid.u} m/s, '
          f'CFL={grid.CFL} -> dt={dt} s ---')
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
    u_faces = np.empty((N + 1, t.size))
    for k in range(t.size):
        for i in range(N):
            Yfis[i, :, k] = renormalize(Yn[i, :, k], f, Ns, norm)
        u_faces[:, k] = sources_and_velocity(t[k], Yn[:, :, k])[1]

    M00, M10, M01 = Yfis[:, 0, :], Yfis[:, i_M10, :], Yfis[:, i_M01, :]
    Rs = Constants.R / PM[0]
    T = funH.thermoPT.T(t, Models.temperature)
    p_g = M01 / M10 * Rs * T
    R_mean = (3 / 4 / np.pi * M10 / M00) ** (1 / 3)

    res = SimpleNamespace(t=t, z=z, z_face=z_face, pl=pl, Y=Yfis, M00=M00, M10=M10, M01=M01,
                          p_g=p_g, R_mean=R_mean, u_face=u_faces, sol=sol)

    if var_u:
        # Controllo di conservazione: a regime u*(1-alpha) (flusso volumetrico di liquido) costante
        u_c = 0.5 * (u_faces[:-1, -1] + u_faces[1:, -1])
        print('u alle facce a fine run [m/s]:', np.round(u_faces[:, -1], 5))
        print('u*(1-M10) per cella a fine run:', np.round(u_c * (1 - M10[:, -1]), 5))

    res.diag = _diagnostics(res, grid, magma, rho_l, p_in, var_u)
    _print_diagnostics(res, grid)

    figs = _make_figures(res)
    figs.append(_make_diagnostic_figure(res))
    if save_out_figure:
        tag = 'Conduit' if not var_u else 'ConduitU'
        for i, fig in enumerate(figs, start=1):
            fig.savefig(f'{tag}_Fig{i}.jpg', dpi=300)
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

    fig3, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    axs[0].plot(res.z, res.pl * 1e-6, 'o--', label='Liquid')
    axs[0].plot(res.z, res.p_g[:, -1] * 1e-6, 'o-', label='Gas')
    axs[0].set_ylabel('Pressure [MPa]')
    axs[0].legend(loc='best')
    axs[1].plot(res.z, res.M10[:, -1], 'o-')
    axs[1].set_ylabel('Gas volume fraction (M$_{10}$)')
    axs[2].plot(res.z, res.R_mean[:, -1], 'o-')
    axs[2].set_ylabel('Average radius [m]')
    axs[3].plot(res.z_face, res.u_face[:, -1], 'o-')
    axs[3].set_ylabel('Velocity at cell faces [m/s]')
    for a in axs:
        a.set_xlabel('z [m]')
    fig3.suptitle('Profili a fine simulazione')
    fig3.tight_layout()

    return [fig1, fig2, fig3]


def _diagnostics(res, grid, magma, rho_l, p_in, var_u):
    """Grandezze per capire in che regime lavora il sistema (tutte a fine simulazione, tranne il ritardo)."""
    N = res.z.size
    mu = magma.liquid.mu
    t = res.t
    u_c = 0.5 * (res.u_face[:-1, :] + res.u_face[1:, :])        # velocita' al centro cella (N, nt)
    u_end = u_c[:, -1]
    phi_in = res.M10[0, 0]                                      # stato di ingresso = stato iniziale

    # 1) Ritardo del gas rispetto al liquido: > 0 vuol dire gas ancora sovrappressurizzato
    lag = res.p_g - res.pl[:, None]                             # [Pa], (N, nt)

    # 2) Riferimento di Boyle: bolle in equilibrio meccanico, gas ideale, massa di gas per bolla
    #    costante, tensione superficiale trascurata  ->  volume della bolla ~ 1/p.
    #    Il numero di bolle per volume di liquido non cambia, quindi g = phi/(1-phi) scala come v.
    g_in = phi_in / (1 - phi_in)
    g_eq = g_in * p_in / res.pl
    phi_boyle = g_eq / (1 + g_eq)

    # 3) Tempi caratteristici per cella
    tau_relax = 4 * mu / (3 * res.p_g[:, -1])                   # rilassamento viscoso (inverso di |autovalore|)
    tau_decomp = res.pl / (rho_l * Constants.g * u_end)         # tempo su cui la pressione cambia di un fattore ~e
    t_cell = grid.dz / u_end                                    # tempo di attraversamento della cella
    De = tau_relax / tau_decomp                                 # <<1: equilibrio (Boyle); >~1: dinamica di crescita

    # 4) Flusso volumetrico di liquido u(1-phi), normalizzato all'ingresso: deve valere 1 a regime
    flux = u_c * (1 - res.M10) / (res.u_face[0, :] * (1 - phi_in))

    # 5) Si e' raggiunto il regime stazionario? variazione relativa di M10 nella cella piu' alta
    k90 = np.searchsorted(t, 0.9 * t[-1])
    rel_change = abs(res.M10[-1, -1] - res.M10[-1, k90]) / res.M10[-1, -1]

    return SimpleNamespace(lag=lag, phi_boyle=phi_boyle, tau_relax=tau_relax, tau_decomp=tau_decomp,
                           t_cell=t_cell, De=De, flux=flux, rel_change=rel_change,
                           t_transit=np.sum(t_cell), phi_in=phi_in)


def _print_diagnostics(res, grid):
    d = res.diag
    N = res.z.size
    print('\n=== Diagnostica ===')
    print(f'Tempo di transito totale: {d.t_transit:.3g} s   (tout = {res.t[-1]:.3g} s)')
    print(f'Variazione relativa di M10 (cella alta) nell\'ultimo 10% del tempo: {d.rel_change:.2e}'
          + ('   -> regime raggiunto' if d.rel_change < 1e-3 else '   -> NON a regime: aumenta tout'))
    print(' cella   z[m]   p_l[kPa]  p_g-p_l[Pa]  phi      phi_Boyle  tau_rel[s]  tau_dec[s]  De      u(1-phi)/rif')
    for i in range(N):
        print(f' {i:3d}  {res.z[i]:7.1f} {res.pl[i]*1e-3:9.1f} {d.lag[i, -1]:11.1f}  {res.M10[i, -1]:.5f}  '
              f'{d.phi_boyle[i]:.5f}   {d.tau_relax[i]:9.3g}  {d.tau_decomp[i]:9.3g}  {d.De[i]:.2e}  {d.flux[i, -1]:.4f}')


def _make_diagnostic_figure(res):
    d = res.diag
    t, N = res.t, res.z.size
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))

    for i in range(N):
        axs[0].plot(t, d.lag[i] / res.pl[i] * 100, label=f'cella {i}')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('(p$_g$ - p$_l$) / p$_l$  [%]')
    axs[0].set_title('Ritardo del gas')
    axs[0].legend(loc='best')

    axs[1].plot(res.z, res.M10[:, -1], 'o-', label='simulazione')
    axs[1].plot(res.z, d.phi_boyle, 's--', label='Boyle (equilibrio)')
    axs[1].set_xlabel('z [m]')
    axs[1].set_ylabel('Gas volume fraction')
    axs[1].set_title('Confronto con Boyle')
    axs[1].legend(loc='best')

    axs[2].semilogy(res.z, d.tau_relax, 'o-', label=r'$\tau$ rilassamento')
    axs[2].semilogy(res.z, d.tau_decomp, 's-', label=r'$\tau$ decompressione')
    axs[2].semilogy(res.z, d.t_cell, '^-', label='transito cella')
    axs[2].set_xlabel('z [m]')
    axs[2].set_ylabel('Tempo [s]')
    axs[2].set_title('Scale di tempo')
    axs[2].legend(loc='best')

    axs[3].plot(res.z, d.flux[:, -1], 'o-')
    axs[3].axhline(1.0, color='k', lw=0.8, ls='--')
    axs[3].set_xlabel('z [m]')
    axs[3].set_ylabel('u(1 - M$_{10}$) / valore di ingresso')
    axs[3].set_title('Conservazione (deve valere 1)')

    fig.tight_layout()
    return fig
