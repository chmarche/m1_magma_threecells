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
    if Ns not in (1, 2):
        raise NotImplementedError('Implementate una (H2O) o due (H2O, CO2) specie volatili.')
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
    # Massa disciolta di ogni specie per unita' di volume totale; massa di gas delle prime Ns-1 specie
    c0 = np.atleast_1d(initial.yl) * rho_l * (1 - y0[i_M10])
    cg0 = np.atleast_1d(initial.yg)[:Ns - 1] * y0[i_M01]
    Ytot0 = np.concatenate([y0, c0, cg0])

    class Norm:
        pass

    norm = Norm()
    norm.w_0 = y0[0]
    norm.v_0 = y0[i_M10] / y0[0]
    norm.m_0 = y0[i_M01] / y0[0]
    if Ns == 1:
        norm.yl = initial.yl                           # (scala storica dei casi 1 e 2)
        norm.yg = initial.yg
    else:
        norm.yl = c0                                   # scale = valori iniziali: variabili O(1)
        norm.yg = cg0
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
    T = funH.thermoPT.T(t, Models.temperature)

    # Specie: massa disciolta cs e massa di gas cg (per m3 di miscela), forma (Ns, N, nt)
    cs = Yfis[:, num:num + Ns, :].transpose(1, 0, 2)
    cg = np.empty_like(cs)
    cg[:Ns - 1] = Yfis[:, num + Ns:num + 2 * Ns - 1, :].transpose(1, 0, 2)
    cg[Ns - 1] = M01 - cg[:Ns - 1].sum(axis=0)         # l'ultima specie e' il complemento
    yg_all = cg / M01                                  # composizione del gas (Ns, N, nt)
    Rs = Constants.R * np.sum(yg_all / PM[:, None, None], axis=0)
    p_g = M01 / M10 * Rs * T
    R_mean = (3 / 4 / np.pi * M10 / M00) ** (1 / 3)

    res = SimpleNamespace(t=t, z=z, z_face=z_face, pl=pl, Y=Yfis, M00=M00, M10=M10, M01=M01,
                          c=cs[0], cs=cs, cg=cg, yg=yg_all, names=list(magma.species.name),
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

    mass_on = (Models.massGrowth.name != 'zero')
    if mass_on:
        res.mass = _mass_diagnostics(res, grid, magma, rho_l, funH, Models)
        _print_mass_diagnostics(res)
        figs.append(_make_mass_figure(res))
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


def _mass_diagnostics(res, grid, magma, rho_l, funH, Models):
    """Diagnostica del trasferimento di massa per ogni specie: sovrasaturazione, flussi conservati,
    tempo di trasferimento di massa."""
    Ns, N, nt = res.cs.shape
    T = funH.thermoPT.T(0.0, Models.temperature)
    u_c = 0.5 * (res.u_face[:-1, :] + res.u_face[1:, :])
    u_b = res.u_face[0, :]
    rho_mix = res.M01 + (1 - res.M10) * rho_l

    yl = res.cs / (rho_l * (1 - res.M10))                  # frazione in massa disciolta (Ns, N, nt)
    ytot = (res.cs + res.cg) / rho_mix                     # frazione in massa totale di volatili

    # Solubilita' alla pressione del liquido (equilibrio con una bolla alla p_l): usa i totali locali
    y_eq = np.empty((Ns, N, nt))
    for i in range(N):
        for k in range(nt):
            for s_ in range(Ns):
                y_eq[s_, i, k] = funH.thermoPT.yl(res.pl[i], T, ytot[:, i, k], Models.saturation, s_)
    d_y = yl - y_eq                                        # > 0: sovrasaturo

    # Invarianti a regime, normalizzati al valore di ingresso (devono valere 1):
    #   - flusso di ogni specie (disciolta + gas):   u (c_s + cg_s)
    #   - flusso di fuso senza volatili (secco):     u (rho_l (1-phi) - sum_s c_s)
    J_s = u_c * (res.cs + res.cg)
    J_s = J_s / (u_b[0] * (res.cs[:, 0, 0] + res.cg[:, 0, 0]))[:, None, None]
    cdis = res.cs.sum(axis=0)
    J_d = u_c * (rho_l * (1 - res.M10) - cdis)
    J_d = J_d / (u_b[0] * (rho_l * (1 - res.M10[0, 0]) - cdis[0, 0]))

    # Tempo di trasferimento di massa per specie: massa di quella specie nella bolla / |flusso|
    tau_mass = np.empty((Ns, N))
    for i in range(N):
        v = res.M10[i, -1] / res.M00[i, -1]
        m = res.M01[i, -1] / res.M00[i, -1]
        for s_ in range(Ns):
            psi = funH.massGrowth.Psi(v, m, yl[:, i, -1], res.yg[:, i, -1], ytot[:, i, -1],
                                      res.pl[i], T, s_, magma, funH, Models)
            tau_mass[s_, i] = m * res.yg[s_, i, -1] / abs(psi) if psi != 0 else np.inf
    return SimpleNamespace(yl=yl, y_eq=y_eq, d_y=d_y, ytot=ytot, J_s=J_s, J_w=J_s[0], J_d=J_d,
                           tau_mass=tau_mass)


def _print_mass_diagnostics(res):
    d, ms = res.diag, res.mass
    for s_, name in enumerate(res.names):
        print(f'\n=== Diagnostica trasferimento di massa ({name}) ===')
        print(' cella   z[m]   yl[-]       y_eq(p_l)   yl-y_eq      tau_mass[s]  t_cella[s]  J_specie/rif  J_secco/rif')
        for i in range(res.z.size):
            print(f' {i:3d}  {res.z[i]:7.1f}  {ms.yl[s_, i, -1]:.6f}  {ms.y_eq[s_, i, -1]:.6f}  '
                  f'{ms.d_y[s_, i, -1]:+.6f}   {ms.tau_mass[s_, i]:10.3g}  {d.t_cell[i]:10.3g}   '
                  f'{ms.J_s[s_, i, -1]:.4f}        {ms.J_d[i, -1]:.4f}')
    if len(res.names) > 1:
        print('\nFrazione in massa di ' + res.names[0] + ' nel gas a fine run: '
              + ', '.join(f'{x:.4f}' for x in res.yg[0, :, -1]))


def _make_mass_figure(res):
    ms = res.mass
    t, N, Ns = res.t, res.z.size, len(res.names)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    ref = ms.yl[:, 0, 0]                                   # contenuto disciolto in ingresso, per specie
    colors = [f'C{i}' for i in range(N)]
    styles = ['-', '--', ':']

    for s_, name in enumerate(res.names):
        axs[0].plot(res.z, ms.yl[s_, :, -1] / ref[s_], 'o-', color=f'C{s_}', label=f'{name} disciolta')
        axs[0].plot(res.z, ms.y_eq[s_, :, -1] / ref[s_], 's--', color=f'C{s_}', label=f'{name} solubilita\' (p$_l$)')
    axs[0].set_xlabel('z [m]')
    axs[0].set_ylabel('y$_l$ / y$_l$(ingresso)')
    axs[0].legend(loc='best', fontsize=8)

    for i in range(N):
        for s_ in range(Ns):
            axs[1].plot(t, ms.d_y[s_, i] / ref[s_], color=colors[i], ls=styles[s_],
                        label=f'cella {i}, {res.names[s_]}')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('(y$_l$ - y$_{eq}$) / y$_l$(ingresso)  (>0: sovrasaturo)')
    axs[1].legend(loc='best', fontsize=7)

    for i in range(N):
        axs[2].plot(t, res.M01[i], color=colors[i], label=f'cella {i}')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('M$_{01}$: gas mass per m$^3$ [kg/m$^3$]')
    axs[2].legend(loc='best')

    for s_, name in enumerate(res.names):
        axs[3].plot(res.z, ms.J_s[s_, :, -1], 'o-', label=f'flusso {name}')
    axs[3].plot(res.z, ms.J_d[:, -1], 's-', label='flusso di fuso secco')
    axs[3].axhline(1.0, color='k', lw=0.8, ls='--')
    axs[3].set_xlabel('z [m]')
    axs[3].set_ylabel('flusso / valore di ingresso')
    axs[3].set_title('Conservazione (devono valere 1)')
    axs[3].legend(loc='best', fontsize=8)

    fig.tight_layout()
    return fig
