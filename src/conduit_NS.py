from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from . import models
from .constants import Constants
from .momord import momord
from .momsources import add_dilution, sources_monodisperse
from .equilibrium import boyle_initial_state


# ----------------------------------------------------------------------------------------------
# Pressione
# ----------------------------------------------------------------------------------------------
def pressure_z(z, z_top, p_top, rho_l):
    """Pressione idrostatica del liquido alla quota z: aumenta andando in profondita'."""
    return p_top + rho_l * Constants.g * (z_top - z)


def pressure_momentum(a, dz, p_top, rho):
    """Pressione dalla quantita' di moto: integrazione dall'alto (p_top) verso il basso.
    a = accelerazione materiale al centro cella (N,); rho = densita' (scalare o (N,)).
    Ritorna (p al centro cella, p alle facce)."""
    dp = rho * (Constants.g + a) * dz                         # salto di pressione di ogni cella
    p_face = p_top + np.concatenate([np.cumsum(dp[::-1])[::-1], [0.0]]) #somma il dp ad ogni cella dal fondo al top
    return 0.5 * (p_face[:-1] + p_face[1:]), p_face #restituisce p in mezzo e poi p alla faccia


def boyle_column(p_top, dz, N, rho_l, bubble_state):
    """Colonna statica con bolle in equilibrio di Boyle, dall'alto (p_top) verso il basso.

    bubble_state(p_in) -> (M00, M10, M01) dello stato di ingresso alla pressione p_in.
    Per ogni p_in: Nb = M00/(1-M10) (bolle per volume di fuso) e la massa per bolla m restano
    fissi, il volume cala come 1/p:  g = Nb v = phi/(1-phi) = g_in p_in / p.
    rho_mix(p) = (1 - phi) (m Nb + rho_l).   La pressione al fondo dipende da p_in (piu' p_in,
    piu' gas, meno densita'): si itera p_in = p_fondo(p_in), contrazione molto rapida.
    Ritorna (p_in, p centri cella (N,), p facce (N+1,))."""
    def column(p_in):
        M00, M10, M01 = bubble_state(p_in)
        Nb, v_in, m = M00 / (1 - M10), M10 / M00, M01 / M00

        def rho(p):
            g_ = Nb * v_in * p_in / p
            return (m * Nb + rho_l) / (1 + g_)            # (1 - phi) (m Nb + rho_l), 1 - phi = 1/(1+g)
        p_f = np.empty(N + 1)
        p_f[N] = p_top
        p_c = np.empty(N)
        for i in range(N - 1, -1, -1):
            pc = p_f[i + 1]
            for _ in range(50):                       # p_c = p_faccia_sopra + rho(p_c) g dz / 2
                pc_new = p_f[i + 1] + 0.5 * rho(pc) * Constants.g * dz
                if abs(pc_new - pc) < 1e-9 * pc_new:
                    pc = pc_new
                    break
                pc = pc_new
            p_c[i] = pc
            p_f[i] = p_f[i + 1] + rho(pc) * Constants.g * dz
        return p_c, p_f

    p_in = p_top + rho_l * Constants.g * N * dz          # primo tentativo: solo liquido
    for _ in range(100):
        p_c, p_f = column(p_in)
        if abs(p_f[0] - p_in) < 1e-9 * p_in:
            break
        p_in = p_f[0]
    return p_f[0], p_c, p_f


def build_funH(Models, initial):
    Ns = len(Models.diffusivity.name)
    return SimpleNamespace(
        thermoPT=models.get_thermo_pt(Models.pressure.name, Models.temperature.name,
                                      Models.saturation.name),
        properties=models.get_properties(Models.surfaceTension.name, Models.diffusivity.name, Ns),
        massGrowth=models.get_mass_growth(Models.massGrowth.name),
        volumeGrowth=models.get_volume_growth(Models.volumeGrowth.name),
        nucleation=models.get_nucleation(Models.nucleation.name),
        coalescence=models.get_coalescence(Models.coalescence.name),
        initMethod=models.get_init_method(initial.method),
    )


# ----------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------
def solve_conduit(time, grid, cqmom, initial, magma, Models, verbose=True):
    """Integra il condotto. Ritorna `res` con, per N celle e nt istanti:

      t (nt,)  z (N,)  z_face (N+1,)
      Y (N, 4, nt) stato fisico [M00, M10, M01, c];  M00, M10, M01, c (N, nt)
      p_g (N, nt) pressione del gas;  R_mean (N, nt) raggio medio
      pl_t (N, nt), p_face (N+1, nt)  pressione del liquido a centro cella / alle facce
      pl (N,) = pl_t a fine run;  pl_hydro (N,), p_face_hydro (N+1,) profilo idrostatico
      u_face (N+1, nt) velocita' alle facce;  p_in, n_noconv, sol
    """
    if len(magma.species.name) != 1:
        raise NotImplementedError('Implementata una sola specie volatile (H2O).')
    if (cqmom.N1, cqmom.N2) != (1, 1):
        raise NotImplementedError('Implementato solo il caso monodisperso (N1 = N2 = 1).')
    if Models.nucleation.name != 'zero':
        raise NotImplementedError('Nucleazione non implementata nel condotto.')

    vel_model = getattr(grid, 'velocityModel', 'constant')
    if vel_model not in ('constant', 'continuity'):
        raise ValueError(f"velocityModel '{vel_model}' non riconosciuto: 'constant' o 'continuity'")
    var_u = (vel_model == 'continuity')

    p_model = getattr(grid, 'pressureModel', 'hydrostatic')
    if p_model not in ('hydrostatic', 'momentum'):
        raise ValueError(f"pressureModel '{p_model}' non riconosciuto: 'hydrostatic' o 'momentum'")
    dyn_p = (p_model == 'momentum')
    if dyn_p and not var_u:
        raise ValueError("pressureModel 'momentum' richiede velocityModel 'continuity'")
    unsteady = bool(getattr(grid, 'unsteady', True))
    maxit = int(getattr(grid, 'maxit', 20))
    density = getattr(grid, 'density', 'mixture')
    if density not in ('mixture', 'liquid'):
        raise ValueError(f"density '{density}' non riconosciuto: 'mixture' o 'liquid'")

    inlet = getattr(grid, 'inlet', 'face')
    if inlet not in ('face', 'ghost'):
        raise ValueError(f"inlet '{inlet}' non riconosciuto: 'face' o 'ghost'")

    funH = build_funH(Models, initial)
    rho_l = magma.liquid.rho_l
    dz = grid.dz

    # Geometria
    N = grid.N
    z_top = grid.z0 + N * dz
    z = grid.z0 + (np.arange(N) + 0.5) * dz            # centri cella, 0 = la piu' profonda
    z_face = grid.z0 + np.arange(N + 1) * dz           # facce delle celle (N+1)

    # Pressione del liquido: profilo idrostatico (e' la pressione se pressureModel = 'hydrostatic',
    # altrimenti e' il punto di partenza dell'iterazione) e pressione dello stato di ingresso
    pl_hydro = pressure_z(z, z_top, grid.p_top, rho_l)
    p_face_hydro = pressure_z(z_face, z_top, grid.p_top, rho_l)
    f = momord(cqmom.N1, cqmom.N2)

    def bubble_state(p0):
        # Stato di ingresso: bolle in equilibrio meccanico con il liquido alla pressione p0.
        # (initMethod legge solo p0 e T0: non si modifica l'oggetto Models dell'utente.)
        Models_in = SimpleNamespace(pressure=SimpleNamespace(p0=p0), temperature=Models.temperature)
        return funH.initMethod(f, initial, magma, funH, Models_in)

    if dyn_p and density == 'mixture' and inlet == 'face':
        p_in, p_start, p_face_start = boyle_column(grid.p_top, dz, N, rho_l, bubble_state)
    else:
        z_in = grid.z0 if inlet == 'face' else grid.z0 - 0.5 * dz
        p_in = pressure_z(z_in, z_top, grid.p_top, rho_l)
        p_start = pl_hydro
    y0 = bubble_state(p_in)
    c0 = initial.yl * rho_l * (1 - y0[1])              # massa disciolta per unita' di volume
    Yin_phys = np.concatenate([y0, [c0]])              # [M00, M10, M01, c]
    scale = Yin_phys                                   # scala numerica del solver
    nvar = Yin_phys.size
    init_state = getattr(grid, 'initialState', 'inlet')
    if init_state not in ('inlet', 'equilibrium'):
        raise ValueError(f"initialState '{init_state}' non riconosciuto: 'inlet' o 'equilibrium'")
    if init_state == 'inlet':
        y_init = np.ones(N * nvar)                     # tutte le celle partono dallo stato di ingresso
    else:
        # Boyle: m per bolla e Nb = M00/(1-M10) costanti, v ~ 1/p. Con g = Nb v = phi/(1-phi):
        # M10 = g/(1+g), M00 = Nb (1-M10), M01 = m M00, c = yl rho_l (1-M10)
        M00_i, M10_i, M01_i, _ = Yin_phys
        Nb, m_in, v_in = M00_i / (1 - M10_i), M01_i / M00_i, M10_i / M00_i
        v = v_in * p_in / p_start
        M10 = Nb * v / (1 + Nb * v)
        M00 = Nb * (1 - M10)
        Y_eq = np.column_stack([M00, M10, m_in * M00, initial.yl * rho_l * (1 - M10)])
        y_init = (Y_eq / scale).ravel()
    Yin = np.ones(nvar)                                # stato di ingresso, normalizzato

    def eval_S(t, Y, pl_loc):
        """Derivata locale (N, nvar), cioe' sorgenti + dilatazione, in variabili normalizzate,
        e divergenza della velocita' (N,), a pressione del liquido pl_loc."""
        Yp = Y * scale
        T = funH.thermoPT.T(t, Models.temperature)
        S, div = sources_monodisperse(Yp, pl_loc, T, funH, magma, Models)
        return add_dilution(S, Yp, div) / scale, div

    def faces_from_div(div):
        """Velocita' alle N+1 facce: u_{i+1/2} = u_{i-1/2} + dz * div_u,i  (u alla base = grid.u)."""
        if not var_u:
            return np.full(N + 1, grid.u)
        return np.concatenate([[grid.u], grid.u + dz * np.cumsum(div)])

    n_noconv = [0]                                     # contatore di mancate convergenze

    def sources_and_velocity(t, Y):
        """Derivata locale (N, nvar), velocita' alle N+1 facce, pressione del liquido al centro
        cella (N) e alle facce (N+1)."""
        if not dyn_p:
            S, div = eval_S(t, Y, pl_hydro)
            return S, faces_from_div(div), pl_hydro, p_face_hydro

        # pressione dalla quantita' di moto: punto fisso su p_l, partendo sempre dall'idrostatica
        # (cosi' la funzione non dipende dalla storia delle chiamate). La densita' dipende solo
        # dallo stato Y, non da p_l: e' fuori dal ciclo.
        if density == 'mixture':
            Yp = Y * scale
            rho = Yp[:, 2] + (1 - Yp[:, 1]) * rho_l     # M01 + (1 - M10) rho_l
        else:
            rho = rho_l
        pl_it = p_start.copy()
        for _ in range(maxit):
            S, div = eval_S(t, Y, pl_it)
            u_face = faces_from_div(div)
            u_c = 0.5 * (u_face[:-1] + u_face[1:])
            a = u_c * np.diff(u_face) / dz              # termine convettivo u du/dz

            if unsteady:
                # du/dt a z fissata: u_{i+1/2} = u_in + dz*sum(div_u), quindi
                # du_{i+1/2}/dt = dz*sum(d div_u/dt); d div_u/dt dalla derivata direzionale
                # di div_u lungo dY/dt (differenza finita in avanti, pl_it fissata)
                Yup = np.vstack([Yin, Y[:-1]])
                Ydot = -u_face[:-1, None] * (Y - Yup) / dz + S
                mx = np.max(np.abs(Ydot))
                if mx > 0.0:
                    eps = min(1.0, 1e-6 / mx)
                    _, div2 = eval_S(t, Y + eps * Ydot, pl_it)
                    du_dt = np.concatenate([[0.0], dz * np.cumsum((div2 - div) / eps)])
                    a = a + 0.5 * (du_dt[:-1] + du_dt[1:])

            pl_new, p_face = pressure_momentum(a, dz, grid.p_top, rho)
            err = np.max(np.abs(pl_new - pl_it) / pl_new)
            pl_it = pl_new
            if err < 1e-10:
                break
        else:
            n_noconv[0] += 1
        return S, u_face, pl_it, p_face

    def rhs(t, yflat):
        Y = yflat.reshape(N, nvar)
        Yup = np.vstack([Yin, Y[:-1]])                 # upwind: cella a monte (sotto)
        S, u_face, _, _ = sources_and_velocity(t, Y)
        return (S - u_face[:-1, None] * (Y - Yup) / dz).ravel()

    dt = grid.CFL * dz / grid.u
    if verbose:
        print(f'\n--- Conduit: N={N}, dz={dz} m, u_base={grid.u} m/s, CFL={grid.CFL} -> dt={dt} s ---')
        print(f'velocita\': {vel_model};  pressione: {p_model}'
              + (f' (termine non stazionario: {unsteady}, densita\': {density})' if dyn_p else '')
              + f';  ingresso: {inlet}')
        print(f'p_l idrostatica celle [Pa] = {pl_hydro},  p_in = {p_in:.1f} Pa')

    sol = solve_ivp(rhs, (0.0, time.tout), y_init, method='BDF', max_step=dt,
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f'Integrazione fallita: {sol.message}')
    if verbose:
        print(f'--- Done ({sol.t.size} steps) ---')

    # Post-processing: variabili fisiche e, a ogni istante, velocita' e pressione
    t = sol.t
    nt = t.size
    Yn = sol.y.reshape(N, nvar, nt)
    Yfis = Yn * scale[None, :, None]
    u_faces = np.empty((N + 1, nt))
    pl_t = np.empty((N, nt))                           # pressione del liquido, al centro cella
    p_face_t = np.empty((N + 1, nt))                   # e alle facce (la faccia 0 e' il fondo)
    for k in range(nt):
        _, u_faces[:, k], pl_t[:, k], p_face_t[:, k] = sources_and_velocity(t[k], Yn[:, :, k])
    if n_noconv[0] and verbose:
        print(f'ATTENZIONE: punto fisso su p_l non convergito in {n_noconv[0]} valutazioni')

    M00, M10, M01, c = Yfis[:, 0, :], Yfis[:, 1, :], Yfis[:, 2, :], Yfis[:, 3, :]
    Rs = Constants.R / np.atleast_1d(magma.species.PM)[0]
    T = funH.thermoPT.T(t, Models.temperature)
    return SimpleNamespace(
        t=t, z=z, z_face=z_face, Y=Yfis, M00=M00, M10=M10, M01=M01, c=c,
        p_g=M01 / M10 * Rs * T, R_mean=(3 / 4 / np.pi * M10 / M00) ** (1 / 3),
        u_face=u_faces, pl_t=pl_t, p_face=p_face_t, pl=pl_t[:, -1],
        pl_hydro=pl_hydro, p_face_hydro=p_face_hydro, p_in=p_in, n_noconv=n_noconv[0],
        sol=sol, funH=funH)


# ----------------------------------------------------------------------------------------------
# Solver + diagnostica + grafici
# ----------------------------------------------------------------------------------------------
def magma_conduit(time, grid, cqmom, initial, magma, Models, save_out_figure=False, show=True):
    res = solve_conduit(time, grid, cqmom, initial, magma, Models)
    rho_l = magma.liquid.rho_l
    var_u = (getattr(grid, 'velocityModel', 'constant') == 'continuity')

    if var_u:
        # Controllo di conservazione: a regime u*(1-alpha) (flusso volumetrico di liquido) costante
        u_c = 0.5 * (res.u_face[:-1, -1] + res.u_face[1:, -1])
        print('u alle facce a fine run [m/s]:', np.round(res.u_face[:, -1], 5))
        print('u*(1-M10) per cella a fine run:', np.round(u_c * (1 - res.M10[:, -1]), 5))

    res.diag = _diagnostics(res, grid, magma, rho_l, res.p_in)
    _print_diagnostics(res, grid)

    figs = _make_figures(res)
    figs.append(_make_diagnostic_figure(res))
    figs.append(make_pressure_figure(res))

    if Models.massGrowth.name != 'zero':
        res.mass = _mass_diagnostics(res, grid, magma, rho_l, res.funH, Models)
        _print_mass_diagnostics(res)
        figs.append(_make_mass_figure(res))
    if save_out_figure:
        tag = 'Conduit' if not var_u else 'ConduitU'
        for i, fig in enumerate(figs, start=1):
            fig.savefig(f'{tag}_Fig{i}.jpg', dpi=300)
    if show:
        plt.show()
    return res


# ----------------------------------------------------------------------------------------------
# Diagnostica
# ----------------------------------------------------------------------------------------------
def _diagnostics(res, grid, magma, rho_l, p_in):
    """Grandezze per capire in che regime lavora il sistema (tutte a fine simulazione, tranne il ritardo)."""
    N = res.z.size
    mu = magma.liquid.mu
    t = res.t
    u_c = 0.5 * (res.u_face[:-1, :] + res.u_face[1:, :])        # velocita' al centro cella (N, nt)
    u_end = u_c[:, -1]
    phi_in = res.M10[0, 0]                                      # stato di ingresso = stato iniziale

    # 1) Ritardo del gas rispetto al liquido: > 0 vuol dire gas ancora sovrappressurizzato
    lag = res.p_g - res.pl_t                                    # [Pa], (N, nt)

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
    dp = res.pl_t - res.pl_hydro[:, None]
    print(f'p_l - p_idrostatica: max |dp| = {np.max(np.abs(dp)):.4g} Pa '
          f'({np.max(np.abs(dp) / res.pl_hydro[:, None]):.2e} in valore relativo);  '
          f'pressione sul fondo a fine run = {res.p_face[0, -1]:.1f} Pa '
          f'(idrostatica {res.p_face_hydro[0]:.1f} Pa)')
    print(f'Tempo di transito totale: {d.t_transit:.3g} s   (tout = {res.t[-1]:.3g} s)')
    print(f'Variazione relativa di M10 (cella alta) nell\'ultimo 10% del tempo: {d.rel_change:.2e}'
          + ('   -> regime raggiunto' if d.rel_change < 1e-3 else '   -> NON a regime: aumenta tout'))
    print(' cella   z[m]   p_l[kPa]  p_g-p_l[Pa]  phi      phi_Boyle  tau_rel[s]  tau_dec[s]  De      u(1-phi)/rif')
    for i in range(N):
        print(f' {i:3d}  {res.z[i]:7.1f} {res.pl[i]*1e-3:9.1f} {d.lag[i, -1]:11.1f}  {res.M10[i, -1]:.5f}  '
              f'{d.phi_boyle[i]:.5f}   {d.tau_relax[i]:9.3g}  {d.tau_decomp[i]:9.3g}  {d.De[i]:.2e}  {d.flux[i, -1]:.4f}')


def _mass_diagnostics(res, grid, magma, rho_l, funH, Models):
    """Diagnostica del trasferimento di massa (specie singola, H2O): sovrasaturazione, flussi conservati,
    tempo di trasferimento di massa."""
    N = res.z.size
    T = funH.thermoPT.T(0.0, Models.temperature)
    u_c = 0.5 * (res.u_face[:-1, :] + res.u_face[1:, :])
    u_b = res.u_face[0, :]

    # Frazione in massa di H2O disciolta e solubilita' alla pressione del liquido
    yl = res.c / (rho_l * (1 - res.M10))
    y_eq = funH.thermoPT.yl(res.pl, T, None, Models.saturation)        # (N,)
    d_y = yl - funH.thermoPT.yl(res.pl_t, T, None, Models.saturation)  # > 0: sovrasaturo

    # Invarianti a regime, normalizzati al valore di ingresso (devono valere 1):
    #   - flusso di acqua totale (disciolta + gas):      u (c + M01)
    #   - flusso di fuso senza volatili (secco):         u (rho_l (1-phi) - c)
    J_w = u_c * (res.c + res.M01)
    J_w = J_w / (u_b[0] * (res.c[0, 0] + res.M01[0, 0]))
    J_d = u_c * (rho_l * (1 - res.M10) - res.c)
    J_d = J_d / (u_b[0] * (rho_l * (1 - res.M10[0, 0]) - res.c[0, 0]))

    # Tempo di trasferimento di massa: tempo per raddoppiare la massa di gas di una bolla
    tau_mass = np.empty(N)
    for i in range(N):
        v = res.M10[i, -1] / res.M00[i, -1]
        m = res.M01[i, -1] / res.M00[i, -1]
        psi = funH.massGrowth.Psi(v, m, np.array([yl[i, -1]]), np.array([1.0]), None,
                                  res.pl[i], T, 0, magma, funH, Models)
        tau_mass[i] = m / abs(psi) if psi != 0 else np.inf
    return SimpleNamespace(yl=yl, y_eq=y_eq, d_y=d_y, J_w=J_w, J_d=J_d, tau_mass=tau_mass)


def _print_mass_diagnostics(res):
    d, ms = res.diag, res.mass
    print('\n=== Diagnostica trasferimento di massa (H2O) ===')
    print(' cella   z[m]   yl[-]     y_eq(p_l)  yl-y_eq    tau_mass[s]  t_cella[s]  J_acqua/rif  J_secco/rif')
    for i in range(res.z.size):
        print(f' {i:3d}  {res.z[i]:7.1f}  {ms.yl[i, -1]:.5f}  {ms.y_eq[i]:.5f}   {ms.d_y[i, -1]:+.5f}   '
              f'{ms.tau_mass[i]:10.3g}  {d.t_cell[i]:10.3g}   {ms.J_w[i, -1]:.4f}      {ms.J_d[i, -1]:.4f}')


# ----------------------------------------------------------------------------------------------
# Grafici
# ----------------------------------------------------------------------------------------------
def _make_figures(res):
    t, N = res.t, res.z.size
    labels = [f'cella {i} (z={res.z[i]:.1f} m)' for i in range(N)]

    fig1, ax = plt.subplots(figsize=(6.5, 5))
    for i in range(N):
        l, = ax.plot(t, res.p_g[i] * 1e-6, label=labels[i])
        ax.plot(t, res.pl_t[i] * 1e-6, color=l.get_color(), ls='--', lw=0.8)
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


def make_pressure_figure(res):
    """Figura della pressione del liquido dal fondo del condotto alla superficie (p_top = p atmosferica).

    Richiede in res: t, z_face, p_face (N+1, nt; la faccia 0 e' il fondo, la faccia N la superficie),
    p_face_hydro (N+1,).
    """
    t = res.t
    nf = res.z_face.size                      # N + 1 facce
    p = res.p_face * 1e-3                     # kPa
    p_hyd = res.p_face_hydro * 1e-3

    def nome(j):
        if j == 0:
            return f'fondo (z={res.z_face[j]:.1f} m)'
        if j == nf - 1:
            return f'superficie (z={res.z_face[j]:.1f} m)'
        return f'faccia {j} (z={res.z_face[j]:.1f} m)'

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))

    # 1) profilo p(z) dal fondo alla superficie, a tempi scelti
    t_sel = np.unique(np.concatenate([[0.0], np.geomspace(max(t[1], 1e-3), t[-1], 5)]))
    for ts in t_sel:
        k = int(np.argmin(np.abs(t - ts)))
        axs[0].plot(res.z_face, p[:, k], 'o-', ms=4, label=f't = {t[k]:.3g} s')
    axs[0].plot(res.z_face, p_hyd, 'k--', lw=0.9, label='idrostatica')
    axs[0].axhline(p[-1, -1], color='gray', lw=0.6, ls=':')
    axs[0].set_xlabel('z [m]  (fondo -> superficie)')
    axs[0].set_ylabel('Liquid pressure [kPa]')
    axs[0].set_title('Profilo di pressione lungo il condotto')
    axs[0].legend(loc='best', fontsize=7)

    # 2) pressione nel tempo a ogni faccia, dal fondo alla superficie
    for j in range(nf):
        axs[1].plot(t, p[j], label=nome(j))
    axs[1].set_xscale('symlog', linthresh=1e-2)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Liquid pressure [kPa]')
    axs[1].set_title('Pressione nel tempo (dal fondo alla superficie)')
    axs[1].legend(loc='best', fontsize=7)

    # 3) scarto dall'idrostatica (la superficie e' fissata a p_top, scarto nullo)
    for j in range(nf - 1):
        axs[2].plot(t, res.p_face[j] - res.p_face_hydro[j], label=nome(j))
    axs[2].set_xscale('symlog', linthresh=1e-2)
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('p - p$_{idrostatica}$ [Pa]')
    axs[2].set_title('Scarto dall\'idrostatica')
    axs[2].legend(loc='best', fontsize=7)

    fig.tight_layout()
    return fig


def _make_diagnostic_figure(res):
    d = res.diag
    t, N = res.t, res.z.size
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))

    for i in range(N):
        axs[0].plot(t, d.lag[i] / res.pl_t[i] * 100, label=f'cella {i}')
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


def _make_mass_figure(res):
    ms = res.mass
    t, N = res.t, res.z.size
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))

    axs[0].plot(res.z, ms.yl[:, -1], 'o-', label='disciolta (y$_l$)')
    axs[0].plot(res.z, ms.y_eq, 's--', label='solubilita\' alla p$_l$')
    axs[0].set_xlabel('z [m]')
    axs[0].set_ylabel('Mass fraction of H$_2$O in the melt')
    axs[0].legend(loc='best')

    for i in range(N):
        axs[1].plot(t, ms.d_y[i], label=f'cella {i}')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('y$_l$ - y$_{eq}$(p$_l$)  (>0: supersaturated)')
    axs[1].legend(loc='best')

    for i in range(N):
        axs[2].plot(t, res.M01[i], label=f'cella {i}')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('M$_{01}$: gas mass per m$^3$ [kg/m$^3$]')
    axs[2].legend(loc='best')

    axs[3].plot(res.z, ms.J_w[:, -1], 'o-', label='flusso di acqua')
    axs[3].plot(res.z, ms.J_d[:, -1], 's-', label='flusso di fuso secco')
    axs[3].axhline(1.0, color='k', lw=0.8, ls='--')
    axs[3].set_xlabel('z [m]')
    axs[3].set_ylabel('flusso / valore di ingresso')
    axs[3].set_title('Conservazione (devono valere 1)')
    axs[3].legend(loc='best')

    fig.tight_layout()
    return fig
