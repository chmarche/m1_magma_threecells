import numpy as np

from .constants import Constants
from .cqmom_2d_adaptive import cqmom_2d_adaptive
from .normalization import normalized, renormalize


def momsources(t, y_state, funH, cqmom, magma, Models, initial):
    """Termini sorgente dei momenti e delle specie 
    I momenti vengono prima invertiti con il CQMOM, poi si calcola l'effetto della
    crescita di volume e di massa delle bolle.
    Restituisce la derivata temporale dello stato normalizzato (per solve_ivp).
    """
    f = cqmom.f
    num = len(f)
    Ns = len(magma.species.name)
    i_M10 = 1                    # posizione di M10 (gas volume fraction)
    i_M01 = 2 * cqmom.N1         # posizione di M01 (massa di gas per unita' di volume)
    rho_l = magma.liquid.rho_l
    PM = np.atleast_1d(magma.species.PM)

    # Pressione e temperatura
    pl = funH.thermoPT.p(t, Models.pressure)
    T = funH.thermoPT.T(t, Models.temperature)

    # Rinormalizzazione
    y = renormalize(y_state, f, Ns, initial.norm)

    # Frazioni in massa
    yl = y[num:num + Ns] / rho_l / (1 - y[i_M10])
    yg = np.zeros(Ns)
    sumyg = 0.0
    for i in range(Ns - 1):
        yg[i] = y[num + Ns + i] / y[i_M01]
        sumyg += yg[i]
    yg[Ns - 1] = 1 - sumyg

    # Pesi e ascisse (v = volume, m = massa)
    w, v, m, nout = cqmom_2d_adaptive(y, cqmom.N1, cqmom.N2)

    # Momenti 0,0 / 1,0 / 0,1
    M00 = np.sum(w)
    M10 = np.sum(w[:nout] * v[:nout])
    M01 = np.sum(w[:nout] * m[:nout])

    rho_mix = M01 + (1 - M10) * rho_l

    # Frazioni in massa totali dei volatili
    ytot = (y[num:num + Ns]
            + np.concatenate([y[num + Ns:num + 2 * Ns - 1], [yg[Ns - 1] * y[i_M01]]])) / rho_mix

    # Inizializzazione dei termini sorgente
    s_volu = np.zeros(num)
    s_mass = np.zeros(num)
    s_mass_species = np.zeros((Ns, num))
    s_U = np.zeros(num)
    s_nucl = np.zeros(num)
    s_coal = np.zeros(num)       # coalescenza non calcolata (come nel MATLAB)

    # Eq dei momenti
    for z in range(num):
        k = f[z, 0]
        l = f[z, 1]

        for i in range(nout):
            grow_v = funH.volumeGrowth.Gamma(v[i], m[i], yl, yg, ytot, pl, T, magma, funH, Models)
            for p in range(Ns):
                grow_m = funH.massGrowth.Psi(v[i], m[i], yl, yg, ytot, pl, T, p, magma, funH, Models)
                s_mass_species[p, z] += l * w[i] * grow_m * v[i] ** k * m[i] ** (l - 1)
            s_volu[z] += k * w[i] * grow_v * v[i] ** (k - 1) * m[i] ** l
        s_mass[z] = np.sum(s_mass_species[:, z])

        J, v_c, m_c = funH.nucleation.J(yl, yg, ytot, pl, T, magma, funH, Models)
        s_nucl[z] = J * v_c ** k * m_c ** l

    # Divergenza della velocita' (termine di diluizione/espansione)
    #   dM10/dt = S10 + M10*divU ,  dM01/dt = S01 + M01*divU ,
    #   divU = (dM01/dt - rho_l*dM10/dt) / rho_mix ,   rho_mix = M01 + (1 - M10)*rho_l
    #   =>  divU * rho_l = S01 - rho_l*S10   =>   divU = S01/rho_l - S10
    # dove S10, S01 sono le sorgenti senza il termine di divergenza.
    S10 = s_coal[i_M10] + s_volu[i_M10] + s_mass[i_M10] + s_nucl[i_M10]
    S01 = s_coal[i_M01] + s_volu[i_M01] + s_mass[i_M01] + s_nucl[i_M01]
    divU = S01 / rho_l - S10

    for z in range(num):
        k = f[z, 0]
        l = f[z, 1]
        s_U[z] = np.sum(w[:nout] * v[:nout] ** k * m[:nout] ** l) * divU

    # Derivata temporale dei momenti
    s = s_coal + s_volu + s_mass + s_U + s_nucl

    # Eq di trasporto della specie
    # Composizione di equilibrio del gas
    Rs = Constants.R * np.sum(yg / PM)
    p_g = M01 / M10 * Rs * T
    ygs = np.zeros(Ns)
    for p in range(Ns - 1):
        ygs[p] = funH.thermoPT.yg(p_g, T, ytot, Models.saturation)
    ygs[Ns - 1] = 1 - np.sum(ygs)

    # Specie disciolte (massa di specie disciolta / volume totale)
    dcdt = -(s_mass_species[:, i_M01] + ygs * s_nucl[i_M01]) + (1 - M10) * rho_l * yl * divU

    # Specie gassose (massa di specie gassosa / volume totale)
    dcgdt = np.zeros(Ns - 1)
    for p in range(Ns - 1):
        dcgdt[p] = s_mass_species[p, i_M01] + ygs[p] * s_nucl[i_M01] + yg[p] * s_U[i_M01]

    return normalized(np.concatenate([s, dcdt, dcgdt]), f, Ns, initial.norm)
