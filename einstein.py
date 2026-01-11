import numpy as np
import matplotlib.pyplot as plt


# Physical constants (SI)
G_SI = 6.67430e-11
c_SI = 299792458.0
M_sun_SI = 1.989e30

# Project scaling: mass unit = M_sun, length unit = GM_sun/c^2 (geometric units)
L0 = G_SI * M_sun_SI / c_SI**2  # meters

# ============================
# Reference point for K
# P/c^2 = 1.5689e31 dyn/cm^2 when rho_r = 1e13 g/cm^3
# We'll treat p := P/c^2 as a mass-density-equivalent and convert to SI (kg/m^3).
# ============================
P_over_c2_ref_cgs_dyn_cm2 = 1.5689e31
rho_r_ref_cgs_g_cm3 = 1.0e13

DYN_PER_CM2_TO_PA = 0.1        # 1 dyn/cm^2 = 0.1 Pa
G_CM3_TO_KG_M3 = 1000.0        # 1 g/cm^3 = 1000 kg/m^3

rho_r_ref_SI = rho_r_ref_cgs_g_cm3 * G_CM3_TO_KG_M3
P_over_c2_ref_SI = (P_over_c2_ref_cgs_dyn_cm2 * DYN_PER_CM2_TO_PA) / (c_SI**2)  # kg/m^3

def si_mass_density_to_code(rho_SI: float) -> float:
    """
    Convert SI mass density (kg/m^3) into dimensionless code units.
    In geometric units: rho_geom [1/m^2] = (G/c^2) * rho_SI
    Then nondimensionalize by multiplying by L0^2.
    """
    rho_geom = (G_SI / c_SI**2) * rho_SI
    return rho_geom * (L0**2)

rho_r_ref_code = si_mass_density_to_code(rho_r_ref_SI)
p_ref_code = si_mass_density_to_code(P_over_c2_ref_SI)

# ============================
# EOS
# p = K rho_r^Gamma
# rho = rho_r + p/(Gamma-1)
# ============================
def K_from_reference(Gamma: float) -> float:
    return p_ref_code / (rho_r_ref_code ** Gamma)

def eos_from_p(p: float, K: float, Gamma: float):
    """Return (rho, rho_r) from pressure p (all in code units)."""
    if p <= 0.0:
        return 0.0, 0.0
    rho_r = (p / K) ** (1.0 / Gamma)
    rho = rho_r + p / (Gamma - 1.0)
    return rho, rho_r

# ============================
# TOV + baryonic mass RHS 
# State y = [m, nu, p, mP]
# ============================
def tov_rhs_with_mp(r: float, y: np.ndarray, K: float, Gamma: float) -> np.ndarray:
    m, nu, p, mP = y

    if p <= 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    rho, rho_r = eos_from_p(p, K, Gamma)

    denom = r * (r - 2.0 * m)
    if denom <= 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    A = (m + 4.0 * np.pi * r**3 * p) / denom

    dm = 4.0 * np.pi * r**2 * rho
    dnu = 2.0 * A
    dp = -(rho + p) * A

    one_minus_2m_over_r = 1.0 - 2.0 * m / r
    if one_minus_2m_over_r <= 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    dmp = 4.0 * np.pi * (one_minus_2m_over_r ** (-0.5)) * (r**2) * rho_r

    return np.array([dm, dnu, dp, dmp])

def rk4_step(f, r, y, h, K, Gamma):
    k1 = f(r, y, K, Gamma)
    k2 = f(r + 0.5*h, y + 0.5*h*k1, K, Gamma)
    k3 = f(r + 0.5*h, y + 0.5*h*k2, K, Gamma)
    k4 = f(r + h, y + h*k3, K, Gamma)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================
# Integrate one star given central pressure p_c (code units)
# Initial conditions: m(0)=0, p(0)=p_c, nu(0)=0, mP(0)=0
# Stop at p=0.
# ============================
def integrate_star_pc(
    p_c: float,
    Gamma: float,
    r_max: float = 120.0,
    h_min: float = 1e-6,
    h_max: float = 5e-3,
):
    """
    Returns (M_solar, R_km, MP_solar, Delta, status)
    status in {"ok","horizon","nan","no_surface"}.
    """
    K = K_from_reference(Gamma)

    if not np.isfinite(p_c) or p_c <= 0.0:
        return (np.nan, np.nan, np.nan, np.nan, "nan")

    # start slightly away from r=0
    r = 1e-6
    rho_c, rho_r_c = eos_from_p(p_c, K, Gamma)

    # small-r series starts
    m0 = (4.0/3.0) * np.pi * rho_c * r**3
    mP0 = (4.0/3.0) * np.pi * rho_r_c * r**3
    nu0 = 0.0

    y = np.array([m0, nu0, p_c, mP0], dtype=float)

    prev_r = r
    prev_y = y.copy()

    while r < r_max and y[2] > 0.0:
        prev_r = r
        prev_y = y.copy()

        m, nu, p, mP = y
        if not (np.isfinite(m) and np.isfinite(p) and np.isfinite(nu) and np.isfinite(mP)):
            return (np.nan, np.nan, np.nan, np.nan, "nan")

        if (r - 2.0*m) <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan, "horizon")

        # variable step: small near center, capped outside
        h = 0.02 * r
        h = min(h, h_max)
        h = max(h, h_min)

        y = rk4_step(tov_rhs_with_mp, r, y, h, K, Gamma)
        r += h

        if not (np.isfinite(y[0]) and np.isfinite(y[2]) and np.isfinite(y[3])):
            return (np.nan, np.nan, np.nan, np.nan, "nan")
        if (r - 2.0*y[0]) <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan, "horizon")

    if y[2] > 0.0:
        return (np.nan, np.nan, np.nan, np.nan, "no_surface")

    # interpolate to surface p=0 using last two steps
    p1 = prev_y[2]
    p2 = y[2]  # <= 0
    if p1 <= 0.0:
        r_surf = prev_r
        m_surf = prev_y[0]
        mP_surf = prev_y[3]
    else:
        frac = p1 / (p1 - p2)
        r_surf = prev_r + frac * (r - prev_r)
        m_surf = prev_y[0] + frac * (y[0] - prev_y[0])
        mP_surf = prev_y[3] + frac * (y[3] - prev_y[3])

    M_solar = m_surf
    MP_solar = mP_surf
    R_km = (r_surf * L0) / 1000.0

    if not (np.isfinite(M_solar) and M_solar > 0 and np.isfinite(MP_solar)):
        return (np.nan, np.nan, np.nan, np.nan, "nan")

    Delta = (MP_solar - M_solar) / M_solar
    return (M_solar, R_km, MP_solar, Delta, "ok")

# ============================
# Choose a p_c sweep range using rho_r factors around the reference rho_r_ref_code
# ============================
def pc_sweep_grid(Gamma: float, n_points: int = 90, rho_factor_min: float = 1e-2, rho_factor_max: float = 1e6):
    K = K_from_reference(Gamma)
    rho_r_min = rho_r_ref_code * rho_factor_min
    rho_r_max = rho_r_ref_code * rho_factor_max
    p_min = K * (rho_r_min ** Gamma)
    p_max = K * (rho_r_max ** Gamma)
    return np.logspace(np.log10(p_min), np.log10(p_max), n_points)

def sweep_models_for_gamma(Gamma: float, n_points: int = 90):
    """
    One sweep per Gamma. Reuse outputs for all plots.
    """
    pc_grid = pc_sweep_grid(Gamma, n_points=n_points)

    R = np.full(n_points, np.nan)
    M = np.full(n_points, np.nan)
    MP = np.full(n_points, np.nan)
    Delta = np.full(n_points, np.nan)
    status = np.array([""] * n_points, dtype=object)

    for i, pc in enumerate(pc_grid):
        Mi, Ri, MPi, Di, si = integrate_star_pc(pc, Gamma)
        M[i] = Mi
        R[i] = Ri
        MP[i] = MPi
        Delta[i] = Di
        status[i] = si

    return {
        "Gamma": Gamma,
        "pc": pc_grid,
        "R_km": R,
        "M": M,
        "MP": MP,
        "Delta": Delta,
        "status": status
    }


# ============================
def main():
    gammas = [1.3569, 2.7138]

    # --- Run one sweep per Gamma and store results ---
    results = []
    for Gamma in gammas:
        results.append(sweep_models_for_gamma(Gamma, n_points=90))

    # ----------------------------
    # Part (a): M-R curves
    # ----------------------------
    fig_a, axes_a = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    for ax, res in zip(axes_a, results):
        good = (res["status"] == "ok") & np.isfinite(res["R_km"]) & np.isfinite(res["M"])
        ax.plot(res["R_km"][good], res["M"][good], marker="o", linestyle="-", color="black",
                label=f"Gamma = {res['Gamma']}")
        ax.set_xlabel("R (km)")
        ax.set_ylabel("M ($M_\\odot$)")
        ax.grid(True)
        ax.legend()

    fig_a.suptitle("Neutron Star Mass–Radius Curves from TOV")
    fig_a.tight_layout()
    plt.show()

    # ----------------------------
    # Part (b1): Delta vs R
    # ----------------------------
    fig_b1, axes_b1 = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    for ax, res in zip(axes_b1, results):
        good = (res["status"] == "ok") & np.isfinite(res["R_km"]) & np.isfinite(res["Delta"])
        ax.plot(res["R_km"][good], res["Delta"][good], marker="o", linestyle="-", color="black",
                label=f"Gamma = {res['Gamma']}")
        ax.set_xlabel("R (km)")
        ax.set_ylabel(r"$\Delta = (M_P - M)/M$")
        ax.grid(True)
        ax.legend()

    fig_b1.suptitle("Fractional Binding Energy vs Radius")
    fig_b1.tight_layout()
    plt.show()

    # ----------------------------
    # Part (b2): M vs MP
    # ----------------------------
    fig_b2, axes_b2 = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    for ax, res in zip(axes_b2, results):
        good = (res["status"] == "ok") & np.isfinite(res["M"]) & np.isfinite(res["MP"])
        ax.plot(res["MP"][good], res["M"][good], marker="o", linestyle="-", color="black",
                label=f"Gamma = {res['Gamma']}")
        ax.set_xlabel(r"$M_P$ ($M_\odot$)")
        ax.set_ylabel(r"$M$ ($M_\odot$)")
        ax.grid(True)
        ax.legend()

    fig_b2.suptitle("Gravitational Mass vs Baryonic Mass")
    fig_b2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
