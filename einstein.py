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

def code_density_to_si(rho_code: float) -> float:
    """
    Invert si_mass_density_to_code:
      rho_code = (G/c^2) * rho_SI * L0^2
      -> rho_SI = rho_code * (c^2/G) / L0^2
    """
    return rho_code * (c_SI**2 / G_SI) / (L0**2)

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
# TOV + baryonic mass RHS (Part a + Part b)
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

    # baryonic mass equation uses rho_r (rest-mass density)
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
# Returns central total density rho_c (SI) for Part (c)
# ============================
def integrate_star_pc(
    p_c: float,
    Gamma: float,
    r_max: float = 120.0,
    h_min: float = 1e-6,
    h_max: float = 5e-3,
):
    """
    Returns:
      (M_solar, R_km, MP_solar, Delta, rho_c_SI, status)
    """
    K = K_from_reference(Gamma)

    if not np.isfinite(p_c) or p_c <= 0.0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, "nan")

    rho_c_code, rho_r_c_code = eos_from_p(p_c, K, Gamma)
    rho_c_SI = code_density_to_si(rho_c_code)

    r = 1e-6
    m0 = (4.0/3.0) * np.pi * rho_c_code * r**3
    mP0 = (4.0/3.0) * np.pi * rho_r_c_code * r**3
    nu0 = 0.0

    y = np.array([m0, nu0, p_c, mP0], dtype=float)

    prev_r = r
    prev_y = y.copy()

    while r < r_max and y[2] > 0.0:
        prev_r = r
        prev_y = y.copy()

        m, nu, p, mP = y
        if not (np.isfinite(m) and np.isfinite(p) and np.isfinite(nu) and np.isfinite(mP)):
            return (np.nan, np.nan, np.nan, np.nan, rho_c_SI, "nan")

        if (r - 2.0*m) <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan, rho_c_SI, "horizon")

        h = 0.02 * r
        h = min(h, h_max)
        h = max(h, h_min)

        y = rk4_step(tov_rhs_with_mp, r, y, h, K, Gamma)
        r += h

        if not (np.isfinite(y[0]) and np.isfinite(y[2]) and np.isfinite(y[3])):
            return (np.nan, np.nan, np.nan, np.nan, rho_c_SI, "nan")
        if (r - 2.0*y[0]) <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan, rho_c_SI, "horizon")

    if y[2] > 0.0:
        return (np.nan, np.nan, np.nan, np.nan, rho_c_SI, "no_surface")

    # interpolate to surface p=0
    p1 = prev_y[2]
    p2 = y[2]
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
        return (np.nan, np.nan, np.nan, np.nan, rho_c_SI, "nan")

    Delta = (MP_solar - M_solar) / M_solar
    return (M_solar, R_km, MP_solar, Delta, rho_c_SI, "ok")

# ============================
# p_c sweep grid
# ============================
def pc_sweep_grid(Gamma: float, n_points: int = 120, rho_factor_min: float = 1e-2, rho_factor_max: float = 1e6):
    K = K_from_reference(Gamma)
    rho_r_min = rho_r_ref_code * rho_factor_min
    rho_r_max = rho_r_ref_code * rho_factor_max
    p_min = K * (rho_r_min ** Gamma)
    p_max = K * (rho_r_max ** Gamma)
    return np.logspace(np.log10(p_min), np.log10(p_max), n_points)

def sweep_models_for_gamma(Gamma: float, n_points: int = 120):
    pc_grid = pc_sweep_grid(Gamma, n_points=n_points)

    R = np.full(n_points, np.nan)
    M = np.full(n_points, np.nan)
    MP = np.full(n_points, np.nan)
    Delta = np.full(n_points, np.nan)
    rho_c_SI = np.full(n_points, np.nan)
    status = np.array([""] * n_points, dtype=object)

    for i, pc in enumerate(pc_grid):
        Mi, Ri, MPi, Di, rhoi, si = integrate_star_pc(pc, Gamma)
        M[i] = Mi
        R[i] = Ri
        MP[i] = MPi
        Delta[i] = Di
        rho_c_SI[i] = rhoi
        status[i] = si

    return {
        "Gamma": Gamma,
        "pc": pc_grid,
        "R_km": R,
        "M": M,
        "MP": MP,
        "Delta": Delta,
        "rho_c_SI": rho_c_SI,
        "status": status
    }

def split_stable_unstable_by_Mmax(res: dict):
    good = (res["status"] == "ok") & np.isfinite(res["M"]) & np.isfinite(res["rho_c_SI"]) & (res["rho_c_SI"] > 0)
    idx = np.where(good)[0]
    if idx.size < 3:
        stable = np.zeros_like(good, dtype=bool)
        unstable = np.zeros_like(good, dtype=bool)
        return good, stable, unstable, None

    i_local_max = np.nanargmax(res["M"][idx])
    i_max = idx[i_local_max]

    stable = np.zeros_like(good, dtype=bool)
    unstable = np.zeros_like(good, dtype=bool)

    stable[idx[idx <= i_max]] = True
    unstable[idx[idx > i_max]] = True
    return good, stable, unstable, i_max

def masked(arr, mask):
    out = np.array(arr, dtype=float)
    out[~mask] = np.nan
    return out


# ============================
def main():
    gammas = [1.3569, 2.7138]

    results = []
    for Gamma in gammas:
        results.append(sweep_models_for_gamma(Gamma, n_points=120))

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

    # ----------------------------
    # Part (c): M vs rho_c (log x-axis)
    # ----------------------------
    fig_c, axes_c = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    for ax, res in zip(axes_c, results):
        good, stable, unstable, i_max = split_stable_unstable_by_Mmax(res)

        ax.set_xscale("log")

        # no markers
        ax.plot(masked(res["rho_c_SI"], stable), masked(res["M"], stable),
                linestyle="-", color="black", label="Stable")
        ax.plot(masked(res["rho_c_SI"], unstable), masked(res["M"], unstable),
                linestyle="--", color="black", label="Unstable")

        ax.set_xlabel(r"$\rho_c$ (kg/m$^3$)")
        ax.set_ylabel(r"$M$ ($M_\odot$)")
        ax.set_title(f"$\Gamma$ = {res['Gamma']}")
        ax.legend()

        # Major grid only 
        ax.grid(True, which="major")
        ax.grid(False, which="minor")

        if i_max is not None and np.isfinite(res["M"][i_max]):
            print(
                f"Gamma={res['Gamma']}: "
                f"M_max = {res['M'][i_max]:.6g} M_sun at "
                f"R = {res['R_km'][i_max]:.6g} km, "
                f"rho_c = {res['rho_c_SI'][i_max]:.6g} kg/m^3"
            )

    fig_c.suptitle(r"Stability from $dM/d\rho_c$ and the Maximum Mass")
    fig_c.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
