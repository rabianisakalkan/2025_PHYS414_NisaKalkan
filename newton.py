import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  

# ============================================================
# Constants (CGS)
# ============================================================
G = 6.67430e-8
M_sun = 1.98847e33
R_earth = 6.371e8

# Eq. (11) constants (CGS)
c = 2.99792458e10          # cm/s
hbar = 1.054571817e-27     # erg*s
m_e = 9.1093837015e-28     # g
m_u = 1.66053906660e-24    # g
mu_e = 2.0

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("white_dwarf_data.csv")
logg = df["logg"].to_numpy()
M_msun = df["mass"].to_numpy()

# ============================================================
# Part (b): Convert logg -> radius and plot M vs R
# ============================================================
g = 10**logg
M_cgs = M_msun * M_sun
R_cm = np.sqrt(G * M_cgs / g)
R_rearth = R_cm / R_earth

plt.figure()
plt.scatter(R_rearth, M_msun, s=12, color="red")
plt.xlabel(r"$R_\oplus$")
plt.ylabel(r"$M_\odot$")
plt.title("White Dwarf Mass vs Radius")
plt.grid(True)
plt.show()

# ============================================================
# Lane–Emden solver (RK4)
# ============================================================
def lane_emden_constants(n, h=1e-3, xi_max=50.0, eps=1e-6):
    xi = eps
    theta = 1 - eps**2/6 + n*eps**4/120
    u = -eps/3 + n*eps**3/30  # u = dtheta/dxi

    def rhs(xi, theta, u):
        dtheta = u
        if xi == 0.0:
            du = -theta**n
        else:
            du = -(2/xi)*u - (theta**n if theta > 0 else 0.0)
        return dtheta, du

    prev_xi, prev_theta, prev_u = xi, theta, u

    while xi < xi_max and theta > 0:
        prev_xi, prev_theta, prev_u = xi, theta, u

        k1t, k1u = rhs(xi, theta, u)
        k2t, k2u = rhs(xi + 0.5*h, theta + 0.5*h*k1t, u + 0.5*h*k1u)
        k3t, k3u = rhs(xi + 0.5*h, theta + 0.5*h*k2t, u + 0.5*h*k2u)
        k4t, k4u = rhs(xi + h, theta + h*k3t, u + h*k3u)

        theta += (h/6)*(k1t + 2*k2t + 2*k3t + k4t)
        u     += (h/6)*(k1u + 2*k2u + 2*k3u + k4u)
        xi    += h

    t = prev_theta / (prev_theta - theta)
    xi1 = prev_xi + t*(xi - prev_xi)
    u1  = prev_u  + t*(u - prev_u)
    return xi1, u1

# ============================================================
# Part (c): Two-stage fit on low-mass subset + plot
# ============================================================
Mmax_Msun = 0.40
sel = M_msun <= Mmax_Msun

x = np.log10(R_cm[sel])    # log10(R [cm])
y = np.log10(M_cgs[sel])   # log10(M [g])

alpha1, b1 = np.polyfit(x, y, 1)          # Fit 1 (free)
n1 = (3 - alpha1) / (1 - alpha1)
q_real = 5*n1 / (1 + n1)
q_int = int(np.rint(q_real))

n_star = q_int / (5 - q_int)              # Fit 2 (q fixed -> slope fixed)
alpha_fixed = (3 - n_star) / (1 - n_star)
b2 = np.mean(y - alpha_fixed*x)
A_fixed = 10**b2

xi_n, theta_p = lane_emden_constants(n_star)

pref_LE = 4*np.pi * (-theta_p) * (xi_n**((n_star+1)/(n_star-1)))
K_star = (4*np.pi*G/(n_star+1)) * ((A_fixed/pref_LE)**((n_star-1)/n_star))

print("Fit 1 (free):")
print("alpha =", alpha1)
print("q_real =", q_real, " -> q_int =", q_int)

print("\nFit 2 (q fixed):")
print("n* =", n_star)
print("alpha_fixed =", alpha_fixed)
print("A_fixed (CGS) =", A_fixed)

print("\nLane–Emden + K*:")
print("xi_n =", xi_n)
print("theta'(xi_n) =", theta_p)
print("K* (cgs) =", K_star)

plt.figure()
plt.scatter(x, y, s=12, label=f"data (M <= {Mmax_Msun} Msun)")
xx = np.linspace(x.min(), x.max(), 200)
plt.plot(xx, alpha1*xx + b1, linewidth=2, label=f"fit1 slope={alpha1:.3f}", color="red")
plt.plot(xx, alpha_fixed*xx + np.log10(A_fixed), linewidth=2,
         label=f"fit2 slope={alpha_fixed:.3f} (q={q_int})", color="magenta")
plt.xlabel("log10(R [cm])")
plt.ylabel("log10(M [g])")
plt.title("Low-mass power-law fit (log–log)")
plt.grid(True)
plt.legend()
plt.show()

# ============================================================
# Central density rho_c (for WDs used in low-mass fit) + print min/max + plot
# ============================================================
R_sel = R_cm[sel]
M_sel = M_msun[sel]

a = R_sel / xi_n
rho_c = (((n_star + 1)*K_star) / (4*np.pi*G*a**2))**(n_star/(n_star - 1))

print("\nCentral densities for WDs in the fit:")
print("rho_c min =", np.min(rho_c), "g/cm^3")
print("rho_c max =", np.max(rho_c), "g/cm^3")

plt.figure()
plt.scatter(M_sel, rho_c, s=14, color="red")
plt.yscale("log")
plt.xlabel(r"$M\,(M_\odot)$")
plt.ylabel(r"$\rho_c\;(\mathrm{g\,cm^{-3}})$")
plt.title("Central density vs Mass (WDs used in low-mass fit)")
plt.grid(True, which="both")
plt.show()

# ============================================================
# Part (d,e): Full EOS + hydrostatic IVP solver (RK4)
# ============================================================
def p_of_rho(rho, C, D, q):
    x = (rho / D)**(1.0 / q)
    return C * (x*(2*x**2 - 3)*np.sqrt(1 + x**2) + 3*np.arcsinh(x))

def dPdrho_analytic(rho, C, D, q):
    if rho <= 0:
        return np.nan
    x = (rho / D)**(1.0 / q)
    s = np.sqrt(1.0 + x**2)
    fp = (6*x**2 - 3)*s + (2*x**4 - 3*x**2 + 3)/s
    return C * fp * (x / (q * rho))

def solve_star(rho_c, C, D, q, dr=2e5, r_max=5e10):
    rho_floor = 1e-12 * D

    def rhs(r, m, rho):
        if r == 0.0:
            return 0.0, 0.0
        dmdr = 4*np.pi * r**2 * rho
        dpdr = -G * m * rho / r**2
        dPdrho = dPdrho_analytic(rho, C, D, q)
        if (not np.isfinite(dPdrho)) or dPdrho <= 0:
            return dmdr, 0.0
        drhodr = dpdr / dPdrho
        return dmdr, drhodr

    r = 0.0
    m = 0.0
    rho = rho_c

    while rho > rho_floor and r < r_max:
        k1m, k1r = rhs(r, m, rho)
        k2m, k2r = rhs(r + 0.5*dr, m + 0.5*dr*k1m, rho + 0.5*dr*k1r)
        k3m, k3r = rhs(r + 0.5*dr, m + 0.5*dr*k2m, rho + 0.5*dr*k2r)
        k4m, k4r = rhs(r + dr,     m + dr*k3m,     rho + dr*k3r)

        m_new = m + (dr/6.0)*(k1m + 2*k2m + 2*k3m + k4m)
        rho_new = rho + (dr/6.0)*(k1r + 2*k2r + 2*k3r + k4r)
        r_new = r + dr

        if rho_new <= rho_floor:
            t = (rho - rho_floor) / (rho - rho_new) if rho != rho_new else 1.0
            R_surf = r + t*dr
            M_surf = m + t*(m_new - m)
            return R_surf, M_surf

        r, m, rho = r_new, m_new, rho_new

    return r, m

# ============================================================
# Part (d): fit D 
# ============================================================
R_obs = R_cm
M_obs = M_cgs
q = q_int  # fixed to 3

D_grid = np.logspace(6, 8, 30)
rho_c_grid_d = np.logspace(5, 9, 20)

errors = []
curves = []

for D in D_grid:
    C = (5.0/8.0) * K_star * (D**(5.0/q))

    R_model = []
    M_model = []
    for rho0 in rho_c_grid_d:
        R_i, M_i = solve_star(rho0, C, D, q)
        R_model.append(R_i)
        M_model.append(M_i)

    R_model = np.array(R_model)
    M_model = np.array(M_model)

    s = np.argsort(R_model)
    R_model = R_model[s]
    M_model = M_model[s]

    cs = CubicSpline(R_model, M_model, extrapolate=False)
    M_pred = cs(R_obs)

    good = np.isfinite(M_pred)
    M_pred_msun = M_pred[good] / M_sun
    M_obs_msun  = M_obs[good] / M_sun

    err = np.sum(((M_pred_msun - M_obs_msun) / M_obs_msun)**2)
    errors.append(err)
    curves.append((D, C, R_model, M_model))

errors = np.array(errors)
best_idx = np.argmin(errors)
D_best, C_best, R_best, M_best = curves[best_idx]

print("\nPart (d) best-fit:")
print("D_best =", D_best, "g/cm^3")
print("C_best =", C_best)

# --- Compare to Eq. (11) theory ---
C_th = (m_e**4 * c**5) / (24 * np.pi**2 * hbar**3)
D_th = (mu_e * m_u * m_e**3 * c**3) / (3 * np.pi**2 * hbar**3)

print("\nEq. (11) theoretical values (mu_e = 2):")
print("C_theory =", C_th)
print("D_theory =", D_th, "g/cm^3")

print("\nComparison (fit / theory):")
print("D_best / D_theory =", D_best / D_th)
print("C_best / C_theory =", C_best / C_th)

print("\nPercent difference:")
print("D percent diff =", 100*(D_best - D_th)/D_th, "%")
print("C percent diff =", 100*(C_best - C_th)/C_th, "%")

# ============================================================
# Part (e): Full M-R curve and Chandrasekhar mass using (C_best, D_best, q=3)
# ============================================================
rho_c_grid_e = np.logspace(4, 11, 120)

R_curve = []
M_curve = []

for rho0 in rho_c_grid_e:
    R_i, M_i = solve_star(rho0, C_best, D_best, q)
    R_curve.append(R_i)
    M_curve.append(M_i)

R_curve = np.array(R_curve)
M_curve = np.array(M_curve)

s = np.argsort(R_curve)
R_curve = R_curve[s]
M_curve = M_curve[s]

idx_max = np.argmax(M_curve)
MCh = M_curve[idx_max]
R_at_MCh = R_curve[idx_max]

print("\nPart (e): Chandrasekhar mass from numerical curve")
print("MCh (max M on curve) =", MCh / M_sun, "M_sun")
print("R at MCh =", R_at_MCh / R_earth, "R_earth")

plt.figure()
plt.plot(R_curve / R_earth, M_curve / M_sun, linewidth=2, color="black", label="model M-R curve")
plt.scatter([R_at_MCh / R_earth], [MCh / M_sun], color="blue", s=40, label="max mass (MCh)")
plt.scatter(R_rearth, M_msun, s=10, color="red", alpha=0.6, label="data")
plt.xlabel(r"$R_\oplus$")
plt.ylabel(r"$M_\odot$")
plt.title("Full M-R curve and maximum mass (Chandrasekhar mass)")
plt.grid(True)
plt.legend()
plt.show()


# Computing theoretical Chandrasekhar mass from n=3 polytrope 

# Lane-Emden constants for n = 3 
xi1_3, thetaP_3 = lane_emden_constants(3.0)          # thetaP_3 = theta'(xi1)
A3 = - (xi1_3**2) * thetaP_3                         # dimensionless constant
K_rel = 2.0 * C_th * (D_th**(-4.0/3.0))
MCh_theory = 4.0*np.pi * A3 * (((3.0 + 1.0) * K_rel) / (4.0*np.pi*G))**(3.0/2.0)

print("\n=== Chandrasekhar mass from n=3 polytrope ===")
print("Lane-Emden (n=3): xi1 =", xi1_3, "  theta'(xi1) =", thetaP_3, "  A3=-xi1^2 theta' =", A3)
print("K_rel =", K_rel)
print("MCh_theory =", MCh_theory / M_sun, "M_sun")

print("\n=== Comparison to numerical MCh ===")
print("MCh_numerical (best-fit EOS) =", MCh / M_sun, "M_sun")
print("ratio (best-fit / theory)   =", (MCh / MCh_theory))
