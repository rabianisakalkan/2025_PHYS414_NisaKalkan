import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Constants (CGS)
# ============================================================
G = 6.67430e-8
M_sun = 1.98847e33
R_earth = 6.371e8

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
plt.scatter(R_rearth, M_msun, s=12,color='red')
plt.xlabel("$R_\odot$")
plt.ylabel("$M_\odot$")
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

    # interpolate to theta=0
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

# Fit 1 (free slope)
alpha1, b1 = np.polyfit(x, y, 1)
n1 = (3 - alpha1) / (1 - alpha1)
q_real = 5*n1 / (1 + n1)
q_int = int(np.rint(q_real))

# Fit 2 (q fixed -> slope fixed)
n_star = q_int / (5 - q_int)
alpha_fixed = (3 - n_star) / (1 - n_star)  
b2 = np.mean(y - alpha_fixed*x)
A_fixed = 10**b2

# Lane–Emden constants for n_star
xi_n, theta_p = lane_emden_constants(n_star)

# Convert A_fixed -> K*
pref = 4*np.pi * (-theta_p) * (xi_n**((n_star+1)/(n_star-1)))
K_star = (4*np.pi*G/(n_star+1)) * ((A_fixed/pref)**((n_star-1)/n_star))

# Print results
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

# Part (c) log-log plot with fit lines
plt.figure()
plt.scatter(x, y, s=12, label=f"data (M <= {Mmax_Msun} $M_\odot$")
xx = np.linspace(x.min(), x.max(), 200)
plt.plot(xx, alpha1*xx + b1, linewidth=2, label=f"fit1 slope={alpha1:.3f}", color='red')
plt.plot(xx, alpha_fixed*xx + np.log10(A_fixed), linewidth=2,
         label=f"fit2 slope={alpha_fixed:.3f} (q={q_int})", color='magenta')
plt.xlabel("log10(R [cm])")
plt.ylabel("log10(M [g])")
plt.title("Low-mass power-law fit (log–log)")
plt.grid(True)
plt.legend()
plt.show()

# ============================================================
# Using n* and Lane–Emden to find central density rho_c
# for the WDs used in the fit, and plotting rho_c vs M
# ============================================================

R_sel = R_cm[sel]
M_sel = M_msun[sel]

a = R_sel / xi_n
rho_c = (((n_star + 1)*K_star) / (4*np.pi*G*a**2))**(n_star/(n_star - 1))  # g/cm^3


plt.figure()
plt.scatter(M_sel, rho_c, s=14, color='r')
plt.yscale("log")

plt.xlabel(r"$M\,(M_\odot)$")
plt.ylabel(r"$\rho_c\;(\mathrm{g\,cm^{-3}})$")
plt.title(r"Central density vs Mass (WDs used in low-mass fit)")

plt.grid(True, which="both")
plt.show()

