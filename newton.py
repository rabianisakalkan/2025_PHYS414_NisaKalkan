import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- constants (CGS) ---
G_cgs = 6.67430e-8        # cm^3 g^-1 s^-2
M_sun = 1.98847e33        # g
R_earth = 6.371e8         # cm

# --- load data  ---
df = pd.read_csv("white_dwarf_data.csv")  

# columns: wdid, logg, mass
logg = df["logg"].astype(float).to_numpy()      # log10(g) in cgs
M_msun = df["mass"].astype(float).to_numpy()    # in solar masses

# --- convert logg -> radius ---
g_cgs = 10**logg                                 # cm/s^2
M_cgs = M_msun * M_sun                           # g
R_cm = np.sqrt(G_cgs * M_cgs / g_cgs)            # cm
R_rearth = R_cm / R_earth                        # Earth radii


# --- plot M vs R ---
plt.figure()
plt.scatter(R_rearth, M_msun, s=12, color='r')
plt.xlabel(r"$R_\oplus$")
plt.ylabel(r"$M_\odot$")
plt.title("White Dwarf Mass vs Radius")
plt.grid(True)
plt.show()
