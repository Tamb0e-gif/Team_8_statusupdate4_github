
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# STAP 1: MEETDATA INLEZEN
# ============================================================

df = pd.read_csv('team_8_meting1.csv')

hoek_graden = df.iloc[:, 0].to_numpy(dtype=float)
aantal_franjes = df.iloc[:, 1].to_numpy(dtype=float)
onzekerheid_N = df.iloc[:, 2].to_numpy(dtype=float)

hoek_radialen = np.radians(hoek_graden)

# Controleren of de meetdata bruikbaar zijn
if not (
    np.all(np.isfinite(hoek_radialen))
    and np.all(np.isfinite(aantal_franjes))
    and np.all(np.isfinite(onzekerheid_N))
):
    raise ValueError("De meetdata bevatten ongeldige waarden.")

if np.any(onzekerheid_N <= 0):
    raise ValueError("Alle onzekerheden in N moeten groter zijn dan nul.")


# ============================================================
# STAP 2: CONSTANTEN EN ONZEKERHEDEN
# ============================================================

d = 0.003
sigma_d = 0.00005

lambda_laser = 533.735e-9
sigma_lambda = 53.8e-9

aantal_simulaties = 10000

rng = np.random.default_rng(42)


# ============================================================
# STAP 3: THEORETISCH MODEL
# ============================================================

def model_franjes(hoek, n, d, lam):
    return (2 * d / lam) * (
        np.sqrt(n**2 - np.sin(hoek)**2)
        - np.cos(hoek)
        + (1 - n)
    )


# Model voor de fit: d en lambda zijn hier vast
def model_fit(hoek, n):
    return model_franjes(hoek, n, d, lambda_laser)


# ============================================================
# STAP 4: FIT UITVOEREN
# ============================================================

# De brekingsindex wordt begrensd op een fysisch plausibel
# interval voor plexiglas. Controleer of de uitkomst niet
# tegen een grens aan zit.
popt, pcov = curve_fit(
    model_fit,
    hoek_radialen,
    aantal_franjes,
    sigma=onzekerheid_N,
    absolute_sigma=True,
    p0=[1.5],
    bounds=(1.0, 3.0),
    method='trf',
    max_nfev=20000
)

n_fit = popt[0]

# Controleer of de fit een eindige covariantiematrix geeft
if np.all(np.isfinite(pcov)):
    sigma_n_fit = np.sqrt(pcov[0, 0])
else:
    sigma_n_fit = np.nan

print("========== OORSPRONKELIJKE FIT ==========")
print(f"Brekingsindex n = {n_fit:.6f}")
print(f"Formele fitonzekerheid = {sigma_n_fit:.6g}")

if np.isclose(n_fit, 1.0, atol=1e-4) or np.isclose(
    n_fit, 3.0, atol=1e-4
):
    print(
        "WAARSCHUWING: de fit zit tegen een grens aan. "
        "Controleer of n goed bepaald kan worden."
    )


# ============================================================
# STAP 5: MONTE CARLO-SIMULATIE
# ============================================================

n_simulaties = []

for i in range(aantal_simulaties):

    # Onzekere grootheden willekeurig variëren
    d_sim = rng.normal(d, sigma_d)
    lambda_sim = rng.normal(lambda_laser, sigma_lambda)

    # Nieuwe mogelijke meetwaarden, inclusief meetonzekerheid
    N_sim = rng.normal(aantal_franjes, onzekerheid_N)

    # Fysisch onmogelijke waarden overslaan
    if d_sim <= 0 or lambda_sim <= 0:
        continue

    def model_sim(hoek, n):
        return model_franjes(
            hoek, n, d_sim, lambda_sim
        )

    try:
        popt_sim, _ = curve_fit(
            model_sim,
            hoek_radialen,
            N_sim,
            sigma=onzekerheid_N,
            absolute_sigma=True,
            p0=[n_fit],
            bounds=(1.0, 3.0),
            method='trf',
            max_nfev=20000
        )

        n_simulaties.append(popt_sim[0])

    except (RuntimeError, ValueError):
        continue


n_simulaties = np.asarray(n_simulaties)

if len(n_simulaties) < 100:
    raise RuntimeError(
        "Te weinig succesvolle simulaties. "
        "Controleer het model en de meetdata."
    )

# Totale onzekerheid uit de spreiding van de simulaties
sigma_n_totaal = np.std(n_simulaties, ddof=1)

print("\n========== MONTE CARLO-RESULTATEN ==========")
print(f"Aantal geslaagde simulaties: {len(n_simulaties)}")
print(f"Onzekerheid uit de fit: {sigma_n_fit:.6g}")
print(f"Totale onzekerheid: {sigma_n_totaal:.6f}")
print(
    f"EINDRESULTAAT: n = {n_fit:.5f} "
    f"+/- {sigma_n_totaal:.5f}"
)

# Waarschuwing bij simulaties die de grenzen bereiken
fractie_grenzen = np.mean(
    np.isclose(n_simulaties, 1.0, atol=1e-4)
    | np.isclose(n_simulaties, 3.0, atol=1e-4)
)

if fractie_grenzen > 0.01:
    print(
        "WAARSCHUWING: meer dan 1% van de simulaties "
        "zit tegen een grens aan. De onzekerheid kan "
        "daardoor niet betrouwbaar worden geschat."
    )



# ============================================================
# STAP 6: FITLIJNEN BEREKENEN
# ============================================================

hoek_fit_graden = np.linspace(
    hoek_graden.min(),
    hoek_graden.max(),
    500
)

hoek_fit_radialen = np.radians(hoek_fit_graden)

# Fit op basis van de meetdata
aantal_franjes_fit = model_fit(
    hoek_fit_radialen,
    n_fit
)

# Theoretische verwachting voor plexiglas: n = 1,49
n_theorie = 1.49

aantal_franjes_theorie = model_franjes(
    hoek_fit_radialen,
    n_theorie,
    d,
    lambda_laser
)


# ============================================================
# STAP 7: PLOT MAKEN
# ============================================================

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 15,
    'legend.fontsize': 10
})

fig, ax = plt.subplots(figsize=(9, 6))

# Meetpunten met foutbalken
ax.errorbar(
    hoek_graden,
    aantal_franjes,
    yerr=onzekerheid_N,
    fmt='o',
    markersize=6,
    color='#A83232',
    ecolor='#777777',
    elinewidth=1.2,
    capsize=4,
    capthick=1.2,
    markeredgecolor='white',
    markeredgewidth=0.7,
    label='Meetdata (Team 8)',
    zorder=3
)

# Fit op basis van de meetdata
ax.plot(
    hoek_fit_graden,
    aantal_franjes_fit,
    color='#2468A2',
    linewidth=2.5,
    label=f'Fit (n = {n_fit:.4f})',
    zorder=2
)

# Theoretische verwachting met n = 1,49
ax.plot(
    hoek_fit_graden,
    aantal_franjes_theorie,
    color='#24855B',
    linewidth=2.2,
    linestyle='--',
    label='Theoretische verwachting (n = 1,49)',
    zorder=1
)

# Labels en titel
ax.set_xlabel('Invalshoek (°)')
ax.set_ylabel('Aantal franjes $N$')
ax.set_title(
    'Bepaling van de brekingsindex van plexiglas',
    pad=15
)

# Resultaten in informatievak
resultaat_tekst = (
    rf'Fit: $n = {n_fit:.4f} \pm {sigma_n_totaal:.4f}$'
    '\n'
    rf'Theoretisch: $n = {n_theorie:.2f}$'
    '\n'
    rf'$d = {d * 1000:.3f}\ \mathrm{{mm}}$'
    '\n'
    rf'$\lambda = {lambda_laser * 1e9:.3f}\ \mathrm{{nm}}$'
)

ax.text(
    0.03,
    0.97,
    resultaat_tekst,
    transform=ax.transAxes,
    va='top',
    ha='left',
    fontsize=10,
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor='#BBBBBB',
        alpha=0.95
    )
)

# Opmaak
ax.grid(
    True,
    linestyle='--',
    linewidth=0.7,
    alpha=0.35
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False, loc='best')

fig.tight_layout()

# Plot opslaan
fig.savefig(
    'team_8_fit_brekingsindex_plot.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
