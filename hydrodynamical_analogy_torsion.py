#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# Griglia e parametri
# ---------------------------
N = 101             # punti per lato 
L = 1.0             # dominio [0, L] x [0, L]
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing='xy')
dx = x[1] - x[0]
dy = y[1] - y[0]

# centro geometrico del contenitore
xc, yc = L/2.0, L/2.0
rho2_center = (X - xc)**2 + (Y - yc)**2

# Parametri fisici
Sigma = 1.0         # Analogia 1: ∇²ψ = -2*Sigma
Omega = 1.0         # Analogia 2: velocità angolare del contenitore
P = 1.0             # Analogia 3: gradiente di pressione (dp/dz)
mu = 1.0            # viscosità

# Risolutore (SOR pointwise stabile)
omega = 1.3         # 1.0 = Gauss–Seidel. 1.3 più veloce, resta stabile.
tol = 1e-6
maxit = 20000
log_every = 500

def bc_zero():
    """Matrice BC con Dirichlet omogenee su ∂C (0 su bordo, NaN all'interno)."""
    BC = np.full((N, N), np.nan, dtype=float)
    BC[0, :]  = 0.0
    BC[-1, :] = 0.0
    BC[:, 0]  = 0.0
    BC[:, -1] = 0.0
    return BC

def solve_poisson_dirichlet_gs(f, BC, omega=1.0, tol=1e-6, maxit=20000, log_every=500):
    """
    Risolve ∇²U = f su griglia cartesiana regolare con Dirichlet (valori in BC).
    Aggiornamento pointwise (Gauss–Seidel / SOR vero): stabile anche con omega>1.
    """
    U = np.zeros_like(f, dtype=float)
    mask_bc = ~np.isnan(BC)
    U[mask_bc] = BC[mask_bc]

    dx2, dy2 = dx*dx, dy*dy
    den = 2.0 * (dx2 + dy2)

    for it in range(1, maxit + 1):
        diff = 0.0
        for i in range(1, N-1):
            for j in range(1, N-1):
                if mask_bc[i, j]:
                    continue
                rhs = ((U[i+1, j] + U[i-1, j]) * dy2 +
                       (U[i, j+1] + U[i, j-1]) * dx2 -
                       f[i, j] * dx2 * dy2)
                new = rhs / den
                U_new = (1.0 - omega) * U[i, j] + omega * new
                d = abs(U_new - U[i, j])
                if d > diff:
                    diff = d
                U[i, j] = U_new

        if math.isnan(diff) or math.isinf(diff):
            raise RuntimeError(f"Divergenza a iter {it} (NaN/Inf)")

        if it % log_every == 0 or it == 1:
            print(f"iter {it}, residuo {diff:.3e}")

        if diff < tol:
            print(f"Convergenza in {it} iter (residuo {diff:.3e})")
            break
    else:
        print(f"Attenzione: non convergente entro {maxit} iter, ultimo residuo={diff:.3e}")

    return U

# ---------------------------
# Analogia 1: ∇²ψ = -2*Sigma, ψ=0 su ∂C
# ---------------------------
BC = bc_zero()
f1 = -2.0 * Sigma * np.ones((N, N), dtype=float)
print("\n--- Analogia 1 (ψ, contenitore fisso) ---")
psi1 = solve_poisson_dirichlet_gs(f1, BC, omega=omega, tol=tol, maxit=maxit, log_every=log_every)

# velocità dal campo ψ: v = (-ψ_y, ψ_x)
psi1_y, psi1_x = np.gradient(psi1, dy, dx)
v1x = -psi1_y
v1y =  psi1_x

# ---------------------------
# Analogia 2: ψ = (Ω/2)|r-r_c|^2 + Φ, ∇²Φ = -2Ω, Φ=0 su ∂C
# ---------------------------
f2 = -2.0 * Omega * np.ones((N, N), dtype=float)
print("\n--- Analogia 2 (Φ, contenitore rotante) ---")
Phi = solve_poisson_dirichlet_gs(f2, BC, omega=omega, tol=tol, maxit=maxit, log_every=log_every)

# campo relativo: v_rel = (-Φ_y, Φ_x)
Phi_y, Phi_x = np.gradient(Phi, dy, dx)
vrel_x = -Phi_y
vrel_y =  Phi_x

# campo rigido (rotazione attorno al centro del dominio!)
vrig_x = -Omega * (Y - yc)
vrig_y =  Omega * (X - xc)

# campo assoluto nel laboratorio
vabs_x = vrel_x + vrig_x
vabs_y = vrel_y + vrig_y

# ψ assoluta (se serve)
psi_abs = 0.5 * Omega * rho2_center + Phi

# ---------------------------
# Analogia 3: Poiseuille come problema armonico
# v3 = Ξ + (P/4μ) ρ²,   ∇²Ξ=0,   Ξ = -(P/4μ)ρ² su ∂C
# ---------------------------

# condizione al contorno per Ξ
BC_Xi = np.full((N, N), np.nan, dtype=float)
BC_Xi[0, :]  = -(P/(4*mu)) * rho2_center[0, :]
BC_Xi[-1, :] = -(P/(4*mu)) * rho2_center[-1, :]
BC_Xi[:, 0]  = -(P/(4*mu)) * rho2_center[:, 0]
BC_Xi[:, -1] = -(P/(4*mu)) * rho2_center[:, -1]

# termine di sorgente nullo (Ξ armonica)
f3 = np.zeros((N, N), dtype=float)

print("\n--- Analogia 3 (Ξ armonica, Poiseuille) ---")
Xi = solve_poisson_dirichlet_gs(f3, BC_Xi, omega=omega, tol=tol, maxit=maxit, log_every=log_every)

# ricostruzione del profilo
v3 = Xi + (P/(4*mu)) * rho2_center

# ---------------------------
# Figura 1: analogia 1
# ---------------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))

# Analogia 1: streamlines
ax1.streamplot(x, y, v1x, v1y, density=1.2, linewidth=0.9)
ax1.set_title("Analogia 1: Contenitore fisso")
ax1.set_aspect('equal', 'box'); ax1.set_xlim(0, L); ax1.set_ylim(0, L)
ax1.set_xlabel('x'); ax1.set_ylabel('y')

plt.tight_layout()
plt.savefig('analogia1.png', dpi=300)

# ---------------------------
# Figura 2: analogia 2
# ---------------------------

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))

# Analogia 2 (osservatore solidale): Φ + streamlines di v_rel (opzionale)
ax2[0].contour(X, Y, Phi, levels=25)
ax2[0].streamplot(x, y, vrel_x, vrel_y, density=1.2, linewidth=0.9)
ax2[0].set_title("Analogia 2: frame solidale")
ax2[0].set_aspect('equal', 'box'); ax2[0].set_xlim(0, L); ax2[0].set_ylim(0, L)
ax2[0].set_xlabel('x'); ax2[0].set_ylabel('y')

# Analogia 2 (osservatore fisso): streamlines del campo assoluto
ax2[1].streamplot(x, y, vabs_x, vabs_y, density=1.2, linewidth=0.9)
ax2[1].set_title("Analogia 2: campo assoluto")
ax2[1].set_aspect('equal', 'box'); ax2[1].set_xlim(0, L); ax2[1].set_ylim(0, L)
ax2[1].set_xlabel('x'); ax2[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('analogia2.png', dpi=300)

# ---------------------------
# Figura 3: Poiseuille 
# ---------------------------

fig3, ax3 = plt.subplots(figsize=(6, 5))
cf = ax3.contourf(X, Y, v3, levels=30)
plt.colorbar(cf, ax=ax3)
ax3.set_title("Analogia 3: v3 (Poiseuille)")
ax3.set_aspect('equal', 'box')
ax3.set_xlim(0, L); ax3.set_ylim(0, L)
ax3.set_xlabel('x'); ax3.set_ylabel('y')
plt.tight_layout()
plt.savefig('poiseuille.png', dpi=300)
plt.show()