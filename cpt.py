import numpy as np
import matplotlib.pyplot as plt

# params
N = 22
R = 500.0
kappa = 0.1
amp = 0.5
sigma = 50.0
maxit = 20
damp = 1.0
tol = 1e-12

# collocation (flip so xi[0] is r=0 left, xi[-1] is r=R right)
xi = np.cos(np.pi * np.arange(N) / (N - 1))[::-1]
ri = R * (xi + 1.0) / 2.0

# nonlinear coefficient functions on collocation points
K  = (kappa**2 * amp**2 + amp**2) * np.exp(-2.0 * (ri**2) / sigma**2)
dk = 4.0 * (ri**2 / sigma**4) * amp**2 * np.exp(-2.0 * (ri**2) / sigma**2)

# initial coefficient vector (in coefficient space)
a = np.zeros(N)
a[0] = 1.0

# bases function (returns PHI, PHIX, PHIXX scaled to r-derivatives)
def bases(x, nbases):
    PHI = np.zeros(N)
    PHIX = np.zeros(N)
    PHIXX = np.zeros(N)

    t = np.arccos(x)
    C = np.cos(t)
    S = np.sin(t)
    for j in range(0, N):
        n = j
        TN = np.cos(n * t)
        TNT = - n * np.sin(n * t)              # d/dt cos(nt)
        TNTT = - (n**2) * TN
        if abs(S) < 1e-14:
            TNX = 0.0
            TNXX = 0.0
        else:
            TNX = - TNT / S
            TNXX = TNTT / (S*S) - (TNT * C) / (S*S*S)

        PHI[j] = TN
        PHIX[j] = TNX
        PHIXX[j] = TNXX
    # scale to derivatives wrt r: d/dt -> d/dx -> d/dr, your previously used scale factors:
    return PHI, 2.0 * PHIX / R, PHIXX * 4.0 / R**2

# Precompute basis rows for all collocation points once (optional but efficient)
PHI_rows = np.zeros((N, N))
PHIX_rows = np.zeros((N, N))
PHIXX_rows = np.zeros((N, N))
for i in range(N):
    PHI_rows[i, :], PHIX_rows[i, :], PHIXX_rows[i, :] = bases(xi[i], N)

# Newton/Picard iterations
for it in range(maxit):
    # compute physical values at collocation points: u_i = sum_j a_j * T_j(x_i)
    u_vals = PHI_rows @ a
    ur_vals = PHIX_rows @ a   # if you need first derivative values

    # assemble H and residual F
    H = np.zeros((N, N))
    F = np.zeros(N)

    for i in range(1, N-1):   # interior rows
        de   = PHI_rows[i, :]      # basis values T_j(x_i)
        deri = PHIX_rows[i, :]     # d/dr basis row
        deri2= PHIXX_rows[i, :]    # d2/dr2 basis row

        # scalar Jacobian coefficient at point i:
        # nonlinear term was 5π u^4 K + π dk u  => derivative wrt u = 20π u^3 K + π dk
        Ci = 20.0 * np.pi * (u_vals[i]**3) * K[i] + np.pi * dk[i]

        # operator row: second derivative + (2/r) first derivative + Ci * identity-on-u
        H[i, :] = deri2 + (2.0 / ri[i]) * deri + Ci * de

        # residual at current iterate: L(u)
        F[i] = (deri2 @ a) + (2.0 / ri[i]) * (deri @ a) \
               + 5.0 * np.pi * (u_vals[i]**4) * K[i] + np.pi * dk[i] * u_vals[i]

    # boundary rows: enforce homogeneous Dirichlet on the correction delta
    H[0, :] = PHI_rows[0, :];   F[0]  = 0.0
    H[-1,:] = PHI_rows[-1,:];   F[-1] = 0.0

    # Solve H dpsi = -F
    try:
        dpsi = np.linalg.solve(H, -F)
    except np.linalg.LinAlgError as e:
        print("Linear solve failed at iter", it, " — matrix may be singular.")
        raise

    a = a + damp * dpsi
    print(f"iter {it:2d}, ||dpsi|| = {np.linalg.norm(dpsi):.3e}, ||F|| = {np.linalg.norm(F):.3e}")

    if np.linalg.norm(dpsi) < tol:
        print("Converged.")
        break

# evaluate spectral polynomial for plotting
def psi_eval(x, a):
    t = np.arccos(x)
    return sum(a[j] * np.cos(j * t) for j in range(len(a)))

x_plot = np.linspace(-1, 1, 400)
y_plot = [psi_eval(x, a) for x in x_plot]

plt.plot(x_plot, y_plot, label="Spectral Solution")
plt.legend()
plt.xlabel("x"); plt.ylabel("ψ(x)")
plt.title("Chebyshev Pseudospectral")
plt.show()
