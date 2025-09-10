import numpy as np
N=22
nbases=N-2
xi=np.zeros(N)
def chebychev(n,x):
    if n==0:
        return 1
    elif n==1:
        return x
    else:
        return 2*x*chebychev(n-1,x)-chebychev(n-2,x)

def bases(x,nbases):
    PHI = np.zeros(N)
    PHIX = np.zeros(N)
    PHIXX = np.zeros(N)

    t=np.arccos(x)
    C=np.cos(t)
    S=np.sin(t)
    for j in range(0, N):
        n=j
        TN = np.cos(n * t)
        TNT = - n * np.sin(n * t)              # d/dt cos(nt) = -n sin(nt)
        TNTT = - (n**2) * TN
        if abs(S) < 1e-14:   # very close to endpoints
            TNX = 0.0
            TNXX = 0.0
        else:
            TNX = - TNT / S
            TNXX = TNTT / (S*S) - (TNT * C) / (S*S*S)

        PHI[j]=TN
        PHIX[j]=TNX
        PHIXX[j]=TNXX
    return [PHI,PHIX,PHIXX]
H=np.zeros([N,N])
for i in range(N):
    xi[i]=np.cos(np.pi*i/(N-1))
for i in range(1,N-1):
    _,_,deri=bases(xi[i],nbases)
    H[i,:]=deri

H[0,:]=bases(xi[0],nbases)[0]
H[-1,:]=bases(xi[-1],nbases)[0]
Rhs=np.zeros(N)
Rhs[0]=10
Rhs[-1]=1


a = np.linalg.solve(H, Rhs)   # Chebyshev coefficients    

def psi_eval(x, a):
    t = np.arccos(x)
    val = 0.0
    for j in range(N):
        val += a[j] * np.cos(j * t)
    return val

x_plot = np.linspace(-1, 1, 400)
y_plot = [psi_eval(x, a) for x in x_plot]

import matplotlib.pyplot as plt
plt.plot(x_plot, y_plot, label="Spectral Solution")
plt.scatter([-1, 1], [1, 10], color='red', zorder=5, label="Boundary conds")
plt.legend()
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Chebyshev Pseudospectral Solution of Laplace Problem")
plt.show()
