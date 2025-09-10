import numpy as np
N=22
R = 500.0
kappa = 0.1
amp = 0.01
sigma = 50.0
nbases=N-2
xi=np.zeros(N)
ri=np.zeros_like(xi)

initv=np.zeros(N)
initv[0]=1
a=initv

R=500
def chebychevtorad(x):
    return R*(x+1)/2
def radtochebychev(r):
    return (2*r/R)-1

xi = np.cos(np.pi*np.arange(N)/(N-1))[::-1]
ri=chebychevtorad(xi)


K=kappa**2*amp**2*np.exp(-2*(ri**2)/sigma**2) + amp**2*np.exp(-2*(ri**2)/sigma**2)
dk=4*(ri**2/sigma**4) *amp**2 *np.exp(-2*(ri**2)/sigma**2)

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
    return [PHI,2*PHIX/R,PHIXX*4/R**2]

H=np.zeros([N,N])

for iter in range(10):
    for i in range(1,N-1):
        de,deri,deri2=bases(xi[i],nbases)
        H[i,:]=deri2+(2/chebychevtorad(xi[i]))*deri +5*np.pi*((a.T@de)**4)*K[i] +np.pi*dk[i]
    G=np.zeros([N,N])

    for i in range(1,N-1):
        de,deri,deri2=bases(xi[i],nbases)
        G[i,:]=-1*(deri2+(2/chebychevtorad(xi[i]))*deri +np.pi*((a.T@de)**5)*K[i] +np.pi*dk[i]*a[i]) 
    G[0,:]=bases(xi[0],nbases)[1]
    G[-1,:]=bases(xi[-1],nbases)[0]
    H[0,:]=bases(xi[0],nbases)[1]
    H[-1,:]=bases(xi[-1],nbases)[0]*-1/chebychevtorad(xi[-1])
    print(H)
    dpsi=np.linalg.solve(H,G@ a.T)
    print(np.linalg.norm(dpsi))
    a=a+dpsi


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
#plt.scatter([-1, 1], [1, 10], color='red', zorder=5, label="Boundary conds")
plt.legend()
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Chebyshev Pseudospectral Solution of Laplace Problem")
plt.show()
