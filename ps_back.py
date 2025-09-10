import numpy as np
N=500
R = 500.0
kappa = 0.1
amp = 0.001
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
        if x==1:   # very close to endpoints
            TNX = n**2
            TNXX = 0.0
        elif x==-1:
            TNX=(-1)**(n-1) *n**2
            TNXX=0.0
        else:
            TNX = - TNT / S
            TNXX = TNTT / (S*S) - (TNT * C) / (S*S*S)

        PHI[j]=TN
        PHIX[j]=TNX
        PHIXX[j]=TNXX
    return [PHI,2*PHIX/R,PHIXX*4/R**2]

H=np.zeros([N,N])
G=np.zeros(N)
def chebychev(n,x):
    if n==0:
        return 1
    elif n==1:
        return x
    else:
        return 2*x*chebychev(n-1,x)-chebychev(n-2,x)
diffmat=np.zeros([N,N])
diffmat1=np.zeros([N,N])
diffmat2=np.zeros([N,N])
for i in range(N):
    diffmat[i,:]=bases(xi[i],nbases)[0]
    diffmat1[i,:]=bases(xi[i],nbases)[1]
    diffmat2[i,:]=bases(xi[i],nbases)[2]

cond=True
while(cond==True) :
    u=diffmat@a.T
    u=u.T
    for i in range(1,N-1):
        H[i,:]=diffmat2[i,:]+(2/chebychevtorad(xi[i]))*diffmat1[i,:] +5*np.pi*((u[i])**4)*K[i]*diffmat[i,:] +np.pi*dk[i]*diffmat[i,:]
        G[i]=-1*(diffmat2[i,:]@ a+(2/chebychevtorad(xi[i]))*diffmat1[i,:] @ a +np.pi*((u[i])**5)*K[i] +np.pi*dk[i]*u[i]) 
   
    H[0,:]=diffmat1[0,:]
    H[-1,:]=diffmat1[-1,:]+diffmat[-1,:]/R
    G[0]=-(diffmat1[0,:] @ a-0.0)
    G[-1]=-(diffmat1[-1,:]@a+diffmat[-1,:]@ a/R-(1/R))
    #print(H)
    dpsi=np.linalg.solve(H,G)
    a=a+dpsi
    if (np.linalg.norm(dpsi) < 1e-13):
        cond=False
    #print(cond)

def psi_eval(x, a):
    t = np.arccos(x)
    val = 0.0
    for j in range(N):
        val += a[j] * np.cos(j * t)
    return val


x_plot = xi
y_plot = [psi_eval(x, a) for x in x_plot]

# Map x to r
r_plot = chebychevtorad(x_plot)

import matplotlib.pyplot as plt
plt.plot(r_plot, y_plot, label="Spectral Solution in r")
plt.xlabel("r")
plt.ylabel("ψ(r)")
plt.title("Chebyshev Pseudospectral Solution vs r")
plt.legend()
plt.show()
plt.savefig("AMP001.PNG")

n = np.arange(len(a))

plt.figure(figsize=(6,4))
plt.semilogy(n, np.abs(a), 'o-')
plt.xlabel("Chebyshev mode n")
plt.ylabel("|a[n]|")
plt.title("Decay of Chebyshev coefficients N="+str(N))
plt.grid(True, which='both')
plt.savefig(f"Spectral_coeff_{N}.png")
plt.show()
