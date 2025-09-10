# Pseudospectral Initial Data Solver

This repository contains Python implementations of the **pseudospectral method** for solving boundary value problems in numerical relativity and related areas.  
The method uses **Chebyshev polynomials** on Gauss–Lobatto collocation points to expand smooth functions and enforce differential equations at discrete grid points.  

Currently, the repository includes:

- `ps_back.py` — Solver for generating **initial data for a massive complex scalar field** using a pseudospectral Newton iteration scheme.  
- `ps_poisson.py` — A simpler **Poisson equation solver** implemented with the same pseudospectral framework, useful for testing and benchmarking.  
- `test.pdf` — A test pdf file containing sufficient theory regarding the approaches used
---

## 🔑 Features
- Chebyshev pseudospectral discretization on Gauss–Lobatto points  
- Affine map to compactify the domain from \([-1,1]\) to \([0,R]\)  
- Automatic construction of derivative matrices \(D, D^2\) using Chebyshev recurrence relations  
- Handles boundary conditions via **row replacement** in the collocation system  
- Nonlinear solver for scalar field equations (Newton–Raphson)  
- Linear solver for Poisson-type equations  


