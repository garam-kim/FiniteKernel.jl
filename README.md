# kernelherding.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://garam-kim.github.io/FiniteKernel.jl/dev/)



This package is a toolbox for finite-dimensional kernel herding using the FrankWolfe.jl package.

For Matern Kernel in the infinite-dimensional setting, move to [$\texttt{MaternKernel.jl}$](https://github.com/garam-kim/MaternKernel.jl).



## Overview

We focus on a specific kernel studied in ([Bach et al.](https://icml.cc/2012/papers/683.pdf)), that has a finite-dimensional feature space.
Let $\mathcal{Y} = \{-1, 1\}^d $ be a feature space with dimension $d \in \mathbb{N}$ and

```math
\mathcal{H}:= \left\lbrace f \colon \mathcal{Y} \to \mathbb{R} \mid f(y) = \langle f, \Phi(y) \rangle_\mathcal{H}, \text{ where } \Phi(y)=(y, yy^T) \right\rbrace
```
a Reproducing Kernel Hilbert Space (RKHS) with inner product $\langle \cdot, \cdot \rangle_\mathcal{H}$ defined by
```math
\langle f, g \rangle_\mathcal{H} := \sum_{i = 1}^{d} f_i(y)g_i(y)
```
for $f, g \in \mathcal{H}$, where $f = [f_1, f_2, \ldots, f_d]$. For $x \in \mathcal{X}$, the feature map $\Phi(x)=(x,xx^T)$ is composed of $x$ and of all of its pairwise products $xx^T$. Then, the reproducing kernel of $\mathcal{H}$ is indeed
```math
k(x, y) =\langle \Phi(x), \Phi(y)\rangle_\mathcal{H} = \sum_{i=1}^{d} \Phi_i(x) \Phi_i(y)= \langle x, y\rangle_2,
```
for $x,y \in \mathcal{X}$. In particular, it holds that
```math
k(x,y) = \big\langle k(z, x), k(z, y) \big\rangle_\mathcal{H}
```
for any $x, y, z\in \mathcal{X}$.  

We denote $\mathcal{M} \subset \mathcal{H}$ the marginal polytope defined by $\mathcal{M} = \text{conv}(\{ \Phi(x) \ |\  x \in \mathcal{X}\}$. In this setting, we compute the expectation 
```math
\mu : = \mathbb{E}_{p(x)} \Phi(x) = \sum_{i=1}^{2^d} p_i(x)\Phi_i(x) \in \mathcal{M}.
```
Since LMO always returns an element of the form $\Phi(x) \in \mathcal{M}$ for $ x \in \mathcal{X}$, the iterate $g_t$ constructed with FW is of the form $\sum_{i=1}^t w_i \Phi(x_i)$. The associated empirical mean with corresponding empirical distribution $\hat{p}(x)$ is defined by
```math
\hat{\mu} := \mathbb{E}_{\hat{p}(x)}\Phi(x) = \sum_{i=1}^t w_i\Phi(x_i) = g_t.
```
Thus,  $\texttt{FiniteKernel.jl}$ finds empirical distribution by solving a convex optimization problem of the form
```math
\min_{g \in \mathcal{M}}J(g) = \frac{1}{2}\Vert g-\mu \Vert^2.
```

For more mathematical description and the results, refer to [report.pdf](https://github.com/garam-kim/FiniteKernel.jl/blob/main/report.pdf).



## Installation

The most recent release is available via:

```julia
Pkg.add(url="https://github.com/garam-kim/FiniteKernel.jl/")
```



