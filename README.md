# FiniteKernel.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://garam-kim.github.io/FiniteKernel.jl/dev/)


Repository for finite-dimensional kernel herding using [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) package. We focus on a specific kernel setting in Section 5.3 of [Bach et al.](https://icml.cc/2012/papers/683.pdf), that has a finite dimensional feature space $\Phi(x)=(x,xx^T)$ for $x \in [-1,1]^d$. 

For Matern Kernel in infinite-dimensional setting, move to [MaternKernel.jl](https://github.com/garam-kim/MaternKernel.jl).


## Installation

The most recent release is available via:

```julia
Pkg.add(url="https://github.com/garam-kim/FiniteKernel.jl/")
```
