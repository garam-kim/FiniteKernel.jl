module FiniteKernel
using FrankWolfe
using LinearAlgebra

include("iterates.jl")
export KernelHerdingIterate, KernelHerdingGradient, MarginalPolytope, create_loss_function_gradient, ZeroMeanElement, NonZeroMeanElement, get_distribution
end