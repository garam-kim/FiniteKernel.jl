module FiniteKernel
using FrankWolfe
using LinearAlgebra

include("iterates.jl")
export KernelHerdingIterate, KernelHerdingGradient, MarginalPolytopeWahba, create_loss_function_gradient, ZeroMeanElement, compute_mu
end