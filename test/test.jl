using FiniteKernel
using Test

# Elementary operations 

@testset "Addition" begin
    x = KernelHerdingIterate([0.1, 0.7], [[0.3, 0.32], [-0.3, 0.35]])
    y = x + x
    @test y.weights == [0.2, 1.4]
    @test y.vertices == [[0.3, 0.32], [-0.3, 0.35]]

    w = KernelHerdingIterate([1.0], [[-0.3, 0.35]])
    z = w + x
    @test z.weights == [1.7, 0.1]
    @test z.vertices == [[-0.3, 0.35], [0.3, 0.32]]
end

@testset "Scalar multiplication" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [[0.3, 0.32], [-0.9, 0.7], [1.0, -0.14]])
    y = x * 0.5
    @test y.weights == [0.05, 0.1, 0.35]
end

@testset "Subtraction" begin
    x = KernelHerdingIterate([0.1, 0.9], [[0.3, 0.35], [0.2, 0.8]])    
    y = KernelHerdingIterate([0.2], [[0.3, 0.0]])
    z = x - y
    @test z.weights == [0.1, 0.9, -0.2]
    @test z.vertices == [[0.3, 0.35], [0.2, 0.8], [0.3, 0.0]]

    w = z - x
    @test w.weights == [0.0, 0.0, -0.2]
    @test w.vertices == [[0.3, 0.35], [0.2, 0.8], [0.3, 0.0]]
end


# Basic linear algebra

@testset "dot with itself" begin
    x = KernelHerdingIterate([0.5, 0.5], [[0.0, 0.5], [0.2, 0.8]])
    y = dot(x, x)
    @assert y ≈ 0.43250000000000005

    x = KernelHerdingIterate([1.0], [[0.0, 0.2]])
    y = dot(x, x)
    @assert y ≈ 0.04

    x = KernelHerdingIterate([0.5, 0.5], [[0.0, 0.5], [0.2, 0.8]])
    y = KernelHerdingIterate([0.15, 0.2, 0.8], [[0.1, 1.0], [-0.3, 0.35], [-0.5, 0.5]])
    z = dot(x,y)
    @assert z ≈ 0.35850000000000004
end

@testset "dot with gradient" begin
    # Gradient with ZeroMeanElement
    x = KernelHerdingIterate([0.2], [[0.1, 0.0]])
    y = KernelHerdingGradient(x, ZeroMeanElement()) 
    z = dot(x, y)
    @assert z ≈ 0.0004000000000000002

    w = KernelHerdingIterate([0.2, 0.7], [[0.1, 0.0], [-0.5, -0.5]])
    y = KernelHerdingGradient(x, ZeroMeanElement())
    z = dot(w, y)
    @assert z ≈ -0.006600000000000001

    # Gradient with NonZeroMeanElement
    x = KernelHerdingIterate([0.2], [[0.1, -0.9]])
    w = KernelHerdingIterate([0.15, 0.8], [[0.1, 1.0], [-0.5, 0.5]])
    mu = NonZeroMeanElement([0.1, 0.2, 0.3, 0.4])
    y = KernelHerdingGradient(x, mu) 
    z = dot(w, y)
    @assert z ≈ -0.1507
end