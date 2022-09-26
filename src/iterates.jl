"""
Kernel herding iterate.
"""
struct KernelHerdingIterate{T}
    weights::Vector{T}
    vertices ::Vector{Vector{T}}
end

"""
MeanElement μ must implement dot with a functional.
"""
abstract type MeanElement
end

# # Different types of MeanElements
"""
μ = 0.
"""
struct ZeroMeanElement <: MeanElement
end

"""
μ =/= 0.
"""
struct NonZeroMeanElement{T} <: MeanElement
    p::Vector{T}
end

"""
The gradient < x - μ, . > is represented by x and μ.
"""
mutable struct KernelHerdingGradient{T, D <: MeanElement}
    x:: KernelHerdingIterate{T}
    mu:: D
end

"""
The marginal polytope of the Wahba kernel.
"""
struct MarginalPolytope <: FrankWolfe.LinearMinimizationOracle
    number_iterations:: Int
end




"""
Basic arithmetic for KernelHerdingIterate.
"""
LinearAlgebra.dot(y, x::KernelHerdingIterate) = LinearAlgebra.dot(x, y)

# # Addition 
function Base.:+(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    x = copy(x1)
    for (idx2, p2) in enumerate(x2.vertices)
        found = false
        for (idx, p) in enumerate(x.vertices)
            if p == p2 
                x.weights[idx] += x2.weights[idx2]
                found = true
            end
        end
        if !found
            push!(x.weights, x2.weights[idx2])
            push!(x.vertices, x2.vertices[idx2])
        end
    end
    return x
end

# # Multiplication
function Base.:*(x::KernelHerdingIterate, scalar::Real)
    w = copy(x)
    w.weights .*= scalar
    return w
end

# # Multiplication, different order.
function Base.:*(scalar::Real, x::KernelHerdingIterate)
    w = copy(x)
    w.weights .*= scalar
    return w
end

# # Subtraction
function Base.:-(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    return x1 + (-1) * x2
end


# # Evaluates the Kernel over the space.
function kernel_evaluation(y1, y2)
    return dot(y1,y2)
end

"""
Scalar product for two KernelHerdingIterates.
"""
function LinearAlgebra.dot(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    w_1 = x1.weights
    w_2 = x2.weights
    v_1 = x1.vertices
    v_2 = x2.vertices
    w_matrix = w_1 * w_2'
    p_matrix = [kernel_evaluation(v1, v2) for v1 in v_2 for v2 in v_1]
    scalar_product = dot(w_matrix, p_matrix)
    return scalar_product
end

"""
Scalar product for KernelHerdingIterate with KernelHerdingGradient.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate, g::KernelHerdingGradient)
    scalar_product = dot(g.x, x)
    scalar_product -= dot(g.mu, x)
    return scalar_product
end

"""
Scalar product for KernelHerdingIterate with ZeroMeanElement.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate{T}, mu::ZeroMeanElement) where T
    return zero(T)
end

"""
Scalar product for KernelHerdingIterate with NonZeroMeanElement.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate, mu::NonZeroMeanElement)
    w = x.weights
    v = x.vertices
    p = mu.p
    dim = length(v[1])
    w_matrix = w * p'
    p_matrix = [kernel_evaluation(v1, v2) for v1 in region_vertices(dim) for v2 in v]
    scalar_product = dot(w_matrix, p_matrix)
    return scalar_product
end



"""
Norm of ZeroMeanElement.
"""
function LinearAlgebra.norm(mu::ZeroMeanElement)
    return zero(Float64)
end

"""
Norm of NonZeroMeanElement.
"""
function LinearAlgebra.norm(mu::NonZeroMeanElement)
    p = mu.p
    dim = Int(log2(length(p)))
    v =  region_vertices(dim)
    w_matrix = p * p'
    p_matrix = [kernel_evaluation(v1, v2) for v1 in v for v2 in v]
    scalar_product = dot(w_matrix, p_matrix)
    return sqrt(scalar_product)
end



# # Technical replacements for the kernel herding setting

function Base.similar(x::KernelHerdingIterate, ::Type{T}) where T
    return KernelHerdingIterate([T(1.0)], [copy(x.vertices[1])])
end

function Base.similar(x::KernelHerdingIterate{T}) where T
    return Base.similar(x, T)
end

function Base.eltype(x::KernelHerdingIterate{T}) where T
    return T
end

function Base.copy(x::KernelHerdingIterate{T}) where T
    return KernelHerdingIterate(copy(x.weights), copy(x.vertices))
end

function Base.copy!(x::KernelHerdingIterate{T}, y::KernelHerdingIterate{T}) where T
    copy!(x.weights, y.weights)
    copy!(x.vertices, y.vertices)
end



# Gradient, loss, and extreme point computations

"""
Creates the loss function and the gradient function, given a MeanElement μ, that is,
    1/2 || x - μ ||_H²      and     < x - μ, . >,
respectively.
"""
function create_loss_function_gradient(mu::MeanElement)
    
    mu_squared = norm(mu)^2
    function evaluate_loss(x::KernelHerdingIterate)
        l = dot(x, x)
        l += mu_squared
        l -= 2 * dot(x, mu)
        l /= 2
        return l
    end
    function evaluate_gradient(g::KernelHerdingGradient, x::KernelHerdingIterate)
        g.x = x
        g.mu = mu
    return g
    end
    return evaluate_loss, evaluate_gradient
end


"""
Constructs the vertices of the observation space.
"""
function region_vertices(dim)
    if dim == 1
        vertices = [[x] for x in [1.0, -1.0]]

    elseif dim == 2
        vertices = [[x,y] for x in [1.0, -1.0] for y in [1.0, -1.0]]

    elseif dim == 3
        vertices = [[x,y,z] for x in [1.0, -1.0] for y in [1.0, -1.0] for z in [1.0, -1.0]]  
   
    else dim == 4
        vertices = [[x,y,z,w] for x in [1.0, -1.0] for y in [1.0, -1.0] for z in [1.0, -1.0] for w in [1.0, -1.0]]  
    end
    return vertices
end


"""
Computes the extreme point in the Frank-Wolfe algorithm for kernel herding.
"""
function FrankWolfe.compute_extreme_point(lmo::MarginalPolytope, direction::KernelHerdingGradient; kw...)
    optimal_value = Inf
    optimal_vertex = nothing
    current_vertex = nothing
    dim = length(direction.x.vertices[1])
    candidates = region_vertices(dim)

    for (idx, vertex) in enumerate(candidates)
        current_vertex = KernelHerdingIterate([1.0], [vertex])
        current_value = dot(direction, current_vertex)
        if current_value < optimal_value
            optimal_vertex = current_vertex
            optimal_value = current_value
        end
    end
    @assert(optimal_vertex !== nothing, "This should never happen.")
    return optimal_vertex
end


"""
Returns valid distribution of the corresponding dimension.
"""
function get_distribution(p)
    distribution = p / sum(p)
    @assert(sum(distribution) ≈ 1, "The vector p still does not correspond to a distribution.")
    return distribution
end
