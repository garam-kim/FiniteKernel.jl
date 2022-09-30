using FrankWolfe
using Plots
using LinearAlgebra
using Random
using FiniteKernel
include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))

# ## Finite-dimensional kernel herding
# We focus on a specific kernel studied in ([Bach et al.](https://icml.cc/2012/papers/683.pdf)), that has a finite dimensional feature space.
# Let $\mathcal{Y} = \{-1, 1\}^d $ and
# ```math
# \mathcal{H}:= \left\lbrace f \colon \mathcal{Y} \to \mathbb{R} \mid f(y) = \langle f, \Phi(y) \rangle_\mathcal{H}, \text{ where} \Phi(y)=(y, yy^T) \right\rbrace.
# ```
# The feature map is composed of $y$ and of all of its pairwise products $yy^T$. i.e., $\Phi(y) = (y, yy^T) = (y_1, y_2, y_1^2, y_1y_2, y_2y_1, y_2^2)$ for $y = (y_1, y_2) \in \{-1, 1\}^2$.
# For $w, x \in \mathcal{H}$,
# ```math
# \langle w, x \rangle_\mathcal{H} := \sum_{i = 1}^{d} w_i(y)x_i(y)
# ```
# defines inner product and $(\mathcal{H}, \langle \cdot, \cdot \rangle_{\mathcal{H}})$ is a Hilbert space. 
# Moreover, the Hilbert space $\mathcal{H}$ is also a Reproducing Kernel Hilbert Space (RKHS) having the reproducing kernel
# ```math
# k(y, z) = \langle k(y, x), k(z, x) \rangle_\mathcal{H}
# ```
# for all $x, y, z \in \mathcal{Y}$.

# In this set-up, we compute the expectation 
# ```math
# \mu(z) : = \mathbb{E}_{p(y)} \Phi(y)(z) = \sum_{i=1}^{2^d} p(y_i)\Phi(y_i)(z) \in \mathcal{C}.
# ```
# such that $\sum_{i=1}^{2^d}p(y_i) = 1$.



# ### Set-up

# Below, we consider the case d = 2.
dim = 2

# We compare different Frank-Wolfe algorithms for kernel herding in the Hilbert space $\mathcal{H}$:
# the Frank-Wolfe algorithm with open loop step-size rule $\eta_t = \frac{2}{t+2}$ (FW-OL),and the Frank-Wolfe algorithm with short-step (FW-SS).
# The LMO in the here-presented kernel herding problem is implemented searching over $\mathcal{Y} = \{-1, 1\}^d$.

max_iterations = 2000
max_iterations_lmo = 2^dim
lmo = MarginalPolytope(max_iterations_lmo)


# ### Uniform distribution
# First, we consider the uniform distribution $p = 1$, which results in the mean element being zero, that is, $\mu = 0$.

mu = ZeroMeanElement()
iterate = KernelHerdingIterate([1.0], [ones(Float64, dim)])
gradient = KernelHerdingGradient(iterate, mu)
f, grad = create_loss_function_gradient(mu)

FW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
FW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(2), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)

# We plot the result
data = [FW_OL[end], FW_SS[end]]
labels = ["FW-OL", "FW-SS"]
plot_trajectories(data, labels, xscalelog=true)

# Observe that FW-SS converges linearly and convergence rate of $\mathcal{O}(1/t^2)$ for FW-OL.


# ### Non-uniform distribution
# Second, we consider a non-uniform distribution $p(y)$ where 
# ```math
# \sum_{i=1}^{2^d}p(y_i) = 1.
# ```
# Hence, we start with an arbitrary vectors:
rho = rand(Float64, 2^dim)
# We then normalize the vectors to obtain a $p$ that is indeed a distribution.
p = get_distribution(rho)

# We then run the experiments.
mu = NonZeroMeanElement(p)
iterate = KernelHerdingIterate([1.0], [ones(Float64, dim)])
gradient = KernelHerdingGradient(iterate, mu)
f, grad = create_loss_function_gradient(mu)

FW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
FW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(2), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)

# We plot the result
data = [FW_OL[end], FW_SS[end]]
labels = ["FW-OL", "FW-SS"]
plot_trajectories(data, labels, xscalelog=true)


# ## Conclusion

# We presented two experiments which show how to use Frank-Wolfe algorithms to solve optimization problems in finite-dimensional Hilbert spaces.

# ## References

# Bach, F., Lacoste-Julien, S. and Obozinski, G., 2012, June. On the Equivalence between Herding and Conditional Gradient Algorithms. [In ICML 2012 International Conference on Machine Learning.](https://icml.cc/2012/papers/683.pdf)

# Wirth, E., Kerdreux, T. and Pokutta, S., 2022. Acceleration of Frank-Wolfe algorithms with open loop step-sizes. [arXiv preprint arXiv:2205.12838.](https://arxiv.org/pdf/2205.12838.pdf)
