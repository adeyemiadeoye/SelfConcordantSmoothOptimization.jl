# SelfConcordantSmoothOptimization.jl

- [SelfConcordantSmoothOptimization.jl](#selfconcordantsmoothoptimizationjl)
  - [Installation](#installation)
  - [Usage example](#usage-example)
      - [A simple sparse logistic regression problem](#a-simple-sparse-logistic-regression-problem)
  - [Implementation details and recommendations](#implementation-details-and-recommendations)
  - [Citing](#citing)
  - [Contributing](#contributing)

`SelfConcordantSmoothOptimization.jl` is a Julia package that implements the self-concordant regularization (SCORE) technique for (nonsmooth) convex optimization. In particular, `SelfConcordantSmoothOptimization.jl` considers problems of the form

<p align="center">
minimize f(x) + g(x)
</p>

where $\mathrm{f}\colon \mathbb{R}^n \to \mathbb{R}$ is smooth and convex, and $\mathrm{g}\colon \mathbb{R}^n \to \mathbb{R}$, which may be nonsmooth, is proper, closed and convex. The smooth part $\mathrm{f}$ defines the problem's objective function, such as quantifying a data-misfit, while the nonsmooth part $\mathrm{g}$ imposes certain properties, such as sparsity, on the decision variable $\mathrm{x}$. Please see [Implementation details and recommendations](#implementation-details-and-recommendations) for functions that are currently supported for each implemented algorithm.

## Installation
Install the package via Julia's `REPL` with
```
julia> ]
pkg> add SelfConcordantSmoothOptimization
```

## Usage example
#### A simple sparse logistic regression problem
```julia
# Load the package
using SelfConcordantSmoothOptimization

using Random, Distributions, SparseArrays

# Generate a random data
Random.seed!(1234)
n, m = 100, 50;
A = Matrix(sprandn(n, m, 0.01));
y_prob = 1 ./ (1 .+ exp.(-A * zeros(m)));
y = rand.(Bernoulli.(y_prob));
unique_y = unique(y); 
y = map(x -> x==unique_y[1] ? -1 : 1, y);
x0 = randn(m);

# Define objective function and choose problem parameters
f(x) = 1/m*sum(log.(1 .+ exp.(-y .* (A*x))));

# choose problem parameters
reg_name = "l1";
λ = 0.4;
μ = 1.0;
hμ = PHuberSmootherL1L2(μ);

# set problem
model = Problem(A, y, x0, f, λ);

# Choose method and run the solver
method = ProxNSCORE();
solution = iterate!(method, model, reg_name, hμ; max_iter=100, x_tol=1e-6, f_tol=1e-6);
# get the solution x
solution.x
```
To use the `ProxGGNSCORE` algorithm, a model output function $\mathcal{M}(A,x)$ is required
(Note that it is essential to define the function f in two ways to use `ProxGGNSCORE`; Julia will decide which one to use at any instance -- thanks to the multiple dispatch feature):
```julia
# f as defined above
f(x) = 1/m*sum(log.(1 .+ exp.(-y .* (A*x))));
# f as a function of y and yhat
f(y, yhat) = -1/m*sum(y .* log.(yhat) .+ (1 .- y) .* log.(1 .- yhat))
# where yhat = Mfunc(A, x) is defined by the model output function
Mfunc(A, x) = 1 ./ (1 .+ exp.(-A*x))
# set problem
model = Problem(A, y, x0, f, λ; out_fn=Mfunc);

# Choose method and run the solver
method = ProxGGNSCORE();
solution = iterate!(method, model, reg_name, hμ; max_iter=100, x_tol=1e-6, f_tol=1e-6);
# get the solution x
solution.x
```
By default, this package computes derivatives using [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl). But users can supply functions that compute the derivates involved in the algorithms. This has at least two benefits:
1. Faster computations
2. Avoid the need to define $\mathrm{f}$ twice when using `ProxGGNSCORE`

In the example above:
```julia
f(x) = 1/m*sum(log.(1 .+ exp.(-y .* (A*x))));

S(x) = exp.(-y .* (A*x));
# gradient of f wrt x:
function grad_fx(x)
    Sx = S(x)
    return -A' * (y .* (Sx ./ (1 .+ Sx))) / m
end;
# Hessian of f wrt x:
function hess_fx(x)
    Sx = 1 ./ (1 .+ S(x))
    W = Diagonal(Sx .* (1 .- Sx))
    hess = A' * W * A
    return hess / m
end;

# The following are used by ProxGGNSCORE
# Jacobian of yhat wrt x:
jac_yx(yhat) = vec(yhat .* (1 .- yhat)) .* A;
# gradient of \ell wrt yhat:
grad_fy(yhat) = (-y ./ yhat .+ (1 .- y) ./ (1 .- yhat))/m;
# Hessian of \ell wrt yhat:
hess_fy(yhat) = Diagonal((y ./ yhat.^2 + (1 .- y) ./ (1 .- yhat).^2)/m);
```
```julia
# Now (for ProxNSCORE):
model_n = Problem(A, y, x0, f, λ; grad_fx=grad_fx, hess_fx=hess_fx);
method_n = ProxNSCORE();
sol_n = iterate!(method_n, model_n, reg_name, hμ; max_iter=100, x_tol=1e-6, f_tol=1e-6);
sol_n.x
```
```julia
# And for ProxGGNSCORE (does not require hess_fx):
model_ggn = Problem(A, y, x0, f, λ; out_fn=Mfunc, grad_fx=grad_fx, jac_yx=jac_yx, grad_fy=grad_fy, hess_fy=hess_fy);
method_ggn = ProxGGNSCORE();
sol_ggn = iterate!(method_ggn, model_ggn, reg_name, hμ; max_iter=100, x_tol=1e-6, f_tol=1e-6);
sol_ggn.x
```
TODO: Add a simple sparse group lasso example... (for now, see example from paper in `/examples/paper`).

## Implementation details and recommendations
Below is a summary of functions $\mathrm{f}$ supported by the algorithms implemented in the package:

| Algorithm      	| Supported $\mathrm{f}$                                                                                                                                                                                                                                 |
|----------------	|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| `ProxNSCORE`   	| <li>Any twice continuously differentiable function.</li>                                                                                                                                                                                        		|
| `ProxGGNSCORE` 	| <li>Any twice continuously differentiable function that can be expressed in the form $f(x) =  \sum\limits_{i=1}^{m}\ell(y_i,\hat{y}_i)$ where $\ell$ is a loss function that measures a data-misfit.</li> <li>Requires a model $\mathcal{M}(A,x)$ that computes the predictions $\hat{y}_i$.</li> 		|
| `ProxQNSCORE`  	| <li>Any twice continuously differentiable function (currently not recommended).</li> <li>Currently not recommended.</li>                                                                                                                                                                                       		|


As the package name and description imply, the implemented algorithms use a generalized self-concordant smooth approximation $\mathrm{g_s}$ of $\mathrm{g}$ in their procedures. The algorithms do this for specific regularization functions that are specified by `reg_name` that takes a string value in the `iterate!` function. We summarize below currently implemented regularization functions, as well as the corresponding smoothing functions $\mathrm{g_s}$.

| `reg_name` value 	| Implemented $\mathrm{g_s}$ function(s)                                                                                                                                              	| Remark(s)                                                                                           		|
|------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------------------------------------------------------------------------------------------------	|
| `"l1"`           	| <li>`PHuberSmootherL1L2(μ)`</li> <li>`OsBaSmootherL1L2(μ)`</li>	| $\mathrm{\mu}>0$                                                                                             	|
| `"l2"`           	| <li>`PHuberSmootherL1L2(μ)`</li> <li>`OsBaSmootherL1L2(μ)`</li>                                                                       	| $\mathrm{\mu}>0$                                                                                             	|
| `"gl"`       	| <li>`PHuberSmootherGL(μ, model)`</li>                                                                          	| For sparse group lasso regularizer <br> $\mathrm{\mu}>0$ 	|
| `"indbox"`       	| <li>`PHuberSmootherIndBox(lb,ub,μ)`</li>                                                                          	| `lb`: lower bound in the box constraints <br> `ub`: upper bound in the box constraints <br> $\mathrm{\mu}>0$ 	|

[!NOTE]<br>`PHuberSmootherL1L2`, `PHuberSmootherGL`, `PHuberSmootherIndBox` are highly recommended.

For more details and insights on the approach implemented in this package, please see the associated paper in [Citing](#citing) below.

## Citing
If you use `SelfConcordantSmoothOptimization.jl` in your work, particularly the algorithms listed above, we kindly request that you cite the following paper:
```
@article{adeoye2023self,
  title={Self-concordant Smoothing for Convex Composite Optimization},
  author={Adeoye, Adeyemi D and Bemporad, Alberto},
  journal={arXiv preprint arXiv:2309.01781},
  year={2023}
}
```

## Contributing
Please use the [Github issue tracker](https://github.com/adeyemiadeoye/SelfConcordantSmoothOptimization.jl/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) for reporting any issues. All types of issues are welcome including bug reports, feature requests, implementation for a specific research problem, etc.
