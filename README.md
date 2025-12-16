# SelfConcordantSmoothOptimization.jl

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2309.01781" alt="arXiv" target="_blank"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License" target="_blank"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

<p align="center">
    <img width=70% src="media/prox-SCORE.svg">
</p>

- [SelfConcordantSmoothOptimization.jl](#selfconcordantsmoothoptimizationjl)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Usage examples](#usage-examples)
      - [Sparse logistic regression](#sparse-logistic-regression)
      - [Sparse-group lasso example](#sparse-group-lasso-example)
      - [Box-constrained example](#box-constrained-example)
  - [Additional details](#additional-details)
    - [Some optional arguments](#some-optional-arguments)
  - [Citing](#citing)
  - [Acknowledgements](#acknowledgements)
  - [Issues and bug reports](#issues-and-bug-reports)

`SelfConcordantSmoothOptimization.jl` is a Julia package that implements the self-concordant regularization (SCORE) technique for solving (nonsmooth) convex optimization with quasi-Newton directions. In particular, `SelfConcordantSmoothOptimization.jl` considers problems of the form

<p align="center">
minimize f(x) + g(x)
</p>

where $\mathrm{f}\colon \mathbb{R}^n \to \mathbb{R}$ is smooth and convex, and $\mathrm{g}\colon \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$, which may be nonsmooth, is proper, closed and convex. The smooth part $\mathrm{f}$ defines a loss function, such as quantifying a data-misfit, while the nonsmooth part $\mathrm{g}$ can be used to impose structures, such as sparsity, on the decision variable $\mathrm{x}$. Please see [Additional details](#additional-details).


Most parts of this package is also available in python/JAX at [PySCSOpt](https://github.com/adeyemiadeoye/PySCSOpt).

## Installation
Install the package via Julia's `REPL` with (dependencies can be found in `Project.toml`)
```julia
using Pkg
Pkg.add("SelfConcordantSmoothOptimization")
```

## Quick start
```julia
# (regularized) rosenbrock
using SelfConcordantSmoothOptimization

f(x) = 100.0*(x[2] - x[1]^2)^2 + (1.0 - x[1])^2

using Random
Random.seed!(1234)
x0 = randn(2)

reg_name = "l1"             # l1-regularization (g(x) = λ‖x‖_1)
λ = 1e-8                    # l1-regularization scaling value (for illustration)
μ = 1.0                     # smoothing/regularization parameter
hμ = PHuberSmootherL1L2(μ)  # smoothing/regularization function

problem = Problem(x0, f, λ)

# use the limited-memory BFGS-SCORE method
method = ProxLQNSCORE(use_prox=true, m=10) # m is the memory size for L-BFGS approximation
solution = iterate!(method, problem, reg_name, hμ; verbose=2)

# get returned solution
solution.x
```
A gradient function for $\mathrm{f}$ can also be provided as follows:
```julia
function grad_fx(x)
    ... # compute and return gradient of f at x
end
problem = Problem(x0, f, λ; grad_fx=grad_fx)
```
See [Usage examples](#usage-examples) for more details.

`solution` has the following fields:
```text
solution.x             The solution vector
solution.obj           History of f + g value
solution.fval          History of f value
solution.pri_res_norm  History of primal residual norm
solution.fvaltest      History of f values on test set
solution.rel           History of relative errors
solution.objrel        History of objective relative errors
solution.metricvals    History of metric values
solution.times         History of times in seconds (last entry is total time)
solution.epochs        Number of epochs/iterations run
solution.model         The problem model used
```

## Usage examples

#### Sparse logistic regression
```julia
# Load the package
using SelfConcordantSmoothOptimization

using Random, Distributions, SparseArrays

# Generate a random data
Random.seed!(1234)
n, m = 100, 50
A = Matrix(sprandn(n, m, 0.01))
y_prob = 1 ./ (1 .+ exp.(-A * zeros(m)))
y = rand.(Bernoulli.(y_prob))
unique_y = unique(y)
y = map(x -> x==unique_y[1] ? -1 : 1, y)
x0 = randn(m)

# Since data is involved, define f as a function of A, y and x
f(A, y, x) = 1/m*sum(log.(1 .+ exp.(-y .* (A*x))))

reg_name = "l1"   # regularization function name
λ = 1e-1          # regularization scale, can be adjusted
μ = 1.0           # smoothing parameter, can be adjusted
hμ = PHuberSmootherL1L2(μ) # smoothing function

# set problem
problem = Problem(A, y, x0, f, λ);

# Choose method and run the solver (see ProxGGNSCORE below)
method = ProxNSCORE();
solution = iterate!(method, problem, reg_name, hμ; max_epoch=100, x_tol=1e-6, f_tol=1e-6);
# get the solution x
solution.x
```
`max_epoch` refers to the maximum number of iterations. The use of "epoch" here is merely for consistency with the terminology used in "supervised learning".

To use the `ProxGGNSCORE` algorithm, a model output function $\mathcal{M}(A,x)$ is required
(Note that it is essential to define the function f in two ways to use `ProxGGNSCORE`; Julia will decide which one to use at any instance -- thanks to the multiple dispatch feature):
```julia
# f as defined above
f(A, y, x) = 1/m*sum(log.(1 .+ exp.(-y .* (A*x))));
# f as a function of y and yhat
f(y, yhat) = -1/m*sum(y .* log.(yhat) .+ (1 .- y) .* log.(1 .- yhat))
# where yhat = Mfunc(A, x) is defined by the model output function
Mfunc(A, x) = vcat([1 ./ (1 .+ exp.(-A*x))]...)
# set problem
problem = Problem(A, y, x0, f, λ; out_fn=Mfunc);

# Choose method and run the solver
method = ProxGGNSCORE();
solution = iterate!(method, problem, reg_name, hμ; max_epoch=100, x_tol=1e-6, f_tol=1e-6);
# get the solution x
solution.x
```
By default, this package computes derivatives using [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl). But users can supply functions that compute the derivates involved in the algorithms. This has at least the benefit of not having to define $\mathrm{f}$ twice when using `ProxGGNSCORE`.

In the above example, this can be done as follows:
```julia
problem_ggn = Problem(A, y, x0, f, λ; out_fn=Mfunc, grad_fx=grad_fx, jac_yx=jac_yx, grad_fy=grad_fy, hess_fy=hess_fy)
```
where `grad_fx`, `jac_yx`, `grad_fy` and `hess_fy` are defined as:
```julia
function grad_fx(A, y, x)
    ... # compute and return gradient of f w.r.t x
end

function jac_yx(A, y, yhat, x)
    ... # compute and return Jacobian of model output function Mfunc w.r.t x
end

function grad_fy(A, y, yhat)
    ... # compute and return gradient of f w.r.t yhat
end

function hess_fy(A, y, yhat)
    ... # compute and return Hessian of f w.r.t yhat
end
```
For `ProxLQNSCORE` and `ProxNSCORE`,
```julia
problem_lqn = Problem(A, y, x0, f, λ; grad_fx=grad_fx)
problem_n = Problem(A, y, x0, f, λ; grad_fx=grad_fx, hess_fx=hess_fx)
```
respectively, where `grad_fx` and `hess_fx` are defined as:
```julia
function grad_fx(A, y, x)
    ... # compute and return gradient of f w.r.t x
end

function hess_fx(A, y, x)
    ... # compute and return Hessian of f w.r.t x
end
```

#### Sparse-group lasso example
We generate some random data using the `make_group_lasso_problem` utility function available in the python port. We require the `PythonCall.jl` package for this.
```julia
using PythonCall; pyscs = pyimport("pyscsopt")
using SelfConcordantSmoothOptimization
using Random

m = 50
n = 100
grpsize = 10
p_active = 0.1  # 10% of groups/features active
A, y, x_true, x0, groups, ind, P = pyscs.utils.make_group_lasso_problem(
                                          m=m, n=n, grpsize=grpsize,
                                          p_active=p_active, noise_std=0.1, seed=1234,
                                          group_weights=1.0, use_const_grpsize=true, corr=0.5
                                          )
ind = pyconvert(Matrix,ind)
ind[1:2,:] .+= 1
G = pyconvert(Vector, P.G) .+ 1
P = get_P(n, G, ind)
A = pyconvert(Matrix, A)
y, x0, x_true = pyconvert.(Vector, (y, x0, x_true));

function f(A, y, x)
    return 0.5*sum((A*x .- y).^2)/m
end

λ1 = 1e-8   # l1
λ2 = 1      # group lasso
λ = [λ1, λ2]
μ = 1e-2

Random.seed!(1234)
x0 = randn(n)
reg_name = "gl"

problem = Problem(A, y, x0, f, λ; P=P)
hμ = PHuberSmootherGL(μ, problem) # group lasso smoother takes problem

method_lqn = ProxLQNSCORE(use_prox=true, ss_type=1, m=10)
sol_lqn = iterate!(method_lqn, problem, reg_name, hμ; verbose=2, max_epoch=100, α=1)
```
Note that for `ProxGGNSCORE`, we would also need to define
```julia
function f(y, ŷ)
    return 0.5*sum((ŷ .- y).^2)/m
end

function out_fn(A, x)
    return A*x
end
```
and define `problem` as
```julia
problem = Problem(A, y, x0, f, λ; out_fn=out_fn, P=P)
```

#### Box-constrained example
```julia
using SelfConcordantSmoothOptimization
using LinearAlgebra
using Random

nvar = 10
Random.seed!(1234)
Q = randn(nvar, nvar)
Q = LowerTriangular(Q)
Q = Q + Q' - Diagonal(diag(Q))
Q = Q + nvar * I(nvar)
c = ones(nvar)

f(x) = 0.5 * dot(x, Q * x) + dot(c, x)

x0 = randn(nvar)
λ = 1
reg_name = "indbox"
C_set = (-0.5, Inf)
μ = 1e1
hμ = LogExpSmootherIndBox(C_set[1], C_set[2], μ)
problem = Problem(x0, f, λ; C_set=C_set)

method = ProxLQNSCORE(use_prox=true, ss_type=1, m=10)
solution = iterate!(method, problem, reg_name, hμ; verbose=2)
```

## Additional details
Some details about the implemented algorithms and available options are provided below.

| Algorithm      	| Remark(s)                                                                                                                                                                                                                                 |
|----------------	|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| `ProxLQNSCORE`  	| <li>Supports any continuously differentiable function $\mathrm{f}$ (with respect to $\mathrm{x}$).</li>                                                                                                                                                                                       		|
| `ProxGGNSCORE` 	| <li>Supports $\mathrm{f}$ that can be expressed in the form $f(x) =  \sum\limits_{i=1}^{m}\ell(y_i,\hat{y}_i)$ where $\ell$ measures a data-misfit.</li> <li>Requires a model $\mathcal{M}(A,x)$ that computes the predictions $\hat{y}_i$.</li> <li> All components must be adequately differentiable. </li> 	|
| `ProxNSCORE`   	| <li>Supports twice continuously differentiable function $\mathrm{f}$.</li>                                                                                                                                                                                        		|


### Some optional arguments
| Arg      	| Description & usage                                                                                                                                                                                                                                 |
|----------------	|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| `ss_type`   	| <li>Value `1`: If `L` (Lipschitz constant of f) is set in `Problem`, this will set $\mathrm{\alpha}=$ `min{1/L, 1}`. If `L` is not set, $\mathrm{\alpha}$ takes the value set in `iterate!`. Otherwise, it takes a default value $0.5$.</li> <li>Value `2` uses the "inverse" of Barzilai-Borwein method to set $\mathrm{\alpha}$.</li> <li>Value `3` uses a line-search method to choose $\mathrm{\alpha}$.</li> <li>Default value: `1`.</li> <li>Example: `method = ProxLQNSCORE(ss_type=1)`</li> <li>NOTE: $\mathrm{\alpha}$ is not the step size. It uses a value in $(0,1]$ and for scaling the effective step size.</li>                                                                                                                                                                                      		|
| `use_prox` 	| <li>Value `true` uses the proximal scheme as described in the paper.</li> <li>Value `false` skips the proximal step and takes only the associated quasi-Newton step.</li> <li>Default value: `true`.</li> <li>Example: `method = ProxLQNSCORE(use_prox=true)`</li>                                                                                                                                                                                       		|
| `m`        	| <li>Memory size for the limited-memory quasi-Newton method `ProxLQNSCORE`.</li> <li>Default value: `10`.</li> <li>Example: `method = ProxLQNSCORE(m=20)`</li>                                                                                                                                                                                                                                                       		|

The implemented algorithms use a generalized self-concordant smooth approximation $\mathrm{g_s}$ of $\mathrm{g}$ in their procedures. The algorithms do this for specific regularization functions that are specified by `reg_name` that takes a string value in `iterate!`. The smoothing is done using a function $\mathrm{h_\mu}$. We summarize below currently implemented $\mathrm{h_\mu}$ functions for each available `reg_name`.

| `reg_name` 	| Implemented $h_\mu$ function(s)                                                                                                                                              	| Remark(s)                                                                                           		|
|------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------------------------------------------------------------------------------------------------	|
| `"l1"`           	| <li>`PHuberSmootherL1L2(μ)`</li> <li>`OsBaSmootherL1L2(μ)`</li>	| <li>$\mathrm{\mu}>0$</li> <li> $\mathrm{\lambda}$ defined in `Problem` is a scalar                                                                                             	|
| `"l2"`           	| <li>`PHuberSmootherL1L2(μ)`</li> <li>`OsBaSmootherL1L2(μ)`</li>                                                                       	| <li>$\mathrm{\mu}>0$</li> <li> $\mathrm{\lambda}$ defined in `Problem` is a scalar                                                                                             	|
| `"gl"`       	| <li>`PHuberSmootherGL(μ, problem)`</li>                                                                          	| <li> For sparse group lasso regularizer </li> <li>$\mathrm{\mu}>0$ and `problem` is an instance of `Problem`</li> <li> $\mathrm{\lambda}$ defined in `Problem` can be given as `[lambda1, lambda2]` or `(lambda1, lambda2)`.	|
| `"indbox"`       	| <li>`LogExpSmootherIndBox(lb,ub,μ)`</li><li>`ExponentialSmootherIndBox(lb,ub,μ)`</li> <li>`PHuberSmootherIndBox(lb,ub,μ)`</li>                                                                          	| `lb`: lower bound on the optimization variable <br> `ub`: upper bound the optimization variable <br> $\mu > 0$ <br> `lb` and `ub` can either be scalars or vectors of the same length as the optimization variable <br> Users are also expected to set `C_set` in `Problem`, where `C_set` can be given as `[lb, ub]`, `(lb, ub)`, or defined via the `IntervalSets.jl` package.	|

## Citing
This software was coded as part of the research presented in the following paper. It is distributed without any warranties or guarantees. If you use this package, please cite the paper.
```
@article{AdeBem2025,
      title={Self-concordant smoothing in proximal quasi-Newton algorithms for large-scale convex composite optimization}, 
      author={Adeyemi D. Adeoye and Alberto Bemporad},
      year={2025},
      eprint={2309.01781},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2309.01781}, 
}
```

## Acknowledgements
The authors acknowledge the funding received from the European Union (ERC Advanced Research Grant COMPACT, No. 101141351). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union
or the European Research Council. Neither the European Union nor the granting authority can be held responsible for
them.

## Issues and bug reports
Please use the [Github issue tracker](https://github.com/adeyemiadeoye/SelfConcordantSmoothOptimization.jl/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) for reporting any issues. All types of issues are welcome including bug reports, feature requests, implementation for a specific research problem, etc.
