# Fitting

The fit function can be used to fit a model to a recorded series with either the Whittle or debiased Whittle likelihoods using the interior point Newton method as implemented in [*Optim.jl*](https://github.com/JuliaNLSolvers/Optim.jl).

First load the package:
```@example
using OceanWaveSpectralFitting
```

As a basic example, consider a univariate Gaussian process with JONSWAP spectral density function. We can simulate such a process with the following:
```@example
using OceanWaveSpectralFitting # hide
n = 1000
Δ = 1
nreps = 1
α = 0.7
ωₚ = 1.1
γ = 3.3
r = 5.0
ts = simulate_gp(JONSWAP{1}(),n,Δ,nreps)[1]
```
The function `simulate_gp` will simulate a vector of `nreps` series, which is why we recover the first of these to get one series.

We can now fit a model with the following:
```@example
using OceanWaveSpectralFitting # hide
n = 1000 # hide
Δ = 1 # hide
nreps = 1 # hide
ts = simulate_gp(JONSWAP{1},n,Δ,nreps)[1] # hide
x₀ = [α,ωₚ,γ,r] .+ 0.1
res = fit(ts,model=JONSWAP{1},x₀=x₀)
```
Here `x₀` is the vector of initial parameter guesses.
For more options, see below:

```@docs
fit
```