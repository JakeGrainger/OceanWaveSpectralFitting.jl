# Fitting

The fit function can be used to fit a model to a recorded series with either the Whittle or debiased Whittle likelihoods using the interior point Newton method as implemented in [*Optim.jl*](https://github.com/JuliaNLSolvers/Optim.jl).

First load the package:
```@example
using OceanWaveSpectralFitting
```

As a basic example, consider a univariate Gaussian process with JONSWAP spectral density function. We can simulate such a process with the following:
```@example
import Random # hide
Random.seed!(1234) # hide
using OceanWaveSpectralFitting # hide
n = 2304
Δ = 1/1.28
nreps = 1
α = 0.7
ωₚ = 1.1
γ = 3.3
r = 5.0
ts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1]
```
In this case, we have simulated a Gaussian process with JONSWAP spectral density function with parameters `α = 0.7, ωₚ = 1.1, γ = 3.3, r = 5.0` of length `2304` sampled every `1/1.28` seconds. This corresponds to 30 minutes of data.
The function `simulate_gp` will simulate a vector of `nreps` series, which is why we recover the first of these to get one series.

We can now fit a model with the following:
```@example
import Random # hide
Random.seed!(1234) # hide
using OceanWaveSpectralFitting # hide
n = 2304 # hide
Δ = 1/1.28 # hide
nreps = 1 # hide
α = 0.7 # hide
ωₚ = 1.1 # hide
γ = 3.3 # hide
r = 5.0 # hide
ts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1] # hide
x₀ = [α,ωₚ,γ,r] .+ 0.1
res = fit(ts,model=JONSWAP{1},x₀=x₀)
```
Here `x₀` is the vector of initial parameter guesses. 
The estimated parameter vector can be recovered by doing:
```@example
import Random # hide
Random.seed!(1234) # hide
using OceanWaveSpectralFitting # hide
n = 2304 # hide
Δ = 1/1.28 # hide
nreps = 1 # hide
α = 0.7 # hide
ωₚ = 1.1 # hide
γ = 3.3 # hide
r = 5.0 # hide
ts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1] # hide
x₀ = [α,ωₚ,γ,r] .+ 0.1 # hide
res = fit(ts,model=JONSWAP{1},x₀=x₀) # hide
x̂ = res.minimizer
```

The full example is:

```@example
import Random # hide
Random.seed!(1234) # hide
using OceanWaveSpectralFitting
n = 2304
Δ = 1/1.28
nreps = 1
α = 0.7
ωₚ = 1.1
γ = 3.3
r = 5.0
ts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1]
x₀ = [α,ωₚ,γ,r] .+ 0.1
res = fit(ts,model=JONSWAP{1},x₀=x₀)
x̂ = res.minimizer
```

For more options, see below:

```@docs
fit
```