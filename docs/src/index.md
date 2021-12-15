```@meta
CurrentModule = OceanWaveSpectralFitting
```

# OceanWaveSpectralFitting
The [*OceanWaveSpectralFitting*](https://github.com/JakeGrainger/OceanWaveSpectralFitting.jl) package for fitting parametric spectral wave models to recorded time series using Whittle and debiased Whittle likelihood inference. This package extends the functionality of [*WhittleLikelihoodInference.jl*](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl) to include models for ocean waves, and adds a fit function which uses solvers from [*Optim.jl*](https://github.com/JuliaNLSolvers/Optim.jl) with the option to use dpss tapers from [*DSP.jl*](https://github.com/JuliaDSP/DSP.jl).

Includes the following models:
- **JONSWAP**: the JONSWAP spectra for vertical displacement,
- **JS\_BWG\_HNE**: a model for the heave, northwards and eastwards displacement of a partical on the water surface when the waves have a JONSWAP marginal spectral density function and bimodal wrapped Gaussian spreading function.
- **JS\_BWG\_HNE\_DL**: a depth limited version of **JS\_BWG\_HNE**.