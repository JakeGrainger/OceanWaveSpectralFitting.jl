```@meta
CurrentModule = OceanWaveSpectralFitting
```

# OceanWaveSpectralFitting
The [*OceanWaveSpectralFitting*](https://github.com/JakeGrainger/OceanWaveSpectralFitting.jl) package for fitting parametric spectral wave models to recorded time series using Whittle and debiased Whittle likelihood inference. This package extends the functionality of [*WhittleLikelihoodInference.jl*](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl) to include models for ocean waves.

Includes the following models:
- **JONSWAP**: the JONSWAP spectra for vertical displacement,
- **JS\_BWG\_HNE**: a model for the heave, northwards and eastwards displacement of a particle on the water surface when the waves have a JONSWAP marginal spectral density function and bimodal wrapped Gaussian spreading function.
- **JS\_BWG\_HNE\_DL**: a depth limited version of **JS\_BWG\_HNE**.