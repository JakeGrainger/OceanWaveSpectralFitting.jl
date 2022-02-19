# OceanWaveSpectralFitting

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JakeGrainger.github.io/OceanWaveSpectralFitting.jl/stable)
[![Build Status](https://github.com/JakeGrainger/OceanWaveSpectralFitting.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JakeGrainger/OceanWaveSpectralFitting.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JakeGrainger/OceanWaveSpectralFitting.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JakeGrainger/OceanWaveSpectralFitting.jl)

A julia package for fitting parametric spectral wave models to recorded time series using Whittle and debiased Whittle likelihood inference. This package extends the functionality of [*WhittleLikelihoodInference.jl*](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl) to include models for ocean waves, and adds a fit function which uses solvers from [*Optim.jl*](https://github.com/JuliaNLSolvers/Optim.jl) with the option to use dpss tapers from [*DSP.jl*](https://github.com/JuliaDSP/DSP.jl).

Includes the following models:
- **JONSWAP**: the JONSWAP spectra for vertical displacement,
- **JS\_BWG\_HNE**: a model for the heave, northwards and eastwards displacement of a partical on the water surface when the waves have a JONSWAP marginal spectral density function and bimodal wrapped Gaussian spreading function.
- **JS\_BWG\_HNE\_DL**: a depth limited version of **JS\_BWG\_HNE**.


## References

Sykulski, A.M., Olhede, S.C., Guillaumin, A.P., Lilly, J.M., Early, J.J. (2019). The debiased Whittle likelihood. *Biometrika* 106 (2), 251–266.

Grainger, J. P., Sykulski, A. M., Jonathan, P., and Ewans, K. (2021). Estimating the parameters of ocean wave
spectra. *Ocean Engineering*, 229:108934.

Hasselmann, K., Barnett, T., Bouws, E., Carlson, H., Cartwright, D., Enke, K., Ewing, J., Gienapp, H., Hasselmann, D., Kruseman, P., Meerburg, A., M  ̈uller, P., Olbers, D., Richter, K., Sell, W., and Walden, H. (1973). Measurements of wind-wave growth and swell decay during the Joint North Sea Wave Project (JONSWAP).
Erg ̈anzungsheft 8-12.

Ewans, K. C. (1998). Observations of the directional spectrum of fetch-limited waves. Journal of Physical Oceanography, 28:495–512.
