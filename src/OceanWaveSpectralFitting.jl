module OceanWaveSpectralFitting

using Reexport
@reexport using WhittleLikelihoodInference

include("models/univariate/JONSWAP.jl")

export JONSWAP

end
