module OceanWaveSpectralFitting

using Reexport
@reexport using WhittleLikelihoodInference
using Optim, DSP

include("models/univariate/JONSWAP.jl")
include("models/multivariate/JS_BWG_HNE.jl")
include("models/multivariate/JS_BWG_HNE_DL.jl")
include("fit.jl")

export JONSWAP, JS_BWG_HNE, JS_BWG_HNE_DL

end
