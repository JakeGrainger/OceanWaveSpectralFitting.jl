module OceanWaveSpectralFitting

using Reexport
@reexport using WhittleLikelihoodInference
import WhittleLikelihoodInference: UnknownAcvTimeSeriesModel, checkparameterlength
import Base: @propagate_inbounds
using Optim, DSP

include("models/univariate/JONSWAP.jl")
include("models/univariate/generaljonswap.jl")
include("models/multivariate/JS_BWG_HNE.jl")
include("models/multivariate/JS_BWG_HNE_DL.jl")
include("models/multivariate/JS_WG_HNE_DL.jl")
include("fit.jl")

export JONSWAP, GeneralJONSWAP, JS_BWG_HNE, JS_BWG_HNE_DL, JS_WG_HNE_DL, fit

end
