var documenterSearchIndex = {"docs":
[{"location":"univariate/#Univariate","page":"Univariate","title":"Univariate","text":"","category":"section"},{"location":"univariate/#JONSWAP","page":"Univariate","title":"JONSWAP","text":"","category":"section"},{"location":"univariate/","page":"Univariate","title":"Univariate","text":"JONSWAP","category":"page"},{"location":"univariate/#OceanWaveSpectralFitting.JONSWAP","page":"Univariate","title":"OceanWaveSpectralFitting.JONSWAP","text":"JONSWAP{K}(α,ωₚ,γ,r)\nJONSWAP{K}(x)\n\n4 parameter JONSWAP model for vertical displacement.\n\nType parameter\n\nThe type parameter K is a non-negative integer representing the ammount of aliasing to be done. This is useful as some devices have high-pass filters which limit the ammount of aliasing that will be present in a recorded series. In particular, a JONSWAP{K} model for a series recorded every Δ seconds would be appropriate for a series sampled every Δ seconds and filtered with a high-pass filter at (2K-1)π/Δ.\n\nArguments\n\nα: The scale parameter.\nωₚ: The peak angular frequency.\nγ: The peak enhancement factor.\nr: The tail decay index.\n\nVector constructor arguments\n\nx: Vector of parameters [α,ωₚ,γ,r].\n\nBackground\n\nThe JONSWAP spectral density function is \n\nf(ω) = αω^-rexpleft -fracr4left(fracωωₚright)^-4right γ^δ(ω)\n\nwhere\n\nδ(ω) = expleft-tfrac12 (007+002cdotmathbb1_ωₚω)^2left (tfracωωₚ-1right )^2right\n\n\n\n\n\n","category":"type"},{"location":"fit/#Fitting","page":"Fitting","title":"Fitting","text":"","category":"section"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"The fit function can be used to fit a model to a recorded series with either the Whittle or debiased Whittle likelihoods using the interior point Newton method as implemented in Optim.jl.","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"First load the package:","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"using OceanWaveSpectralFitting","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"As a basic example, consider a univariate Gaussian process with JONSWAP spectral density function. We can simulate such a process with the following:","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"import Random # hide\nRandom.seed!(1234) # hide\nusing OceanWaveSpectralFitting # hide\nn = 2304\nΔ = 1/1.28\nnreps = 1\nα = 0.7\nωₚ = 1.1\nγ = 3.3\nr = 5.0\nts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1]","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"In this case, we have simulated a Gaussian process with JONSWAP spectral density function with parameters α = 0.7, ωₚ = 1.1, γ = 3.3, r = 5.0 of length 2304 sampled every 1/1.28 seconds. This corresponds to 30 minutes of data. The function simulate_gp will simulate a vector of nreps series, which is why we recover the first of these to get one series.","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"We can now fit a model with the following:","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"import Random # hide\nRandom.seed!(1234) # hide\nusing OceanWaveSpectralFitting # hide\nn = 2304 # hide\nΔ = 1/1.28 # hide\nnreps = 1 # hide\nα = 0.7 # hide\nωₚ = 1.1 # hide\nγ = 3.3 # hide\nr = 5.0 # hide\nts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1] # hide\nx₀ = [α,ωₚ,γ,r] .+ 0.1\nres = fit(ts,model=JONSWAP{1},x₀=x₀)","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"Here x₀ is the vector of initial parameter guesses.  The estimated parameter vector can be recovered by doing:","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"import Random # hide\nRandom.seed!(1234) # hide\nusing OceanWaveSpectralFitting # hide\nn = 2304 # hide\nΔ = 1/1.28 # hide\nnreps = 1 # hide\nα = 0.7 # hide\nωₚ = 1.1 # hide\nγ = 3.3 # hide\nr = 5.0 # hide\nts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1] # hide\nx₀ = [α,ωₚ,γ,r] .+ 0.1 # hide\nres = fit(ts,model=JONSWAP{1},x₀=x₀) # hide\nx̂ = res.minimizer","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"The full example is:","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"import Random # hide\nRandom.seed!(1234) # hide\nusing OceanWaveSpectralFitting\nn = 2304\nΔ = 1/1.28\nnreps = 1\nα = 0.7\nωₚ = 1.1\nγ = 3.3\nr = 5.0\nts = simulate_gp(JONSWAP{1}(α,ωₚ,γ,r),n,Δ,nreps)[1]\nx₀ = [α,ωₚ,γ,r] .+ 0.1\nres = fit(ts,model=JONSWAP{1},x₀=x₀)\nx̂ = res.minimizer","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"For more options, see below:","category":"page"},{"location":"fit/","page":"Fitting","title":"Fitting","text":"fit","category":"page"},{"location":"fit/#OceanWaveSpectralFitting.fit","page":"Fitting","title":"OceanWaveSpectralFitting.fit","text":"fit(ts,Δ;model::Type{<:TimeSeriesModel},x₀,lowerΩcutoff,upperΩcutoff,x_lowerbounds,x_upperbounds,method,taper)\nfit(timeseries::TimeSeries;model::Type{<:TimeSeriesModel},x₀,lowerΩcutoff,upperΩcutoff,x_lowerbounds,x_upperbounds,method,taper)\n\nFit a wave model using IPNewton method from Optim.jl.\n\nArguments\n\nts: n by D matrix containing the timeseries (or vector if D=1), where n is the number of observations and D is the number of series.\nΔ: The sampling rate, which should be a positive real number.\ntimeseries: Can be provided in place of ts and Δ.\nmodel: The model which will be fitted. Should be a type (not a realisation of the model) e.g. JONSWAP{k} not JONSWAP{K}(x).\nx₀: The initial parameter guess.\nlowerΩcutoff: The lower cutoff for the frequency range to be used in fitting. Default is 0.\nupperΩcutoff: The upper cutoff for the frequency range to be used in fitting. Default is Inf.\nx_lowerbounds: The lower bounds on the parameter space. If nothing is provided (the default) then these are set to default values based on the model.\nx_upperbounds: The upper bounds on the parameter space. If nothing is provided (the default) then these are set to default values based on the model.\nmethod: Either :Whittle or :debiasedWhittle.\ntaper: The choice of tapering to be used. This should be nothing (in which case no taper is used) or dpss_nw where nw time-bandwith product (see DSP.dpss for more details).\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = OceanWaveSpectralFitting","category":"page"},{"location":"#OceanWaveSpectralFitting","page":"Home","title":"OceanWaveSpectralFitting","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The OceanWaveSpectralFitting package for fitting parametric spectral wave models to recorded time series using Whittle and debiased Whittle likelihood inference. This package extends the functionality of WhittleLikelihoodInference.jl to include models for ocean waves, and adds a fit function which uses solvers from Optim.jl with the option to use dpss tapers from DSP.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Includes the following models:","category":"page"},{"location":"","page":"Home","title":"Home","text":"JONSWAP: the JONSWAP spectra for vertical displacement,\nJS_BWG_HNE: a model for the heave, northwards and eastwards displacement of a partical on the water surface when the waves have a JONSWAP marginal spectral density function and bimodal wrapped Gaussian spreading function.\nJS_BWG_HNE_DL: a depth limited version of JS_BWG_HNE.","category":"page"},{"location":"multivariate/#Multivariate","page":"Multivariate","title":"Multivariate","text":"","category":"section"},{"location":"multivariate/#Displacement-buoys-(heave-north-east)","page":"Multivariate","title":"Displacement buoys (heave-north-east)","text":"","category":"section"},{"location":"multivariate/#JONSWAP-with-bimodal-wrapped-Gaussian","page":"Multivariate","title":"JONSWAP with bimodal wrapped Gaussian","text":"","category":"section"},{"location":"multivariate/","page":"Multivariate","title":"Multivariate","text":"JS_BWG_HNE","category":"page"},{"location":"multivariate/#OceanWaveSpectralFitting.JS_BWG_HNE","page":"Multivariate","title":"OceanWaveSpectralFitting.JS_BWG_HNE","text":"JS_BWG_HNE{K}(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ)\nJS_BWG_HNE{K}(x)\n\n9 parameter model for vertical, northward and eastward displacement of a partical in waves with JONSWAP marginal spectra and bimodal wrapped Gaussian spreading.\n\nType parameter\n\nThe type parameter K is a non-negative integer representing the ammount of aliasing to be done. This is useful as some devices have high-pass filters which limit the ammount of aliasing that will be present in a recorded series. In particular, a JS_BWG_HNE{K} model for a series recorded every Δ seconds would be appropriate for a series sampled every Δ seconds and filtered with a high-pass filter at (2K-1)π/Δ.\n\nArguments\n\nα: The scale parameter.\nωₚ: The peak angular frequency.\nγ: The peak enhancement factor.\nr: The tail decay index.\nϕₘ: The mean direction.\nβ: The limiting peak separation.\nν: The peak separation shape.\nσₗ: The limiting angular width.\nσᵣ: The angular width shape.\n\nVector constructor arguments\n\nx: Vector of parameters [α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ].\n\nBackground\n\nThe model is a transformation of a model for the frequency-direction spectra of a wave process. Such a frequency-direction spectra, denoted f(ωϕ) is typically decomposed by writing:\n\nf(ωϕ) = f(ω)D(ωϕ)\n\nwhere f(ω) is the marginal spectral density function and D(ωϕ) is the spreading function. This model uses a JONSWAP marginal sdf and a bimodal wrapped Gaussian spreading function. The JONSWAP spectral density function is \n\nf(ω) = αω^-rexpleft -fracr4left(fracωωₚright)^-4right γ^δ(ω)\n\nwhere\n\nδ(ω) = expleft-tfrac12 (007+002cdotmathbb1_ωₚω)^2left (tfracωωₚ-1right )^2right\n\nThe bimodal wrapped Gaussian spreading function is \n\nD(ωϕ) = frac12σ(ω)sqrt2πsumlimits_k=-^ sumlimits_i=1^2 expleft-frac12left(fracϕ-ϕ_mi(ω)-2pi kσ(ω)right)^2right\n\nwhere\n\nbeginalign*\n    ϕ_m1(ω) = ϕ_m + β exp-ν min(ω_pω1)2 \n    ϕ_m2(ω) = ϕ_m - β exp-ν min(ω_pω1)2 \n    σ(ω) = σ_l - fracσ_r3left 4left(fracω_pωright)^2 - left(fracω_pωright)^8 right\nendalign*\n\n\n\n\n\n","category":"type"},{"location":"multivariate/#JONSWAP-with-bimodal-wrapped-Gaussian-(depth-limited)","page":"Multivariate","title":"JONSWAP with bimodal wrapped Gaussian (depth limited)","text":"","category":"section"},{"location":"multivariate/","page":"Multivariate","title":"Multivariate","text":"JS_BWG_HNE_DL","category":"page"},{"location":"multivariate/#OceanWaveSpectralFitting.JS_BWG_HNE_DL","page":"Multivariate","title":"OceanWaveSpectralFitting.JS_BWG_HNE_DL","text":"JS_BWG_HNE_DL{K,H}(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ)\nJS_BWG_HNE_DL{K,H}(x)\n\nDepth limited version of the JS_BWG_HNE model.\n\nType parameter\n\nThe additional type parameter H is the water depth (m).\n\nMore information\n\nFor more information, see the documentation for JS_BWG_HNE.\n\n\n\n\n\n","category":"type"}]
}