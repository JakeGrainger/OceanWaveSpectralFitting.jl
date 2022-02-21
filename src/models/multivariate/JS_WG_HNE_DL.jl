@doc raw"""
    JS_WG_HNE{K,H}(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ)
    JS_WG_HNE{K,H}(x)

9 parameter model for vertical, northward and eastward displacement of a partical in waves with JONSWAP marginal spectra and bimodal wrapped Gaussian spreading.

# Type parameter
The type parameter `K` is a non-negative integer representing the ammount of aliasing to be done.
This is useful as some devices have high-pass filters which limit the ammount of aliasing that will be present in a recorded series.
In particular, a `JS_WG_HNE{K,H}` model for a series recorded every `Δ` seconds would be appropriate for a series sampled every `Δ` seconds and filtered with a high-pass filter at `(2K-1)π/Δ`.
The second type parameter `H` is the water depth.

# Arguments
- `α`: The scale parameter.
- `ωₚ`: The peak angular frequency.
- `γ`: The peak enhancement factor.
- `r`: The tail decay index.
- `ϕₘ`: The mean direction.
- `σ`: The width of the spreading function.

# Vector constructor arguments
- `x`: Vector of parameters `[α,ωₚ,γ,r,ϕₘ,σ]`.

# Background
The model is a transformation of a model for the frequency-direction spectra of a wave process.
Such a frequency-direction spectra, denoted ``f(ω,ϕ)`` is typically decomposed by writing:
```math
f(ω,ϕ) = f(ω)D(ω,ϕ)
```
where ``f(ω)`` is the marginal spectral density function and ``D(ω,ϕ)`` is the spreading function.
This model uses a JONSWAP marginal sdf and a bimodal wrapped Gaussian spreading function.
The JONSWAP spectral density function is 
```math
f(ω) = αω^{-r}\exp\left \{-\frac{r}{4}\left(\frac{|ω|}{ωₚ}\right)^{-4}\right \}γ^{δ(|ω|)}
```
where
```math
δ(ω) = \exp\left\{-\tfrac{1}{2 (0.07+0.02\cdot\mathbb{1}_{ωₚ>|ω|})^2}\left (\tfrac{|ω|}{ωₚ}-1\right )^2\right\}.
```
The bimodal wrapped Gaussian spreading function is 
```math
D(ω,ϕ) = \frac{1}{σ(ω)\sqrt{2π}}\sum\limits_{k=-∞}^{∞} \exp\left\{-\frac{1}{2}\left(\frac{ϕ-ϕ_{m}-2\pi k}{σ}\right)^2\right\}
```

"""
struct JS_WG_HNE_DL{K,H} <: UnknownAcvTimeSeriesModel{3,Float64}
    α::Float64
    ωₚ::Float64
    γ::Float64
    r::Float64
    ϕₘ::Float64
    σ::Float64

    r_over4::Float64
    ωₚ²::Float64
    ωₚ³::Float64
    ωₚ⁴::Float64
    logγ::Float64
    cosϕₘ::Float64
    sinϕₘ::Float64
    cos2ϕₘ::Float64
    sin2ϕₘ::Float64
    invexp2σ²::Float64
    invexphalfσ²::Float64
    function JS_WG_HNE_DL{K,H}(α,ωₚ,γ,r,ϕₘ,σ) where {K,H}
        α > 0 || throw(ArgumentError("JS_WG_HNE_DL requires α > 0"))
        ωₚ > 0 || throw(ArgumentError("JS_WG_HNE_DL requires ωₚ > 0"))
        γ >= 1 || throw(ArgumentError("JS_WG_HNE_DL requires γ > 1"))
        r > 1 || throw(ArgumentError("JS_WG_HNE_DL requires r > 1"))
        σ >= 0 || throw(ArgumentError("JS_WG_HNE_DL requires σ > 0"))
        new{K,H}(α,ωₚ,γ,r,ϕₘ,σ,
        r/4,ωₚ^2,ωₚ^3,ωₚ^4,log(γ),
        cos(ϕₘ),sin(ϕₘ),cos(2ϕₘ),sin(2ϕₘ),exp(-2.0 * σ^2),exp(-0.5 * σ^2)
        )
    end
    function JS_WG_HNE_DL{K,H}(x::AbstractVector{Float64}) where {K,H}
        @boundscheck checkparameterlength(x,JS_WG_HNE_DL{K,H})
        @inbounds JS_WG_HNE_DL{K,H}(x[1], x[2], x[3], x[4], x[5], x[6])
    end
end
JS_WG_HNE_DL(x::AbstractVector{Float64}) = JS_WG_HNE_DL(ones(6)...)
JS_WG_HNE_DL(α,ωₚ,γ,r,ϕₘ,σ) = error("JS_WG_HNE_DL process requires the ammount of aliasing specified as a type parameter. Use JS_WG_HNE_DL{K,H}() where K ∈ N₀.")

WhittleLikelihoodInference.npars(::Type{JS_WG_HNE_DL{K,H}}) where {K,H} = 6
WhittleLikelihoodInference.nalias(::JS_WG_HNE_DL{K,H}) where {K,H} = K

lowerbounds(::Type{JS_WG_HNE_DL{K,H}}) where {K,H} = [0,0,1,1,-Inf,0,]
upperbounds(::Type{JS_WG_HNE_DL{K,H}}) where {K,H} = [Inf,Inf,Inf,Inf,Inf,Inf]

@propagate_inbounds @fastmath function WhittleLikelihoodInference.add_sdf!(out, model::JS_WG_HNE_DL{K,H}, ω) where {K,H}
    @boundscheck checkbounds(out,1:6)
    @inbounds begin
        signω = sign(ω)
        ω = abs(ω)
        if ω > 1e-10
            α,ωₚ,γ,r = model.α,model.ωₚ,model.γ,model.r
        
            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            ω_over_ωₚ = ω / ωₚ
            δ = exp(-1 / (2σ1²) * (ω_over_ωₚ - 1)^2)
            ω⁻⁴ = ω^(-4)
            ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
            sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2

            cos2part = model.cos2ϕₘ * model.invexp2σ²
            imexppart = sdf * 1.0im * signω * model.invexphalfσ²

            tanhkh = tanh(approx_dispersion(ω, H))
            inv_tanhkh = inv(tanhkh)
            inv_tanhkh² = inv_tanhkh^2
            
            out[1] += sdf
            out[2] += imexppart * model.cosϕₘ * inv_tanhkh
            out[3] += imexppart * model.sinϕₘ * inv_tanhkh
            out[4] += sdf * (0.5 + 0.5cos2part) * inv_tanhkh²
            out[5] += sdf * model.sin2ϕₘ *0.5*model.invexp2σ² * inv_tanhkh²
            out[6] += sdf * (0.5 - 0.5cos2part) * inv_tanhkh²
        end
    end
    return nothing
end

@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::JS_WG_HNE_DL{K,H}, ω::Real) where {K,H}
    @boundscheck checkbounds(out,1:6,1:6)
    @inbounds begin
        signω = sign(ω)
        ω = abs(ω)
        if ω > 1e-10
            α,ωₚ,γ,r,σ = model.α,model.ωₚ,model.γ,model.r,model.σ

            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            ω_over_ωₚ = ω / ωₚ
            δ = exp(-1 / (2σ1²) * (ω_over_ωₚ - 1)^2)
            ω⁻⁴ = ω^(-4)
            ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
            sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
            
            ∂S∂α = sdf / α
            ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*model.ωₚ³*ω⁻⁴)
            ∂S∂γ = sdf * δ / γ
            ∂S∂r = sdf * (-log(ω)-ωₚ⁴_over_ω⁴/4)

            tanhkh = tanh(approx_dispersion(ω, H))
            inv_tanhkh = inv(tanhkh)
            inv_tanhkh² = inv_tanhkh^2

            cos2part = model.cos2ϕₘ * model.invexp2σ²
            imexppart = 1.0im * signω * model.invexphalfσ²
            dirxx = 0.5 + 0.5cos2part
            diryy = 0.5 - 0.5cos2part
            dirxz = imexppart * model.cosϕₘ
            diryz = imexppart * model.sinϕₘ
            diryx = model.sin2ϕₘ *0.5 * model.invexp2σ²
            ## α
            out[1, 1] +=∂S∂α
            out[2, 1] +=∂S∂α * dirxz * inv_tanhkh
            out[3, 1] +=∂S∂α * diryz * inv_tanhkh
            out[4, 1] +=∂S∂α * dirxx * inv_tanhkh²
            out[5, 1] +=∂S∂α * diryx * inv_tanhkh²
            out[6, 1] +=∂S∂α * diryy * inv_tanhkh²
            ## ωₚ
            out[1, 2] +=∂S∂ωₚ
            out[2, 2] +=∂S∂ωₚ * dirxz * inv_tanhkh
            out[3, 2] +=∂S∂ωₚ * diryz * inv_tanhkh
            out[4, 2] +=∂S∂ωₚ * dirxx * inv_tanhkh²
            out[5, 2] +=∂S∂ωₚ * diryx * inv_tanhkh²
            out[6, 2] +=∂S∂ωₚ * diryy * inv_tanhkh²
            ## γ
            out[1, 3] +=∂S∂γ
            out[2, 3] +=∂S∂γ * dirxz * inv_tanhkh
            out[3, 3] +=∂S∂γ * diryz * inv_tanhkh
            out[4, 3] +=∂S∂γ * dirxx * inv_tanhkh²
            out[5, 3] +=∂S∂γ * diryx * inv_tanhkh²
            out[6, 3] +=∂S∂γ * diryy * inv_tanhkh²
            ## r
            out[1, 4] +=∂S∂r
            out[2, 4] +=∂S∂r * dirxz * inv_tanhkh
            out[3, 4] +=∂S∂r * diryz * inv_tanhkh
            out[4, 4] +=∂S∂r * dirxx * inv_tanhkh²
            out[5, 4] +=∂S∂r * diryx * inv_tanhkh²
            out[6, 4] +=∂S∂r * diryy * inv_tanhkh²
            ## ϕₘ
            horTemp = sdf * model.sin2ϕₘ * model.invexp2σ²
            # out[1, 4] += 0
            out[2, 5] +=-1im * sdf * model.sinϕₘ * model.invexphalfσ² * signω * inv_tanhkh
            out[3, 5] +=1im * sdf * model.cosϕₘ * model.invexphalfσ² * signω * inv_tanhkh
            out[4, 5] +=-horTemp * inv_tanhkh²
            out[5, 5] +=sdf * model.cos2ϕₘ * model.invexp2σ² * inv_tanhkh²
            out[6, 5] +=horTemp  * inv_tanhkh²
            ## σ
            sigtemp = 2.0σ * sdf * cos2part * inv_tanhkh²
            # out[1, 6] += 0
            out[2, 6] +=-σ * sdf * dirxz * inv_tanhkh
            out[3, 6] +=-σ * sdf * diryz * inv_tanhkh
            out[4, 6] +=-sigtemp
            out[5, 6] +=-4.0σ * sdf * diryx  * inv_tanhkh²
            out[6, 6] += sigtemp
        end
    end
    return nothing
end