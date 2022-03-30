@doc raw"""
    JONSWAP{K}(α,ωₚ,γ,r)
    JONSWAP{K}(x)

4 parameter JONSWAP model for vertical displacement.

# Type parameter
The type parameter `K` is a non-negative integer representing the ammount of aliasing to be done.
This is useful as some devices have high-pass filters which limit the ammount of aliasing that will be present in a recorded series.
In particular, a `JONSWAP{K}` model for a series recorded every `Δ` seconds would be appropriate for a series sampled every `Δ` seconds and filtered with a high-pass filter at `(2K-1)π/Δ`.

# Arguments
- `α`: The scale parameter.
- `ωₚ`: The peak angular frequency.
- `γ`: The peak enhancement factor.
- `r`: The tail decay index.

# Vector constructor arguments
- `x`: Vector of parameters `[α,ωₚ,γ,r]`.

# Background
The JONSWAP spectral density function is 
```math
f(ω) = αω^{-r}\exp\left \{-\frac{r}{4}\left(\frac{|ω|}{ωₚ}\right)^{-4}\right \}γ^{δ(|ω|)}
```
where
```math
δ(ω) = \exp\left\{-\tfrac{1}{2 (0.07+0.02\cdot\mathbb{1}_{ωₚ>|ω|})^2}\left (\tfrac{|ω|}{ωₚ}-1\right )^2\right\}.
```

"""
struct JONSWAP{K} <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{1,Float64}
    α::Float64
    ωₚ::Float64
    γ::Float64
    r::Float64

    r_over4::Float64
    ωₚ²::Float64
    ωₚ³::Float64
    ωₚ⁴::Float64
    ωₚ⁶::Float64
    logγ::Float64
    function JONSWAP{K}(α,ωₚ,γ,r) where {K}
        α > 0 || throw(ArgumentError("JONSWAP requires α > 0"))
        ωₚ > 0 || throw(ArgumentError("JONSWAP requires ωₚ > 0"))
        γ >= 1 || throw(ArgumentError("JONSWAP requires γ > 1"))
        r > 1 || throw(ArgumentError("JONSWAP requires r > 1"))
        new{K}(α,ωₚ,γ,r,r/4,ωₚ^2,ωₚ^3,ωₚ^4,ωₚ^6,log(γ))
    end
    function JONSWAP{K}(x::AbstractVector{Float64}) where {K}
        @boundscheck checkparameterlength(x,JONSWAP{K})
        @inbounds JONSWAP{K}(x[1], x[2], x[3], x[4])
    end
end

# functions to throw informative error if type parameter not provided
JONSWAP(x::AbstractVector{Float64}) = JONSWAP(1,1,1,1)
JONSWAP(α,ωₚ,γ,r) = error("JONSWAP process requires the ammount of aliasing specified as a type parameter. Use JONSWAP{K}() where K ∈ N.")

WhittleLikelihoodInference.npars(::Type{JONSWAP{K}}) where {K} = 4
WhittleLikelihoodInference.nalias(::JONSWAP{K}) where {K} = K

WhittleLikelihoodInference.lowerbounds(::Type{JONSWAP{K}}) where {K} = [0,0,1,1]
WhittleLikelihoodInference.upperbounds(::Type{JONSWAP{K}}) where {K} = [Inf,Inf,Inf,Inf]

@inline @fastmath function WhittleLikelihoodInference.sdf(model::JONSWAP{K}, ω::Real) where {K}
    α,ωₚ,γ,r = model.α,model.ωₚ,model.γ,model.r
    ω = abs(ω)
    if ω > 1e-10
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
        ωₚ⁴_over_ω⁴ = model.ωₚ⁴ / (ω^4)
        return (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
    else
        return 0.0
    end
end

@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::JONSWAP{K}, ω::Real) where {K}
    @boundscheck checkbounds(out,1:4)
    @inbounds begin
        ω = abs(ω)
        if ω > 1e-10
            α,ωₚ,γ,r = model.α,model.ωₚ,model.γ,model.r
            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
            ω⁻⁴ = ω^(-4)
            ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
            sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
            
            ∂S∂α = sdf / α
            ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*model.ωₚ³*ω⁻⁴)
            ∂S∂γ = sdf * δ / γ
            ∂S∂r = sdf * (-log(ω)-ωₚ⁴_over_ω⁴/4)

            out[1] += ∂S∂α
            out[2] += ∂S∂ωₚ
            out[3] += ∂S∂γ
            out[4] += ∂S∂r
        end # 0 otherwise
    end
    return nothing
end

@propagate_inbounds @fastmath function WhittleLikelihoodInference.hess_add_sdf!(out, model::JONSWAP{K}, ω::Real) where {K}
    @boundscheck checkbounds(out,1:10)
    @inbounds begin
        ω = abs(ω)
        if ω > 1e-10
            α,ωₚ,γ,r = model.α,model.ωₚ,model.γ,model.r
            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
            ω⁻⁴ = ω^(-4)
            ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
            ωₚ³_over_ω⁴ = model.ωₚ³ * ω⁻⁴
            sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
            
            δωlogγ_over_σ1² = δ*model.logγ * ω / σ1²
            ∂S∂ωₚUsepart = (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*ωₚ³_over_ω⁴)
            ∂S∂ωₚ = sdf * ∂S∂ωₚUsepart
            δ_over_γ = δ / γ
            ∂S∂γ = sdf * δ_over_γ
            ∂r_part = (-log(ω)-ωₚ⁴_over_ω⁴/4)
            ∂S∂r = sdf * ∂r_part
            
            ∂S∂αωₚ = ∂S∂ωₚ / α
            ∂S∂αγ = ∂S∂γ / α
            ∂S∂αr = ∂S∂r / α
            
            ∂S∂ωₚ2 = ∂S∂ωₚ * ∂S∂ωₚUsepart + sdf * (δωlogγ_over_σ1² *((-3ω + 2ωₚ)/model.ωₚ⁴ + ω * (ω-ωₚ)^2/model.ωₚ⁶/σ1²) - 3r*model.ωₚ²*ω⁻⁴)
            ∂S∂ωₚγ = ∂S∂γ * ∂S∂ωₚUsepart + sdf * δ_over_γ * ω / σ1²*(ω-ωₚ)/model.ωₚ³
            ∂S∂ωₚr = ∂S∂r * ∂S∂ωₚUsepart - sdf * model.ωₚ³*ω⁻⁴
            ∂S∂γ2 = δ_over_γ * (∂S∂γ - sdf/γ)
            ∂S∂γr = ∂S∂r * δ_over_γ
            ∂S∂r2 = ∂S∂r * ∂r_part

            # out[1] += 0
            out[2] += ∂S∂αωₚ
            out[3] += ∂S∂αγ
            out[4] += ∂S∂αr
            out[5] += ∂S∂ωₚ2
            out[6] += ∂S∂ωₚγ
            out[7] += ∂S∂ωₚr
            out[8] += ∂S∂γ2
            out[9] += ∂S∂γr
            out[10] += ∂S∂r2
        end # 0 otherwise
    end

    return nothing
end