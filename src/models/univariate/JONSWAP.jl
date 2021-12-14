struct JONSWAP{K} <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{1}
    α::Float64
    ωₚ::Float64
    γ::Float64
    r::Float64

    r_over4::Float64
    ωₚ³::Float64
    logγ::Float64
    function JONSWAP{K}(α,ωₚ,γ,r) where {K}
        α > 0 || throw(ArgumentError("JONSWAP requires α > 0"))
        ωₚ > 0 || throw(ArgumentError("JONSWAP requires ωₚ > 0"))
        γ > 1 || throw(ArgumentError("JONSWAP requires γ > 1"))
        r > 1 || throw(ArgumentError("JONSWAP requires r > 1"))
        new(α,ωₚ,γ,r,r/4,ωₚ^3,log(γ))
    end
end

WhittleLikelihoodInference.npars(::Type{JONSWAP}) = 4
WhittleLikelihoodInference.nalias(::JONSWAP{K}) = K

function WhittleLikelihoodInference.sdf(model::JONSWAP, ω::Real)
    α,ωₚ,γ,r = model.α,model.ωₚ,model.γ,model.r
    ω = abs(ω)
    if ω > 1e-10
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
        ωₚ⁴_over_ω⁴ = (ωₚ / ω)^(4)
        return (α * ω^(-r) * exp(-(r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
    else
        return 0.0
    end
end

function grad_add_sdf!(out, model::JONSWAP, ω::Real)
    ω = abs(ω)
    if ω > 1e-10
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
        ωₚ⁴_over_ω⁴ = (ωₚ / ω)^(4)
        sdf = (α * ω^(-r) * exp(-(r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
        
        ∂S∂α = sdf / α
        ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*ωₚ⁴_over_ω⁴/ωₚ)
        ∂S∂γ = sdf * δ / γ
        ∂S∂r = sdf * (-log(ω)-ωₚ⁴_over_ω⁴/4)

        out[1] += ∂S∂α
        out[2] += ∂S∂ωₚ
        out[3] += ∂S∂γ
        out[4] += ∂S∂r
    end # 0 otherwise

    return nothing
end