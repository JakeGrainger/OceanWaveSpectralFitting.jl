struct GeneralJONSWAP{K} <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{1,Float64}
    α::Float64
    ωₚ::Float64
    γ::Float64
    r::Float64
    s::Float64

    r_over_s::Float64
    ωₚ²::Float64
    ωₚ³::Float64
    ωₚˢ::Float64
    ωₚˢ⁻¹::Float64
    logγ::Float64
    s⁻¹::Float64
    logωₚminuss⁻¹::Float64
    function GeneralJONSWAP{K}(α,ωₚ,γ,r,s) where {K}
        α > 0 || throw(ArgumentError("GeneralJONSWAP requires α > 0"))
        ωₚ > 0 || throw(ArgumentError("GeneralJONSWAP requires ωₚ > 0"))
        γ >= 1 || throw(ArgumentError("GeneralJONSWAP requires γ > 1"))
        r > 1 || throw(ArgumentError("GeneralJONSWAP requires r > 1"))
        s > 0 || throw(ArgumentError("GeneralJONSWAP requires s > 0"))
        new{K}(α,ωₚ,γ,r,s, r/s,ωₚ^2,ωₚ^3,ωₚ^s,ωₚ^(s-1),log(γ),inv(s),log(ωₚ)-inv(s))
    end
    function GeneralJONSWAP{K}(x::AbstractVector{Float64}) where {K}
        @boundscheck checkparameterlength(x,GeneralJONSWAP{K})
        @inbounds GeneralJONSWAP{K}(x[1], x[2], x[3], x[4], x[5])
    end
end

# functions to throw informative error if type parameter not provided
GeneralJONSWAP(x::AbstractVector{Float64}) = GeneralJONSWAP(1,1,1,1,1)
GeneralJONSWAP(α,ωₚ,γ,r,s) = error("GeneralJONSWAP process requires the ammount of aliasing specified as a type parameter. Use GeneralJONSWAP{K}() where K ∈ N.")

WhittleLikelihoodInference.npars(::Type{GeneralJONSWAP{K}}) where {K} = 5
WhittleLikelihoodInference.nalias(::GeneralJONSWAP{K}) where {K} = K

lowerbounds(::Type{GeneralJONSWAP{K}}) where {K} = [0,0,1,1,0]
upperbounds(::Type{GeneralJONSWAP{K}}) where {K} = [Inf,Inf,Inf,Inf,Inf]

@inline @fastmath function WhittleLikelihoodInference.sdf(model::GeneralJONSWAP{K}, ω::Real) where {K}
    α,ωₚ,γ,r,s = model.α,model.ωₚ,model.γ,model.r,model.s
    ω = abs(ω)
    if ω > 1e-10
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
        ωₚˢ_over_ωˢ = model.ωₚˢ / (ω^s)
        return (α * ω^(-r) * exp(-(model.r_over_s) * ωₚˢ_over_ωˢ) * γ^δ) / 2
    else
        return 0.0
    end
end

@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::GeneralJONSWAP{K}, ω::Real) where {K}
    @boundscheck checkbounds(out,1:5)
    @inbounds begin
        ω = abs(ω)
        if ω > 1e-10
            α,ωₚ,γ,r,s = model.α,model.ωₚ,model.γ,model.r,model.s
            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
            ω⁻ˢ = ω^(-s)
            ωₚˢ_over_ωˢ = model.ωₚˢ * ω⁻ˢ
            ∂S∂α = ω^(-r) * exp(-(model.r_over_s) * ωₚˢ_over_ωˢ) * γ^δ / 2
            sdf = α * ∂S∂α
            
            ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*model.ωₚˢ⁻¹*ω⁻ˢ)
            ∂S∂γ = sdf * δ / γ
            ∂S∂r = sdf * (-log(ω)-ωₚˢ_over_ωˢ*model.s⁻¹)
            ∂S∂s = sdf * model.r_over_s * ωₚˢ_over_ωˢ * (log(ω)-model.logωₚminuss⁻¹)

            out[1] += ∂S∂α
            out[2] += ∂S∂ωₚ
            out[3] += ∂S∂γ
            out[4] += ∂S∂r
            out[5] += ∂S∂s
        end # 0 otherwise
    end
    return nothing
end