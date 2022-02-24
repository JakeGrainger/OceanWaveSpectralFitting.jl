struct Gaussian{K} <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{3,Float64}
    α::Float64
    ωₚ::Float64
    σ::Float64

    norm::Float64
    halfσ⁻²::Float64
    function Gaussian{K}(α,ωₚ,σ) where {K}
        α > 0 || throw(ArgumentError("α must be > 0."))
        ωₚ > 0 || throw(ArgumentError("ωₚ must be > 0."))
        σ > 0 || throw(ArgumentError("σ must be > 0."))
        new(α,ωₚ,σ,α/(σ*sqrt(2π)),inv(2σ^2))
    end
    function Gaussian{K}(x) where {K}
        @boundscheck checkparameterlength(x,Gaussian{K})
        @inbounds Gaussian{K}(x[1],x[2],x[3])
    end
end

Base.@propagate_inbounds @fastmath function WhittleLikelihoodInference.add_sdf!(out, model::Gaussian, ω::Real)
    @inbounds begin
        ω = abs(ω)
        return model.norm*exp(-(model.ωₚ-ω)^2*model.halfσ⁻²)
    end
    nothing
end

WhittleLikelihoodInference.npars(::Type{Gaussian{K}}) where {K} = 3
WhittleLikelihoodInference.nalias(::Gaussian{K}) where {K} = K

OceanWaveSpectralFitting.lowerbounds(::Type{Gaussian{K}}) where {K} = [0,  0, 0]
OceanWaveSpectralFitting.upperbounds(::Type{Gaussian{K}}) where {K} = [Inf,Inf,Inf]

Gaussian(x::AbstractVector{Float64}) = GeneralJONSWAP(1,1,1)
Gaussian(α,ωₚ,σ) = error("Gaussian process requires the ammount of aliasing specified as a type parameter. Use Gaussian{K}() where K ∈ N.")

Base.@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::Gaussian, ω::Real)
    @boundscheck checkbounds(out,1:3)
    @inbounds begin
        ω = abs(ω)
        sdf = model.norm*exp(-(model.ωₚ-ω)^2 * model.halfσ⁻²)
        out[1] += sdf/α
        out[2] -= 2(ω-model.ωₚ) * model.halfσ⁻² * sdf
        out[3] += 2(model.ωₚ-ω)^2 * model.halfσ⁻² * sdf
        out[3] += sdf * ((ω-ωₚ)^2/model.σ^3-1/model.σ)
    end
    nothing
end