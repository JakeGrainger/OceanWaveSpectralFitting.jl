struct Gaussian{K} <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{1,Float64}
    α::Float64
    ωₚ::Float64
    κ::Float64

    norm::Float64
    halfκ⁻²::Float64
    function Gaussian{K}(α,ωₚ,κ) where {K}
        α > 0 || throw(ArgumentError("α must be > 0."))
        ωₚ > 0 || throw(ArgumentError("ωₚ must be > 0."))
        κ > 0 || throw(ArgumentError("κ must be > 0."))
        new(α,ωₚ,κ,α/(κ*sqrt(2π)),inv(2κ^2))
    end
    function Gaussian{K}(x) where {K}
        @boundscheck checkparameterlength(x,Gaussian{K})
        @inbounds Gaussian{K}(x[1],x[2],x[3])
    end
end

@fastmath function WhittleLikelihoodInference.sdf(model::Gaussian, ω::Real)
    return model.norm*exp(-(model.ωₚ-abs(ω))^2*model.halfκ⁻²)
end

WhittleLikelihoodInference.npars(::Type{Gaussian{K}}) where {K} = 3
WhittleLikelihoodInference.nalias(::Gaussian{K}) where {K} = K

OceanWaveSpectralFitting.lowerbounds(::Type{Gaussian{K}}) where {K} = [0,  0, 0]
OceanWaveSpectralFitting.upperbounds(::Type{Gaussian{K}}) where {K} = [Inf,Inf,Inf]

Gaussian(x::AbstractVector{Float64}) = Gaussian(1,1,1)
Gaussian(α,ωₚ,κ) = error("Gaussian process requires the ammount of aliasing specified as a type parameter. Use Gaussian{K}() where K ∈ N.")

Base.@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::Gaussian, ω::Real)
    @boundscheck checkbounds(out,1:3)
    @inbounds begin
        ω = abs(ω)
        sdf = model.norm*exp(-(model.ωₚ-ω)^2 * model.halfκ⁻²)
        out[1] += sdf/model.α
        out[2] += 2(ω-model.ωₚ) * model.halfκ⁻² * sdf
        out[3] += sdf * ((ω-model.ωₚ)^2/model.κ^3-1/model.κ)
    end
    nothing
end