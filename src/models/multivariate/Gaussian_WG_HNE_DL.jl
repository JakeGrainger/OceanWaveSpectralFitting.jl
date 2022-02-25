struct Gaussian_WG_HNE_DL{K,H} <: UnknownAcvTimeSeriesModel{3,Float64}
    α::Float64
    ωₚ::Float64
    κ::Float64
    ϕₘ::Float64
    σ::Float64

    norm::Float64
    halfκ⁻²::Float64
    cosϕₘ::Float64
    sinϕₘ::Float64
    cos2ϕₘ::Float64
    sin2ϕₘ::Float64
    invexp2σ²::Float64
    invexphalfσ²::Float64
    function Gaussian_WG_HNE_DL{K,H}(α,ωₚ,κ,ϕₘ,σ) where {K,H}
        α > 0 || throw(ArgumentError("α must be > 0."))
        ωₚ > 0 || throw(ArgumentError("ωₚ must be > 0."))
        κ > 0 || throw(ArgumentError("κ must be > 0."))
        σ > 0 || throw(ArgumentError("Gaussian_WG_HNE_DL requires σ > 0"))
        new{K,H}(α,ωₚ,κ,ϕₘ,σ,
        α/(κ*sqrt(2π)),inv(2κ^2),
        cos(ϕₘ),sin(ϕₘ),cos(2ϕₘ),sin(2ϕₘ),exp(-2.0 * σ^2),exp(-0.5 * σ^2)
        )
    end
    function Gaussian_WG_HNE_DL{K,H}(x::AbstractVector{Float64}) where {K,H}
        @boundscheck checkparameterlength(x,Gaussian_WG_HNE_DL{K,H})
        @inbounds Gaussian_WG_HNE_DL{K,H}(x[1], x[2], x[3], x[4], x[5])
    end
end
Gaussian_WG_HNE_DL(x::AbstractVector{Float64}) = Gaussian_WG_HNE_DL(ones(5)...)
Gaussian_WG_HNE_DL(α,ωₚ,κ,ϕₘ,σ) = error("Gaussian_WG_HNE_DL process requires the ammount of aliasing specified as a type parameter. Use Gaussian_WG_HNE_DL{K,H}() where K ∈ N₀.")

WhittleLikelihoodInference.npars(::Type{Gaussian_WG_HNE_DL{K,H}}) where {K,H} = 5
WhittleLikelihoodInference.nalias(::Gaussian_WG_HNE_DL{K,H}) where {K,H} = K

lowerbounds(::Type{Gaussian_WG_HNE_DL{K,H}}) where {K,H} = [0,0,0,-Inf,0]
upperbounds(::Type{Gaussian_WG_HNE_DL{K,H}}) where {K,H} = [Inf,Inf,Inf,Inf,Inf]

@propagate_inbounds @fastmath function WhittleLikelihoodInference.add_sdf!(out, model::Gaussian_WG_HNE_DL{K,H}, ω) where {K,H}
    @boundscheck checkbounds(out,1:6)
    @inbounds begin
        signω = sign(ω)
        ω = abs(ω)
        if ω > 1e-10 # note this is due to the depth limiting, and results in a spectra that for some impractical parameters will be meaningfully discontinuous near 0.
            sdf = model.norm*exp(-(model.ωₚ-abs(ω))^2*model.halfκ⁻²)

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

@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::Gaussian_WG_HNE_DL{K,H}, ω::Real) where {K,H}
    @boundscheck checkbounds(out,1:6,1:5)
    @inbounds begin
        signω = sign(ω)
        ω = abs(ω)
        if ω > 1e-10 # note this is due to the depth limiting, and results in a spectra that for some impractical parameters will be meaningfully discontinuous near 0.
            σ = model.σ

            sdf = model.norm*exp(-(model.ωₚ-ω)^2 * model.halfκ⁻²)
            ∂S∂α = sdf/model.α
            ∂S∂ωₚ = 2(ω-model.ωₚ) * model.halfκ⁻² * sdf
            ∂S∂κ = sdf * ((ω-model.ωₚ)^2/model.κ^3-1/model.κ)

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
            out[1, 3] +=∂S∂κ
            out[2, 3] +=∂S∂κ * dirxz * inv_tanhkh
            out[3, 3] +=∂S∂κ * diryz * inv_tanhkh
            out[4, 3] +=∂S∂κ * dirxx * inv_tanhkh²
            out[5, 3] +=∂S∂κ * diryx * inv_tanhkh²
            out[6, 3] +=∂S∂κ * diryy * inv_tanhkh²
            ## ϕₘ
            horTemp = sdf * model.sin2ϕₘ * model.invexp2σ²
            # out[1, 4] += 0
            out[2, 4] +=-1im * sdf * model.sinϕₘ * model.invexphalfσ² * signω * inv_tanhkh
            out[3, 4] +=1im * sdf * model.cosϕₘ * model.invexphalfσ² * signω * inv_tanhkh
            out[4, 4] +=-horTemp * inv_tanhkh²
            out[5, 4] +=sdf * model.cos2ϕₘ * model.invexp2σ² * inv_tanhkh²
            out[6, 4] +=horTemp  * inv_tanhkh²
            ## σ
            sigtemp = 2.0σ * sdf * cos2part * inv_tanhkh²
            # out[1, 5] += 0
            out[2, 5] +=-σ * sdf * dirxz * inv_tanhkh
            out[3, 5] +=-σ * sdf * diryz * inv_tanhkh
            out[4, 5] +=-sigtemp
            out[5, 5] +=-4.0σ * sdf * diryx  * inv_tanhkh²
            out[6, 5] += sigtemp
        end
    end
    return nothing
end