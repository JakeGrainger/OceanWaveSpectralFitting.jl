struct GJS_BWG_HNE_DL{K,H} <: UnknownAcvTimeSeriesModel{3,Float64}
    α::Float64
    ωₚ::Float64
    γ::Float64
    r::Float64
    s::Float64
    ϕₘ::Float64
    β::Float64
    ν::Float64
    σₗ::Float64
    σᵣ::Float64

    r_over_s::Float64
    ωₚ²::Float64
    ωₚ³::Float64
    ωₚˢ::Float64
    ωₚˢ⁻¹::Float64
    logγ::Float64
    s⁻¹::Float64
    logωₚminuss⁻¹::Float64
    σᵣ_over3::Float64
    cosϕₘ::Float64
    sinϕₘ::Float64
    cos2ϕₘ::Float64
    sin2ϕₘ::Float64
    function GJS_BWG_HNE_DL{K,H}(α,ωₚ,γ,r,s,ϕₘ,β,ν,σₗ,σᵣ) where {K,H}
        α > 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires α > 0"))
        ωₚ > 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires ωₚ > 0"))
        γ >= 1 || throw(ArgumentError("GJS_BWG_HNE_DL requires γ > 1"))
        r > 1 || throw(ArgumentError("GJS_BWG_HNE_DL requires r > 1"))
        s > 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires s > 0"))
        β >= 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires β > 0"))
        ν >= 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires ν > 0"))
        σₗ >= 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires σₗ > 0"))
        σᵣ >= 0 || throw(ArgumentError("GJS_BWG_HNE_DL requires σᵣ > 0"))
        new{K,H}(α,ωₚ,γ,r,s,ϕₘ,β,ν,σₗ,σᵣ,
        r/s,ωₚ^2,ωₚ^3,ωₚ^s,ωₚ^(s-1),log(γ),inv(s),log(ωₚ)-inv(s),
        σᵣ/3,cos(ϕₘ),sin(ϕₘ),cos(2ϕₘ),sin(2ϕₘ)
        )
    end
    function GJS_BWG_HNE_DL{K,H}(x::AbstractVector{Float64}) where {K,H}
        @boundscheck checkparameterlength(x,GJS_BWG_HNE_DL{K,H})
        @inbounds GJS_BWG_HNE_DL{K,H}(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])
    end
end

# functions to throw informative error if type parameter not provided
GJS_BWG_HNE_DL(x::AbstractVector{Float64}) = GJS_BWG_HNE_DL(ones(10)...)
GJS_BWG_HNE_DL(α,ωₚ,γ,r,s,ϕₘ,β,ν,σₗ,σᵣ) = error("GJS_BWG_HNE_DL process requires the ammount of aliasing and water depth to be specified as a type parameter. Use GJS_BWG_HNE_DL{K,H}() where K ∈ N₀ and H ∈ R.")

WhittleLikelihoodInference.npars(::Type{GJS_BWG_HNE_DL{K,H}}) where {K,H} = 10
WhittleLikelihoodInference.nalias(::GJS_BWG_HNE_DL{K,H}) where {K,H} = K

WhittleLikelihoodInference.lowerbounds(::Type{GJS_BWG_HNE_DL{K,H}}) where {K,H} = [0,0,1,1,0,-Inf,0,0,0,0]
WhittleLikelihoodInference.upperbounds(::Type{GJS_BWG_HNE_DL{K,H}}) where {K,H} = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]

@propagate_inbounds @fastmath function WhittleLikelihoodInference.add_sdf!(out, model::GJS_BWG_HNE_DL{K,H}, ω::Real) where {K,H}
    @boundscheck checkbounds(out,1:6)
    @inbounds begin
        s_om = sign(ω)
        ω = abs(ω)
        
        if ω > 1e-10
            tanhkh = tanh(approx_dispersion(ω, H))
            inv_tanhkh = inv(tanhkh)
            inv_tanhkh² = inv_tanhkh^2
            α,ωₚ,γ,r,s,β,ν,σₗ = model.α,model.ωₚ,model.γ,model.r,model.s,model.β,model.ν,model.σₗ
            
            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            ω_over_ωₚ = ω / ωₚ
            ωₚ_over_ω = 1/ω_over_ωₚ

            δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
            ω⁻ˢ = ω^(-s)
            ωₚˢ_over_ωˢ = model.ωₚˢ * ω⁻ˢ
            ∂S∂α = ω^(-r) * exp(-(model.r_over_s) * ωₚˢ_over_ωˢ) * γ^δ / 2
            sdf = α * ∂S∂α

            PeakSep = β / exp(ν * ((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0))
            σ = σₗ - model.σᵣ_over3*(4*(ωₚ_over_ω)^2 - (ωₚ_over_ω)^8)
            σ² = σ^2

            cosPS = cos(PeakSep)
            exp2part = (2 * exp(2.0 * σ²))
            cos2part = (model.cos2ϕₘ*cosPS) / exp2part
            imexppart = sdf * 1im * s_om / exp(0.5 * σ²) * cos(PeakSep/2)

            out[1] += sdf
            out[2] += imexppart * model.cosϕₘ * inv_tanhkh
            out[3] += imexppart * model.sinϕₘ * inv_tanhkh
            out[4] += sdf * (0.5 + cos2part) * inv_tanhkh²
            out[5] += sdf * model.sin2ϕₘ * cosPS / exp2part * inv_tanhkh²
            out[6] += sdf * (0.5 - cos2part) * inv_tanhkh²
        end # add zero otherwise
    end
    return nothing
end

@propagate_inbounds @fastmath function WhittleLikelihoodInference.grad_add_sdf!(out, model::GJS_BWG_HNE_DL{K,H}, ω::Real) where {K,H}
    @boundscheck checkbounds(out,1:6,1:10)
    @inbounds begin
        s_om = sign(ω)
        ω = abs(ω)
        
        if ω > 1e-10
            tanhkh = tanh(approx_dispersion(ω, H))
            inv_tanhkh = inv(tanhkh)
            inv_tanhkh² = inv_tanhkh^2
            α,ωₚ,γ,r,s,β,ν,σₗ,σᵣ = model.α,model.ωₚ,model.γ,model.r,model.s,model.β,model.ν,model.σₗ,model.σᵣ
            
            σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
            ω_over_ωₚ = ω / ωₚ
            ωₚ_over_ω = 1/ω_over_ωₚ

            δ = exp(-1 / (2σ1²) * (ω / ωₚ - 1)^2)
            ω⁻ˢ = ω^(-s)
            ωₚˢ_over_ωˢ = model.ωₚˢ * ω⁻ˢ
            ∂S∂α = ω^(-r) * exp(-(model.r_over_s) * ωₚˢ_over_ωˢ) * γ^δ / 2
            sdf = α * ∂S∂α
            
            ∂S∂α = sdf/α
            ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*model.ωₚˢ⁻¹*ω⁻ˢ)
            ∂S∂γ = sdf * δ / γ
            ∂S∂r = sdf * (-log(ω)-ωₚˢ_over_ωˢ*model.s⁻¹)
            ∂S∂s = sdf * model.r_over_s * ωₚˢ_over_ωˢ * (log(ω)-model.logωₚminuss⁻¹)
            
            PeakSep = β / exp(ν * ((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0))
            σ = σₗ - model.σᵣ_over3*(4*(ωₚ_over_ω)^2 - (ωₚ_over_ω)^8)
            σ² = σ^2

            cosϕₘ = model.cosϕₘ
            sinϕₘ = model.sinϕₘ
            cos2ϕₘ = model.cos2ϕₘ
            sin2ϕₘ = model.sin2ϕₘ
            cosPShalf = cos(PeakSep/2)
            cosPS = cos(PeakSep)
            sinPShalf = sin(PeakSep/2)
            sinPS = sin(PeakSep)

            exp2part = (exp(2.0 * σ²))
            cos2part = (cos2ϕₘ*cosPS) / exp2part
            exphalf = exp(0.5 * σ²)
            imexppart = 1im * s_om / exphalf * cosPShalf

            # hzz = 1
            hxx = (0.5 + cos2part/2)
            hyy = (0.5 - cos2part/2)
            hxz = imexppart * cosϕₘ
            hyz = imexppart * sinϕₘ
            hyx = sin2ϕₘ * cosPS / exp2part / 2
            
            #### h_derivs
            ## ϕₘ
            hyxpart∂ϕₘ = cosPS / exp2part
            hxx∂ϕₘ = -sin2ϕₘ * hyxpart∂ϕₘ
            hyy∂ϕₘ = -hxx∂ϕₘ
            hzpart∂ϕₘ = -1im * s_om * cosPShalf / exphalf
            hxz∂ϕₘ = sinϕₘ * hzpart∂ϕₘ
            hyz∂ϕₘ = -cosϕₘ * hzpart∂ϕₘ
            hyx∂ϕₘ = cos2ϕₘ * hyxpart∂ϕₘ

            ## β, ν
            PeakSep∂β = PeakSep / β
            PeakSep∂ν = -((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0) * β / exp(ν * ((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0))
            hx∂βνtemp = sinPS / 2 / exp2part
            hz∂βνtemp = -0.5im * sinPShalf / exphalf * s_om

            hx∂βt = hx∂βνtemp * PeakSep∂β
            hx∂νt = hx∂βνtemp * PeakSep∂ν
            hz∂βt = hz∂βνtemp * PeakSep∂β
            hz∂νt = hz∂βνtemp * PeakSep∂ν

            hxx∂β = -cos2ϕₘ * hx∂βt
            hyy∂β = -hxx∂β
            hxz∂β = cosϕₘ * hz∂βt
            hyz∂β = sinϕₘ * hz∂βt
            hyx∂β = -sin2ϕₘ * hx∂βt

            hxx∂ν = -cos2ϕₘ * hx∂νt
            hyy∂ν = -hxx∂ν
            hxz∂ν = cosϕₘ * hz∂νt
            hyz∂ν = sinϕₘ * hz∂νt
            hyx∂ν = -sin2ϕₘ * hx∂νt

            ## σₗ, σᵣ
            # ∂σ∂σₗ = 1
            ∂σ∂σᵣ = (σ-σₗ)/σᵣ
            hx∂σtemp = cosPS * 2 / exp2part * σ
            hz∂σtemp = -1im * cosPShalf / exphalf * σ * s_om

            hx∂σᵣt = hx∂σtemp * ∂σ∂σᵣ
            hz∂σᵣt = hz∂σtemp * ∂σ∂σᵣ

            hxx∂σₗ = -cos2ϕₘ * hx∂σtemp
            hyy∂σₗ = -hxx∂σₗ
            hxz∂σₗ = cosϕₘ * hz∂σtemp
            hyz∂σₗ = sinϕₘ * hz∂σtemp
            hyx∂σₗ = -sin2ϕₘ * hx∂σtemp

            hxx∂σᵣ = -cos2ϕₘ * hx∂σᵣt
            hyy∂σᵣ = -hxx∂σᵣ
            hxz∂σᵣ = cosϕₘ * hz∂σᵣt
            hyz∂σᵣ = sinϕₘ * hz∂σᵣt
            hyx∂σᵣ = -sin2ϕₘ * hx∂σᵣt

            ## ωₚ
            PeakSep∂ωₚ = -(ω > ωₚ) * ν/ω * β * exp(-ν*ωₚ/ω)
            ∂σ∂ωₚ = (-8σᵣ/3*(ωₚ/(ω^2)-ωₚ^7/(ω^8)))
            
            hx∂ωₚt = (sinPS / 2 * PeakSep∂ωₚ + 2*cosPS*σ*∂σ∂ωₚ)/ exp2part 
            hz∂ωₚt = -1im * (sinPShalf / 2 * PeakSep∂ωₚ + cosPShalf*σ*∂σ∂ωₚ)/ exphalf * s_om

            hxx∂ωₚ = -cos2ϕₘ * hx∂ωₚt
            hyy∂ωₚ = -hxx∂ωₚ
            hxz∂ωₚ = cosϕₘ * hz∂ωₚt
            hyz∂ωₚ = sinϕₘ * hz∂ωₚt
            hyx∂ωₚ = -sin2ϕₘ * hx∂ωₚt

            ## α
            out[1, 1] += ∂S∂α
            out[2, 1] += ∂S∂α * hxz * inv_tanhkh
            out[3, 1] += ∂S∂α * hyz * inv_tanhkh
            out[4, 1] += ∂S∂α * hxx * inv_tanhkh²
            out[5, 1] += ∂S∂α * hyx * inv_tanhkh²
            out[6, 1] += ∂S∂α * hyy * inv_tanhkh²

            ## ωₚ
            out[1, 2] += ∂S∂ωₚ
            out[2, 2] += (∂S∂ωₚ * hxz + sdf * hxz∂ωₚ) * inv_tanhkh
            out[3, 2] += (∂S∂ωₚ * hyz + sdf * hyz∂ωₚ) * inv_tanhkh
            out[4, 2] += (∂S∂ωₚ * hxx + sdf * hxx∂ωₚ) * inv_tanhkh²
            out[5, 2] += (∂S∂ωₚ * hyx + sdf * hyx∂ωₚ) * inv_tanhkh²
            out[6, 2] += (∂S∂ωₚ * hyy + sdf * hyy∂ωₚ) * inv_tanhkh²

            ## γ
            out[1, 3] += ∂S∂γ
            out[2, 3] += ∂S∂γ * hxz * inv_tanhkh
            out[3, 3] += ∂S∂γ * hyz * inv_tanhkh
            out[4, 3] += ∂S∂γ * hxx * inv_tanhkh²
            out[5, 3] += ∂S∂γ * hyx * inv_tanhkh²
            out[6, 3] += ∂S∂γ * hyy * inv_tanhkh²

            ## r
            out[1, 4] += ∂S∂r
            out[2, 4] += ∂S∂r * hxz * inv_tanhkh
            out[3, 4] += ∂S∂r * hyz * inv_tanhkh
            out[4, 4] += ∂S∂r * hxx * inv_tanhkh²
            out[5, 4] += ∂S∂r * hyx * inv_tanhkh²
            out[6, 4] += ∂S∂r * hyy * inv_tanhkh²

            ## s
            out[1, 5] += ∂S∂s
            out[2, 5] += ∂S∂s * hxz * inv_tanhkh
            out[3, 5] += ∂S∂s * hyz * inv_tanhkh
            out[4, 5] += ∂S∂s * hxx * inv_tanhkh²
            out[5, 5] += ∂S∂s * hyx * inv_tanhkh²
            out[6, 5] += ∂S∂s * hyy * inv_tanhkh²

            ## ϕₘ
            # out[1, 5] += 0
            out[2, 6] += sdf * hxz∂ϕₘ * inv_tanhkh
            out[3, 6] += sdf * hyz∂ϕₘ * inv_tanhkh
            out[4, 6] += sdf * hxx∂ϕₘ * inv_tanhkh²
            out[5, 6] += sdf * hyx∂ϕₘ * inv_tanhkh²
            out[6, 6] += sdf * hyy∂ϕₘ * inv_tanhkh²

            ## β
            # out[1, 7] += 0
            out[2, 7] += sdf * hxz∂β * inv_tanhkh
            out[3, 7] += sdf * hyz∂β  * inv_tanhkh
            out[4, 7] += sdf * hxx∂β  * inv_tanhkh²
            out[5, 7] += sdf * hyx∂β * inv_tanhkh²
            out[6, 7] += sdf * hyy∂β * inv_tanhkh²

            ## ν
            # out[1, 8] += 0
            out[2, 8] += sdf * hxz∂ν * inv_tanhkh
            out[3, 8] += sdf * hyz∂ν  * inv_tanhkh
            out[4, 8] += sdf * hxx∂ν  * inv_tanhkh²
            out[5, 8] += sdf * hyx∂ν * inv_tanhkh²
            out[6, 8] += sdf * hyy∂ν * inv_tanhkh²

            ## σₗ
            # out[1, 9] += 0
            out[2, 9] += sdf * hxz∂σₗ * inv_tanhkh
            out[3, 9] += sdf * hyz∂σₗ * inv_tanhkh
            out[4, 9] += sdf * hxx∂σₗ * inv_tanhkh²
            out[5, 9] += sdf * hyx∂σₗ * inv_tanhkh²
            out[6, 9] += sdf * hyy∂σₗ * inv_tanhkh²

            ## σᵣ
            # out[1, 10] += 0
            out[2, 10] += sdf * hxz∂σᵣ * inv_tanhkh
            out[3, 10] += sdf * hyz∂σᵣ * inv_tanhkh
            out[4, 10] += sdf * hxx∂σᵣ * inv_tanhkh²
            out[5, 10] += sdf * hyx∂σᵣ * inv_tanhkh²
            out[6, 10] += sdf * hyy∂σᵣ * inv_tanhkh²

        end # add zero otherwise
    end
    return nothing
end