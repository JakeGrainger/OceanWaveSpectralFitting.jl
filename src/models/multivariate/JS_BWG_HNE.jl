struct JS_BWG_HNE{K} <: UnknownAcvTimeSeriesModel{3}
    α::Float64
    ωₚ::Float64
    γ::Float64
    r::Float64
    ϕₘ::Float64
    β::Float64
    ν::Float64
    σₗ::Float64
    σᵣ::Float64

    r_over4::Float64
    ωₚ²::Float64
    ωₚ³::Float64
    ωₚ⁴::Float64
    ωₚ⁶::Float64
    logγ::Float64
    σᵣ_over3::Float64
    cosϕₘ::Float64
    sinϕₘ::Float64
    cos2ϕₘ::Float64
    sin2ϕₘ::Float64
    function JS_BWG_HNE{K}(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ) where {K}
        α > 0 || throw(ArgumentError("JS_BWG_HNE requires α > 0"))
        ωₚ > 0 || throw(ArgumentError("JS_BWG_HNE requires ωₚ > 0"))
        γ > 1 || throw(ArgumentError("JS_BWG_HNE requires γ > 1"))
        r > 1 || throw(ArgumentError("JS_BWG_HNE requires r > 1"))
        β > 0 || throw(ArgumentError("JS_BWG_HNE requires β > 0"))
        ν > 0 || throw(ArgumentError("JS_BWG_HNE requires ν > 0"))
        σₗ > 0 || throw(ArgumentError("JS_BWG_HNE requires σₗ > 0"))
        σᵣ > 0 || throw(ArgumentError("JS_BWG_HNE requires σᵣ > 0"))
        new{K}(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ,
        r/4,ωₚ^2,ωₚ^3,ωₚ^4,ωₚ^6,log(γ),
        σᵣ/3,cos(ϕₘ),sin(ϕₘ),cos(2ϕₘ),sin(2ϕₘ)
        )
    end
    function JS_BWG_HNE{K}(x::AbstractVector{Float64}) where {K}
        length(x) == npars(JS_BWG_HNE{K}) || throw(ArgumentError("JS_BWG_HNE process has $(npars(JS_BWG_HNE{K})) parameters, but $(length(x)) were provided."))
        @inbounds JS_BWG_HNE{K}(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    end
end

# functions to throw informative error if type parameter not provided
JS_BWG_HNE(x::AbstractVector{Float64}) = JS_BWG_HNE(ones(9)...)
JS_BWG_HNE(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ) = error("JS_BWG_HNE process requires the ammount of aliasing specified as a type parameter. Use JS_BWG_HNE{K}() where K ∈ N.")

WhittleLikelihoodInference.npars(::Type{JS_BWG_HNE{K}}) where {K} = 9
WhittleLikelihoodInference.nalias(::JS_BWG_HNE{K}) where {K} = K

function WhittleLikelihoodInference.add_sdf!(out, model::JS_BWG_HNE{K}, ω::Real) where {K}
    s_om = sign(ω)
    ω = abs(ω)

    if ω > 1e-10
        α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ = model.α,model.ωₚ,model.γ,model.r,model.ϕₘ,model.β,model.ν,model.σₗ,model.σᵣ
        
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        ω_over_ωₚ = ω / ωₚ
        δ = exp(-1 / (2σ1²) * (ω_over_ωₚ - 1)^2)
        ω⁻⁴ = ω^(-4)
        ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
        sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
        
        PeakSep = β / exp(ν * ((ω > ωₚ)*((1/ω_over_ωₚ) - 1.0) + 1.0))
        σ = σₗ - model.σᵣ_over3*(4*(1/ω_over_ωₚ)^2 - (ωₚ⁴_over_ω⁴)^2)
        σ² = σ^2

        cosPS = cos(PeakSep)
        exp2part = (2 * exp(2.0 * σ²))
        cos2part = (model.cos2ϕₘ*cosPS) / exp2part
        imexppart = sdf * 1im * s_om / exp(0.5 * σ²) * cos(PeakSep/2)

        out[1] += sdf
        out[2] += imexppart * model.cosϕₘ
        out[3] += imexppart * model.sinϕₘ
        out[4] += sdf * (0.5 + cos2part)
        out[5] += sdf * model.sin2ϕₘ * cosPS / exp2part
        out[6] += sdf * (0.5 - cos2part)
    end # add zero otherwise

    return nothing
end

function WhittleLikelihoodInference.grad_add_sdf!(out, model::JS_BWG_HNE{K}, ω::Real) where {K}
    s_om = sign(ω)
    ω = abs(ω)
    
    if ω > 1e-10
        α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ = model.α,model.ωₚ,model.γ,model.r,model.ϕₘ,model.β,model.ν,model.σₗ,model.σᵣ
        
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
        
        PeakSep = β / exp(ν * ((ω > ωₚ)*((1/ω_over_ωₚ) - 1.0) + 1.0))
        σ = σₗ - model.σᵣ_over3*(4*(1/ω_over_ωₚ)^2 - (ωₚ⁴_over_ω⁴)^2)
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
        PeakSep∂ν = -((ω > ωₚ)*((ωₚ / ω) - 1.0) + 1.0) * β / exp(ν * ((ω > ωₚ)*((ωₚ / ω) - 1.0) + 1.0))
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
        out[2, 1] += ∂S∂α * hxz
        out[3, 1] += ∂S∂α * hyz
        out[5, 1] += ∂S∂α * hyx
        out[4, 1] += ∂S∂α * hxx
        out[6, 1] += ∂S∂α * hyy

        ## ωₚ
        out[1, 2] += ∂S∂ωₚ
        out[2, 2] += ∂S∂ωₚ * hxz + sdf * hxz∂ωₚ
        out[3, 2] += ∂S∂ωₚ * hyz + sdf * hyz∂ωₚ
        out[4, 2] += ∂S∂ωₚ * hxx + sdf * hxx∂ωₚ
        out[5, 2] += ∂S∂ωₚ * hyx + sdf * hyx∂ωₚ
        out[6, 2] += ∂S∂ωₚ * hyy + sdf * hyy∂ωₚ

        ## γ
        out[1, 3] += ∂S∂γ
        out[2, 3] += ∂S∂γ * hxz
        out[3, 3] += ∂S∂γ * hyz
        out[4, 3] += ∂S∂γ * hxx
        out[5, 3] += ∂S∂γ * hyx
        out[6, 3] += ∂S∂γ * hyy

        ## r
        out[1, 4] += ∂S∂r
        out[2, 4] += ∂S∂r * hxz
        out[3, 4] += ∂S∂r * hyz
        out[4, 4] += ∂S∂r * hxx
        out[5, 4] += ∂S∂r * hyx
        out[6, 4] += ∂S∂r * hyy

        ## ϕₘ
        # out[1, 5] += 0
        out[2, 5] += sdf * hxz∂ϕₘ
        out[3, 5] += sdf * hyz∂ϕₘ
        out[4, 5] += sdf * hxx∂ϕₘ
        out[5, 5] += sdf * hyx∂ϕₘ
        out[6, 5] += sdf * hyy∂ϕₘ

        ## β
        # out[1, 6] += 0
        out[2, 6] += sdf * hxz∂β
        out[3, 6] += sdf * hyz∂β
        out[4, 6] += sdf * hxx∂β
        out[5, 6] += sdf * hyx∂β
        out[6, 6] += sdf * hyy∂β

        ## ν
        # out[1, 7] += 0
        out[2, 7] += sdf * hxz∂ν
        out[3, 7] += sdf * hyz∂ν
        out[4, 7] += sdf * hxx∂ν
        out[5, 7] += sdf * hyx∂ν
        out[6, 7] += sdf * hyy∂ν

        ## σₗ
        # out[1, 8] += 0
        out[2, 8] += sdf * hxz∂σₗ
        out[3, 8] += sdf * hyz∂σₗ
        out[4, 8] += sdf * hxx∂σₗ
        out[5, 8] += sdf * hyx∂σₗ
        out[6, 8] += sdf * hyy∂σₗ

        ## σᵣ
        # out[1, 9] += 0
        out[2, 9] += sdf * hxz∂σᵣ
        out[3, 9] += sdf * hyz∂σᵣ
        out[4, 9] += sdf * hxx∂σᵣ
        out[5, 9] += sdf * hyx∂σᵣ
        out[6, 9] += sdf * hyy∂σᵣ

    end # add zero otherwise

    return nothing
end