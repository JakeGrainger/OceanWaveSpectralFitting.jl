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
JS_BWG_HNE(α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ) = error("JS_BWG_HNE process requires the ammount of aliasing specified as a type parameter. Use JS_BWG_HNE{K}() where K ∈ N₀.")

WhittleLikelihoodInference.npars(::Type{JS_BWG_HNE{K}}) where {K} = 9
WhittleLikelihoodInference.nalias(::JS_BWG_HNE{K}) where {K} = K

function WhittleLikelihoodInference.add_sdf!(out, model::JS_BWG_HNE{K}, ω::Real) where {K}
    s_om = sign(ω)
    ω = abs(ω)

    if ω > 1e-10
        α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ = model.α,model.ωₚ,model.γ,model.r,model.ϕₘ,model.β,model.ν,model.σₗ,model.σᵣ
        
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        ω_over_ωₚ = ω / ωₚ
        ωₚ_over_ω = 1/ω_over_ωₚ
        δ = exp(-1 / (2σ1²) * (ω_over_ωₚ - 1)^2)
        ω⁻⁴ = ω^(-4)
        ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
        sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
        
        PeakSep = β / exp(ν * ((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0))
        σ = σₗ - model.σᵣ_over3*(4*(ωₚ_over_ω)^2 - (ωₚ⁴_over_ω⁴)^2)
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
        ωₚ_over_ω = 1/ω_over_ωₚ
        δ = exp(-1 / (2σ1²) * (ω_over_ωₚ - 1)^2)
        ω⁻⁴ = ω^(-4)
        ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
        sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
        
        ∂S∂α = sdf / α
        ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*model.ωₚ³*ω⁻⁴)
        ∂S∂γ = sdf * δ / γ
        ∂S∂r = sdf * (-log(ω)-ωₚ⁴_over_ω⁴/4)
        
        PeakSep = β / exp(ν * ((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0))
        σ = σₗ - model.σᵣ_over3*(4*(ωₚ_over_ω)^2 - (ωₚ⁴_over_ω⁴)^2)
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

function WhittleLikelihoodInference.hess_add_sdf!(out, model::JS_BWG_HNE{K}, ω::Real) where {K}
    
    s_om = sign(ω)
    ω = abs(ω)
    
    if ω > 1e-10
        α,ωₚ,γ,r,ϕₘ,β,ν,σₗ,σᵣ = model.α,model.ωₚ,model.γ,model.r,model.ϕₘ,model.β,model.ν,model.σₗ,model.σᵣ
        
        σ1² = 0.0049 + 0.0032 * (ω > ωₚ)
        ω_over_ωₚ = ω / ωₚ
        ωₚ_over_ω = 1/ω_over_ωₚ
        δ = exp(-1 / (2σ1²) * (ω_over_ωₚ - 1)^2)
        ω⁻⁴ = ω^(-4)
        ωₚ⁴_over_ω⁴ = model.ωₚ⁴ * ω⁻⁴
        sdf = (α * ω^(-r) * exp(-(model.r_over4) * ωₚ⁴_over_ω⁴) * γ^δ) / 2
        
        ∂S∂α = sdf / α
        ∂S∂ωₚ = sdf * (δ*model.logγ * ω / σ1² *(ω-ωₚ) / model.ωₚ³ - r*model.ωₚ³*ω⁻⁴)
        ∂S∂γ = sdf * δ / γ
        ∂S∂r = sdf * (-log(ω)-ωₚ⁴_over_ω⁴/4)
        
        PeakSep = β / exp(ν * ((ω > ωₚ)*((ωₚ_over_ω) - 1.0) + 1.0))
        σ = σₗ - model.σᵣ_over3*(4*(ωₚ_over_ω)^2 - (ωₚ⁴_over_ω⁴)^2)
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

        ## Peak seperation derivatives
        PeakSep∂β∂ν = PeakSep∂ν/β
        PeakSep∂β∂ωₚ = PeakSep∂ωₚ/β
        PeakSep∂β2 = 0
        PeakSep∂ν∂ωₚ = -(ω > ωₚ) * (ωₚ_over_ω * PeakSep∂ωₚ + PeakSep/ω)
        PeakSep∂ν2 = (((ωₚ_over_ω)^2 - 1) * (ω > ωₚ) + 1) * PeakSep
        PeakSep∂ωₚ2 = (ν/ω) ^ 2 * PeakSep * (ω > ωₚ)
        ## sigma derivatives
        ## most of these are 0, except
        ∂σ∂σᵣ∂ωₚ = ∂σ∂ωₚ / σᵣ
        ∂σ∂ωₚ2 = (-8σᵣ/3*(1/(ω^2)-7*(model.ωₚ⁶)/(ω^8)))

        ## g1
        g1 = -0.5 / exp2part * sinPS
        g1_β = g1 * PeakSep∂β
        g1_ν = g1 * PeakSep∂ν
        g1_ωₚ = g1 * PeakSep∂ωₚ

        minus_exp2part_over2 = -1/exp2part/2
        g1_β2 = minus_exp2part_over2 * PeakSep∂β^2 * cosPS
        g1_βν = minus_exp2part_over2 * (PeakSep∂β∂ν * sinPS + PeakSep∂β * PeakSep∂ν * cosPS)
        g1_βσₗ = 2/exp2part * sinPS * PeakSep∂β * σ
        g1_βσᵣ = g1_βσₗ * ∂σ∂σᵣ

        g1_ν2 = minus_exp2part_over2 * (PeakSep∂ν2 * sinPS + PeakSep∂ν^2 * cosPS)
        g1_νσₗ = 2/exp2part * sinPS * PeakSep∂ν * σ
        g1_νσᵣ = g1_νσₗ * ∂σ∂σᵣ

        g1_ωₚ2 = 1/exp2part * (-sinPS/2 * PeakSep∂ωₚ2 + PeakSep∂ωₚ * (2σ * ∂σ∂ωₚ * sinPS - 0.5 * PeakSep∂ωₚ * cosPS))

        g1_ωₚβ = minus_exp2part_over2 * (PeakSep∂β∂ωₚ * sinPS + PeakSep∂β * PeakSep∂ωₚ * cosPS)
        g1_ωₚν = minus_exp2part_over2 * (PeakSep∂ν∂ωₚ * sinPS + PeakSep∂ωₚ * PeakSep∂ν * cosPS)
        g1_ωₚσₗ = 2/exp2part * sinPS * PeakSep∂ωₚ * σ
        g1_ωₚσᵣ = g1_ωₚσₗ * ∂σ∂σᵣ
        
        ## g2
        g2_σₗ = -2 / exp2part * cosPS * σ
        g2_σᵣ = g2_σₗ * ∂σ∂σᵣ
        g2_ωₚ = g2_σₗ * ∂σ∂ωₚ

        one_minus_4σ² = (1 - 4σ²)
        g2_σₗ2 = -2 * cosPS / exp2part * one_minus_4σ²
        g2_σₗσᵣ = g2_σₗ2 * ∂σ∂σᵣ 
        g2_σᵣ2 = g2_σₗσᵣ * ∂σ∂σᵣ
        g2_ωₚ2 = 2/exp2part * (PeakSep∂ωₚ * sinPS * ∂σ∂ωₚ * σ - cosPS * (∂σ∂ωₚ2 * σ + ∂σ∂ωₚ^2 * one_minus_4σ²))

        g2_ωₚβ = 2/exp2part * sinPS * PeakSep∂β * ∂σ∂ωₚ * σ
        g2_ωₚν = 2/exp2part * sinPS * PeakSep∂ν * ∂σ∂ωₚ * σ
        g2_ωₚσₗ = -2 * cosPS / exp2part * (∂σ∂ωₚ * one_minus_4σ²)
        g2_ωₚσᵣ = -2 * cosPS / exp2part * (σ * ∂σ∂σᵣ∂ωₚ + ∂σ∂σᵣ * ∂σ∂ωₚ * one_minus_4σ²)

        ## g3
        g3 = s_om * 0.5im / exphalf * sinPShalf
        g3_β = g3 * PeakSep∂β
        g3_ν = g3 * PeakSep∂ν
        g3_ωₚ = g3 * PeakSep∂ωₚ

        g3_β2 = s_om * 0.5im / exphalf * (PeakSep∂β2 * sinPShalf + PeakSep∂β^2 * cosPShalf / 2)
        g3_βν = s_om * 0.5im / exphalf * (PeakSep∂β∂ν * sinPShalf + PeakSep∂β * PeakSep∂ν * cosPShalf / 2)
        g3_ν2 = s_om * 0.5im / exphalf * (PeakSep∂ν2 * sinPShalf + PeakSep∂ν^2 * cosPShalf / 2)
        g3_ωₚ2 = s_om * 0.5im / exphalf * ((PeakSep∂ωₚ2 - σ * ∂σ∂ωₚ * PeakSep∂ωₚ) * sinPShalf + PeakSep∂ωₚ^2*cosPShalf/2)
        g3_βσₗ = -s_om * 0.5im / exphalf * sinPShalf * PeakSep∂β * σ
        g3_βσᵣ = g3_βσₗ * ∂σ∂σᵣ
        g3_νσₗ = -s_om * 0.5im / exphalf * sinPShalf * PeakSep∂ν * σ
        g3_νσᵣ = g3_νσₗ * ∂σ∂σᵣ

        g3_ωₚβ = s_om * 1im/exphalf/2 * (PeakSep∂β∂ωₚ * sinPShalf + PeakSep∂β * PeakSep∂ωₚ * cosPShalf/2)
        g3_ωₚν = s_om * 1im/exphalf/2 * (PeakSep∂ν∂ωₚ * sinPShalf + PeakSep∂ωₚ * PeakSep∂ν * cosPShalf/2)
        g3_ωₚσₗ = -s_om * 1im/exphalf/2 * sinPShalf * PeakSep∂ωₚ * σ
        g3_ωₚσᵣ = g3_ωₚσₗ * ∂σ∂σᵣ
        ## g4
        g4 = s_om * 1im / exphalf * cosPShalf * σ
        g4_σₗ = g4
        g4_σᵣ = g4 * ∂σ∂σᵣ
        g4_ωₚ = g4 * ∂σ∂ωₚ

        g4_σₗ2 = s_om * 1im * cosPShalf / exphalf * (1-σ²) 
        g4_σₗσᵣ = g4_σₗ2 * ∂σ∂σᵣ 
        g4_σᵣ2 = g4_σₗσᵣ * ∂σ∂σᵣ 
        g4_ωₚ2 = -s_om * 1im / exphalf * (PeakSep∂ωₚ * ∂σ∂ωₚ * sinPShalf * σ/2 - cosPShalf * (∂σ∂ωₚ2*σ + ∂σ∂ωₚ^2 * (1 - σ²)) ) 

        g4_ωₚβ = -s_om * 0.5im/exphalf * sinPShalf * PeakSep∂β * ∂σ∂ωₚ * σ
        g4_ωₚν = -s_om * 0.5im/exphalf * sinPShalf * PeakSep∂ν * ∂σ∂ωₚ * σ
        g4_ωₚσₗ = s_om * 1im * cosPShalf / exphalf * (∂σ∂ωₚ * (1 - σ²))
        g4_ωₚσᵣ = s_om * 1im * cosPShalf / exphalf * (σ * ∂σ∂σᵣ∂ωₚ + ∂σ∂σᵣ * ∂σ∂ωₚ * (1 - σ²))
        
        ## finished required g functions
        
        #### hxx

        ## ϕₘ (6)
        hxx∂ϕₘbit = -2sin2ϕₘ
        hxx∂ϕₘ2 = -2cos2ϕₘ * hyxpart∂ϕₘ
        hxx∂ϕₘ∂β = hxx∂ϕₘbit * g1_β
        hxx∂ϕₘ∂ν = hxx∂ϕₘbit * g1_ν
        hxx∂ϕₘ∂σₗ = hxx∂ϕₘbit * g2_σₗ
        hxx∂ϕₘ∂σᵣ = hxx∂ϕₘbit * g2_σᵣ
        hxx∂ϕₘ∂ωₚ = hxx∂ϕₘbit * (g1_ωₚ + g2_ωₚ)
        ## ωₚ (5)
        hxx∂ωₚ2 = cos2ϕₘ * (g1_ωₚ2 + g2_ωₚ2)
        hxx∂ωₚ∂β = cos2ϕₘ * (g1_ωₚβ + g2_ωₚβ)
        hxx∂ωₚ∂ν = cos2ϕₘ * (g1_ωₚν + g2_ωₚν)
        hxx∂ωₚ∂σₗ = cos2ϕₘ * (g1_ωₚσₗ + g2_ωₚσₗ)
        hxx∂ωₚ∂σᵣ = cos2ϕₘ * (g1_ωₚσᵣ + g2_ωₚσᵣ)

        ## β (4)
        hxx∂β2 = cos2ϕₘ * g1_β2
        hxx∂β∂ν = cos2ϕₘ * g1_βν
        hxx∂β∂σₗ = cos2ϕₘ * g1_βσₗ
        hxx∂β∂σᵣ = cos2ϕₘ * g1_βσᵣ
        ## ν (3)
        hxx∂ν2 = cos2ϕₘ * g1_ν2
        hxx∂ν∂σₗ = cos2ϕₘ * g1_νσₗ
        hxx∂ν∂σᵣ = cos2ϕₘ * g1_νσᵣ
        ## σₗ (2)
        hxx∂σₗ2 = cos2ϕₘ * g2_σₗ2
        hxx∂σₗ∂σᵣ = cos2ϕₘ * g2_σₗσᵣ
        ## σᵣ (1)
        hxx∂σᵣ2 = cos2ϕₘ * g2_σᵣ2

        #### hyx

        ## ϕₘ (6)
        hyx∂ϕₘ2 = -2sin2ϕₘ * hyxpart∂ϕₘ
        hyx∂ϕₘ∂β = 2cos2ϕₘ * g1_β
        hyx∂ϕₘ∂ν = 2cos2ϕₘ * g1_ν
        hyx∂ϕₘ∂σₗ = 2cos2ϕₘ * g2_σₗ
        hyx∂ϕₘ∂σᵣ = 2cos2ϕₘ * g2_σᵣ
        hyx∂ϕₘ∂ωₚ = 2cos2ϕₘ * (g1_ωₚ + g2_ωₚ)
        ## ωₚ (5)
        hyx∂ωₚ2 = sin2ϕₘ * (g1_ωₚ2 + g2_ωₚ2)
        hyx∂ωₚ∂β = sin2ϕₘ * (g1_ωₚβ + g2_ωₚβ)
        hyx∂ωₚ∂ν = sin2ϕₘ * (g1_ωₚν + g2_ωₚν)
        hyx∂ωₚ∂σₗ = sin2ϕₘ * (g1_ωₚσₗ + g2_ωₚσₗ)
        hyx∂ωₚ∂σᵣ = sin2ϕₘ * (g1_ωₚσᵣ + g2_ωₚσᵣ)

        ## β (4)
        hyx∂β2 = sin2ϕₘ * g1_β2
        hyx∂β∂ν = sin2ϕₘ * g1_βν
        hyx∂β∂σₗ = sin2ϕₘ * g1_βσₗ
        hyx∂β∂σᵣ = sin2ϕₘ * g1_βσᵣ
        ## ν (3)
        hyx∂ν2 = sin2ϕₘ * g1_ν2
        hyx∂ν∂σₗ = sin2ϕₘ * g1_νσₗ
        hyx∂ν∂σᵣ = sin2ϕₘ * g1_νσᵣ
        ## σₗ (2)
        hyx∂σₗ2 = sin2ϕₘ * g2_σₗ2
        hyx∂σₗ∂σᵣ = sin2ϕₘ * g2_σₗσᵣ
        ## σᵣ (1)
        hyx∂σᵣ2 = sin2ϕₘ * g2_σᵣ2
        
        #### hxz

        ## ϕₘ (6)
        hxz∂ϕₘ2 = cosϕₘ * hzpart∂ϕₘ # minus already added here
        hxz∂ϕₘbit = sinϕₘ # double negative
        hxz∂ϕₘ∂β = hxz∂ϕₘbit * g3_β
        hxz∂ϕₘ∂ν = hxz∂ϕₘbit * g3_ν
        hxz∂ϕₘ∂σₗ = hxz∂ϕₘbit * g4_σₗ
        hxz∂ϕₘ∂σᵣ = hxz∂ϕₘbit * g4_σᵣ
        hxz∂ϕₘ∂ωₚ = hxz∂ϕₘbit * (g3_ωₚ + g4_ωₚ)
        ## ωₚ (5)
        hxz∂ωₚ2 = -cosϕₘ * (g3_ωₚ2 + g4_ωₚ2)
        hxz∂ωₚ∂β = -cosϕₘ * (g3_ωₚβ + g4_ωₚβ)
        hxz∂ωₚ∂ν = -cosϕₘ * (g3_ωₚν + g4_ωₚν)
        hxz∂ωₚ∂σₗ = -cosϕₘ * (g3_ωₚσₗ + g4_ωₚσₗ)
        hxz∂ωₚ∂σᵣ = -cosϕₘ * (g3_ωₚσᵣ + g4_ωₚσᵣ)

        ## β (4)
        hxz∂β2 = -cosϕₘ * g3_β2
        hxz∂β∂ν = -cosϕₘ * g3_βν
        hxz∂β∂σₗ = -cosϕₘ * g3_βσₗ
        hxz∂β∂σᵣ = -cosϕₘ * g3_βσᵣ
        ## ν (3)
        hxz∂ν2 = -cosϕₘ * g3_ν2
        hxz∂ν∂σₗ = -cosϕₘ * g3_νσₗ
        hxz∂ν∂σᵣ = -cosϕₘ * g3_νσᵣ
        ## σₗ (2)
        hxz∂σₗ2 = -cosϕₘ * g4_σₗ2
        hxz∂σₗ∂σᵣ = -cosϕₘ * g4_σₗσᵣ
        ## σᵣ (1)
        hxz∂σᵣ2 = -cosϕₘ * g4_σᵣ2
        
        #### hyz

        ## ϕₘ (6)
        hyz∂ϕₘ2 = sinϕₘ * hzpart∂ϕₘ
        hyz∂ϕₘbit = -cosϕₘ  # add in minus for conjugate
        hyz∂ϕₘ∂β = hyz∂ϕₘbit * g3_β
        hyz∂ϕₘ∂ν = hyz∂ϕₘbit * g3_ν
        hyz∂ϕₘ∂σₗ = hyz∂ϕₘbit * g4_σₗ
        hyz∂ϕₘ∂σᵣ = hyz∂ϕₘbit * g4_σᵣ
        hyz∂ϕₘ∂ωₚ = hyz∂ϕₘbit * (g3_ωₚ + g4_ωₚ)
        ## ωₚ (5)
        hyz∂ωₚ2 = -sinϕₘ * (g3_ωₚ2 + g4_ωₚ2)
        hyz∂ωₚ∂β = -sinϕₘ * (g3_ωₚβ + g4_ωₚβ)
        hyz∂ωₚ∂ν = -sinϕₘ * (g3_ωₚν + g4_ωₚν)
        hyz∂ωₚ∂σₗ = -sinϕₘ * (g3_ωₚσₗ + g4_ωₚσₗ)
        hyz∂ωₚ∂σᵣ = -sinϕₘ * (g3_ωₚσᵣ + g4_ωₚσᵣ)

        ## β (4)
        hyz∂β2 = -sinϕₘ * g3_β2
        hyz∂β∂ν = -sinϕₘ * g3_βν
        hyz∂β∂σₗ = -sinϕₘ * g3_βσₗ
        hyz∂β∂σᵣ = -sinϕₘ * g3_βσᵣ
        ## ν (3)
        hyz∂ν2 = -sinϕₘ * g3_ν2
        hyz∂ν∂σₗ = -sinϕₘ * g3_νσₗ
        hyz∂ν∂σᵣ = -sinϕₘ * g3_νσᵣ
        ## σₗ (2)
        hyz∂σₗ2 = -sinϕₘ * g4_σₗ2
        hyz∂σₗ∂σᵣ = -sinϕₘ * g4_σₗσᵣ
        ## σᵣ (1)
        hyz∂σᵣ2 = -sinϕₘ * g4_σᵣ2

        #### out

        ## α2
        # out[1, 1] += 0
        # out[2, 1] += 0
        # out[3, 1] += 0
        # out[4, 1] += 0
        # out[5, 1] += 0
        # out[6, 1] += 0

        ## α ωₚ
        ∂S∂αωₚ = ∂S∂ωₚ / α
        out[1, 2] += ∂S∂αωₚ
        out[2, 2] += ∂S∂αωₚ * hxz + ∂S∂α * hxz∂ωₚ
        out[3, 2] += ∂S∂αωₚ * hyz + ∂S∂α * hyz∂ωₚ
        out[4, 2] += ∂S∂αωₚ * hxx + ∂S∂α * hxx∂ωₚ
        out[5, 2] += ∂S∂αωₚ * hyx + ∂S∂α * hyx∂ωₚ
        out[6, 2] += ∂S∂αωₚ * hyy + ∂S∂α * hyy∂ωₚ

        ## α γ
        ∂S∂αγ = ∂S∂γ / α
        out[1, 3] += ∂S∂αγ
        out[2, 3] += ∂S∂αγ * hxz
        out[3, 3] += ∂S∂αγ * hyz
        out[4, 3] += ∂S∂αγ * hxx
        out[5, 3] += ∂S∂αγ * hyx
        out[6, 3] += ∂S∂αγ * hyy

        ## α r
        ∂S∂αr = ∂S∂r / α
        out[1, 4] += ∂S∂αr
        out[2, 4] += ∂S∂αr * hxz
        out[3, 4] += ∂S∂αr * hyz
        out[4, 4] += ∂S∂αr * hxx
        out[5, 4] += ∂S∂αr * hyx
        out[6, 4] += ∂S∂αr * hyy

        ## α ϕₘ
        # out[1, 5] += 0
        out[2, 5] += ∂S∂α * hxz∂ϕₘ
        out[3, 5] += ∂S∂α * hyz∂ϕₘ
        out[4, 5] += ∂S∂α * hxx∂ϕₘ
        out[5, 5] += ∂S∂α * hyx∂ϕₘ
        out[6, 5] += ∂S∂α * hyy∂ϕₘ

        ## α β
        # out[1, 6] += 0
        out[2, 6] += ∂S∂α * hxz∂β
        out[3, 6] += ∂S∂α * hyz∂β
        out[4, 6] += ∂S∂α * hxx∂β
        out[5, 6] += ∂S∂α * hyx∂β
        out[6, 6] += ∂S∂α * hyy∂β

        ## α ν
        # out[1, 7] += 0
        out[2, 7] += ∂S∂α * hxz∂ν
        out[3, 7] += ∂S∂α * hyz∂ν
        out[4, 7] += ∂S∂α * hxx∂ν
        out[5, 7] += ∂S∂α * hyx∂ν
        out[6, 7] += ∂S∂α * hyy∂ν

        ## α σₗ
        # out[1, 8] += 0
        out[2, 8] += ∂S∂α * hxz∂σₗ
        out[3, 8] += ∂S∂α * hyz∂σₗ
        out[4, 8] += ∂S∂α * hxx∂σₗ
        out[5, 8] += ∂S∂α * hyx∂σₗ
        out[6, 8] += ∂S∂α * hyy∂σₗ

        ## α σᵣ
        # out[1, 9] += 0
        out[2, 9] += ∂S∂α * hxz∂σᵣ
        out[3, 9] += ∂S∂α * hyz∂σᵣ
        out[4, 9] += ∂S∂α * hxx∂σᵣ
        out[5, 9] += ∂S∂α * hyx∂σᵣ
        out[6, 9] += ∂S∂α * hyy∂σᵣ

        ## ωₚ2
        ∂S∂ωₚ2part = δ * log(γ) * ω / σ1²
        ∂S∂ωₚUsepart = (∂S∂ωₚ2part * (ω-ωₚ) / (ωₚ^3)) - (r * (ωₚ^3) / (ω^4))
        ∂S∂ωₚ2 = ∂S∂ωₚ * ∂S∂ωₚUsepart + sdf * (∂S∂ωₚ2part *((-3ω + 2ωₚ)/ωₚ^4 + ω * (ω-ωₚ)^2/ωₚ^6/σ1²) - 3r*ωₚ^2/ω^4)
        out[1, 10] += ∂S∂ωₚ2
        out[2, 10] += ∂S∂ωₚ2 * hxz + 2∂S∂ωₚ * hxz∂ωₚ + sdf*hxz∂ωₚ2
        out[3, 10] += ∂S∂ωₚ2 * hyz + 2∂S∂ωₚ * hyz∂ωₚ + sdf*hyz∂ωₚ2
        out[4, 10] += ∂S∂ωₚ2 * hxx + 2∂S∂ωₚ * hxx∂ωₚ + sdf*hxx∂ωₚ2
        out[5, 10] += ∂S∂ωₚ2 * hyx + 2∂S∂ωₚ * hyx∂ωₚ + sdf*hyx∂ωₚ2
        out[6, 10] += ∂S∂ωₚ2 * hyy + 2∂S∂ωₚ * hyy∂ωₚ - sdf*hxx∂ωₚ2
        
        ## ωₚγ
        ∂S∂ωₚγ = ∂S∂γ * ∂S∂ωₚUsepart + sdf * δ / γ * ω / σ1²*(ω-ωₚ)/ωₚ^3
        out[1, 11] += ∂S∂ωₚγ
        out[2, 11] += ∂S∂ωₚγ * hxz + ∂S∂γ * hxz∂ωₚ
        out[3, 11] += ∂S∂ωₚγ * hyz + ∂S∂γ * hyz∂ωₚ
        out[4, 11] += ∂S∂ωₚγ * hxx + ∂S∂γ * hxx∂ωₚ
        out[5, 11] += ∂S∂ωₚγ * hyx + ∂S∂γ * hyx∂ωₚ
        out[6, 11] += ∂S∂ωₚγ * hyy + ∂S∂γ * hyy∂ωₚ

        ## ωₚr
        ∂S∂ωₚr = ∂S∂r * ∂S∂ωₚUsepart - sdf * ωₚ^3/ω^4
        out[1, 12] += ∂S∂ωₚr
        out[2, 12] += ∂S∂ωₚr * hxz + ∂S∂r * hxz∂ωₚ
        out[3, 12] += ∂S∂ωₚr * hyz + ∂S∂r * hyz∂ωₚ
        out[4, 12] += ∂S∂ωₚr * hxx + ∂S∂r * hxx∂ωₚ
        out[5, 12] += ∂S∂ωₚr * hyx + ∂S∂r * hyx∂ωₚ
        out[6, 12] += ∂S∂ωₚr * hyy + ∂S∂r * hyy∂ωₚ

        ## ωₚϕₘ
        # out[1, 13] += 0
        out[2, 13] += ∂S∂ωₚ * hxz∂ϕₘ + sdf * hxz∂ϕₘ∂ωₚ
        out[3, 13] += ∂S∂ωₚ * hyz∂ϕₘ + sdf * hyz∂ϕₘ∂ωₚ
        out[4, 13] += ∂S∂ωₚ * hxx∂ϕₘ + sdf * hxx∂ϕₘ∂ωₚ
        out[5, 13] += ∂S∂ωₚ * hyx∂ϕₘ + sdf * hyx∂ϕₘ∂ωₚ
        out[6, 13] += ∂S∂ωₚ * hyy∂ϕₘ - sdf * hxx∂ϕₘ∂ωₚ

        ## ωₚβ
        # out[1, 14] += 0
        out[2, 14] += ∂S∂ωₚ * hxz∂β + sdf * hxz∂ωₚ∂β
        out[3, 14] += ∂S∂ωₚ * hyz∂β + sdf * hyz∂ωₚ∂β
        out[4, 14] += ∂S∂ωₚ * hxx∂β + sdf * hxx∂ωₚ∂β
        out[5, 14] += ∂S∂ωₚ * hyx∂β + sdf * hyx∂ωₚ∂β
        out[6, 14] += ∂S∂ωₚ * hyy∂β - sdf * hxx∂ωₚ∂β

        ## ωₚν
        # out[1, 15] += 0
        out[2, 15] += ∂S∂ωₚ * hxz∂ν + sdf * hxz∂ωₚ∂ν
        out[3, 15] += ∂S∂ωₚ * hyz∂ν + sdf * hyz∂ωₚ∂ν
        out[4, 15] += ∂S∂ωₚ * hxx∂ν + sdf * hxx∂ωₚ∂ν
        out[5, 15] += ∂S∂ωₚ * hyx∂ν + sdf * hyx∂ωₚ∂ν
        out[6, 15] += ∂S∂ωₚ * hyy∂ν - sdf * hxx∂ωₚ∂ν

        ## ωₚσₗ
        # out[1, 16] += 0
        out[2, 16] += ∂S∂ωₚ * hxz∂σₗ + sdf * hxz∂ωₚ∂σₗ
        out[3, 16] += ∂S∂ωₚ * hyz∂σₗ + sdf * hyz∂ωₚ∂σₗ
        out[4, 16] += ∂S∂ωₚ * hxx∂σₗ + sdf * hxx∂ωₚ∂σₗ
        out[5, 16] += ∂S∂ωₚ * hyx∂σₗ + sdf * hyx∂ωₚ∂σₗ
        out[6, 16] += ∂S∂ωₚ * hyy∂σₗ - sdf * hxx∂ωₚ∂σₗ

        ## ωₚσᵣ
        # out[1, 17] += 0
        out[2, 17] += ∂S∂ωₚ * hxz∂σᵣ + sdf * hxz∂ωₚ∂σᵣ
        out[3, 17] += ∂S∂ωₚ * hyz∂σᵣ + sdf * hyz∂ωₚ∂σᵣ
        out[4, 17] += ∂S∂ωₚ * hxx∂σᵣ + sdf * hxx∂ωₚ∂σᵣ
        out[5, 17] += ∂S∂ωₚ * hyx∂σᵣ + sdf * hyx∂ωₚ∂σᵣ
        out[6, 17] += ∂S∂ωₚ * hyy∂σᵣ - sdf * hxx∂ωₚ∂σᵣ

        ## γ2
        ∂S∂γ2 = δ/γ * (∂S∂γ - sdf/γ)
        out[1, 18] += ∂S∂γ2
        out[2, 18] += ∂S∂γ2 * hxz
        out[3, 18] += ∂S∂γ2 * hyz
        out[4, 18] += ∂S∂γ2 * hxx
        out[5, 18] += ∂S∂γ2 * hyx
        out[6, 18] += ∂S∂γ2 * hyy

        ## γr
        ∂S∂γr = δ/γ * ∂S∂r
        out[1, 19] += ∂S∂γr
        out[2, 19] += ∂S∂γr * hxz
        out[3, 19] += ∂S∂γr * hyz
        out[4, 19] += ∂S∂γr * hxx
        out[5, 19] += ∂S∂γr * hyx
        out[6, 19] += ∂S∂γr * hyy

        ## γϕₘ
        # out[1, 20] += 0
        out[2, 20] += ∂S∂γ * hxz∂ϕₘ
        out[3, 20] += ∂S∂γ * hyz∂ϕₘ
        out[4, 20] += ∂S∂γ * hxx∂ϕₘ
        out[5, 20] += ∂S∂γ * hyx∂ϕₘ
        out[6, 20] += ∂S∂γ * hyy∂ϕₘ

        ## γβ
        # out[1, 21] += 0
        out[2, 21] += ∂S∂γ * hxz∂β
        out[3, 21] += ∂S∂γ * hyz∂β
        out[4, 21] += ∂S∂γ * hxx∂β
        out[5, 21] += ∂S∂γ * hyx∂β
        out[6, 21] += ∂S∂γ * hyy∂β

        ## γν
        # out[1, 22] += 0
        out[2, 22] += ∂S∂γ * hxz∂ν
        out[3, 22] += ∂S∂γ * hyz∂ν
        out[4, 22] += ∂S∂γ * hxx∂ν
        out[5, 22] += ∂S∂γ * hyx∂ν
        out[6, 22] += ∂S∂γ * hyy∂ν

        ## γσₗ
        # out[1, 23] += 0
        out[2, 23] += ∂S∂γ * hxz∂σₗ
        out[3, 23] += ∂S∂γ * hyz∂σₗ
        out[4, 23] += ∂S∂γ * hxx∂σₗ
        out[5, 23] += ∂S∂γ * hyx∂σₗ
        out[6, 23] += ∂S∂γ * hyy∂σₗ

        ## γσᵣ
        # out[1, 24] += 0
        out[2, 24] += ∂S∂γ * hxz∂σᵣ
        out[3, 24] += ∂S∂γ * hyz∂σᵣ
        out[4, 24] += ∂S∂γ * hxx∂σᵣ
        out[5, 24] += ∂S∂γ * hyx∂σᵣ
        out[6, 24] += ∂S∂γ * hyy∂σᵣ
        
        ##  r2
        ∂S∂r2 = (-log(ω) - 0.25*(ωₚ/ω)^4) * ∂S∂r
        out[1, 25] += ∂S∂r2
        out[2, 25] += ∂S∂r2 * hxz
        out[3, 25] += ∂S∂r2 * hyz
        out[4, 25] += ∂S∂r2 * hxx
        out[5, 25] += ∂S∂r2 * hyx
        out[6, 25] += ∂S∂r2 * hyy

        ## rϕₘ
        # out[1, 26] += 0
        out[2, 26] += ∂S∂r * hxz∂ϕₘ
        out[3, 26] += ∂S∂r * hyz∂ϕₘ
        out[4, 26] += ∂S∂r * hxx∂ϕₘ
        out[5, 26] += ∂S∂r * hyx∂ϕₘ
        out[6, 26] += ∂S∂r * hyy∂ϕₘ

        ## rβ
        # out[1, 27] += 0
        out[2, 27] += ∂S∂r * hxz∂β
        out[3, 27] += ∂S∂r * hyz∂β
        out[4, 27] += ∂S∂r * hxx∂β
        out[5, 27] += ∂S∂r * hyx∂β
        out[6, 27] += ∂S∂r * hyy∂β

        ## rν
        # out[1, 28] += 0
        out[2, 28] += ∂S∂r * hxz∂ν
        out[3, 28] += ∂S∂r * hyz∂ν
        out[4, 28] += ∂S∂r * hxx∂ν
        out[5, 28] += ∂S∂r * hyx∂ν
        out[6, 28] += ∂S∂r * hyy∂ν

        ## rσₗ
        # out[1, 29] += 0
        out[2, 29] += ∂S∂r * hxz∂σₗ
        out[3, 29] += ∂S∂r * hyz∂σₗ
        out[4, 29] += ∂S∂r * hxx∂σₗ
        out[5, 29] += ∂S∂r * hyx∂σₗ
        out[6, 29] += ∂S∂r * hyy∂σₗ

        ## rσᵣ
        # out[1, 30] += 0
        out[2, 30] += ∂S∂r * hxz∂σᵣ
        out[3, 30] += ∂S∂r * hyz∂σᵣ
        out[4, 30] += ∂S∂r * hxx∂σᵣ
        out[5, 30] += ∂S∂r * hyx∂σᵣ
        out[6, 30] += ∂S∂r * hyy∂σᵣ

        ## ϕₘ2
        # out[1, 31] += 0
        out[2, 31] += sdf * hxz∂ϕₘ2
        out[3, 31] += sdf * hyz∂ϕₘ2
        out[4, 31] += sdf * hxx∂ϕₘ2
        out[5, 31] += sdf * hyx∂ϕₘ2
        out[6, 31] += sdf * -hxx∂ϕₘ2

        ## ϕₘβ
        # out[1, 32] += 0
        out[2, 32] += sdf * hxz∂ϕₘ∂β
        out[3, 32] += sdf * hyz∂ϕₘ∂β
        out[4, 32] += sdf * hxx∂ϕₘ∂β
        out[5, 32] += sdf * hyx∂ϕₘ∂β
        out[6, 32] += sdf * -hxx∂ϕₘ∂β

        ## ϕₘν
        # out[1, 33] += 0
        out[2, 33] += sdf * hxz∂ϕₘ∂ν
        out[3, 33] += sdf * hyz∂ϕₘ∂ν
        out[4, 33] += sdf * hxx∂ϕₘ∂ν    
        out[5, 33] += sdf * hyx∂ϕₘ∂ν
        out[6, 33] += sdf * -hxx∂ϕₘ∂ν

        ## ϕₘσₗ
        # out[1, 34] += 0
        out[2, 34] += sdf * hxz∂ϕₘ∂σₗ
        out[3, 34] += sdf * hyz∂ϕₘ∂σₗ
        out[4, 34] += sdf * hxx∂ϕₘ∂σₗ
        out[5, 34] += sdf * hyx∂ϕₘ∂σₗ
        out[6, 34] += sdf * -hxx∂ϕₘ∂σₗ

        ## ϕₘσᵣ
        # out[1, 35] += 0
        out[2, 35] += sdf * hxz∂ϕₘ∂σᵣ
        out[3, 35] += sdf * hyz∂ϕₘ∂σᵣ
        out[4, 35] += sdf * hxx∂ϕₘ∂σᵣ
        out[5, 35] += sdf * hyx∂ϕₘ∂σᵣ
        out[6, 35] += sdf * -hxx∂ϕₘ∂σᵣ

        ## β2
        # out[1, 36] += 0
        out[2, 36] += sdf * hxz∂β2
        out[3, 36] += sdf * hyz∂β2
        out[4, 36] += sdf * hxx∂β2
        out[5, 36] += sdf * hyx∂β2
        out[6, 36] += sdf * -hxx∂β2

        ## βν
        # out[1, 37] += 0
        out[2, 37] += sdf * hxz∂β∂ν
        out[3, 37] += sdf * hyz∂β∂ν
        out[4, 37] += sdf * hxx∂β∂ν
        out[5, 37] += sdf * hyx∂β∂ν
        out[6, 37] += sdf * -hxx∂β∂ν

        ## βσₗ
        # out[1, 38] += 0
        out[2, 38] += sdf * hxz∂β∂σₗ
        out[3, 38] += sdf * hyz∂β∂σₗ
        out[4, 38] += sdf * hxx∂β∂σₗ
        out[5, 38] += sdf * hyx∂β∂σₗ
        out[6, 38] += sdf * -hxx∂β∂σₗ

        ## βσᵣ
        # out[1, 39] += 0
        out[2, 39] += sdf * hxz∂β∂σᵣ
        out[3, 39] += sdf * hyz∂β∂σᵣ
        out[4, 39] += sdf * hxx∂β∂σᵣ
        out[5, 39] += sdf * hyx∂β∂σᵣ
        out[6, 39] += sdf * -hxx∂β∂σᵣ

        ## ν2
        # out[1, 40] += 0
        out[2, 40] += sdf * hxz∂ν2
        out[3, 40] += sdf * hyz∂ν2
        out[4, 40] += sdf * hxx∂ν2
        out[5, 40] += sdf * hyx∂ν2
        out[6, 40] += sdf * -hxx∂ν2

        ## νσₗ
        # out[1, 41] += 0
        out[2, 41] += sdf * hxz∂ν∂σₗ
        out[3, 41] += sdf * hyz∂ν∂σₗ
        out[4, 41] += sdf * hxx∂ν∂σₗ
        out[5, 41] += sdf * hyx∂ν∂σₗ
        out[6, 41] += sdf * -hxx∂ν∂σₗ

        ## νσᵣ
        # out[1, 42] += 0
        out[2, 42] += sdf * hxz∂ν∂σᵣ
        out[3, 42] += sdf * hyz∂ν∂σᵣ
        out[4, 42] += sdf * hxx∂ν∂σᵣ
        out[5, 42] += sdf * hyx∂ν∂σᵣ
        out[6, 42] += sdf * -hxx∂ν∂σᵣ

        ## σₗ2
        # out[1, 43] += 0
        out[2, 43] += sdf * hxz∂σₗ2
        out[3, 43] += sdf * hyz∂σₗ2
        out[4, 43] += sdf * hxx∂σₗ2
        out[5, 43] += sdf * hyx∂σₗ2
        out[6, 43] += sdf * -hxx∂σₗ2

        ## σₗσᵣ
        # out[1, 44] += 0
        out[2, 44] += sdf * hxz∂σₗ∂σᵣ
        out[3, 44] += sdf * hyz∂σₗ∂σᵣ
        out[4, 44] += sdf * hxx∂σₗ∂σᵣ
        out[5, 44] += sdf * hyx∂σₗ∂σᵣ
        out[6, 44] += sdf * -hxx∂σₗ∂σᵣ

        ## σᵣ2
        # out[1, 45] += 0
        out[2, 45] += sdf * hxz∂σᵣ2
        out[3, 45] += sdf * hyz∂σᵣ2
        out[4, 45] += sdf * hxx∂σᵣ2
        out[5, 45] += sdf * hyx∂σᵣ2
        out[6, 45] += sdf * -hxx∂σᵣ2

    end # otherwise add 0

    return nothing
end