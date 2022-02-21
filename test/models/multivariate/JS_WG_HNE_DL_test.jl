@testset "JS_WG_HNE_DL" begin
    θ₀ = [0.7, 0.9, 3.3, 5.0, 0.3, 0.4]
    ω = 1.1
    @testset "Preallocation" begin
        model = JS_WG_HNE_DL{1,40}(θ₀)
        @test model.r_over4 == model.r/4
        @test model.ωₚ² == model.ωₚ^2
        @test model.ωₚ³ == model.ωₚ^3
        @test model.ωₚ⁴ == model.ωₚ^4
        @test model.logγ == log(model.γ)
        @test model.cosϕₘ == cos(model.ϕₘ)
        @test model.sinϕₘ == sin(model.ϕₘ)
        @test model.cos2ϕₘ == cos(2model.ϕₘ)
        @test model.sin2ϕₘ == sin(2model.ϕₘ)
        @test model.invexp2σ² == exp(-2.0 * model.σ^2)
        @test model.invexphalfσ² == exp(-0.5 * model.σ^2)
        @test model.r_over4 == model.r/4
        @test model.ωₚ² == model.ωₚ^2
        @test model.ωₚ³ == model.ωₚ^3
        @test model.ωₚ⁴ == model.ωₚ^4
        @test model.logγ == log(model.γ)
        @test model.cosϕₘ == cos(model.ϕₘ)
        @test model.sinϕₘ == sin(model.ϕₘ)
        @test model.cos2ϕₘ == cos(2model.ϕₘ)
        @test model.sin2ϕₘ == sin(2model.ϕₘ)
    end
    @testset "Gradient" begin
        @test approx_gradient(θ -> sdf(JS_WG_HNE_DL{1,40}(θ), ω), θ₀)      ≈ grad_sdf(JS_WG_HNE_DL{1,40}(θ₀), ω)
        @test approx_gradient(θ -> sdf(JS_WG_HNE_DL{1,40}(θ), 0), θ₀)      ≈ grad_sdf(JS_WG_HNE_DL{1,40}(θ₀), 0)
    end
    @testset "coherancy" begin
        S = sdf(JS_WG_HNE_DL{1,40}(θ₀), ω)
        @test coherancy(JS_WG_HNE_DL{1,40}(θ₀), ω) ≈ [S[i,j]/sqrt(S[i,i]*S[j,j]) for i in 1:size(S,1), j in 1:size(S,2)]
    end
    @testset "Error handling" begin
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(-0.7,0.9, 3.3, 5.0, 0, 0.4)
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(0.7,-0.9, 3.3, 5.0, 0, 0.4)
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(0.7, 0.9, 0.8, 5.0, 0, 0.4) # 0.8 < 1
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(0.7, 0.9, 3.3, 0.9, 0, 0.4) # 0.9 < 1
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(0.7, 0.9, 3.3, 5.0, 0,-0.4)
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(ones(5))
        @test_throws ArgumentError JS_WG_HNE_DL{1,40}(ones(7))
        @test_throws ErrorException JS_WG_HNE_DL(ones(6))
        @test_throws ErrorException JS_WG_HNE_DL(0.7,0.9, 3.3, 5.0, 0, 0.4)
    end
end