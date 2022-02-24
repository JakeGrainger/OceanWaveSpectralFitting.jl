@testset "GJS_BWG_HNE_DL" begin
    θ₀ = [0.7, 0.9, 3.3, 5.0, 4.0, 0.3, 4, 2.7, 0.55, 0.26]
    ω = 1.1
    @testset "Preallocation" begin
        model = GJS_BWG_HNE_DL{1,40}(θ₀)
        @test model.r_over_s == model.r/model.s
        @test model.ωₚ² == model.ωₚ^2
        @test model.ωₚ³ == model.ωₚ^3
        @test model.ωₚˢ == model.ωₚ^model.s
        @test model.ωₚˢ⁻¹ == model.ωₚ^(model.s-1)
        @test model.logγ == log(model.γ)
        @test model.s⁻¹ == 1/(model.s)
        @test model.logωₚminuss⁻¹ == log(model.ωₚ) - 1/model.s
        @test model.σᵣ_over3 == model.σᵣ/3
        @test model.cosϕₘ == cos(model.ϕₘ)
        @test model.sinϕₘ == sin(model.ϕₘ)
        @test model.cos2ϕₘ == cos(2model.ϕₘ)
        @test model.sin2ϕₘ == sin(2model.ϕₘ)
    end
    @testset "Gradient" begin
        @test approx_gradient(θ -> sdf(GJS_BWG_HNE_DL{1,40}(θ), ω), θ₀)      ≈ grad_sdf(GJS_BWG_HNE_DL{1,40}(θ₀), ω)
        @test approx_gradient(θ -> sdf(GJS_BWG_HNE_DL{1,40}(θ), 0), θ₀)      ≈ grad_sdf(GJS_BWG_HNE_DL{1,40}(θ₀), 0)
    end
    @testset "Error handling" begin
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(-0.7,0.9, 3.3, 5.0, 4.0, 0, 4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7,-0.9, 3.3, 5.0, 4.0, 0, 4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 0.8, 5.0, 4.0, 0, 4, 2.7, 0.55, 0.26) # 0.8 <1
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 3.3, 0.9, 4.0, 0, 4, 2.7, 0.55, 0.26) # 0.9 < 1
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 3.3, 5.0,-4.0, 0, 4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 3.3, 5.0, 4.0, 0,-4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 3.3, 5.0, 4.0, 0, 4,-2.7, 0.55, 0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 3.3, 5.0, 4.0, 0, 4, 2.7,-0.55, 0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(0.7, 0.9, 3.3, 5.0, 4.0, 0, 4, 2.7, 0.55,-0.26)
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(ones(9))
        @test_throws ArgumentError GJS_BWG_HNE_DL{1,40}(ones(11))
        @test_throws ErrorException GJS_BWG_HNE_DL(ones(10))
        @test_throws ErrorException GJS_BWG_HNE_DL(0.7, 0.9, 3.3, 5.0, 4.0, 0, 4, 2.7, 0.55, 0.26)
    end
end