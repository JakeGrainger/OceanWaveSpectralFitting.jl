@testset "JS_BWG_HNE" begin
    θ₀ = [0.7, 0.9, 3.3, 5.0, 0.3, 4, 2.7, 0.55, 0.26]
    ω = 1.1
    @testset "Preallocation" begin
        model = JS_BWG_HNE{1}(θ₀)
        @test model.r_over4 == model.r/4
        @test model.ωₚ² == model.ωₚ^2
        @test model.ωₚ³ == model.ωₚ^3
        @test model.ωₚ⁴ == model.ωₚ^4
        @test model.ωₚ⁶ == model.ωₚ^6
        @test model.logγ == log(model.γ)
        @test model.σᵣ_over3 == model.σᵣ/3
        @test model.cosϕₘ == cos(model.ϕₘ)
        @test model.sinϕₘ == sin(model.ϕₘ)
        @test model.cos2ϕₘ == cos(2model.ϕₘ)
        @test model.sin2ϕₘ == sin(2model.ϕₘ)
    end
    @testset "Gradient" begin
        @test approx_gradient(θ -> sdf(JS_BWG_HNE{1}(θ), ω), θ₀)      ≈ grad_sdf(JS_BWG_HNE{1}(θ₀), ω)
        @test approx_gradient(θ -> sdf(JS_BWG_HNE{1}(θ), 0), θ₀)      ≈ grad_sdf(JS_BWG_HNE{1}(θ₀), 0)
    end
    @testset "Hessian" begin
        @test approx_hessian( θ -> grad_sdf(JS_BWG_HNE{1}(θ), ω), θ₀) ≈ hess_sdf(JS_BWG_HNE{1}(θ₀), ω)
        @test approx_hessian( θ -> grad_sdf(JS_BWG_HNE{1}(θ), 0), θ₀) ≈ hess_sdf(JS_BWG_HNE{1}(θ₀), 0)
    end
    @testset "coherancy" begin
        S = sdf(JS_BWG_HNE{1}(θ₀), ω)
        @test coherancy(JS_BWG_HNE{1}(θ₀), ω) ≈ [S[i,j]/sqrt(S[i,i]*S[j,j]) for i in 1:size(S,1), j in 1:size(S,2)]
    end
    @testset "Error handling" begin
        @test_throws ArgumentError JS_BWG_HNE{1}(-0.7,0.9, 3.3, 5.0, 0, 4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7,-0.9, 3.3, 5.0, 0, 4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7, 0.9, 0.8, 5.0, 0, 4, 2.7, 0.55, 0.26) # 0.8 <1
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7, 0.9, 3.3, 0.9, 0, 4, 2.7, 0.55, 0.26) # 0.9 < 1
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7, 0.9, 3.3, 5.0, 0,-4, 2.7, 0.55, 0.26)
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7, 0.9, 3.3, 5.0, 0, 4,-2.7, 0.55, 0.26)
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7, 0.9, 3.3, 5.0, 0, 4, 2.7,-0.55, 0.26)
        @test_throws ArgumentError JS_BWG_HNE{1}(0.7, 0.9, 3.3, 5.0, 0, 4, 2.7, 0.55,-0.26)
        @test_throws ArgumentError JS_BWG_HNE{1}(ones(8))
        @test_throws ArgumentError JS_BWG_HNE{1}(ones(10))
        @test_throws ErrorException JS_BWG_HNE(ones(9))
        @test_throws ErrorException JS_BWG_HNE(0.7, 0.9, 3.3, 5.0, 0, 4, 2.7, 0.55, 0.26)
    end
end