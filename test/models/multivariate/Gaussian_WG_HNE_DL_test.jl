@testset "Gaussian_WG_HNE_DL" begin
    θ₀ = [0.7, 0.9, 0.1, 0.3, 0.3]
    ω = 1.1
    @testset "Preallocation" begin
        model = Gaussian_WG_HNE_DL{1,40}(θ₀)
        @test model.norm == model.α/model.κ/sqrt(2π)
        @test model.halfκ⁻² == inv(2model.κ^2)
        @test model.cosϕₘ == cos(model.ϕₘ)
        @test model.sinϕₘ == sin(model.ϕₘ)
        @test model.cos2ϕₘ == cos(2model.ϕₘ)
        @test model.sin2ϕₘ == sin(2model.ϕₘ)
        @test model.invexp2σ² == exp(-2model.σ^2)
        @test model.invexphalfσ² == exp(-0.5model.σ^2)
    end
    @testset "Gradient" begin
        @test approx_gradient(θ -> sdf(Gaussian_WG_HNE_DL{1,40}(θ), ω), θ₀)      ≈ grad_sdf(Gaussian_WG_HNE_DL{1,40}(θ₀), ω)
        @test approx_gradient(θ -> sdf(Gaussian_WG_HNE_DL{1,40}(θ), 0), θ₀)      ≈ grad_sdf(Gaussian_WG_HNE_DL{1,40}(θ₀), 0)
    end
    @testset "Error handling" begin
        @test_throws ArgumentError Gaussian_WG_HNE_DL{1,40}(-0.7, 0.9, 0.1, 0.3, 0.3)
        @test_throws ArgumentError Gaussian_WG_HNE_DL{1,40}(0.7, -0.9, 0.1, 0.3, 0.3)
        @test_throws ArgumentError Gaussian_WG_HNE_DL{1,40}(0.7,  0.9, -0.1, 0.3, 0.3) 
        @test_throws ArgumentError Gaussian_WG_HNE_DL{1,40}(0.7,  0.9, 0.1, 0.3, -0.3)
        @test_throws ArgumentError Gaussian_WG_HNE_DL{1,40}(ones(4))
        @test_throws ArgumentError Gaussian_WG_HNE_DL{1,40}(ones(6))
        @test_throws ErrorException Gaussian_WG_HNE_DL(ones(5))
        @test_throws ErrorException Gaussian_WG_HNE_DL(0.7, 0.9, 0.1, 0.3, 0.3)
    end
end