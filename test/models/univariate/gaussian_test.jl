@testset "Gaussian" begin
    θ₀ = [0.7, 0.9, 0.1]
    ω = 1.0
    
    @testset "Gradient" begin
        @test approx_gradient_uni(θ -> sdf(Gaussian{1}(θ), ω), θ₀)      ≈ grad_sdf(Gaussian{1}(θ₀), ω)
    end

    @testset "Error handling" begin
        @test_throws ArgumentError Gaussian{1}(-0.7, 0.9, 0.1)
        @test_throws ArgumentError Gaussian{1}(0.7, -0.9, 0.1)
        @test_throws ArgumentError Gaussian{1}(0.7, 0.9, -0.1)
        @test_throws ArgumentError Gaussian{1}(ones(2))
        @test_throws ArgumentError Gaussian{1}(ones(4))
        @test_throws ErrorException Gaussian(ones(3))
        @test_throws ErrorException Gaussian(0.7, 0.9, 0.1)
    end
end