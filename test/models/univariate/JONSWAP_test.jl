@testset "JONSWAP" begin
    θ₀ = [0.7, 0.9, 3.3, 5.0]
    ω = 1.0
    @testset "sdf" begin
        @testset "Gradient" begin
            @test approx_gradient_uni(θ -> sdf(JONSWAP{1}(θ), ω), θ₀)      ≈ grad_sdf(JONSWAP{1}(θ₀), ω)
        end
        @testset "Hessian" begin
            @test approx_hessian_uni( θ -> grad_sdf(JONSWAP{1}(θ), ω), θ₀) ≈ hess_sdf(JONSWAP{1}(θ₀), ω)
        end
    end
    @testset "Error handling" begin
        @test_throws ArgumentError JONSWAP{K}(-0.7, 0.9, 3.3, 5.0)
        @test_throws ArgumentError JONSWAP{K}(0.7, -0.9, 3.3, 5.0)
        @test_throws ArgumentError JONSWAP{K}(0.7, 0.9, 0.8, 5.0)
        @test_throws ArgumentError JONSWAP{K}(0.7, 0.9, 3.3, 0.9)
        @test_throws ArgumentError JONSWAP{K}(ones(3))
        @test_throws ArgumentError JONSWAP{K}(ones(5))
        @test_throws ErrorException JONSWAP(ones(4))
        @test_throws ErrorException JONSWAP(0.7, 0.9, 3.3, 5.0)
    end
end