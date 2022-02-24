@testset "GeneralJONSWAP" begin
    θ₀ = [0.7, 0.9, 3.3, 5.0, 4.0]
    ω = 1.0
    
    @testset "Gradient" begin
        @test approx_gradient_uni(θ -> sdf(GeneralJONSWAP{1}(θ), ω), θ₀)      ≈ grad_sdf(GeneralJONSWAP{1}(θ₀), ω)
    end

    @testset "Error handling" begin
        @test_throws ArgumentError GeneralJONSWAP{1}(-0.7, 0.9, 3.3, 5.0, 4.0)
        @test_throws ArgumentError GeneralJONSWAP{1}(0.7, -0.9, 3.3, 5.0, 4.0)
        @test_throws ArgumentError GeneralJONSWAP{1}(0.7, 0.9, 0.8, 5.0, 4.0)
        @test_throws ArgumentError GeneralJONSWAP{1}(0.7, 0.9, 3.3, 0.9, 4.0)
        @test_throws ArgumentError GeneralJONSWAP{1}(0.7, 0.9, 3.3, 5.0, -1)
        @test_throws ArgumentError GeneralJONSWAP{1}(ones(4))
        @test_throws ArgumentError GeneralJONSWAP{1}(ones(6))
        @test_throws ErrorException GeneralJONSWAP(ones(5))
        @test_throws ErrorException GeneralJONSWAP(0.7, 0.9, 3.3, 5.0, 4.0)
    end
end