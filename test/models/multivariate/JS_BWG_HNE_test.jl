@testset "JS_BWG_HNE" begin
    θ₀ = [0.7, 0.9, 3.3, 5.0, 0, 4, 2.7, 0.55, 0.26]
    ω = 1.1
    @testset "Gradient" begin
        @test approx_gradient(θ -> sdf(JS_BWG_HNE{1}(θ), ω), θ₀)      ≈ grad_sdf(JS_BWG_HNE{1}(θ₀), ω)
        @test approx_gradient(θ -> sdf(JS_BWG_HNE{1}(θ), 0), θ₀)      ≈ grad_sdf(JS_BWG_HNE{1}(θ₀), 0)
    end
    # @testset "Hessian" begin
    #     @test approx_hessian( θ -> grad_sdf(JS_BWG_HNE{1}(θ), ω), θ₀) ≈ hess_sdf(JS_BWG_HNE{1}(θ₀), ω)
    # end
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