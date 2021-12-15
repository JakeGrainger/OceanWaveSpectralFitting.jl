import OceanWaveSpectralFitting: fit
@testset "fit" begin
    @testset "Error handling" begin
        @test_throws ArgumentError fit(ones(10,2),1,model=JONSWAP{1},x₀=ones(4))
        @test_throws ArgumentError fit(ones(10,2),1,model=JS_BWG_HNE{1},x₀=ones(9))
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=ones(5))
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=ones(4),lowerΩcutoff = 2, upperΩcutoff = 1)
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=ones(4),x_lowerbounds = ones(3))
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=ones(4),x_upperbounds = ones(3))
        @test_throws ArgumentError fit(ts.ts,ts.Δ,model=JONSWAP{1},x₀=θ,taper="dpss_4")
    end
    @testset "fitting" begin
        θ = [0.7,0.8,3.3,5.0]
        ts = simulate_gp(JONSWAP{1}(θ),1000,1.0,1)[1]
        @test fit(ts.ts,ts.Δ,model=JONSWAP{1},x₀=θ) isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
        @test fit(ts.ts,ts.Δ,model=JONSWAP{1},x₀=θ,taper=:dpss_4) isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
        @test fit(ts.ts,ts.Δ,model=JONSWAP{1},x₀=θ,taper="dpss_4.2") isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
    end
end