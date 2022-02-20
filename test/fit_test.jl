const TwoJONSWAP = JONSWAP{1} + JONSWAP{1}
import OceanWaveSpectralFitting: lowerbounds, upperbounds
@testset "fit" begin
    θ = [0.7,0.8,3.3,5.0]
    θ2 = [0.7,0.8,3.3,5.0,1.2,1.1,3.3,5.0]
    @testset "Error handling" begin
        @test_throws ArgumentError fit(ones(10,2),1,model=JONSWAP{1},x₀=θ)
        @test_throws ArgumentError fit(ones(10,2),1,model=JS_BWG_HNE{1},x₀=[θ;[0,4.0,2.7,0.55,0.26]])
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=[θ;1.1])
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=θ,lowerΩcutoff = 2, upperΩcutoff = 1)
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=θ,x_lowerbounds = ones(3))
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=θ,x_upperbounds = ones(3))
        @test_throws ArgumentError fit(ones(10),1,model=JONSWAP{1},x₀=θ,taper="Dpss_4")
    end
    @testset "fitting" begin
        ts = simulate_gp(JONSWAP{1}(θ),1000,1.0,1)[1]
        @test fit(ts,model=JONSWAP{1},x₀=θ) isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
        @test fit(ts,model=JONSWAP{1},x₀=θ,taper=:dpss_4) isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
        @test fit(ts.ts,ts.Δ,model=JONSWAP{1},x₀=θ,taper="dpss_4.2") isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
        @test lowerbounds(TwoJONSWAP) == [lowerbounds(JONSWAP{1});lowerbounds(JONSWAP{1})]
        @test upperbounds(TwoJONSWAP) == [upperbounds(JONSWAP{1});upperbounds(JONSWAP{1})]
        ts2 = simulate_gp(TwoJONSWAP(θ2),1000,1.0,1)[1]
        @test fit(ts.ts,ts.Δ,model=TwoJONSWAP,x₀=θ2,taper="dpss_4.2") isa OceanWaveSpectralFitting.Optim.MultivariateOptimizationResults
    end
end