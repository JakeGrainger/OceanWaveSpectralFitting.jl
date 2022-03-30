using OceanWaveSpectralFitting
using Documenter

DocMeta.setdocmeta!(OceanWaveSpectralFitting, :DocTestSetup, :(using OceanWaveSpectralFitting); recursive=true)

makedocs(;
    modules=[OceanWaveSpectralFitting],
    authors="Jake Grainger <j.p.grainger2@outlook.com> and contributors",
    repo="https://github.com/JakeGrainger/OceanWaveSpectralFitting.jl/blob/{commit}{path}#{line}",
    sitename="OceanWaveSpectralFitting.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakeGrainger.github.io/OceanWaveSpectralFitting.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Models" => [
            "univariate.md",
            "multivariate.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/JakeGrainger/OceanWaveSpectralFitting.jl",
    devbranch="main",
)
