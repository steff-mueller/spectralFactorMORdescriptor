using SpectralFactorMOR
using Documenter

DocMeta.setdocmeta!(SpectralFactorMOR, :DocTestSetup, :(using SpectralFactorMOR); recursive=true)

makedocs(;
    modules=[SpectralFactorMOR],
    authors="Steffen Müller <steffen.mueller@simtech.uni-stuttgart.de>",
    sitename="spectralFactorMORdescriptor",
    format=Documenter.HTML(;
        canonical="https://steff-mueller.github.io/spectralFactorMORdescriptor",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "SpectralFactorMOR" => "SpectralFactorMOR.md"
    ]
)

deploydocs(;
    repo="github.com/steff-mueller/spectralFactorMORdescriptor",
    devbranch="main",
)
