using SpectralFactorMOR
using DocumenterCitations
using Documenter

DocMeta.setdocmeta!(SpectralFactorMOR, :DocTestSetup, :(using SpectralFactorMOR); recursive=true)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

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
    ],
    plugins=[bib]
)

deploydocs(;
    repo="github.com/steff-mueller/spectralFactorMORdescriptor",
    devbranch="main",
)
