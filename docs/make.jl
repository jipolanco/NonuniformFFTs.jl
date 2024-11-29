using Documenter
using DocumenterCitations
using NonuniformFFTs

# Copy benchmark results to docs/src/benchmarks/
dstdir = joinpath(@__DIR__, "src", "benchmarks")
srcdir = joinpath(@__DIR__, "..", "benchmarks", "plots")
@assert isdir(dstdir)
for fname ∈ readdir(srcdir)
    endswith(".svg")(fname) || continue
    cp(joinpath(srcdir, fname), joinpath(dstdir, fname); force = true)
end

# Bibliography
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

makedocs(;
    sitename = "NonuniformFFTs",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = [
            "assets/citations.css",
            "assets/benchmarks.css",
        ],
    ),
    modules = [NonuniformFFTs],
    pages = [
        "index.md",
        "accuracy.md",
        "benchmarks.md",
        "API.md",
    ],
    plugins = [bib],
    warnonly = [:missing_docs],  # TODO fix this?
)

deploydocs(
    repo = "github.com/jipolanco/NonuniformFFTs.jl",
    forcepush = true,
    push_preview = false,
)
