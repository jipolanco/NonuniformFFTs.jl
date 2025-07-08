using Documenter
using DocumenterCitations
using Downloads: Downloads
using NonuniformFFTs

# Copy benchmark results to docs/src/benchmarks/
srcdir = relpath(joinpath(@__DIR__, "..", "benchmark", "CPU+CUDA", "plots"))
dstdir = relpath(joinpath(@__DIR__, "src", "img"))
mkpath(dstdir)
for fname ∈ readdir(srcdir)
    endswith(".svg")(fname) || continue
    srcfile = joinpath(srcdir, fname)
    dstfile = joinpath(dstdir, fname)
    @info "Copying $srcfile -> $dstfile"
    cp(srcfile, dstfile; force = true)
end

# Bibliography
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

assets = [
    asset("assets/citations.css"; islocal = true),
    asset("assets/benchmarks.css"; islocal = true),
]

# Try to download latest version of simpleanalytics script.
try
    script = "assets/sa.js"
    dst = joinpath(@__DIR__, "src", script)
    Downloads.download("https://scripts.simpleanalyticscdn.com/latest.js", dst)
    attributes = Dict(:async => "", Symbol("data-collect-dnt") => "true")
    push!(assets, asset(script; attributes, islocal = true))
catch e
    @warn "Failed downloading asset" e
end

makedocs(;
    sitename = "NonuniformFFTs",
    format = Documenter.HTML(;
        prettyurls = true,
        assets,
    ),
    modules = [NonuniformFFTs],
    pages = [
        "index.md",
        "examples.md",
        "accuracy.md",
        "benchmarks.md",
        "API.md",
    ],
    plugins = [bib],
    # warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/jipolanco/NonuniformFFTs.jl",
    forcepush = true,
    push_preview = true,
)
