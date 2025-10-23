using Documenter
using DocumenterVitepress
using DocumenterCitations
using Downloads: Downloads
using NonuniformFFTs

function copy_svg_images(dstdir, srcdir)
    mkpath(dstdir)
    for fname ∈ readdir(srcdir)
        endswith(".svg")(fname) || continue
        srcfile = joinpath(srcdir, fname)
        dstfile = joinpath(dstdir, fname)
        @info "Copying $srcfile -> $dstfile"
        cp(srcfile, dstfile; force = true)
    end
end

# Copy benchmark results to docs/src/benchmarks/
srcdir = relpath(joinpath(@__DIR__, "..", "benchmark", "CPU+CUDA", "plots"))
dstdir = relpath(joinpath(@__DIR__, "src", "img", "CUDA"))
copy_svg_images(dstdir, srcdir)
srcdir = relpath(joinpath(@__DIR__, "..", "benchmark", "CPU+AMDGPU", "plots"))
dstdir = relpath(joinpath(@__DIR__, "src", "img", "AMDGPU"))
copy_svg_images(dstdir, srcdir)

# Bibliography
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

# assets = [
#     asset("assets/citations.css"; islocal = true),
#     asset("assets/benchmarks.css"; islocal = true),
# ]

# Try to download latest version of simpleanalytics script.
try
    script = "public/sa.js"
    dst = joinpath(@__DIR__, "src", script)
    Downloads.download("https://scripts.simpleanalyticscdn.com/latest.js", dst)
    # attributes = Dict(:async => "", Symbol("data-collect-dnt") => "true")
    # push!(assets, asset(script; attributes, islocal = true))
catch e
    @warn "Failed downloading asset" e
end

makedocs(;
    sitename = "NonuniformFFTs",
    authors = "Juan Ignacio Polanco",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/jipolanco/NonuniformFFTs.jl",
        devbranch = "master",
        devurl = "dev",
        # assets,
    ),
    # format = Documenter.HTML(;
    #     prettyurls = true,
    #     assets,
    # ),
    modules = [NonuniformFFTs],
    pages = [
        "index.md",
        "examples.md",
        "accuracy.md",
        "benchmarks.md",
        "API.md",
    ],
    plugins = [bib],
    warnonly = true,
    # warnonly = [:missing_docs],
)

# Documenter.deploydocs(
#     repo = "github.com/jipolanco/NonuniformFFTs.jl",
#     forcepush = true,
#     push_preview = true,
# )

@show readdir("build/1")

# DocumenterVitepress.dev_docs("build")  # use this to see docs locally

DocumenterVitepress.deploydocs(
    repo = "github.com/jipolanco/NonuniformFFTs.jl",
    devbranch = "master",
    push_preview = true,
)
