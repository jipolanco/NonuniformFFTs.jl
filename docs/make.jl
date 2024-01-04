using Documenter
using NonuniformFFTs

makedocs(;
    sitename = "NonuniformFFTs",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    modules = [NonuniformFFTs],
    pages = [
        "index.md",
        "API.md",
    ],
    warnonly = [:missing_docs],  # TODO fix this?
)

deploydocs(
    repo = "github.com/jipolanco/NonuniformFFTs.jl",
    forcepush = true,
    push_preview = false,
)
