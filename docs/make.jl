using Documenter
using NonuniformFFTs

makedocs(;
    sitename = "NonuniformFFTs",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)

deploydocs(
    repo = "github.com/jipolanco/NonuniformFFTs.jl",
    forcepush = true,
    push_preview = false,
)
