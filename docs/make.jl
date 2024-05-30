using Documenter
using DocumenterVitepress
using NonuniformFFTs

makedocs(;
    sitename = "NonuniformFFTs",
    format = DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/jipolanco/NonuniformFFTs.jl",
    ),
    modules = [NonuniformFFTs],
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Methods" => "methods.md",
        "API" => "api.md",
    ],
    warnonly = true,  # TODO fix this?
)

deploydocs(
    repo = "github.com/jipolanco/NonuniformFFTs.jl",
    forcepush = true,
    push_preview = false,
)
