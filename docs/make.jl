using Documenter, DocumenterVitepress, PureFFT

makedocs(;
    sitename = "PureFFT.jl",
    authors = "el_oso",
    modules = [PureFFT],
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureFFT.jl",
        devbranch = "master",
        devurl = "dev",
    ),
    draft = false,
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Manual (FFTW map)" => "manual.md",
        "Codelet Generator (KB)" => "codelet-generator.md",
        "Performance" => "performance.md",
        "Benchmarks" => "benchmarks.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureFFT.jl",
    devbranch = "master",
    push_preview = true,
)
