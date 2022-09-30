using Documenter
using LinearAlgebra
using FrankWolfe
using FiniteKernel
using SparseArrays

using Literate, Test

DocMeta.setdocmeta!(FiniteKernel, :DocTestSetup, :(using FiniteKernel); recursive=true)

EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
DOCS_EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
DOCS_REFERENCE_DIR = joinpath(@__DIR__, "src", "reference")

function file_list(dir, extension)
    return filter(file -> endswith(file, extension), sort(readdir(dir)))
end

function literate_directory(jl_dir, md_dir)
    for filename in file_list(md_dir, ".md")
        filepath = joinpath(md_dir, filename)
        rm(filepath)
    end
    for filename in file_list(jl_dir, ".jl")
        filepath = joinpath(jl_dir, filename)
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        
        Literate.markdown(
            filepath, md_dir; documenter=true, flavor=Literate.DocumenterFlavor()
        )
    end
    return nothing
end

literate_directory(EXAMPLE_DIR, DOCS_EXAMPLE_DIR)

ENV["GKSwstype"] = "100"

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/garam-kim/FiniteKernel.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)

makedocs(;
    modules=[FiniteKernel],
    authors="Garam Kim <garam.tolba@gmail.com>",
    repo="https://github.com/garam-kim/FiniteKernel.jl/blob/{commit}{path}#{line}",
    sitename="FiniteKernel",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://garam-kim.github.io/FiniteKernel.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        # "Examples" => ".examples/finite_dimensional_kernel_herding.jl",
        "Examples" => [joinpath("examples", f) for f in file_list(DOCS_EXAMPLE_DIR, ".md")],
        "API reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/garam-kim/FiniteKernel.jl",
    devbranch="main",
    push_preview=true,
)