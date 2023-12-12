using Test

# Wraps each test file in a separate module, to avoid definition clashes and to make sure
# that each file can also be run as a standalone script.
macro includetest(path::String)
    modname = Symbol("Mod_" * replace(path, '.' => '_'))
    escname = esc(modname)
    ex = quote
        @info "Running $($path)"
        module $escname
            $escname.include($path)
        end
        using .$modname
    end
    ex.head = :toplevel
    ex
end

@testset "NonuniformFFTs.jl" begin
    @includetest "accuracy.jl"
    @includetest "multidimensional.jl"
    @includetest "uniform_points.jl"
end
