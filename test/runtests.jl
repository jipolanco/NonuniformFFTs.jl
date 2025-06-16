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

@info "Running tests on $(Threads.nthreads()) threads"

@testset "NonuniformFFTs.jl" begin
    @includetest "errors.jl"
    @includetest "approx_window_functions.jl"
    @includetest "accuracy.jl"
    @includetest "multidimensional.jl"
    @includetest "callbacks.jl"
    @includetest "pseudo_gpu.jl"
    @includetest "abstractNFFTs.jl"
    @includetest "uniform_points.jl"
    @includetest "near_2pi.jl"
end
