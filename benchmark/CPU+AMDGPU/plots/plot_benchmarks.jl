using GLMakie
using CairoMakie
using DelimitedFiles
using MathTeXEngine

MathTeXEngine.set_texfont_family!(FontFamily("TeXGyreHeros"))
GLMakie.activate!()
Makie.set_theme!(theme_latexfonts())

function read_timings(io::IO; nufft_type::Int)
    while Char(peek(io)) == '#'
        readline(io)
    end
    data = readdlm(io, Float64)
    (; Np = @view(data[:, 1]), times = @view(data[:, 1 + nufft_type]),)
end

read_timings(fname::AbstractString; kws...) = open(io -> read_timings(io; kws...), fname)

function plot_from_file!(ax::Axis, filename; nufft_type, zorder = nothing, kws...)
    data = read_timings(filename; nufft_type)
    (; Np, times,) = data
    l = scatterlines!(ax, Np, times; kws...)
    l.strokecolor[] = l.color[]
    if zorder !== nothing
        translate!(l, 0, 0, zorder)
    end
    l
end

function plot_benchmark(::Type{T}, results_dir; nufft_type = 1, atomics = false, save_svg = false,) where {T <: Number}
    Ns = (256, 256, 256)
    N = first(Ns)
    Ngrid = prod(Ns)
    Z = complex(T)  # for FINUFFT (complex data only)

    fig = Figure(size = (750, 440))
    ax = Axis(
        fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = L"Number of nonuniform points $N$", xlabelsize = 16,
        ylabel = "Time (s)",
        xticks = LogTicks(0:20), xminorticks = IntervalsBetween(9), xminorticksvisible = true,
        yticks = LogTicks(-8:3), yminorticks = IntervalsBetween(9), yminorticksvisible = true,
        xgridvisible = false, ygridvisible = false,
        limits = (nothing, (1e-3, 1e1)),
    )
    limits_top = lift(ax.finallimits) do lims
        xlims = Makie.xlimits(lims) ./ Ngrid
        ylims = Makie.ylimits(lims)
        xlims, ylims
    end
    ax_top = Axis(
        fig[1, 1];
        xscale = ax.xscale, yscale = ax.yscale,
        xaxisposition = :top,
        # xgridvisible = true, ygridvisible = false,
        xticks = LogTicks(-8:2), xminorticks = IntervalsBetween(9), xminorticksvisible = true,
        xlabel = L"Point density $ρ = N / M^3$", xlabelsize = 16,
        limits = limits_top,
    )
    hidespines!(ax_top)
    hideydecorations!(ax_top; grid = false)

    colours = Makie.wong_colors()

    kws_all = (; nufft_type,)
    kws_cpu = (; marker = :x, markersize = 16, strokewidth = 0,)
    kws_gpu = (; marker = :circle, markersize = 10, strokewidth = 2,)
    kws_gpu_sm = (; kws_gpu..., markercolor = :transparent,)  # open symbols
    kws_nonuniform = (; linestyle = :solid, color = colours[1], zorder = 10,)
    kws_finufft = (; linestyle = :dash, color = colours[2],)

    # Leftmost point of all CPU/GPU curves (for annotating later)
    first_points_cpu = Point2{Float64}[]
    first_points_gpu = Point2{Float64}[]
    last_points_gpu = Point2{Float64}[]

    cpu_suffix = if atomics
        "_atomics"
    else
        ""
    end

    l = plot_from_file!(ax, "$results_dir/NonuniformFFTs_$(N)_$(T)_CPU$(cpu_suffix).dat"; label = "NonuniformFFTs CPU", kws_nonuniform..., kws_cpu..., kws_all...)
    push!(first_points_cpu, l[1][][1])  # get first datapoint in line

    l = plot_from_file!(ax, "$results_dir/NonuniformFFTs_$(N)_$(T)_ROCBackend_global_memory.dat"; label = "NonuniformFFTs GPU", kws_nonuniform..., kws_gpu..., kws_all...)
    push!(first_points_gpu, l[1][][1])  # get first datapoint in line
    push!(last_points_gpu, l[1][][end])  # get last datapoint in line

    l = plot_from_file!(ax, "$results_dir/NonuniformFFTs_$(N)_$(T)_ROCBackend_shared_memory.dat"; label = "NonuniformFFTs GPU (SM)", kws_nonuniform..., kws_gpu_sm..., kws_all...)
    push!(first_points_gpu, l[1][][1])  # get first datapoint in line
    push!(last_points_gpu, l[1][][end])  # get last datapoint in line

    l = plot_from_file!(ax, "$results_dir/FINUFFT_$(N)_$(Z)_CPU.dat"; label = "FINUFFT CPU", kws_finufft..., kws_cpu..., kws_all...)
    push!(first_points_cpu, l[1][][1])  # get first datapoint in line

    # l = plot_from_file!(ax, "$results_dir/CuFINUFFT_$(N)_$(Z)_global_memory.dat"; label = "CuFINUFFT GPU", kws_finufft..., kws_gpu..., kws_all...)
    # push!(first_points_gpu, l[1][][1])  # get first datapoint in line
    # push!(last_points_gpu, l[1][][end])  # get last datapoint in line
    #
    # l = plot_from_file!(ax, "$results_dir/CuFINUFFT_$(N)_$(Z)_shared_memory.dat"; label = "CuFINUFFT GPU (SM)", kws_finufft..., kws_gpu_sm..., kws_all...)
    # push!(first_points_gpu, l[1][][1])  # get first datapoint in line
    # push!(last_points_gpu, l[1][][end])  # get last datapoint in line

    let points = first_points_cpu, text = "CPU"
        x = minimum(p -> p[1], points)  # generally all points x are the same
        y = maximum(p -> p[2], points)
        text!(ax, x, y; text = rich(text; font = :bold), align = (:left, :bottom), offset = (0, 8), fontsize = 16)
    end

    let points = first_points_gpu, text = "GPU"
        x = minimum(p -> p[1], points)  # generally all points x are the same
        y = maximum(p -> p[2], points)
        text!(ax, x, y; text = rich(text; font = :bold), align = (:left, :bottom), offset = (0, 8), fontsize = 16)
    end

    let xs = logrange(0.5, 10.0; length = 3)
        ymin, n = findmin(p -> p[2], last_points_gpu)
        xmin = last_points_gpu[n][1] / Ngrid  # as a density ρ
        scale = ymin / xmin * 0.7
        ys = @. xs * scale
        lines!(ax_top, xs, ys; linestyle = :dash, color = :black, linewidth = 3)
        text!(ax_top, xs[2], ys[2]; text = L"∼N", align = (:left, :top), fontsize = 18)
    end

    Label(
        fig[1, 2][1, 1],
        """
        Type-$(nufft_type) NUFFT
        $(N)³ Fourier modes
        $T data
        6-digit accuracy
        """;
        justification = :left, fontsize = 16, lineheight = 1.2,
    )
    Legend(fig[1, 2][2, 1], ax; framevisible = false, rowgap = 8, labelsize = 14)

    if save_svg
        save("benchmark_$(T)_type$(nufft_type)$(cpu_suffix).svg", fig; backend = CairoMakie)
    end

    fig
end

##

results_dir = "../results.MI300A_adastra"

for T ∈ (Float64, ComplexF64), nufft_type ∈ (1, 2), atomics ∈ (false, true)
    plot_benchmark(T, results_dir; nufft_type, atomics, save_svg = true)
end

fig = plot_benchmark(ComplexF64, results_dir; nufft_type = 1, atomics = true)
