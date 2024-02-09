using Printf
using TOML
using JLD2
using MAT
using Dates
using DelimitedFiles
using LoggingExtras
using GLMakie
using Random
using DescriptorSystems
using PortHamiltonianBenchmarkSystems

function load_toml_config(toml_file::String)
    toml_config = read(toml_file, String)
    @info "Loading configuration from $toml_file:\n$toml_config"
    return TOML.parse(toml_config)
end

function to_symbol_dict(d::Dict{String, Any})
    return Iterators.map(p -> (Symbol(p.first), p.second), d)
end

struct MORMethod
    name::String
    reduce_fn
    color::Symbol
    marker::Symbol
end

struct Result
    name::String
    r::Int64
    irka_result::Union{IRKAResult, Nothing}
    abs_h2_error::Union{Nothing,Float64}
    rom::DescriptorStateSpace
    rom_stable::Bool
    rom_pr::NamedTuple{
        (:ispr, :mineigval, :w), Tuple{Bool, Float64, Float64}}
end

function Base.show(io::IO, res::Result)
    print(io, "$(res.name), r=$(res.r):\n")
    print(io, "\tROM stable: $(res.rom_stable)\n")
    print(io, "\tROM positive real: $(res.rom_pr)\n")
    if (!isnothing(res.abs_h2_error))
        print(io, "\tAbsolute H2 error: $(@sprintf("%.2e", res.abs_h2_error))\n")
    end
    if (!isnothing(res.irka_result))
        irka_result = res.irka_result
        print(io, """
        \tIRKA iterations: $(irka_result.iterations)
        \tIRKA converged: $(irka_result.converged)
        \tIRKA conv crit: $(@sprintf("%.2e", irka_result.conv_crit))
        \tIRKA cycle detected: $(irka_result.cycle_detected)
        \tIRKA cycle crit: $(@sprintf("%.2e", irka_result.cycle_crit))\n""")
    end
end

function Base.show(io::IO, results::Matrix{Result})
    print(io, "Result Summary\n")
    for i in 1:size(results, 1)
        print(io, "=== $(results[i,1].name) ===\n")
        for j in 1:size(results, 2)
            print(io, results[i,j])
        end
    end
    return summary
end

function sigmaplot(systems::AbstractVector{TS};
    w = exp10.(range(-5, stop=2, length=100)), title = "Bodeplot"
) where TS <: AbstractDescriptorStateSpace
    f = Figure()
    ax = Axis(f[1, 1],
        title = title,
        xlabel = "w",
        ylabel = "σ_1(T(iw))",
        xscale = log10,
        yscale = log10
    )
    for (i,sys) in enumerate(systems)
        T(s) = Matrix(sys.C)/(s.*sys.E-sys.A)*sys.B+sys.D
        lines!(ax, w, opnorm.(T.(im.*w), 2), label="G_$i")
    end
    axislegend(ax)
    return f
end

function sigmaplot(sys::AbstractDescriptorStateSpace;
    w = exp10.(range(-5, stop=2, length=100)), title = "Bodeplot")
    return sigmaplot([sys], w=w, title=title)
end

function plot_tf(experiment_id::String, results::Matrix{Result}, fom)
    for res::Result in results
        (; name, r, rom) = res
        f = sigmaplot([fom, rom], title="$name, r=$r ")
        display(f)
        save(datadir(experiment_id, "tf_$(name)_$r.png"), f)

        f = sigmaplot(fom-rom, title="$name, r=$r error") # TODO change y axis desc
        display(f)
        save(datadir(experiment_id, "tf_$(name)_$(r)_err.png"), f)
    end
end

function plot_h2_errors(
    experiment_id::String, methods::Vector{MORMethod}, results::Matrix{Result}
)
    f = Figure()
    ax = Axis(f[1, 1],
        xlabel = "r",
        ylabel = "H2 error",
        yscale = log10
    )

    reduced_orders = map(x -> x.r, results[1,:])
    for (i, item) in enumerate(methods)
        (; name, color, marker) = item
        scatterlines!(
            ax, reduced_orders,
            map(x -> x.abs_h2_error, results[i,:]), label=name,
            color=color, marker=marker
        )
    end

    axislegend(ax)
    display(f)
    save(datadir(experiment_id, "h2_errors.png"), f)
end

function save_results(experiment_id::String, results::Matrix{Result}, fom)
    h2errors_tocsv(experiment_id, results)
    tf_tocsv(experiment_id, results, fom)
    summary_tocsv(experiment_id, results)
    jldsave(datadir(experiment_id, "data.jld2"); results, fom)
end

function h2errors_tocsv(experiment_id::String, results::Matrix{Result})
    for method in eachrow(results)
        open(datadir(experiment_id, "$(method[1].name)_h2errors.csv"), "w") do f
            write(f, "r,abs_h2_error\n")
            writedlm(f, map(x -> (x.r, x.abs_h2_error), method), ',')
        end 
    end
end

function tf_tocsv(experiment_id::String, results::Matrix{Result}, fom)
    w = exp10.(range(-5, stop=2, length=200))
    T(s) = Matrix(fom.C)/(s.*fom.E-fom.A)*fom.B+fom.D
    tf_fom = T.(im.*w)

    open(datadir(experiment_id, "fom_tf.csv"), "w") do f
        write(f, "f,tf\n")
        writedlm(f, [w opnorm.(tf_fom)], ',')
    end

    for result in results
        (; name, r, rom) = result
        Tr(s) = Matrix(rom.C)/(s.*rom.E-rom.A)*rom.B+rom.D
        open(datadir(experiment_id, "$(name)_$(r)_tf.csv"), "w") do f
            tf = Tr.(im.*w)
            tf_err = tf-tf_fom
            write(f, "f,tf,tf_err\n")
            writedlm(f, [w opnorm.(tf) opnorm.(tf_err)], ',')
        end
    end
end

function summary_tocsv(experiment_id::String, results::Matrix{Result})
    open(datadir(experiment_id, "summary.csv"), "w") do f
        write(f, "name,r,abs h2 error,irka conv crit,irka cycle crit,irka iterations,pr mineigval\n")
        for result in results
            (; name, r, abs_h2_error, irka_result, rom_pr) = result
            if (isnothing(irka_result))
                write(f, "$name,$r,$abs_h2_error,,,,$(rom_pr.mineigval)\n")
            else
                write(f, "$name,$r,$abs_h2_error,$(irka_result.conv_crit),$(irka_result.cycle_crit),$(irka_result.iterations),$(rom_pr.mineigval)\n")
            end
        end
    end
end

function get_experiment_logger(experiment_id::String)
    return TeeLogger(
        global_logger(),
        MinLevelLogger(
            FormatLogger(open(datadir(experiment_id, "run.log"), "w")) do io, args only
                println(io, args.group, " | ", "[", args.level, "] ", args.message)
                for (key, val) in args.kwargs
                    println(io, "\t", key, " = ", val)
                end
            end,
            Logging.Info
        )
    )
end

function start_experiment(@nospecialize(f::Function), experiment_id::String)
    mkpath(datadir(experiment_id))
    with_logger(f, get_experiment_logger(experiment_id))
end

"""
Returns a matrix of type `Matrix{Result}`. The columns of the matrix
correspond to the reduced-order size and the rows of the matrix correspond
to the reduction method.
"""
function run_methods(
    f::Function, methods::Vector{MORMethod}, reduced_orders::Vector{Int64}
)::Matrix{Result}
    results = Matrix{Result}(
        undef, (length(methods), length(reduced_orders))
    )
    for (i, method) in enumerate(methods)
        for (j, r) in enumerate(reduced_orders)
            @info "Executing $(method.name), r=$r."
            res::Result = f(method, r)
            results[i,j] = res
        end
    end
    return results
end

projectdir() = dirname(Base.active_project())
projectdir(args...) = joinpath(projectdir(), args...)
datadir(args...) = joinpath(projectdir(), "data", args...)
