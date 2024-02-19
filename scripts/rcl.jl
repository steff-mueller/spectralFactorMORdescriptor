using DescriptorSystems
using Logging
using Dates
using LinearAlgebra
using SparseArrays
using SpectralFactorMOR

include("../src/utils.jl")
include("../src/problems.jl")

Makie.inline!(true)

Base.@kwdef struct RCLConfiguration
    run_problem::String
    reduced_orders::Vector{Int64}
    irka_tries::Int64
    run_methods::Vector{String}
    compute_h2_errors::Bool
    plot_tf::Bool
    dense_riccati_solver::Bool
end

experiment_id = "rcl_$(now())"
results, sys = start_experiment(experiment_id) do
    # Set ENV variable in REPL via:
    #   ENV["RCL_CONFIG"] = "scripts/rcl_dae2_random_mimo.toml"
    f = load_toml_config(get(ENV, "RCL_CONFIG", "scripts/rcl_default.toml"))
    config = RCLConfiguration(; to_symbol_dict(f["rcl"])...)
    irka_options = IRKAOptions(; to_symbol_dict(f["irka_options"])...)

    sys = get_problem(config.run_problem)
    sysinf, syssp, = splitsys(sys)

    @info """Running problem $(config.run_problem)
            (n = $(size(sys.A,1)), m = $(size(sys.B,2)))."""

    Zo, Xo, Zc = compute_gramians(config.run_problem, sys,
        config.dense_riccati_solver)

    methods = [
        MORMethod(
            "irka",
            r -> irka(sys, r, irka_options),
            :blue, :circle
        ),
        MORMethod(
            "sfmor(X_min)",
            r -> sfmor(sys, r, irka_options;
                 X = Xo, compute_factors=:together),
            :orange, :cross
        ),
        MORMethod(
            "prbt",
            r -> prbaltrunc(sys, r, Zc, Zo),
            :red, :rect
        )
    ]
    filter!(m -> m.name in config.run_methods, methods)

    if "sfmor(X_alt)" in config.run_methods # only valid for index-1 RCL system
        # Compute alternative solution to proj. positive real Lur'e equation:
        Tlinv = [I -sys.A_12/Matrix(sys.A_22);
                 spzeros(sys.n_2, sys.n_1) inv(Matrix(sys.A_22))]
        X_alt = Tlinv'*[inv(Matrix(sys.E_11)) spzeros(sys.n_1, sys.n_2);
        spzeros(sys.n_2, sys.n_1) spzeros(sys.n_2,sys.n_2)]*Tlinv
        @assert checkprojlure(sys, X_alt)

        append!(methods, [MORMethod(
            "sfmor(X_alt)",
            r -> sfmor(sys, r, irka_options; 
                 X=X_alt, compute_factors=:together),
            :cyan, :cross
        )])
    end

    global results = run_methods(
        methods, config.reduced_orders
    ) do method::MORMethod, r::Int64
        (; name, reduce_fn) = method
        if name == "prbt"
            rom = reduce_fn(r)
            irka_result = nothing
        else
            rom, irka_result = retry_irka(() -> reduce_fn(r),
                config.irka_tries, syssp)
        end

        rominf, romsp, = splitsys(rom)
        @assert all(dss2rm(todss(rominf)) .≈ dss2rm(todss(sysinf)))

        if config.compute_h2_errors
            abs_h2_error = gh2norm(todss(syssp - romsp))
            @info "Abs H2-error $name, r=$r: $abs_h2_error."
        else
            abs_h2_error = nothing
            rel_h2_error = nothing
        end

        return Result(
            name, r, irka_result,
            abs_h2_error, rom, isastable(rom), ispr(rom)
        )
    end

    save_results(experiment_id, results, sys)

    if config.plot_tf
        plot_tf(experiment_id, results, sys)
    end
    if config.compute_h2_errors
        plot_h2_errors(experiment_id, methods, results)
    end

    @info results
    return results, sys
end;
