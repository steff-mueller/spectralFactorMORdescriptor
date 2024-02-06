module spectralFactorMORdescriptor

using Random
using Logging
using MAT
using PortHamiltonianBenchmarkSystems
using DescriptorSystems

export start_experiment, run_methods, save_results, plot_tf, plot_h2_errors
export load_toml_config, to_symbol_dict
export MORMethod, Result, get_problem, compute_gramians
export SparseDescriptorStateSpace, SemiExplicitIndex1DAE, StaircaseDAE
export splitsys, todss, checkprojlure, toindex0, toindex0sm, is_valid
export isastable, ispr
export lyapc_lradi
export pr_o_gramian_lr, pr_c_gramian_lr, pr_o_gramian, pr_c_gramian
export IRKAOptions, irka, retry_irka, i0interpolate, i1interpolate
export prbaltrunc
export sfmor

include("types.jl")
include("irka.jl")
include("utils.jl")
include("lyapunov.jl")
include("riccati.jl")
include("sfmor.jl")
include("bt.jl")
include("plotting.jl")
include("sys.jl")
include("problems.jl")

end # module spectralFactorMORdescriptor
