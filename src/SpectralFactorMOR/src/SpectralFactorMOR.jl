module SpectralFactorMOR

import Base: +, -
using LinearAlgebra
using SparseArrays
using Random
using Logging
using Memoize
using DescriptorSystems
using MatrixEquations

export SparseDescriptorStateSpace, SemiExplicitIndex1DAE, StaircaseDAE
export splitsys, todss, toindex0, tokronecker
export checkprojlure, is_valid
export isastable, ispr, h2normsp
export lyapc_lradi, lrcf
export pr_o_gramian_lr, pr_c_gramian_lr, pr_o_gramian, pr_c_gramian
export IRKAResult, IRKAOptions, irka, retry_irka, i0interpolate, i1interpolate
export prbaltrunc
export sfmor

# parametric-type variant
abstract type AbstractDescriptorStateSpaceT{Tv} <: AbstractDescriptorStateSpace end

include("types/AlmostKroneckerDAE.jl")
include("types/SemiExplicitIndex1DAE.jl")
include("types/UnstructuredDAE.jl")
include("types/StaircaseDAE.jl")
include("irka.jl")
include("lyapunov.jl")
include("riccati.jl")
include("sfmor.jl")
include("bt.jl")
include("sys.jl")

end # module SpectralFactorMOR
