module PrbtTests

using Test
using Random
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using SpectralFactorMOR
using DescriptorSystems

include("./problems.jl")
using .TestProblems

const problems = [
    ("prbt index 0 SISO", test_setup_DAE0_RCL(1), 1e-3),
    ("prbt index 0 MIMO", test_setup_DAE0_RCL(2), 1e-1),
    ("prbt index 1 SISO", test_setup_DAE1_RCL(1), 1e-3),
    ("prbt index 1 MIMO", test_setup_DAE1_RCL(2), 1e-1),
    ("prbt index 2 SISO", test_setup_DAE2_RCL(1), 1e-2),
    ("prbt index 2 MIMO", test_setup_DAE2_RCL(2), 1e-1)
]

for problem in problems
    testname, sys, tol = problem
    @testset "$testname" begin
        Random.seed!(0)

        Xo = pr_o_gramian(sys)
        Zo = lrcf(Matrix(Xo), 1e-9)'
        Xc = pr_c_gramian(sys)
        Zc = lrcf(Matrix(Xc), 1e-9)'

        sysinf, syssp, = splitsys(sys)
        rom = prbaltrunc(sys, 10, Zc, Zo)
        rominf, romsp, = splitsys(rom)
        @test rom isa AbstractDescriptorStateSpace
        @test all(dss2rm(todss(rominf)) .≈ dss2rm(todss(sysinf)))
        @test h2normsp(syssp - romsp) < tol
        @test ispr(rom)[1]
        @test norm(pr_o_gramian(rom) - pr_c_gramian(rom)) < 1e-7
    end
end

end # module PrbtTests
