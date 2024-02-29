module SfmorTests

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
    ("sfmor index 0 SISO", test_setup_DAE0_RCL(1), 1e-3),
    ("sfmor index 0 MIMO", test_setup_DAE0_RCL(2), 1e-1),
    ("sfmor index 1 SISO", test_setup_DAE1_RCL(1), 1e-3),
    ("sfmor index 1 MIMO", test_setup_DAE1_RCL(2), 1e-1),
    ("sfmor index 2 SISO", test_setup_DAE2_RCL(1), 1e-3),
    ("sfmor index 2 MIMO", test_setup_DAE2_RCL(2), 1e-1)
]

for problem in problems
    testname, sys, tol = problem
    @testset "$testname" begin
        Random.seed!(0)
        sysinf, syssp, = splitsys(sys)
        rom, result = sfmor(sys, 10, IRKAOptions())
        rominf, romsp, = splitsys(rom)
        @test rom isa AbstractDescriptorStateSpace
        @test result isa IRKAResult
        @test all(dss2rm(todss(rominf)) .≈ dss2rm(todss(sysinf)))
        @test h2normsp(syssp - romsp) < tol
        @test ispr(rom)[1]
    end
end

end # module SfmorTests
