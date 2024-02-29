module IrkaTests

using Test
using Random
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using SpectralFactorMOR
using DescriptorSystems

include("./problems.jl")
using .TestProblems

sys = test_setup_DAE1_RCL(1, 3)
(; E, A, B, C, D, M_0) = sys

T(s) = Matrix(C)/(s.*E-A)*B+D

@testset "i1interpolate real" begin
    r = 2
    s = [0.1, 10]
    c = ones(1,r)
    b = ones(1,r)
    Er,Ar,Br,Cr = i1interpolate(E,A,B,C,D,M_0,c,b,s)
    Tr(s) = Cr/(s.*Er-Ar)*Br+M_0
    @test T.(s) ≈ Tr.(s)
end

@testset "i1interpolate complex" begin
    s = [0.1+0.1im, 0.1-0.1im]
    c = [im -im]
    b = [im -im]
    Er,Ar,Br,Cr = i1interpolate(E,A,B,C,D,M_0,c,b,s)
    Tr(s) = Cr/(s.*Er-Ar)*Br+M_0
    @test T.(s) ≈ Tr.(s)
    @test T.(s) ≈ Tr.(s)
end

const problems = [
    ("irka index 0 SISO", test_setup_DAE0_RCL(1), 1e-3),
    ("irka index 0 MIMO", test_setup_DAE0_RCL(2), 1e-1),
    ("irka index 1 SISO", test_setup_DAE1_RCL(1), 1e-3),
    ("irka index 1 MIMO", test_setup_DAE1_RCL(2), 1e-1),
    ("irka index 2 SISO", test_setup_DAE2_RCL(1), 1e-3),
    ("irka index 2 MIMO", test_setup_DAE2_RCL(2), 1e-1)
]

for problem in problems
    testname, sys, tol = problem
    @testset "$testname" begin
        Random.seed!(0)
        sysinf, syssp, = splitsys(sys)
        rom, result = irka(sys, 10, IRKAOptions())
        rominf, romsp, = splitsys(rom)
        @test rom isa AbstractDescriptorStateSpace
        @test result isa IRKAResult
        @test all(dss2rm(todss(rominf)) .≈ dss2rm(todss(sysinf)))
        @test h2normsp(syssp - romsp) < tol
    end
end

end # module IrkaTests
