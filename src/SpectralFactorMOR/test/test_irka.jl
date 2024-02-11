module IrkaTests

using Test
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
        sysinf, syssp, = splitsys(sys)
        rom, result = irka(sys, 10, IRKAOptions())
        @test rom isa DescriptorStateSpace
        @test result isa IRKAResult
        T(s) = Matrix(sys.C)/(s.*sys.E-sys.A)*sys.B+sys.D
        Tr(s) = Matrix(rom.C)/(s.*rom.E-rom.A)*rom.B+rom.D
        @test T(1e6) ≈ Tr(1e6)
        @test norm(T(0)-Tr(0)) < tol
    end
end

end # module IrkaTests
