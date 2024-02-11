module IrkaTests

using Test
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using SpectralFactorMOR

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

end # module IrkaTests
