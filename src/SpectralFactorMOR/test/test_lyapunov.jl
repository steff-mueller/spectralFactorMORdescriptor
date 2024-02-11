module LyapunovTests

using Test
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using DescriptorSystems
using SpectralFactorMOR

include("./problems.jl")
using .TestProblems

sys = test_setup_DAE1_RCL()
(; E, A, B, P_l, P_r) = sys

@testset "lyapc_lradi index 1" begin
    Z = lyapc_lradi(A, E, B, P_l; tol=1e-14, max_iterations=200)
    @test Z isa Matrix{Float64}
    X = Z*Z'
    @test P_r*X*P_r' ≈ X
    err = norm(E*X*A' + A*X*E' + P_l*B*B'*P_l')
    @info "lyapc_lradi error (index 1): $err"
    @test err < 1e-12
end

@testset "lypac_lradi index 0" begin
    sys_i0 = test_setup_DAE0_RCL()
    Z = lyapc_lradi(sys_i0.A, sys_i0.E, sys_i0.B, I; tol=1e-14, max_iterations=200)
    @test Z isa Matrix{Float64}
    X = Z*Z'
    err = norm(sys_i0.E*X*sys_i0.A' + sys_i0.A*X*sys_i0.E' + sys_i0.B*sys_i0.B')
    @info "lyapc_lradi error (index 0): $err"
    @test err < 1e-12
end

@testset "lyapc_lradi index 0 simple" begin
    A = [-1 0; 3 -1]
    E = I
    B = hcat([1; 1])
    Z = lyapc_lradi(A,E,B)
    @test Z isa Matrix{Float64}
    X = Z*Z'
    err = norm(E*X*A'+A*X*E'+B*B')
    @test err < 1e-12
end

end # module LyapunovTests
