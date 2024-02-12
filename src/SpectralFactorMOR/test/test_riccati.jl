module RiccatiTests

using Test
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using DescriptorSystems
using SpectralFactorMOR

include("./problems.jl")
using .TestProblems

sys = test_setup_DAE1_RCL(1)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

@testset "pr_o_gramian_lr index 1 SISO" begin
    Z = pr_o_gramian_lr(sys)
    X = Z*Z'
    @test P_l'*X*P_l ≈ X
    err = norm(A'*X*E + E'*X*A
        + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
    @test err < 1e-12
    @info "err pr_o_gramian_lr (index 1): $err"
end

@testset "pr_c_gramian_lr index 1 SISO " begin
    Z = pr_c_gramian_lr(sys)
    X = Z*Z'
    @test P_r*X*P_r' ≈ X
    err = norm(A*X*E'+E*X*A'
        + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
    @test err < 1e-12
    @info "err pr_c_gramian_lr (index 1): $err"
end

@testset "pr_o_gramian_lr index 0 SISO" begin
    sys_i0 = toindex0(sys)
    compute_err(X) = norm(sys_i0.A'*X*sys_i0.E + sys_i0.E'*X*sys_i0.A 
        + (sys_i0.C'-sys_i0.E'*X*sys_i0.B)
        /(sys_i0.D+sys_i0.D')*(sys_i0.C-sys_i0.B'*X*sys_i0.E))

    Z = pr_o_gramian_lr(sys_i0)
    err = compute_err(Z*Z')
    @info "err pr_o_gramian_lr (index 0): $err"
    @test err < 1e-12
end

sys = test_setup_DAE1_RCL(2)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

@testset "pr_o_gramian index 1 MIMO" begin
    X = pr_o_gramian(sys)
    @test P_l'*X*P_l ≈ X
    err = norm(A'*X*E + E'*X*A
        + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
    @test err < 1e-6
    @info "err pr_o_gramian MIMO (index 1): $err"
end

@testset "pr_c_gramian index 1 MIMO" begin
    X = pr_c_gramian(sys)
    @test P_r*X*P_r' ≈ X
    err = norm(A*X*E'+E*X*A'
        + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
    @test err < 1e-6
    @info "err pr_c_gramian MIMO (index 1): $err"
end

sys = test_setup_DAE2_RCL(1)
@test is_valid(sys)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

@testset "pr_o_gramian index 2 SISO" begin
    X = pr_o_gramian(sys)
    @test P_l'*X*P_l ≈ X
    err = norm(A'*X*E + E'*X*A
        + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
    @test err < 1e-4
    @info "err pr_o_gramian SISO (index 2): $err"
end

@testset "pr_c_gramian index 2 SISO" begin
    X = pr_c_gramian(sys)
    @test P_r*X*P_r' ≈ X
    err = norm(A*X*E'+E*X*A'
        + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
    @test err < 1e-5
    @info "err pr_c_gramian SISO (index 2): $err"
end

sys = test_setup_DAE2_RCL(2)
@test is_valid(sys)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

@testset "pr_o_gramian index 2 MIMO" begin
    X = pr_o_gramian(sys)
    @test P_l'*X*P_l ≈ X
    err = norm(A'*X*E + E'*X*A
        + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
    @test err < 1e-4
    @info "err pr_o_gramian MIMO (index 2): $err"
end

@testset "pr_c_gramian index 2 MIMO" begin
    X = pr_c_gramian(sys)
    @test P_r*X*P_r' ≈ X
    err = norm(A*X*E'+E*X*A'
        + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
    @test err < 1e-4
    @info "err pr_c_gramian MIMO (index 2): $err"
end

end # module RiccatiTests
