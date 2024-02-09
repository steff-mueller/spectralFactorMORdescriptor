using Test
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using DescriptorSystems
using SpectralFactorMOR

function riccati_test_setup_DAE1_RCL(m)
    E, J, R, Q, G = setup_DAE1_RCL_LadderNetwork_sparse(ns=50, m=m)
    A = (J-R)*Q
    B = G
    C = SparseMatrixCSC(G')
    # add regularization term such that M_0+M_0' is nonsingular for m=2
    D = m==2 ? 1e-12*[1 0; 0 1] : zeros(1,1)
    n_1 = nnz(E)
    return SemiExplicitIndex1DAE(E,A,B,C,D,n_1)
end

function riccati_test_setup_DAE2_RCL(m)
    E, J, R, G = setup_DAE2_RCL_LadderNetwork_sparse(ns = 50, m = m)

    # Transform into staircase form [AchAM2021]
    s, UE = eigen(Diagonal(E); sortby=x -> -x)
    rank_E = count(s .!= 0.0)
    n = size(E, 1)
    n_1 = n_4 = 1
    n_2 = rank_E - 1
    n_3 = n - n_1 - n_2 - n_4

    if m == 2
        # Re-order the last two columns/rows for MIMO case    
        UE = (UE * [I spzeros(n-2, 2);
                    spzeros(1, n-2) 0 1;
                    spzeros(1, n-2) 1 0])
    end

    return StaircaseDAE(
        E = sparse(Diagonal(s)),
        A = sparse(UE'*(J-R)*UE),
        B = sparse(UE'*G),
        C = sparse(G'*UE),
        # add regularization term such that M_0+M_0' is nonsingular
        D = 1e-12*Matrix(I, m, m),
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    )
end

sys = riccati_test_setup_DAE1_RCL(1)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

Z = pr_o_gramian_lr(sys)
X = Z*Z'
@test P_l'*X*P_l ≈ X
err = norm(A'*X*E + E'*X*A
    + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
@test err < 1e-12
@info "err pr_o_gramian_lr (index 1): $err"

Z = pr_c_gramian_lr(sys)
X = Z*Z'
@test P_r*X*P_r' ≈ X
err = norm(A*X*E'+E*X*A'
    + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
@test err < 1e-12
@info "err pr_c_gramian_lr (index 1): $err" 

sys_i0 = toindex0(sys)
compute_err(X) = norm(sys_i0.A'*X*sys_i0.E + sys_i0.E'*X*sys_i0.A 
    + (sys_i0.C'-sys_i0.E'*X*sys_i0.B)
       /(sys_i0.D+sys_i0.D')*(sys_i0.C-sys_i0.B'*X*sys_i0.E))

Z = pr_o_gramian_lr(sys_i0)
err = compute_err(Z*Z')
@info "err pr_o_gramian_lr (index 0): $err"
@test err < 1e-12

sys = riccati_test_setup_DAE1_RCL(2)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

X = pr_o_gramian(sys)
@test P_l'*X*P_l ≈ X
err = norm(A'*X*E + E'*X*A
    + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
@test err < 1e-6
@info "err pr_o_gramian MIMO (index 1): $err"

X = pr_c_gramian(sys)
@test P_r*X*P_r' ≈ X
err = norm(A*X*E'+E*X*A'
    + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
@test err < 1e-6
@info "err pr_c_gramian MIMO (index 1): $err"

sys = riccati_test_setup_DAE2_RCL(1)
@test is_valid(sys)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

X = pr_o_gramian(sys)
@test P_l'*X*P_l ≈ X
err = norm(A'*X*E + E'*X*A
    + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
@test err < 1e-5
@info "err pr_o_gramian SISO (index 2): $err"

X = pr_c_gramian(sys)
@test P_r*X*P_r' ≈ X
err = norm(A*X*E'+E*X*A'
    + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
@test err < 1e-5
@info "err pr_c_gramian SISO (index 2): $err"

sys = riccati_test_setup_DAE2_RCL(2)
@test is_valid(sys)
(; E, A, B, C, D, P_r, P_l, M_0) = sys

X = pr_o_gramian(sys)
@test P_l'*X*P_l ≈ X
err = norm(A'*X*E + E'*X*A
    + (P_r'*C'-E'*X*B)/(M_0+M_0')*(C*P_r-B'*X*E))
@test err < 1e-4
@info "err pr_o_gramian MIMO (index 2): $err"

X = pr_c_gramian(sys)
@test P_r*X*P_r' ≈ X
err = norm(A*X*E'+E*X*A'
    + (P_l*B-E*X*C')/(M_0+M_0')*(B'*P_l'-C*X*E'))
@test err < 1e-4
@info "err pr_c_gramian MIMO (index 2): $err"
