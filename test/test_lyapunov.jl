using Test
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using DescriptorSystems
using spectralFactorMORdescriptor

E, J, R, Q, G = setup_DAE1_RCL_LadderNetwork_sparse()

A = (J-R)*Q
B = G
C = SparseMatrixCSC(G')
D = zeros(size(G,2),size(G,2))
n = size(A,1)
n_1 = nnz(E)
n_2 = n-n_1
sys = SemiExplicitIndex1DAE(E,A,B,C,D,n_1)

P_r = [
    I spzeros(n_1, n_2);
    -Matrix(sys.A_22)\sys.A_21 spzeros(n_2, n_2)
]
P_l = [
    I -sys.A_12/Matrix(sys.A_22)
    spzeros(n_2, n_1) spzeros(n_2, n_2)
]
Q_r = [
    spzeros(n_1, n_1) spzeros(n_1, n_2);
    Matrix(sys.A_22)\sys.A_21 I
]
Q_l = [
    spzeros(n_1, n_1) sys.A_12/Matrix(sys.A_22);
    spzeros(n_2, n_1) I
]

Z = lyapc_lradi(A, E, B, P_l; tol=1e-14, max_iterations=200)
@test Z isa Matrix{Float64}
X = Z*Z'
@test P_r*X*P_r' ≈ X
err = norm(E*X*A' + A*X*E' + P_l*B*B'*P_l')
@info "lyapc_lradi error (index 1): $err"
@test err < 1e-12

sys_i0 = toindex0(sys)
Z = lyapc_lradi(sys_i0.A, sys_i0.E, sys_i0.B, I; tol=1e-14, max_iterations=200)
@test Z isa Matrix{Float64}
X = Z*Z'
err = norm(sys_i0.E*X*sys_i0.A' + sys_i0.A*X*sys_i0.E' + sys_i0.B*sys_i0.B')
@info "lyapc_lradi error (index 0): $err"
@test err < 1e-12

A = [-1 0; 3 -1]
E = I
B = hcat([1; 1])
Z = lyapc_lradi(A,E,B)
@test Z isa Matrix{Float64}
X = Z*Z'
err = norm(E*X*A'+A*X*E'+B*B')
@test err < 1e-12
