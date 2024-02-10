"""
                |E_11    0  0 0|      | 0   0  0 I|      |B_1|
            E = |  0   E_22 0 0|, A = | 0 A_22 0 0|, B = |B_2|, C = |C_1 C_2 C_3 C_4|
                |  0     0  0 0|      | 0   0  I 0|      |B_3|
                |  0     0  0 0|      |-I   0  0 0|      |B_4|

The matrices E_11 and E_22 are nonsingular.
"""
Base.@kwdef struct AlmostKroneckerDAE{Tv,Ti} <: AbstractSparseDescriptorStateSpace{Tv,Ti}
    E::SparseMatrixCSC{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    B::SparseMatrixCSC{Tv,Ti}
    C::SparseMatrixCSC{Tv,Ti}
    D::Matrix{Tv}
    n_1::Ti
    n_2::Ti
    n_3::Ti
    n_4::Ti
    Ts = 0.0
    n::Ti = size(E, 1)
    m::Ti = size(B, 2)
    idx_1 = 1:n_1
    idx_2 = n_1+1:n_1+n_2
    idx_3 = n_1+n_2+1:n_1+n_2+n_3
    idx_4 = n_1+n_2+n_3+1:n_1+n_2+n_3+n_4
    E_11 = E[idx_1, idx_1]
    E_22 = E[idx_2, idx_2]
    A_22 = A[idx_2, idx_2]
    B_1 = B[idx_1, :]
    B_2 = B[idx_2, :]
    B_3 = B[idx_3, :]
    B_4 = B[idx_4, :]
    C_1 = C[:, idx_1]
    C_2 = C[:, idx_2]
    C_3 = C[:, idx_3]
    C_4 = C[:, idx_4]
    M_0 = Matrix(C_1*B_4 - C_3*B_3 - C_4*B_1 + D) # [SchMMV23]
    M_1 = Matrix(C_4*E_11*B_4) # [SchMMV23]
end
