"""
                |E_11    0  0 0|      |A_11 A_12 A_13 A_14|      |B_1|
            E = |  0   E_22 0 0|, A = |A_21 A_22 A_23   0 |, B = |B_2|, C = |C_1 C_2 C_3 C_4|
                |  0     0  0 0|      |A_31 A_32 A_33   0 |      |B_3|
                |  0     0  0 0|      |A_41   0    0    0 |      |B_4|

    with E_11, E_22, A_14=-A_41^T, A_33 nonsingular
"""
Base.@kwdef struct StaircaseDAE{Tv,Ti} <: AbstractSparseDescriptorStateSpace{Tv,Ti}
    E::SparseMatrixCSC{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    B::SparseMatrixCSC{Tv,Ti}
    C::SparseMatrixCSC{Tv,Ti}
    D::Matrix{Tv}
    n_1::Ti
    n_2::Ti
    n_3::Ti
    n_4::Ti
    n = size(E, 1)
    m = size(B, 2)
    Ts = 0.0
    idx_1 = 1:n_1
    idx_2 = n_1+1:n_1+n_2
    idx_3 = n_1+n_2+1:n_1+n_2+n_3
    idx_4 = n_1+n_2+n_3+1:n_1+n_2+n_3+n_4
    E_11 = E[idx_1, idx_1]
    E_22 = E[idx_2, idx_2]
    A_11 = A[idx_1, idx_1]
    A_21 = A[idx_2, idx_1]
    A_31 = A[idx_3, idx_1]
    A_41 = A[idx_4, idx_1]
    A_12 = A[idx_1, idx_2]
    A_22 = A[idx_2, idx_2]
    A_32 = A[idx_3, idx_2]
    A_13 = A[idx_1, idx_3]
    A_23 = A[idx_2, idx_3]
    A_33 = A[idx_3, idx_3]
    A_14 = A[idx_1, idx_4]
end

"""
Transform `sys` into almost Kronecker form [AchAM2021].
"""
@memoize function tokronecker(sys::StaircaseDAE)
    (; A_11, A_13, A_33, A_31, A_41, A_23, A_21, A_32, A_14, A_12,
       n_1, n_2, n_3, n_4) = sys

    L_A = [sparse(I, n_1, n_1) spzeros(n_1, n_2) -A_13/Matrix(A_33) (-A_11 + A_13/Matrix(A_33)*A_31)/Matrix(A_41);
        spzeros(n_2, n_1) sparse(I, n_2, n_2) -A_23/Matrix(A_33) (-A_21 + A_23/Matrix(A_33)*A_31)/Matrix(A_41);
        spzeros(n_3, n_1) spzeros(n_3, n_2) inv(Matrix(A_33)) -Matrix(A_33)\A_31/Matrix(A_41);
        spzeros(n_4, n_1) spzeros(n_4, n_2) spzeros(n_4, n_3) -inv(Matrix(A_41))]

    dropzeros!(L_A)

    Z_A = [sparse(I, n_1, n_1) spzeros(n_1, n_2) spzeros(n_1, n_3) spzeros(n_1, n_4);
        spzeros(n_2, n_1) sparse(I, n_2, n_2) spzeros(n_2, n_3) spzeros(n_2, n_4);
        spzeros(n_3, n_1) -Matrix(A_33)\A_32 I spzeros(n_3, n_4);
        spzeros(n_4, n_1) Matrix(A_14)\(-A_12 + A_13/Matrix(A_33)*A_32) spzeros(n_4, n_3) inv(Matrix(A_14))]

    return AlmostKroneckerDAE(
        E = dropzeros(L_A*sys.E*Z_A),
        A = dropzeros(L_A*sys.A*Z_A),
        B = L_A*sys.B,
        C = sys.C*Z_A,
        D = sys.D,
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    ), L_A, Z_A
end

function Base.getproperty(sys::StaircaseDAE, sym::Symbol)
    if sym == :P_r
        (; n, n_1, n_2, n_3, n_4) = sys
        _, _, Z_A = tokronecker(sys)
        # TODO simplify: use explicit formula
        sel = Matrix([spzeros(n_1, n);
                      spzeros(n_2, n_1) I spzeros(n_2, n_3+n_4);
                      spzeros(n_3+n_4, n)])
        return sparse(Z_A*sel/Z_A)
    elseif sym == :P_l
        (; n, n_1, n_2, n_3, n_4) = sys
        _, L_A = tokronecker(sys)
        # TODO simplify: use explicit formula
        sel = Matrix([spzeros(n_1, n);
                      spzeros(n_2, n_1) I spzeros(n_2, n_3+n_4);
                      spzeros(n_3+n_4, n)])
        return sparse(L_A\sel*L_A)
    elseif sym == :M_0
        sys_kronecker, = tokronecker(sys)
        return sys_kronecker.M_0
    elseif sym == :M_1
        sys_kronecker, = tokronecker(sys)
        return sys_kronecker.M_1
    else
        return getfield(sys, sym)
    end
end

Base.propertynames(sys::StaircaseDAE) =
    (:P_r, :P_l, :M_0, :M_1, fieldnames(typeof(sys))...)

"""
Assert that `sys` is in valid staircase form.
"""
function is_valid(sys::StaircaseDAE)
    (; E_11, E_22, A_33, A_14, A_41, n_1, n_2, n_3, n_4,
       E, A, idx_1, idx_2, idx_3, idx_4) = sys
    return (
        E==E'
        && rank(E_11) == n_1
        && rank(E_22) == n_2
        && nnz(E[idx_1, idx_2]) == 0
        && nnz(E[:, [idx_3; idx_4]]) == 0
        && rank(A_33) == n_3
        && rank(A_14) == n_4
        && A_14 == -A_41'
        && nnz(A[[idx_2; idx_3; idx_4], idx_4]) == 0
        && nnz(A[idx_4, [idx_2; idx_3; idx_4]]) == 0
    )
end
