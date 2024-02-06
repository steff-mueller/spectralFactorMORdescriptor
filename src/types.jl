using SparseArrays
using Memoize
using DescriptorSystems: AbstractDescriptorStateSpace, DescriptorStateSpace

import Base: +, -

abstract type AbstractSparseDescriptorStateSpace{Tv,Ti} <: AbstractDescriptorStateSpace end

function todss(sys::AbstractSparseDescriptorStateSpace)
    (; E, A, B, C, D) = sys
    return dss(A, E, B, C, D)
end

struct SparseDescriptorStateSpace{Tv,Ti} <: AbstractSparseDescriptorStateSpace{Tv,Ti}
    E::SparseMatrixCSC{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    B::SparseMatrixCSC{Tv,Ti}
    C::SparseMatrixCSC{Tv,Ti}
    D::Matrix{Tv}
    Ts::Float64
end

"""
            | E_11 0 | 
        E = |        | 
            | 0    0 |


            | A_11 A_12 |
        A = |           |
            | A_21 A_22 |

            | B_1 |
        B = |     |
            | B_2 |

        C = | C_1 C_2 |
"""
Base.@kwdef struct SemiExplicitIndex1DAE{Tv,Ti} <: AbstractSparseDescriptorStateSpace{Tv,Ti}
    E::SparseMatrixCSC{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    B::SparseMatrixCSC{Tv,Ti}
    C::SparseMatrixCSC{Tv,Ti}
    D::Matrix{Tv}
    Ts = 0.0
    n_1::Ti
    E_11::SparseMatrixCSC{Tv,Ti} = E[1:n_1,1:n_1]
    A_11::SparseMatrixCSC{Tv,Ti} = A[1:n_1,1:n_1]
    A_12::SparseMatrixCSC{Tv,Ti} = A[1:n_1,n_1+1:end]
    A_21::SparseMatrixCSC{Tv,Ti} = A[n_1+1:end,1:n_1]
    A_22::SparseMatrixCSC{Tv,Ti} = A[n_1+1:end,n_1+1:end]
    B_1::SparseMatrixCSC{Tv,Ti} = B[1:n_1,1:end]
    B_2::SparseMatrixCSC{Tv,Ti} = B[n_1+1:end,1:end]
    C_1::SparseMatrixCSC{Tv,Ti} = C[1:end,1:n_1]
    C_2::SparseMatrixCSC{Tv,Ti} = C[1:end,n_1+1:end]
    n = size(A, 1)
    n_2 = n-n_1
    P_l = [
        I -A_12/Matrix(A_22)
        spzeros(n_2, n_1) spzeros(n_2, n_2)
    ]
    P_r =  [
        I spzeros(n_1, n_2);
        -Matrix(A_22)\A_21 spzeros(n_2, n_2)
    ]
    M_0 = D - C_2/Matrix(A_22)*B_2
end

function SemiExplicitIndex1DAE(
    E::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    B::SparseMatrixCSC{Tv,Ti},
    C::SparseMatrixCSC{Tv,Ti},
    D::Matrix{Tv},
    n_1::Ti
) where {Tv,Ti}
    SemiExplicitIndex1DAE(E=E,A=A,B=B,C=C,D=D,n_1=n_1)
end

function +(sys1::AbstractSparseDescriptorStateSpace{Tv1,Ti}, sys2::DescriptorStateSpace{Tv2}) where {Tv1,Tv2,Ti}
    # We assume that the dense system `sys2` has a much smaller state space dimension than
    # the sparse system `sys1`. Thus, we return a `SparseDescriptorStateSpace` to maintain
    # the sparsity.
    T = promote_type(Tv1, Tv2)
    n1 = size(sys1.A, 1)
    n2 = sys2.nx
    A = [sys1.A  spzeros(T,n1,n2);
         spzeros(T,n2,n1) sys2.A]
    E = [sys1.E  spzeros(T,n1,n2);
         spzeros(T,n2,n1) sys2.E]
    B = [sys1.B ; sys2.B]
    C = [sys1.C sys2.C;]
    D = [sys1.D + sys2.D;]
    return SparseDescriptorStateSpace(E, A, B, C, D, 0.0)
end

function +(sys1::DescriptorStateSpace{Tv1}, sys2::AbstractSparseDescriptorStateSpace{Tv2,Ti}) where {Tv1,Tv2,Ti}
    return sys2 + sys1
end

# TODO add methods for `SemiExplicitIndex1DAE` etc. to preserve structure.
function -(sys::AbstractSparseDescriptorStateSpace{Tv,Ti}) where {Tv,Ti}
    (; E, A, B, C, D) = sys
    return SparseDescriptorStateSpace(E, A, B, -C, -D, 0.0)
end

function -(sys1::AbstractSparseDescriptorStateSpace{Tv1,Ti}, sys2::DescriptorStateSpace{Tv2}) where {Tv1,Tv2,Ti}
    return sys1 + (-sys2)
end

function -(sys1::DescriptorStateSpace{Tv1}, sys2::AbstractSparseDescriptorStateSpace{Tv2,Ti}) where {Tv1,Tv2,Ti}
    return sys1 + (-sys2)
end

"""
    toindex0(sys::SemiExplicitIndex1DAE)

Transform `sys` to a system with index 0.
"""
function toindex0(sys::SemiExplicitIndex1DAE)
    E_i0 = sys.E_11
    A_i0 = sparse(sys.A_11 - sys.A_12/Matrix(sys.A_22)*sys.A_21)
    B_i0 = sparse(sys.B_1 - sys.A_12/Matrix(sys.A_22)*sys.B_2)
    C_i0 = sparse(sys.C_1 - sys.C_2/Matrix(sys.A_22)*sys.A_21)
    D_i0 = sys.D - sys.C_2/Matrix(sys.A_22)*sys.B_2
    return SparseDescriptorStateSpace(E_i0, A_i0, B_i0, C_i0, D_i0, 0.0)
end

function toindex0sm(sys::SemiExplicitIndex1DAE)
    (; E, A, B, C, D) = toindex0(sys)
    n = size(A,1)
    return SemiExplicitIndex1DAE(E, A, B, C, D, n)
end

"""
    splitsys(sys::SemiExplicitIndex1DAE)

Return infinite and strictly proper subsystems.

- `sys1` is the infinite subsystem.
- `sys2` is the strictly proper subsystem.
"""
function splitsys(sys::SemiExplicitIndex1DAE)
    (; E, A, B, C, D, n_1) = sys
    n = size(A,1)
    n_2 = n - n_1

    # TODO Why not sparse?
    Q = [I -sys.A_12/Matrix(sys.A_22);
         spzeros(n_2, n_1) I]
    Z = [I spzeros(n_1, n_2);
         -Matrix(sys.A_22)\sys.A_21 I]

    A = sparse(Q*A*Z)
    B = sparse(Q*B)
    C = sparse(C*Z)
    i1 = 1:n_1; i2 = n_1+1:n
    return (
        SparseDescriptorStateSpace(
            E[i2,i2], A[i2,i2], B[i2,:], C[:,i2], D, 0.0),
        SparseDescriptorStateSpace(
            E[i1,i1], A[i1,i1], B[i1,:], C[:,i1], zeros(size(D)...), 0.0),
        Q, Z
    )
end

"""
Check if `X` satisfies the projected positive real Lur'e equation.
"""
function checkprojlure(sys, X)
    (; E, A, B, C, M_0, P_l, P_r) = sys
    projLure = [
        -A'*X*E-E'*X*A P_r'*C'-E'*X*B;
        C*P_r-B'*X*E M_0+M_0'
    ]
    s = eigvals(Symmetric(Matrix(projLure)))
    return (
        minimum(s) >= -1e-12
        && norm(X'-X) < 1e-14
        && norm(P_l'*X*P_l-X) < 1e-14
    )
end

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
    # TODO Why not sparse already?
    L_A = sparse([I spzeros(n_1, n_2) -A_13/Matrix(A_33) (-A_11 + A_13/Matrix(A_33)*A_31)/Matrix(A_41);
        spzeros(n_2, n_1) I -A_23/Matrix(A_33) (-A_21 + A_23/Matrix(A_33)*A_31)/Matrix(A_41);
        spzeros(n_3, n_1) spzeros(n_3, n_2) inv(Matrix(A_33)) -Matrix(A_33)\A_31/Matrix(A_41);
        spzeros(n_4, n_1) spzeros(n_4, n_2) spzeros(n_4, n_3) -inv(Matrix(A_41))])

    dropzeros!(L_A)

    # TODO Why not sparse already?
    Z_A = sparse([I spzeros(n_1, n_2) spzeros(n_1, n_3) spzeros(n_1, n_4);
        spzeros(n_2, n_1) I spzeros(n_2, n_3) spzeros(n_2, n_4);
        spzeros(n_3, n_1) -Matrix(A_33)\A_32 I spzeros(n_3, n_4);
        spzeros(n_4, n_1) Matrix(A_14)\(-A_12 + A_13/Matrix(A_33)*A_32) spzeros(n_4, n_3) inv(Matrix(A_14))])

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
Assert that `sys` is in staircase form.
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

"""
                |E_11    0  0 0|      | 0   0  0 I|      |B_1|
            E = |  0   E_22 0 0|, A = | 0 A_22 0 0|, B = |B_2|, C = |C_1 C_2 C_3 C_4|
                |  0     0  0 0|      | 0   0  I 0|      |B_3|
                |  0     0  0 0|      |-I   0  0 0|      |B_4|

The matrices E_11 and E_22 are positive definite.
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

"""
    splitsys(sys::AlmostKroneckerDAE)

Return infinite and strictly proper subsystems.

- `sys1` is the infinite subsystem.
- `sys2` is the strictly proper subsystem.
"""
function splitsys(sys::AlmostKroneckerDAE)
    (; E, A, B, C, D, A_22, E_22, B_2, C_2, idx_1, idx_3, idx_4, m) = sys

    idx_inf = [idx_1; idx_3; idx_4]
    return SparseDescriptorStateSpace(
               E[idx_inf, idx_inf], A[idx_inf, idx_inf],
               B[idx_inf,:], C[:,idx_inf], D, 0.0),
           SparseDescriptorStateSpace(
               E_22, A_22, B_2, C_2, zeros(m, m), 0.0)
end

"""
    splitsys(sys::StaircaseDAE)

Return infinite and strictly proper subsystems.

- `sys1` is the infinite subsystem.
- `sys2` is the strictly proper subsystem.
"""
function splitsys(sys::StaircaseDAE)
    sys_kronecker, = tokronecker(sys)
    return splitsys(sys_kronecker)
end
