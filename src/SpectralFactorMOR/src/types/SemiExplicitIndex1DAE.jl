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

The matrix E_11 is nonsingular.
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
    n = size(E_i0, 1)
    return SemiExplicitIndex1DAE(E_i0, A_i0, B_i0, C_i0, D_i0, n)
end
