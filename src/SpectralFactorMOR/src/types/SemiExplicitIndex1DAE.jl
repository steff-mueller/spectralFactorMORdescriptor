"""
Represents a system of the form

```math
E = \\begin{bmatrix}
    E_{11} & 0\\\\
    0 & 0
\\end{bmatrix}, \\quad
A = \\begin{bmatrix}
    A_{11} & A_{12}\\\\
    A_{21} & A_{22}
\\end{bmatrix}
```

with ``E_{11}`` and ``A_{22}`` nonsingular. The system matrices ``B`` and ``C``
are partitioned accordingly as

```math
B = \\begin{bmatrix}
    B_1\\\\
    B_2
\\end{bmatrix}, \\quad
C = \\begin{bmatrix}
    C_1 & C_2
\\end{bmatrix}.
```
"""
Base.@kwdef struct SemiExplicitIndex1DAE{Tv, Ti, T <: AbstractMatrix{Tv}} <: AbstractDescriptorStateSpaceT{Tv}
    E::T
    A::T
    B::T
    C::T
    D::Matrix{Tv}
    n_1::Ti
    Ts = 0.0
    E_11 = E[1:n_1,1:n_1]
    A_11 = A[1:n_1,1:n_1]
    A_12 = A[1:n_1,n_1+1:end]
    A_21 = A[n_1+1:end,1:n_1]
    A_22 = A[n_1+1:end,n_1+1:end]
    B_1 = B[1:n_1,1:end]
    B_2 = B[n_1+1:end,1:end]
    C_1 = C[1:end,1:n_1]
    C_2 = C[1:end,n_1+1:end]
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
    E::T,
    A::T,
    B::T,
    C::T,
    D::Matrix{Tv},
    n_1::Ti
) where {Tv, Ti, T <: AbstractMatrix{Tv}}
    SemiExplicitIndex1DAE(E=E,A=A,B=B,C=C,D=D,n_1=n_1)
end

"""
    toindex0(sys::SemiExplicitIndex1DAE)

Transforms `sys` to a system with index 0.
Returns a system of type [`SemiExplicitIndex1DAE`](@ref).
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
