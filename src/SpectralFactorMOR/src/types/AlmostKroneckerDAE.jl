"""
Represents a system of the form

```math
E = \\begin{bmatrix}
    E_{11} & 0      & 0 & 0\\\\
    0      & E_{22} & 0 & 0\\\\
    0      & 0      & 0 & 0\\\\
    0      & 0      & 0 & 0
\\end{bmatrix}, \\quad
A = \\begin{bmatrix}
    0 & 0      & 0 & I\\\\
    0 & A_{22} & 0 & 0\\\\
    0 & 0      & I & 0\\\\
    -I&0&0&0
\\end{bmatrix}
```

with ``E_{11}`` and ``E_{22}`` nonsingular. The system matrices ``B`` and ``C``
are partitioned accordingly as

```math
B = \\begin{bmatrix}
    B_1\\\\
    B_2\\\\
    B_3\\\\
    B_4
\\end{bmatrix}, \\quad
C = \\begin{bmatrix}
    C_1 & C_2 & C_3 & C_4
\\end{bmatrix}.
```
"""
Base.@kwdef struct AlmostKroneckerDAE{Tv, Ti, T <: AbstractMatrix{Tv}} <: AbstractDAE{Tv}
    E::T
    A::T
    B::T
    C::T
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
    M_0 = Matrix(C_1*B_4 - C_3*B_3 - C_4*B_1 + D)
    M_1 = Matrix(C_4*E_11*B_4)
end
