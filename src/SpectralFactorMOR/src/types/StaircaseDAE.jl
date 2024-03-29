"""
Represents a system of the form

```math
E = \\begin{bmatrix}
    E_{11} & 0      & 0 & 0\\\\
    0      & E_{22} & 0 & 0\\\\
    0      & 0      & 0 & 0\\\\
    0      & 0      & 0 & 0
\\end{bmatrix},\\quad
A = \\begin{bmatrix}
    A_{11} & A_{12} & A_{13} & A_{14}\\\\
    A_{21} & A_{22} & A_{23} & 0\\\\
    A_{31} & A_{32} & A_{33} & 0\\\\
    A_{41} & 0      & 0      & 0
\\end{bmatrix}
```

with ``E_{11}, E_{22}, A_{14}=-A_{41}^T,A_{33}`` nonsingular.
"""
Base.@kwdef struct StaircaseDAE{Tv, Ti, T <: AbstractMatrix{Tv}} <: AbstractDAE{Tv}
    E::T
    A::T
    B::T
    C::T
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
    tokronecker(sys::StaircaseDAE)

Transforms `sys` into almost Kronecker form [AchAM21; Alg. 5](@cite).
Returns a system of type [`AlmostKroneckerDAE`](@ref).

The result is cached using the
[Memoize.jl](https://github.com/JuliaCollections/Memoize.jl) package.
"""
@memoize function tokronecker(sys::StaircaseDAE{Tv}) where {Tv}
    (; A_11, A_13, A_33, A_31, A_41, A_23, A_21, A_32, A_14, A_12,
       n_1, n_2, n_3, n_4) = sys

    L_A = [sparse(I, n_1, n_1) spzeros(n_1, n_2) -A_13/Matrix(A_33) (-A_11 + A_13/Matrix(A_33)*A_31)/Matrix(A_41);
        spzeros(n_2, n_1) sparse(I, n_2, n_2) -A_23/Matrix(A_33) (-A_21 + A_23/Matrix(A_33)*A_31)/Matrix(A_41);
        spzeros(n_3, n_1) spzeros(n_3, n_2) inv(Matrix(A_33)) -Matrix(A_33)\A_31/Matrix(A_41);
        spzeros(n_4, n_1) spzeros(n_4, n_2) spzeros(n_4, n_3) -inv(Matrix(A_41))]

    droptol!(L_A, 2*eps(Tv))

    Z_A = [sparse(I, n_1, n_1) spzeros(n_1, n_2) spzeros(n_1, n_3) spzeros(n_1, n_4);
        spzeros(n_2, n_1) sparse(I, n_2, n_2) spzeros(n_2, n_3) spzeros(n_2, n_4);
        spzeros(n_3, n_1) -Matrix(A_33)\A_32 I spzeros(n_3, n_4);
        spzeros(n_4, n_1) Matrix(A_14)\(-A_12 + A_13/Matrix(A_33)*A_32) spzeros(n_4, n_3) inv(Matrix(A_14))]

    droptol!(Z_A, 2*eps(Tv))

    return AlmostKroneckerDAE(
        E = L_A*sys.E*Z_A,
        A = L_A*sys.A*Z_A,
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
        sel = Matrix([spzeros(n_1, n);
                      spzeros(n_2, n_1) I spzeros(n_2, n_3+n_4);
                      spzeros(n_3+n_4, n)])
        return sparse(Z_A*sel/Z_A)
    elseif sym == :P_l
        (; n, n_1, n_2, n_3, n_4) = sys
        _, L_A = tokronecker(sys)
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
    is_valid(sys::StaircaseDAE)

Asserts that `sys` is in valid staircase form.
"""
function is_valid(sys::StaircaseDAE)
    (; E_11, E_22, A_33, A_14, A_41, n_1, n_2, n_3, n_4,
       E, A, idx_1, idx_2, idx_3, idx_4) = sys
    return (
        n_1 == n_4
        && rank(E_11) == n_1
        && rank(E_22) == n_2
        && iszero(E[idx_1, idx_2])
        && iszero(E[:, [idx_3; idx_4]])
        && rank(A_33) == n_3
        && rank(A_14) == n_4
        && A_14 == -A_41'
        && iszero(A[[idx_2; idx_3; idx_4], idx_4])
        && iszero(A[idx_4, [idx_2; idx_3; idx_4]])
    )
end
