using LinearAlgebra
using MatrixEquations: lyapc, plyapc, arec

"""
    lrcf(X, trunc_tol)

Computes an approximate low-rank Cholesky-like 
factorization of a symmetric positive semi-definite matrix X s.t. X = Z'*Z
(up to a prescribed tolerance trunc_tol).
"""
function lrcf(X, trunc_tol)
    d,L = eigen(Symmetric(X))
    dr,Lr = truncation(d,L, trunc_tol)
    return (Lr*diagm(sqrt.(dr)))'
end

"""
    truncation(d, L, trunc_tol)

Computes a rank revealing factorization for a given LDL-decomposition of
``S = L * diagm(d) * L'`` (up to a prescribed tolerance trunc_tol).

Returns dr,Lr such that ``Lr * diagm(d) * Lr' \\approx S``.
"""
function truncation(d, L, trunc_tol)
    Q,R = qr(L)
    tmp = Symmetric(R*diagm(d)*R')
    d,U = eigen(tmp)
    p = sortperm(d, by=abs, rev=true)
    d = d[p]
    trunc_index = findlast(d/d[1] .>= trunc_tol)
    return d[1:trunc_index], Q*U[:,p[1:trunc_index]]
end

"""
    compress_lr(Z)

Compress `Z` via SVD such that it holds
        Z*Z' ≈ Z_new*Z_new'.
"""
function compress_lr(Z, r::Int)
    U, Σ = svd(Z)
    return U[:,1:r]*Diagonal(Σ[1:r])
end

"""
    prbaltrunc(sys::SemiExplicitIndex1DAE, r::Int)

Positive real balanced truncation for descriptor systems
[DAE_Forum_IV, Algorithm 2].

The parameter `r` corresponds to the reduced order
of the finite part of the system.
"""
function prbaltrunc(sys::SemiExplicitIndex1DAE, r::Int, Z_prc, Z_pro)
    (; E, A, B, C, M_0) = sys

    U, Σ, V = svd(Z_pro'*E*Z_prc)
    U_1 = U[:,1:r]
    Σ_1 = Σ[1:r]
    V_1 = V[:,1:r]

    W = Z_pro*U_1/Diagonal(sqrt.(Σ_1))
    T = Z_prc*V_1/Diagonal(sqrt.(Σ_1))

    Er = W'*E*T
    Ar = W'*A*T
    Br = W'*B
    Cr = C*T
    Dr = M_0
    return dss(Ar, Er, Br, Cr, Dr)
end

function prbaltrunc(sys::StaircaseDAE, r::Int, Z_prc, Z_pro)
    (; E, A, B, C, M_0, M_1, m) = sys

    U, Σ, V = svd(Z_pro'*E*Z_prc)
    U_1 = U[:,1:r]
    Σ_1 = Σ[1:r]
    V_1 = V[:,1:r]

    W = Z_pro*U_1/Diagonal(sqrt.(Σ_1))
    T = Z_prc*V_1/Diagonal(sqrt.(Σ_1))

    romf = dss(W'*A*T, W'*E*T, W'*B, C*T, 0)

    # TODO Generalize construction of `rominf` if rank(Z) > 1
    Z = lrcf(M_1, 10*eps(Float64)) # M_1 = Z'*Z
    rominf = dss(
        [1 0; 0 1], [0 1; 0 0], [zeros(1, m); Z], [-Z' zeros(m, 1)], M_0)
    
    return romf + rominf
end
