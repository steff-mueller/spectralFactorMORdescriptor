"""
    lrcf(X, trunc_tol)

Computes an approximate low-rank Cholesky-like 
factorization of a symmetric positive semi-definite matrix ``X``
s.t. ``X = Z^T Z`` (up to a prescribed tolerance `trunc_tol`).
"""
function lrcf(X, trunc_tol)
    d,L = eigen(Symmetric(X))
    # remove negative eigenvalues (numerical errors)
    idx = findall(v -> v >= 0, d)
    dr, Lr = truncation(d[idx], L[:, idx], trunc_tol)
    return (Lr*diagm(sqrt.(dr)))'
end

"""
    truncation(d, L, trunc_tol) -> (dr, Lr)

Computes a rank revealing factorization for a given LDL-decomposition of
``S = L * \\mathrm{diag}(d) * L^T`` (up to a prescribed tolerance `trunc_tol`)
such that ``L_r * diag(d_r) * L_r^T \\approx S``.
"""
function truncation(d, L, trunc_tol)
    Q,R = qr(L)
    tmp = Symmetric(R*diagm(d)*R')
    d,U = eigen(tmp)
    p = sortperm(d, by=abs, rev=true)
    d = d[p]
    trunc_index = findlast(abs.(d/d[1]) .>= trunc_tol)
    return d[1:trunc_index], Q*U[:,p[1:trunc_index]]
end

"""
    compress_lr(Z, r)

Compresses ``Z`` via SVD such that it holds
``Z Z^T ≈ Z_\\mathrm{new} Z_\\mathrm{new}^T``,
where ``Z_\\mathrm{new}`` is the returned matrix.
"""
function compress_lr(Z, r)
    U, Σ = svd(Z)
    return U[:,1:r]*Diagonal(Σ[1:r])
end

"""
    prbaltrunc(sys::SemiExplicitIndex1DAE, r, Z_prc, Z_pro)

Computes a reduced-order model for the semi-explicit index-1 system `sys`
using positive real balanced truncation [DAE_Forum_IV_mor; Alg. 2](@cite).

The parameter `r` corresponds to the dimension of the reduced-order model.

The parameter `Z_prc` must be a (low-rank) Cholesky factor such that
``Z_\\mathrm{prc}^T Z_\\mathrm{prc}``
is the positive real controllability Gramian.

The parameter `Z_pro` must be a (low-rank) Cholesky factor such that
``Z_\\mathrm{pro}^T Z_\\mathrm{pro}``
is the positive real observability Gramian.
"""
function prbaltrunc(sys::SemiExplicitIndex1DAE, r, Z_prc, Z_pro)
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

"""
    prbaltrunc(sys::StaircaseDAE, r, Z_prc, Z_pro)

Computes a reduced-order model for the staircase system `sys`
using positive real balanced truncation [DAE_Forum_IV_mor; Alg. 2](@cite).

The parameter `r` corresponds to the dimension
of the finite part of the reduced-order model.

The parameter `Z_prc` must be a (low-rank) Cholesky factor such that
``Z_\\mathrm{prc}^T Z_\\mathrm{prc}``
is the positive real controllability Gramian.

The parameter `Z_pro` must be a (low-rank) Cholesky factor such that
``Z_\\mathrm{pro}^T Z_\\mathrm{pro}``
is the positive real observability Gramian.
"""
function prbaltrunc(sys::StaircaseDAE, r, Z_prc, Z_pro)
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
