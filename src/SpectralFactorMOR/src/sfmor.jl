function lure_cholesky(sys, X, method; compute_factors_tol=1e-12)
    (; A, E, B, C, M_0, P_r, n) = sys
    @assert method in (:together, :firstm)
    if method == :together
        W = Matrix([
            -A'*X*E-E'*X*A P_r'*C'-E'*X*B;
            C*P_r-B'*X*E M_0+M_0'
        ])
        K = lrcf(W, compute_factors_tol)
        L = K[:,1:n]
        M = K[:,n+1:end]
    elseif method == :firstm # valid if X is solution of Riccati eq.
        M = cholesky(Symmetric(M_0+M_0')).U
        L = M'\(C*P_r-B'*X*E)
    end
    return L, M
end

"""
A solution to the positive real projected Lur'e equation
can be supplied via the parameter `X`. If no `X` is supplied,
the positive real observability Gramian is computed.
"""
function sfmor(sys::SemiExplicitIndex1DAE, r,
    irka_options::IRKAOptions; X=nothing,
    compute_factors = :together, compute_factors_tol=1e-12
)
    (; A, E, B, C, n_1, M_0, P_r) = sys
    if isnothing(X)
        Z, _, _ = pr_o_gramian_lr(sys)
        X = Z*Z'
    end
    L, M = lure_cholesky(sys, X, compute_factors; compute_factors_tol)

    Σ_H = SemiExplicitIndex1DAE(E, A, B, sparse(L), Matrix(M), n_1)

    # Compute ROM for spectral factor
    rom_H, result = irka(Σ_H, r, irka_options)
    Ar, Er, Br, Lr, Mr = dssdata(rom_H)

    # i1irka preserves the feed-through term,
    # i.e. Mr = M for semi-explicit index-1 systems, since
    # `Σ_H.C_2 = 0`. Thus, we can equivalently choose
    # Dr = M_0 instead of Dr = 0.5*(Mr'*Mr)+skew(M_0).
    # This choice has the advantage that we do not introduce
    # numerical errors from the Cholesky factorization.
    Dr = M_0
    Xr = lyapc(Ar', Er', Lr'*Lr)
    Cr = Br'*Xr*Er + Mr'*Lr

    return dss(Ar, Er, Br, Cr, Dr), result
end

"""
A solution to the positive real projected Lur'e equation
can be supplied via the parameter `X`. If no `X` is supplied,
the positive real observability Gramian is computed.
"""
function sfmor(sys::StaircaseDAE, r, irka_options::IRKAOptions;
    X = nothing, compute_factors = :together, compute_factors_tol=1e-12
)
    if isnothing(X)
        X = pr_o_gramian(sys)
    end
    L, M = lure_cholesky(sys, X, compute_factors; compute_factors_tol)

    (; E, A, B, M_0, M_1, P_r, P_l, n_1, n_2, n_3, n_4, m) = sys
    Σ_H = StaircaseDAE(
        E = E, A = A, B = B,
        C = sparse(L), D = Matrix(M),
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    )

    # Compute ROM for spectral factor
    romf_H, result = irka(Σ_H, r, irka_options; P_r = P_r, P_l = P_l)

    Ar_f, Er_f, Br_f, Lr_f, = dssdata(romf_H)
    Xr_f = lyapc(Ar_f', Er_f', Lr_f'*Lr_f)
    Cr_f = Br_f'*Xr_f*Er_f + M'*Lr_f
    romp = dss(Ar_f, Er_f, Br_f, Cr_f, M_0)

    # TODO Generalize construction of `rominf` if rank(Z) > 1
    Z = lrcf(M_1, 10*eps(Float64)) # M_1 = Z'*Z
    # `M_0` is already included in `romp`.
    rominf = dss([1 0; 0 1], [0 1; 0 0], [zeros(1, m); Z], [-Z' zeros(m, 1)], 0)

    return romp + rominf, result
end
