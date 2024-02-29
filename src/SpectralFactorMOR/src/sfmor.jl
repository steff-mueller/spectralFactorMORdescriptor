function lure_cholesky(sys, X; compute_factors_tol=1e-12)
    (; A, E, B, C, M_0, P_r, n) = sys
    W = Matrix([
        -A'*X*E-E'*X*A P_r'*C'-E'*X*B;
        C*P_r-B'*X*E M_0+M_0'
    ])
    K = lrcf(W, compute_factors_tol)
    L = K[:,1:n]
    M = K[:,n+1:end]
    return L, M
end

"""
    sfmor(
        sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions;
        X=nothing, compute_factors_tol=1e-12
    )

Executes the spectral factor MOR method for a semi-explicit index-1 system.

A solution to the positive real projected Lur'e equation
can be supplied via the parameter `X`. If no `X` is supplied,
the positive real observability Gramian is computed
using [`pr_o_gramian`](@ref).
"""
function sfmor(sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions;
    X=nothing, compute_factors_tol=1e-12
)
    (; A, E, B, n_1, M_0) = sys
    if isnothing(X)
        X = pr_o_gramian(sys)
    end
    L, M = lure_cholesky(sys, X; compute_factors_tol)

    Σ_H = SemiExplicitIndex1DAE(E, A, B, sparse(L), Matrix(M), n_1)

    # Compute ROM for spectral factor
    rom_H, result = irka(Σ_H, r, irka_options)

    Er = rom_H.E; Ar = rom_H.A; Br = rom_H.B; Lr = rom_H.C
    Dr = M_0
    Xr = lyapc(Ar', Er', Lr'*Lr)
    Cr = Br'*Xr*Er + M'*Lr

    return SemiExplicitIndex1DAE(Er, Ar, Br, Cr, Dr, r), result
end

"""
    sfmor(
        sys::StaircaseDAE, r, irka_options::IRKAOptions;
        X = nothing, compute_factors_tol=1e-12
    )

Executes the spectral factor MOR method for a system in staircase form.

A solution to the positive real projected Lur'e equation
can be supplied via the parameter `X`. If no `X` is supplied,
the positive real observability Gramian is computed
using [`pr_o_gramian`](@ref).
"""
function sfmor(sys::StaircaseDAE, r, irka_options::IRKAOptions;
    X = nothing, compute_factors_tol=1e-12
)
    if isnothing(X)
        X = pr_o_gramian(sys)
    end
    L, M = lure_cholesky(sys, X; compute_factors_tol)

    (; E, A, B, M_0, M_1, P_r, P_l, n_1, n_2, n_3, n_4, m) = sys
    Σ_H = StaircaseDAE(
        E = E, A = A, B = B,
        C = sparse(L), D = Matrix(M),
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    )

    # Compute ROM for spectral factor
    romf_H, result = irka(Σ_H, r, irka_options; P_r = P_r, P_l = P_l)

    Er_f = romf_H.E; Ar_f = romf_H.A; Br_f = romf_H.B; Lr_f = romf_H.C
    Xr_f = lyapc(Ar_f', Er_f', Lr_f'*Lr_f)
    Cr_f = Br_f'*Xr_f*Er_f + M'*Lr_f
    romp = UnstructuredDAE(Er_f, Ar_f, Br_f, Cr_f, M_0)

    # TODO Generalize construction of ROM if rank(Z) > 1
    Z = lrcf(M_1, 10*eps(Float64)) # M_1 = Z'*Z
    return StaircaseDAE(
        E = [
            1 zeros(1,r) 0;
            zeros(r,1) romp.E zeros(r,1);
            0 zeros(1,r) 0
        ],
        A = [
            0 zeros(1,r) -1;
            zeros(r,1) romp.A zeros(r,1);
            1 zeros(1,r) 0
        ],
        B = [
            zeros(1, m);
            romp.B;
            Z
        ],
        C = [zeros(m, 1) romp.C Z'],
        D = romp.D,
        n_1 = 1, n_2 = r, n_3 = 0, n_4 = 1 
    ), result
end
