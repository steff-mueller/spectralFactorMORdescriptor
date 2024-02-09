"""
    arec_lr_nwt(F, E, Q, R, P_r=I, P_l=I; 
    conv_tol=1e-12, max_iterations=20, 
    lyapc_lradi_tol=1e-12, lyapc_lradi_max_iterations=150)

Low-rank Newton method [DAE_Forum_IV, Algorithm 9].

    FXE^T + EXF^T + EXQ^TQXE^T + P_l RR^T P_l^T = 0,
    X = P_r X P_r^T.

Returns an approximate solution Z such that

    X ≈ Z*Z'.
"""
function arec_lr_nwt(F, E, Q, R, P_r=I, P_l=I; 
    conv_tol=1e-9, max_iterations=20, 
    lyapc_lradi_tol=1e-14, lyapc_lradi_max_iterations=200
)
    lyapc_opts = (
        tol=lyapc_lradi_tol,
        max_iterations=lyapc_lradi_max_iterations
    )
    N = lyapc_lradi(F, E, R, P_l; lyapc_opts...)
    Z = N
    F_i = F
    err = Inf
    iter = 0
    while err > conv_tol && iter < max_iterations
        iter = iter + 1
        K = E*N*N'*Q'
        F_i = F_i + K*Q*P_r
        # TODO Use Sherman-Morrison-Woodbury formula inside
        # TODO `lyapc_lradi`. See [BenS14, Section 4.3.]
        N = lyapc_lradi(F_i, E, K, P_l; lyapc_opts...)
        Z = compress_lr([Z N])

        # TODO Do not form full matrix for convergence criterion.
        # TODO See [BenS14, Section 4.4.].
        X = Z*Z'
        err = norm(F*X*E'+E*X*F'+E*X*Q'*Q*X*E'+P_l*R*R'*P_l')
        @info "Error in iteration $iter: $err"
    end
    return Z
end

"""
    compress_lr(Z)

Compress `Z` via SVD such that it holds
        Z*Z' ≈ Z_new*Z_new'.
"""
function compress_lr(Z; tol=1e-02 * sqrt(size(Z, 1) * eps(Float64)))
    U, Σ = svd(Z)
    r = findlast(Σ/Σ[1] .> tol)
    return U[:,1:r]*Diagonal(Σ[1:r])
end

"""
    pr_o_gramian(sys)

Computes positive real observability gramian,
i.e., the unique stabilizing solution of

    A^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0')^{-1}(C P_r-B^TXE) = 0,
    X = P_l^T Y P_l,

where P_r, P_l and M_0 are defined by the given system.

Note: Uses `MatrixEquations.garec`.
"""
@memoize function pr_o_gramian(sys::SemiExplicitIndex1DAE)
    (; M_0, n, n_1, n_2) = sys
    _, syssp, Q, = splitsys(sys)
    X, = garec(Matrix(syssp.A), Matrix(syssp.E), syssp.B, 0, -M_0-M_0', 0, -syssp.C') # or other standard method
    X_full = [X spzeros(n_1, n_2); spzeros(n_2, n)]
    return Q'*X_full*Q
end

"""
    pr_c_gramian(sys)

Computes the positive real observability gramian,
i.e., the unique stabilizing solution of

    AXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0')^{-1} (P_l B - EXC^T)^T = 0,
    X = P_r X P_r^T,

where P_r, P_l and M_0 are defined by the given system.

Note: Uses `MatrixEquations.garec`.
"""
@memoize function pr_c_gramian(sys::SemiExplicitIndex1DAE)
    (; M_0, n, n_1, n_2) = sys
    _, syssp, _, Z = splitsys(sys)
    X, = garec(Matrix(syssp.A'), Matrix(syssp.E'), syssp.C', 0, -M_0-M_0', 0, -syssp.B) # or other standard method
    X_full = [X spzeros(n_1, n_2); spzeros(n_2, n)]
    return Z*X_full*Z'
end

"""
    pr_o_gramian(sys)

Computes positive real observability gramian,
i.e., the unique stabilizing solution of

    A^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0')^{-1}(C P_r-B^TXE) = 0,
    X = P_l^T Y P_l,

where P_r, P_l and M_0 are defined by the given system.

Note: Uses `MatrixEquations.garec`.
"""
@memoize function pr_o_gramian(sys::StaircaseDAE)
    sys_kronecker, L_A, = tokronecker(sys)
    (; E_22, A_22, B_2, C_2, M_0, n, n_1, n_2, n_3, n_4) = sys_kronecker
    X, = garec(Matrix(A_22), Matrix(E_22), B_2, 0, -M_0-M_0', 0, -C_2') # or other standard method
    X_full = [spzeros(n_1, n);
              spzeros(n_2, n_1) X spzeros(n_2, n_3+n_4);
              spzeros(n_3+n_4, n)]
    return L_A'*X_full*L_A
end

"""
    pr_c_gramian(sys)

Computes the positive real observability gramian,
i.e., the unique stabilizing solution of

    AXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0')^{-1} (P_l B - EXC^T)^T = 0,
    X = P_r X P_r^T,

where P_r, P_l and M_0 are defined by the given system.

Note: Uses `MatrixEquations.garec`.
"""
@memoize function pr_c_gramian(sys::StaircaseDAE)
    sys_kronecker, _, Z_A = tokronecker(sys)
    (; E_22, A_22, B_2, C_2, M_0, n, n_1, n_2, n_3, n_4) = sys_kronecker
    X, = garec(Matrix(A_22'), Matrix(E_22'), C_2', 0, -M_0-M_0', 0, -B_2) # or other standard method
    X_full = [spzeros(n_1, n);
              spzeros(n_2, n_1) X spzeros(n_2, n_3+n_4);
              spzeros(n_3+n_4, n)]
    return Z_A*X_full*Z_A'
end

"""
    pr_o_gramian_lr(sys)

Computes low-rank factor of positive real observability gramian,
i.e., the unique stabilizing solution of

    A^TXE + E^TXA + (C^T-E^TXB)(D+D')^{-1}(C-B^TXE) = 0.

E is assumed to be nonsingular.
Returns an approximate solution Z such that

    X ≈ Z*Z'.
"""
@memoize function pr_o_gramian_lr(sys::AbstractDescriptorStateSpace)
    (; A,E,B,C,D) = sys
    J = cholesky(Symmetric(D+D'))
    F = A' - C'/(D+D')*B'
    Q = J.L\B'
    R = C'/J.U
    return arec_lr_nwt(F,E,Q,R)
end

"""
    pr_c_gramian_lr(sys)

Computes low-rank factor of positive real controllability gramian,
i.e., the unique stabilizing solution of

    AXE^T + EXA^T + (B - EXC^T) (D+D')^{-1} (B - EXC^T)^T = 0

E is assumed to be nonsingular.
Returns an approximate solution Z such that

    X ≈ Z*Z'.
"""
@memoize function pr_c_gramian_lr(sys::AbstractDescriptorStateSpace)
    (; A,E,B,C,D) = sys
    J = cholesky(Symmetric(D+D'))
    F = A - B/(D+D')*C
    Q = J.L\C
    R = B/J.U
    return arec_lr_nwt(F,E,Q,R)
end

"""
    pr_o_gramian_lr(sys)

Computes low-rank factor of positive real observability gramian,
i.e., the unique stabilizing solution of

    A^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0')^{-1}(C P_r-B^TXE) = 0,
    X = P_l^T Y P_l,

where P_r, P_l and M_0 are defined by the given system.

Returns an approximate solution Z such that

    X ≈ Z*Z'.
"""
@memoize function pr_o_gramian_lr(sys::SemiExplicitIndex1DAE)
    (; A, E, B, C, M_0, P_r, P_l) = sys
    M = cholesky(Symmetric(M_0+M_0'))
    F = A' - P_r'*C'/(M_0+M_0')*B'
    Q = M.L\B'
    R = C'/M.U
    return arec_lr_nwt(F, E', Q, R, P_l', P_r')
end

"""
    pr_c_gramian_lr(sys)

Computes low-rank factor of positive real observability gramian,
i.e., the unique stabilizing solution of

    AXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0')^{-1} (P_l B - EXC^T)^T = 0,
    X = P_r X P_r^T,

where P_r, P_l and M_0 are defined by the given system.

Returns an approximate solution Z such that

    X ≈ Z*Z'.
"""
@memoize function pr_c_gramian_lr(sys::SemiExplicitIndex1DAE)
    (; A, E, B, C, M_0, P_r, P_l) = sys
    M = cholesky(Symmetric(M_0+M_0'))
    F = A - P_l*B/(M_0+M_0')*C
    Q = M.L\C
    R = B/M.U
    return arec_lr_nwt(F, E, Q, R, P_r, P_l)
end
