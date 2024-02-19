"""
    ispr(sys::AbstractDescriptorStateSpace) -> (ispr, mineigval, w)

Checks if the system is positive real by
sampling the Popov function on the imaginary axis
and verifying that the Popov function is positive
semi-definite for all sample points.
"""
function ispr(sys::AbstractDescriptorStateSpace)
    (; E, A, B, C, D) = sys
    T(s) = Matrix(C)/(s.*E-A)*B + D
    popov(w) = Hermitian(T(im*w) + transpose(T(-im*w)))

    w = [
        -exp10.(range(1, stop=-12, length=800));
        exp10.(range(-1, stop=12, length=800))
    ]

    mineigval, idx = findmin(minimum.(eigvals.(popov.(w))))
    return (ispr = mineigval >= 0, mineigval = mineigval, w = w[idx])
end

"""
    isastable(sys::AbstractDescriptorStateSpace)

Checks if `sys` is asymptotically stable.
"""
function isastable(sys::AbstractDescriptorStateSpace)
    sys_poles = filter(s -> !isinf(s), gpole(sys)) # finite eigenvalues
    return maximum(real((sys_poles))) < 0
end

"""
    checkprojlure(sys, X)

Checks if `X` satisfies the projected positive real Lur'e equation,
i. e., if there exist matrices ``L`` and ``M`` such that

```math
\\begin{align*}
	\\begin{bmatrix}
		-A^TXE-E^TXA & P_r^TC^T - E^TXB\\\\
		CP_r - B^TXE & M_0+M_0^T
	\\end{bmatrix} &= \\begin{bmatrix}
		L^T\\\\
        M^T
	\\end{bmatrix} \\begin{bmatrix}
		L & M
	\\end{bmatrix},\\\\
	X=X^T &\\geq 0,\\ X=P_l^TXP_l.
\\end{align*}
```
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
    splitsys(sys::SemiExplicitIndex1DAE) -> (sys1, sys2)

Returns the infinite and strictly proper subsystems of `sys`.

- `sys1` is the infinite subsystem.
- `sys2` is the strictly proper subsystem.
"""
function splitsys(sys::SemiExplicitIndex1DAE)
    (; E, A, B, C, D, n_1) = sys
    n = size(A,1)
    n_2 = n - n_1

    Q = [sparse(I, n_1, n_1) -sys.A_12/Matrix(sys.A_22);
         spzeros(n_2, n_1) sparse(I, n_2, n_2)]
    Z = [sparse(I, n_1, n_1) spzeros(n_1, n_2);
         -Matrix(sys.A_22)\sys.A_21 sparse(I, n_2, n_2)]

    A = Q*A*Z
    B = Q*B
    C = C*Z
    i1 = 1:n_1; i2 = n_1+1:n
    return (
        GenericDescriptorStateSpace(
            E[i2,i2], A[i2,i2], B[i2,:], C[:,i2], D, 0.0),
        GenericDescriptorStateSpace(
            E[i1,i1], A[i1,i1], B[i1,:], C[:,i1], zeros(size(D)...), 0.0),
        Q, Z
    )
end

"""
    splitsys(sys::AlmostKroneckerDAE) -> (sys1, sys2)

Returns the infinite and strictly proper subsystems of `sys`.

- `sys1` is the infinite subsystem.
- `sys2` is the strictly proper subsystem.
"""
function splitsys(sys::AlmostKroneckerDAE)
    (; E, A, B, C, D, A_22, E_22, B_2, C_2, idx_1, idx_3, idx_4, m) = sys

    idx_inf = [idx_1; idx_3; idx_4]
    return GenericDescriptorStateSpace(
               E[idx_inf, idx_inf], A[idx_inf, idx_inf],
               B[idx_inf,:], C[:,idx_inf], D, 0.0),
           GenericDescriptorStateSpace(
               E_22, A_22, B_2, C_2, zeros(m, m), 0.0)
end

"""
    splitsys(sys::StaircaseDAE) -> (sys1, sys2)

Returns the infinite and strictly proper subsystems of `sys`.

- `sys1` is the infinite subsystem.
- `sys2` is the strictly proper subsystem.
"""
function splitsys(sys::StaircaseDAE)
    sys_kronecker, = tokronecker(sys)
    return splitsys(sys_kronecker)
end
