using SparseArrays

"""
    lyapc_lradi(A, E, B, P_l=I; tol=1e-12, max_iterations=150)

LR-ADI method for the projected continuous-time Lyapunov equation
[DAE_Forum_IV, Algorithm 7]

    EXA^T + AXE^T = - P_l BB^T P_l^T,
    X = P_r X P_r^T.

Returns an approximate solution Z such that

    X ≈ Z*Z'.
"""
function lyapc_lradi(A, E, B, P_l=I; tol=1e-12, max_iterations=150)
    n = size(A,1)
    W_0 = Matrix(P_l*B)
    W = W_0
    Z = zeros(n,0)
    shifts = initial_shifts(A,E,W)

    res = Inf
    j = 0
    j_shift = 1
    while res > tol && j < max_iterations
        p = shifts[j_shift]
        if imag(p) == 0
            V = (A + real(p).*E)\W
            W = W - 2*real(p).*E*V
            Z = [Z sqrt(-2*real(p)).*V]
            j = j+1
        else
            V = (A + p.*E)\W
            a = 2*sqrt(-real(p))
            b = real(p)/imag(p)
            W = W - 4*real(p).*E*(real(V)+b.*imag(V))
            Z = [Z a.*(real(V)+b.*imag(V)) a*sqrt(b^2+1).*imag(V)]
            j = j+2
        end
        j_shift = j_shift+1

        res = norm(W'*W)/norm(W_0'*W_0)

        if j_shift > length(shifts)
            shifts = new_shifts(A,E,V,Z,shifts)
            j_shift = 1
        end

        @info "Relative residual at iteration $j: $res"
    end
    return Z
end

"""
    initial_shifts(A,E,B,max_iterations=10,iter=0)

Returns initial shifts for LR-ADI method.

See [Kür16_Doktorarbeit, pp. 92-95] and
https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/lradi.py.
"""
function initial_shifts(A, E, B, max_iterations=10, iter=1)
    k = size(B,2)
    Q = qr(B).Q[:,1:k]
    shifts = eigvals(Q'*A*Q, Q'*E*Q)
    shifts = shifts[real(shifts) .< 0]
    if length(shifts) > 0
        return shifts
    elseif iter < max_iterations
        return initial_shifts(A, E, randn(size(B)),
            max_iterations, iter+1)
    else
        error("Unable to generate initial shifts.")
    end
end

"""
    new_shifts(A,E,V,Z,prev_shifts,subspace_columns=6)

Returns new intermediate shifts for LR-ADI method.

See [Kür16_Doktorarbeit, pp. 92-95] and
https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/lradi.py.
"""
function new_shifts(A, E, V, Z, prev_shifts, subspace_columns=6)
    num_columns = min(subspace_columns * size(V,2), 
        size(Z,1), size(Z,2))
    Q = qr(Z[:,end-num_columns+1:end]).Q[:,1:num_columns]

    shifts = eigvals(Q'*A*Q, Q'*E*Q)
    shifts = shifts[real(shifts) .< 0]
    shifts = shifts[imag(shifts) .>= 0]
    sort!(shifts; by=abs)

    if length(shifts) > 0
        return shifts
    else
        return prev_shifts
    end
end
