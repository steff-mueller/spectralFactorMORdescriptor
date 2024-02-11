Base.@kwdef struct IRKAOptions
    conv_tol = 1e-3
    max_iterations = 50
    cycle_detection_length = 1
    cycle_detection_tol = 0.0
    s_init_start = -1
    s_init_stop = 1
    randomize_s_init = false
end

struct IRKAResult
    iterations::Int64
    converged::Bool
    conv_crit::Float64
    cycle_detected::Bool
    cycle_crit::Float64
    s::Vector{ComplexF64}
    c::Matrix{ComplexF64}
    b::Matrix{ComplexF64}
end

function i0interpolate(E,A,B,C,c,b,s; P_r=I, P_l=I)
    n = size(A,1)
    r = length(s)
    V = zeros(n,r)
    W = zeros(n,r)

    j = 1
    while j <= r
        x = (s[j] .* E - A)\(P_l*B*b[:,j])
        y = (s[j] .* E' - A')\(P_r'*C'*c[:,j])
        if abs(imag(s[j])) > 0
            V[:,j] = real(x)
            W[:,j] = real(y)
            V[:,j+1] = imag(x)
            W[:,j+1] = imag(y)
            j = j+2
        else
            V[:,j] = real(x)
            W[:,j] = real(y)
            j = j+1
        end
    end

    # orthonormalization to prevent numerical ill conditioning
    V = qr(V).Q[:,1:r]
    W = qr(W).Q[:,1:r]

    Er = W'*E*V
    Ar = W'*A*V
    Br = W'*B
    Cr = C*V
    return Er,Ar,Br,Cr
end

function i1interpolate(E,A,B,C,D,Dr,c,b,s; P_r=I, P_l=I)
    n = size(A,1)
    r = length(s)
    V = zeros(Float64, n, r)
    W = zeros(Float64, n, r)
    TV = spzeros(ComplexF64, r, r)
    TW = spzeros(ComplexF64, r, r)
    j = 1
    while j <= r
        x = (s[j] .* E - A)\(P_l*B*b[:,j])
        y = (s[j] .* E' - A')\(P_r'*C'*c[:,j])

        # Make V and W real.
        if any(abs.(imag.(s[j])) .> (10^2)*eps(Float64))
            V[:,j] = real(x); V[:,j+1] = imag(x)
            iv = findfirst(!iszero, b[:,j])
            fv = b[iv,j]/conj(b[iv,j+1])
            TV[j:j+1,j:j+1] = 0.5*[1 -im; 1/fv im/fv]

            W[:,j] = real(y); W[:,j+1] = imag(y)
            iw = findfirst(!iszero, c[:,j])
            fw = c[iw,j]/conj(c[iw,j+1])
            TW[j:j+1,j:j+1] = 0.5*[1 -im; 1/fw im/fw]
            j = j+2
        else
            V[:,j] = real(x)
            W[:,j] = real(y)
            TV[j,j] = TW[j,j] = 1
            j = j+1
        end
    end

    # orthonormalization to prevent numerical ill conditioning
    V_qr = qr(V)
    W_qr = qr(W)
    V = V_qr.Q[:,1:r]
    W = W_qr.Q[:,1:r]  

    Er = transpose(W)*E*V
    Ar = transpose(W)*A*V + real(
         transpose(W_qr.R)\transpose(TW)*transpose(c)*(Dr-D)*b*TV/V_qr.R)
    Br = transpose(W)*B - real(
         transpose(W_qr.R)\transpose(TW)*transpose(c)*(Dr-D))
    Cr = C*V - real((Dr-D)*b*TV/V_qr.R)
    return Er,Ar,Br,Cr
end

function compute_conv_crit(s, s_old)
    sort_by = x -> (sign(real(x)), sign(imag(x)), real(x),imag(x))
    return norm(sort(s, by=sort_by)-sort(s_old, by=sort_by))/norm(s_old)
end

function compute_cycle_crit(s, s_history)
    return findmin(compute_conv_crit.([s], s_history))
end

function update_s_history!(s, s_history, cycle_detection_length)
    prepend!(s_history, [s])
    if length(s_history) > cycle_detection_length
        pop!(s_history)
    end
end

"""
    i0irka(
        sys::Union{DescriptorStateSpace,SparseDescriptorStateSpace},
        r::Int, irka_options::IRKAOptions
    )

IRKA for index-0 systems.
"""
function irka(
    sys::AbstractDescriptorStateSpace, r, irka_options::IRKAOptions;
    P_r=I, P_l=I, Dr=nothing
)
    (; conv_tol, max_iterations, s_init_start, s_init_stop, randomize_s_init,
       cycle_detection_length, cycle_detection_tol) = irka_options
    (; A, E, B, C, D) = sys
    c = size(sys.D,1)==1 ? ones(1,r) : randn(size(D,1),r)
    b = size(sys.D,2)==1 ? ones(1,r) : randn(size(D,2),r)
    sexp = range(s_init_start, stop=s_init_stop, length=r)
    if (randomize_s_init)
        sexp = sexp .+ randn(r)
        @info "irka: Random init sexp" sexp
    end
    s = (10+0*im) .^ sexp
    s_history = [s]

    Er,Ar,Br,Cr = isnothing(Dr) ?
        i0interpolate(E,A,B,C,c,b,s; P_r = P_r, P_l = P_l) :
        i1interpolate(E,A,B,C,D,Dr,c,b,s; P_r = P_r, P_l = P_l)

    iter = 0
    conv_crit = Inf
    cycle_crit = Inf
    local cycle_index::Int
    while (conv_crit > conv_tol && cycle_crit > cycle_detection_tol
        && iter < max_iterations)
        iter = iter+1

        Λ,T = eigen(Ar,Er)
        c = Cr*T
        b = transpose((Er*T)\Br)
        s = -Λ
        conv_crit = compute_conv_crit(s, s_history[1])
        cycle_crit, cycle_index = compute_cycle_crit(s, s_history)
        update_s_history!(s, s_history, cycle_detection_length)

        Er,Ar,Br,Cr = isnothing(Dr) ?
            i0interpolate(E,A,B,C,c,b,s; P_r = P_r, P_l = P_l) :
            i1interpolate(E,A,B,C,D,Dr,c,b,s; P_r = P_r, P_l = P_l)

        @info "Convergence criterion in iteration $iter: $conv_crit"
    end

    cycle_detected = cycle_crit <= cycle_detection_tol && conv_crit > conv_tol
    if cycle_detected
        @warn ("Cycle detected: Same eigenvalues $cycle_index iterations before" *
            " (tol = $cycle_detection_tol)")
    end

    converged = conv_crit < conv_tol
    @info "IRKA iterations" iter converged conv_crit
    return dss(Ar, Er, Br, Cr, isnothing(Dr) ? D : Dr), IRKAResult(
        iter, converged, conv_crit,
        cycle_detected, cycle_crit, s, c, b
    )
end

"""
    i1irka(sys::SemiExplicitIndex1DAE, r::Int, irka_options::IRKAOptions)

IRKA for semiexplicit index-1 DAEs.

Gugercin, S., Stykel, T., & Wyatt, S. (2013). Model Reduction of Descriptor
Systems by Interpolatory Projection Methods. SIAM. https://doi.org/10.1137/130906635
"""
function irka(sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions)
    return irka(sys, r, irka_options; Dr = sys.M_0)
end

"""
    i1irka(sys::StaircasePHDAE, r::Int, irka_options::IRKAOptions)

IRKA for port-Hamiltonian DAEs in staircase form.
"""
function irka(sys::StaircaseDAE, r, irka_options::IRKAOptions)
    (; M_0, M_1, P_r, P_l, m) = sys
    romp, result = irka(sys, r, irka_options; P_r = P_r, P_l = P_l, Dr = M_0)

    # TODO Generalize construction of `rominf` if rank(Z) > 1
    Z = lrcf(M_1, 10*eps(Float64)) # M_1 = Z'*Z
    # `M_0` is already included in `romp`.
    rominf = dss([1 0; 0 1], [0 1; 0 0], [zeros(1, m); Z], [-Z' zeros(m, 1)], 0)
    return romp + rominf, result
end

"""
Retry IRKA `max_retries` times and pick the result with the lowest H2-error.
`syssp` is the strictly proper subsystem of the full-order model.
"""
function retry_irka(run, max_tries, syssp)
    Random.seed!(0)
    local return_rom
    return_irka_result = IRKAResult(
        0, false, Inf, false, Inf,
        [], reshape([], (0,0)), reshape([], (0,0))
    )
    return_abs_h2_error = Inf
    return_stable = false
    for i = 1:max_tries
        @info "retry_irka: Starting $i try."
        rom, irka_result = run()
        @assert rom isa DescriptorStateSpace
        stable = isastable(rom)
        _, romsp = gsdec(rom; job="infinite")
        abs_h2_error = gh2norm(syssp - romsp)
        @info "retry_irka: Try: $i, Abs H2-error $abs_h2_error, Stable $stable."
        if (
            # only set new ROM if "better" than current one:
            (!return_stable || stable)
            && abs_h2_error < return_abs_h2_error
        )
            return_rom = rom
            return_irka_result = irka_result
            return_abs_h2_error = abs_h2_error
            return_stable = stable
        end
    end

    if !return_irka_result.converged
        @warn "retry_irka: IRKA did not converge after $max_tries tries."
    end
    if !return_stable
        @warn "retry_irka: IRKA did not return a stable ROM after $max_tries tries."
    end

    return return_rom, return_irka_result
end
