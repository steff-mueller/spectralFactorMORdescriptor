"""
The struct holds options for the IRKA algorithm.

- The `conv_tol` option sets the convergence tolerance.
- The `max_iterations` option sets the maximum number of iterations.
- The `cycle_detection_length` controls how many past iterations are
  considered for cycle detection. The `cycle_detection_tol` sets
  the cycle detection tolerance. If `cycle_detection_tol` is zero,
  no cycle detection is performed.
- The `s_init_start` and `s_init_stop` control the
  initial interpolation points. The initial interpolation points are
  distributed logarithmically between `10^s_init_start` and `10^s_init_stop`.
- If `randomize_s_init` is true, the initial interpolation points
  are disturbed randomly by `randn`. The variance is set by `randomize_s_var`.
"""
Base.@kwdef struct IRKAOptions
    conv_tol = 1e-3
    max_iterations = 50
    cycle_detection_length = 1
    cycle_detection_tol = 0.0
    s_init_start = -1
    s_init_stop = 1
    randomize_s_init = false
    randomize_s_var = 1.0
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

function i0interpolate(E, A, B, C, c, b, s; P_r=I, P_l=I)
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

function i1interpolate(E, A, B, C, D, Dr, c, b, s; P_r=I, P_l=I)
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
    irka(
        sys::AbstractDescriptorStateSpace, r, irka_options::IRKAOptions;
        P_r=I, P_l=I, Dr=nothing
    )

Computes a reduced-order model for `sys` using the
Iterative rational Krylov Algorithm (IRKA) [AntBG20](@cite).

The parameter `r` corresponds to the dimension of the reduced-order model.
If `Dr=nothing`, the feed-through matrix ``D_r`` of the reduced-order model
is set to the original feed-through matrix ``D`` of the full-order model.
For descriptor systems, the spectral projectors used for interpolation may
be provided via the parameters `P_r` and `P_l` [GugSW13; Sec. 3](@cite).
"""
function irka(
    sys::AbstractDescriptorStateSpace, r, irka_options::IRKAOptions;
    P_r=I, P_l=I, Dr=nothing
)
    (; conv_tol, max_iterations, s_init_start, s_init_stop,
       randomize_s_init, randomize_s_var,
       cycle_detection_length, cycle_detection_tol) = irka_options
    (; A, E, B, C, D) = sys
    c = size(sys.D,1)==1 ? ones(1,r) : randn(size(D,1),r)
    b = size(sys.D,2)==1 ? ones(1,r) : randn(size(D,2),r)
    sexp = range(s_init_start, stop=s_init_stop, length=r)
    if (randomize_s_init)
        sexp = sexp .+ (randomize_s_var .* randn(r))
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

    rom = GenericDescriptorStateSpace(Er, Ar, Br, Cr, isnothing(Dr) ? D : Dr)
    return rom, IRKAResult(
        iter, converged, conv_crit,
        cycle_detected, cycle_crit, s, c, b
    )
end

"""
    irka(sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions)

Computes a reduced-order model for the semi-explicit index-1 system `sys`
using the Iterative rational Krylov Algorithm (IRKA) [AntBG20,GugSW13](@cite).

The parameter `r` corresponds to the dimension of the reduced-order model.
"""
function irka(sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions)
    rom, result = irka(sys, r, irka_options; Dr = sys.M_0)
    return SemiExplicitIndex1DAE(rom.E, rom.A, rom.B, rom.C, rom.D, r), result
end

"""
    irka(sys::StaircaseDAE, r, irka_options::IRKAOptions)

Computes a reduced-order model for the staircase system `sys` using the
Iterative rational Krylov Algorithm (IRKA) [AntBG20,GugSW13](@cite).

The parameter `r` corresponds to the dimension of the reduced-order model.
"""
function irka(sys::StaircaseDAE, r, irka_options::IRKAOptions)
    (; M_0, M_1, P_r, P_l, m) = sys
    romp, result = irka(sys, r, irka_options; P_r = P_r, P_l = P_l, Dr = M_0)

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

"""
    retry_irka(run, max_tries, syssp)

Retries IRKA `max_tries` times and picks the result with the lowest H2-error.
The strictly proper subsystem of the full-order model `syssp` must be supplied 
in order to compute the H2-error. The parameter `run` must be a callable which
executes IRKA.
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
        try
            rom, irka_result = run()
            @assert rom isa AbstractDescriptorStateSpace
            stable = isastable(rom)
            _, romsp, = splitsys(rom)
            abs_h2_error = h2normsp(syssp - romsp)
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
        catch e
            @warn "retry_irka: IRKA run failed." e
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
