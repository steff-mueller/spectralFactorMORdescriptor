function setup_DAE1_RCL(ns::Int64, random::Bool, mimo::Bool)
    Random.seed!(0)
    E, J, R, _, G = setup_DAE1_RCL_LadderNetwork_sparse( # Q=I
        ns = ns,
        r = random
            ? max.(0.2 * (randn(ns+2).+1), 0.001) 
            : 0.2 * ones(ns+2),
        m = mimo ? 2 : 1
    )
    A = (J-R)
    B = G
    C = SparseMatrixCSC(G')
    # add regularization term such that M_0+M_0' is nonsingular for m=2
    D = mimo ? 1e-12*[1 0; 0 1] : zeros(1,1)
    n_1 = nnz(E)
    return SemiExplicitIndex1DAE(E,A,B,C,D,n_1)
end

function setup_DAE2_RCL(ns::Int64, random::Bool, mimo::Bool)
    Random.seed!(0)
    E, J, R, G = setup_DAE2_RCL_LadderNetwork_sparse(
        ns = ns,
        r = random
            ? max.(0.2 * (randn(ns+1).+1), 0.001) 
            : 0.2 * ones(ns+1),
        m = mimo ? 2 : 1
    )

    # Transform into staircase form [AchAM21]
    s, UE = eigen(Diagonal(E); sortby=x -> -x)
    rank_E = count(s .!= 0.0)
    n = size(E, 1)
    n_1 = n_4 = 1
    n_2 = rank_E - 1
    n_3 = n - n_1 - n_2 - n_4
    m = mimo ? 2 : 1

    if mimo
        # Re-order the last two columns/rows for MIMO case    
        UE = (UE * [I spzeros(n-2, 2);
                    spzeros(1, n-2) 0 1;
                    spzeros(1, n-2) 1 0])
    end

    sys = StaircaseDAE(
        E = sparse(Diagonal(s)),
        A = sparse(UE'*(J-R)*UE),
        B = sparse(UE'*G),
        C = sparse(G'*UE),
        # add regularization term such that M_0+M_0' is nonsingular
        D = 1e-12*Matrix(I, m, m),
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    )
    @assert is_valid(sys)
    return sys
end

function read_RCL_1_SISO()
    E, J, R, G = _readphmat("scripts/RCL-1-SISO.mat")
    A = (J-R)
    B = G
    C = SparseMatrixCSC(G')
    D = zeros(1,1)
    n_1 = nnz(E)
    return SemiExplicitIndex1DAE(E, A, B, C, D, n_1)
end

function read_RCL_2_SISO()
    E, J, R, G = _readphmat("scripts/RCL-2-SISO.mat")

    # Transform into staircase form [AchAM21]
    s, UE = eigen(Diagonal(E); sortby=x -> -x)
    rank_E = count(s .!= 0.0)
    n = size(E, 1)
    n_1 = n_4 = 1
    n_2 = rank_E - 1
    n_3 = n - n_1 - n_2 - n_4
    m = 1

    t = sparse(I(n))
    t[1,1] = t[999,999] = 0
    t[999,1] = t[1,999] = 1
    UE = UE*t

    sys = StaircaseDAE(
        E = dropzeros(t'*Diagonal(s)*t),
        A = sparse(UE'*(J-R)*UE),
        B = sparse(UE'*G),
        C = sparse(G'*UE),
        # add regularization term such that M_0+M_0' is nonsingular
        D = 1e-12*Matrix(I, m, m),
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    )
    @assert is_valid(sys)
    return sys
end

function _readphmat(path)
    v = matread(path)
    return v["E"], v["J"], v["R"], v["B"]
end

function get_problem(name)
    problems = Dict(
        "RCL-1-SISO-SIMPLE" => () -> setup_DAE1_RCL(500, false, false),
        "RCL-1-SISO" => () -> read_RCL_1_SISO(),
        "RCL-1-MIMO" => () -> setup_DAE1_RCL(500, true, true),
        "RCL-0-SISO" => () -> toindex0(
            read_RCL_1_SISO()), # index-reduced sys
        "RCL-0-MIMO" => () -> toindex0(
            setup_DAE1_RCL(500, true, true)),  # index-reduced sys
        "RCL-2-SISO" => () -> read_RCL_2_SISO(),
        "RCL-2-MIMO" => () -> setup_DAE2_RCL(500, true, true)
    )
    return problems[name]()
end

function compute_gramians(problem, sys, dense)
    cache_name = string("gramians_", problem, dense ? "_dense" : "_lr", ".mat")
    cache_path = "data/$cache_name"
    if isfile(cache_path)
        @info "Using cached Gramians from disk"
        vars = matread(cache_path)
        return vars["Zo"], vars["Xo"], vars["Zc"]
    end 

    if dense
        Xo = pr_o_gramian(sys)
        Zo = lrcf(Matrix(Xo), 1e-9)'

        Xc = pr_c_gramian(sys)
        Zc = lrcf(Matrix(Xc), 1e-9)'
    else
        Zo = pr_o_gramian_lr(sys)
        Xo = Zo*Zo'

        Zc = pr_c_gramian_lr(sys)
    end

    matwrite(cache_path, Dict(
        "Zo" => Zo, "Xo" => Xo, "Zc" => Zc
    ); compress = true)

    return Zo, Xo, Zc
end
