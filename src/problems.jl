function setup_DAE1_RCL(ns::Int64, random::Bool, mimo::Bool)
    Random.seed!(0)
    E, J, R, Q, G = setup_DAE1_RCL_LadderNetwork_sparse(
        ns = ns,
        r = random
            ? max.(0.2 * (randn(ns+2).+1), 0.001) 
            : 0.2 * ones(ns+2),
        m = mimo ? 2 : 1
    )
    A = (J-R)*Q
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

    # Transform into staircase form [AchAM2021]
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

    return StaircaseDAE(
        E = sparse(Diagonal(s)),
        A = sparse(UE'*(J-R)*UE),
        B = sparse(UE'*G),
        C = sparse(G'*UE),
        # add regularization term such that M_0+M_0' is nonsingular
        D = 1e-12*Matrix(I, m, m),
        n_1 = n_1, n_2 = n_2, n_3 = n_3, n_4 = n_4
    )
end

function read_ph_mat(path)
    v = matread(path)
    E = v["E"]
    A = (v["J"]-v["R"])*v["Q"]
    B = v["B"]
    m = size(B, 2)
    C = sparse(v["B"]')
    D = m == 1 ? reshape([v["S"]+v["N"]], (1,1)) : v["S"]+v["N"]
    n_1 = nnz(E)
    return SemiExplicitIndex1DAE(E,A,B,C,D,n_1)
end

function get_problem(name)
    problems = Dict(
        "DAE1_RCL_Simple_SISO" => () -> setup_DAE1_RCL(500, false, false),
        "DAE1_RCL_Random_SISO" => () -> setup_DAE1_RCL(500, true, false),
        "DAE0_RCL_Simple_SISO" => () ->
            toindex0sm(setup_DAE1_RCL(500, false, false)), # index-reduced sys
        "DAE0_RCL_Random_SISO" => () ->
            toindex0sm(setup_DAE1_RCL(500, true, false)), # index-reduced sys
        "DAE1_RCL_Simple_MIMO" => () -> setup_DAE1_RCL(500, false, true),
        "DAE1_RCL_Random_MIMO" => () -> setup_DAE1_RCL(500, true, true),
        "DAE2_RCL_Simple_SISO" => () -> setup_DAE2_RCL(500, false, false),
        "DAE2_RCL_Random_SISO" => () -> setup_DAE2_RCL(500, true, false),
        "DAE2_RCL_Simple_MIMO" => () -> setup_DAE2_RCL(500, false, true),
        "DAE2_RCL_Random_MIMO" => () -> setup_DAE2_RCL(500, false, true),
        "SchMMV22_FOM-CONS" => () -> read_ph_mat("scripts/SchMMV22_FOM-CONS.mat"),
        "SchMMV22_FOM-RAND" => () -> read_ph_mat("scripts/SchMMV22_FOM-RAND.mat"),
        "SchMMV22_FOM-MIMO" => () -> read_ph_mat("scripts/SchMMV22_FOM-MIMO.mat")
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
