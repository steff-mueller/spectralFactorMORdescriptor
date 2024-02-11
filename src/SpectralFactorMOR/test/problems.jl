module TestProblems

using LinearAlgebra
using SparseArrays
using PortHamiltonianBenchmarkSystems
using SpectralFactorMOR

export test_setup_DAE0_RCL, test_setup_DAE1_RCL, test_setup_DAE2_RCL

function test_setup_DAE1_RCL(m=1, ns=50)
    E, J, R, Q, G = setup_DAE1_RCL_LadderNetwork_sparse(ns = ns, m = m)
    A = (J-R)*Q
    B = G
    C = SparseMatrixCSC(G')
    # add regularization term such that M_0+M_0' is nonsingular for m=2
    D = m==2 ? 1e-12*[1 0; 0 1] : zeros(1,1)
    n_1 = nnz(E)
    return SemiExplicitIndex1DAE(E,A,B,C,D,n_1)
end

test_setup_DAE0_RCL(m=1, ns=50) = toindex0(test_setup_DAE1_RCL(m, ns))

function test_setup_DAE2_RCL(m=1, ns=50)
    E, J, R, G = setup_DAE2_RCL_LadderNetwork_sparse(ns = ns, m = m)

    # Transform into staircase form [AchAM2021]
    s, UE = eigen(Diagonal(E); sortby=x -> -x)
    rank_E = count(s .!= 0.0)
    n = size(E, 1)
    n_1 = n_4 = 1
    n_2 = rank_E - 1
    n_3 = n - n_1 - n_2 - n_4

    if m == 2
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

end # module TestProblems