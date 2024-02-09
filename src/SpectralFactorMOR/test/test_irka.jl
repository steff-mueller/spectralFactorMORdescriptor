using Test
using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using SparseArrays
using SpectralFactorMOR

E, J, R, Q, G = setup_DAE1_RCL_LadderNetwork_sparse(ns=3)
A = (J-R)*Q
B = G
C = SparseMatrixCSC(G')
D = diagm(ones(size(G,2)))
n_1 = nnz(E)
sys = SemiExplicitIndex1DAE(E,A,B,C,D,n_1)

Dr = sys.D - sys.C_2/Matrix(sys.A_22)*sys.B_2
T(s) = Matrix(C)/(s.*E-A)*B+D
Tr(s) = Cr/(s.*Er-Ar)*Br+Dr

r = 2
s = [0.1, 10]
c = ones(1,r)
b = ones(1,r)
Er,Ar,Br,Cr = i1interpolate(E,A,B,C,D,Dr,c,b,s)
@test T.(s) ≈ Tr.(s)

s = [0.1+0.1im, 0.1-0.1im]
c = [im -im]
b = [im -im]
Er,Ar,Br,Cr = i1interpolate(E,A,B,C,D,Dr,c,b,s)
@test T.(s) ≈ Tr.(s)
@test T.(s) ≈ Tr.(s)