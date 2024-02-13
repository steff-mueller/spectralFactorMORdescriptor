var documenterSearchIndex = {"docs":
[{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"CurrentModule = SpectralFactorMOR","category":"page"},{"location":"SpectralFactorMOR/#SpectralFactorMOR","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"Documentation for SpectralFactorMOR.","category":"page"},{"location":"SpectralFactorMOR/#API","page":"SpectralFactorMOR","title":"API","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"","category":"page"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"Modules = [SpectralFactorMOR]","category":"page"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.AlmostKroneckerDAE","page":"SpectralFactorMOR","title":"SpectralFactorMOR.AlmostKroneckerDAE","text":"Represents a system of the form\n\nE = beginbmatrix\n    E_11  0       0  0\n    0       E_22  0  0\n    0       0       0  0\n    0       0       0  0\nendbmatrix quad\nA = beginbmatrix\n    0  0       0  I\n    0  A_22  0  0\n    0  0       I  0\n    -I000\nendbmatrix\n\nwith E_11 and E_22 nonsingular. The system matrices B and C are partitioned accordingly as\n\nB = beginbmatrix\n    B_1\n    B_2\n    B_3\n    B_4\nendbmatrix quad\nC = beginbmatrix\n    C_1  C_2  C_3  C_4\nendbmatrix\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.SemiExplicitIndex1DAE","page":"SpectralFactorMOR","title":"SpectralFactorMOR.SemiExplicitIndex1DAE","text":"Represents a system of the form\n\nE = beginbmatrix\n    E_11  0\n    0  0\nendbmatrix quad\nA = beginbmatrix\n    A_11  A_12\n    A_21  A_22\nendbmatrix\n\nwith E_11 nonsingular. The system matrices B and C are partitioned accordingly as\n\nB = beginbmatrix\n    B_1\n    B_2\nendbmatrix quad\nC = beginbmatrix\n    C_1  C_2\nendbmatrix\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.StaircaseDAE","page":"SpectralFactorMOR","title":"SpectralFactorMOR.StaircaseDAE","text":"Represents a system of the form\n\nE = beginbmatrix\n    E_11  0       0  0\n    0       E_22  0  0\n    0       0       0  0\n    0       0       0  0\nendbmatrixquad\nA = beginbmatrix\n    A_11  A_12  A_13  A_14\n    A_21  A_22  A_23  0\n    A_31  A_32  A_33  0\n    A_41  0       0       0\nendbmatrix\n\nwith E_11 E_22 A_14=-A_41^TA_33 nonsingular.\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.arec_lr_nwt","page":"SpectralFactorMOR","title":"SpectralFactorMOR.arec_lr_nwt","text":"arec_lr_nwt(\n    F, E, Q, R, P_r=I, P_l=I; \n    conv_tol=1e-12, max_iterations=20, \n    lyapc_lradi_tol=1e-12, lyapc_lradi_max_iterations=150\n)\n\nSolves the Riccati equation\n\nFXE^T + EXF^T + EXQ^TQXE^T + P_l RR^T P_l^T = 0quad\nX = P_r X P_r^T\n\nusing the low-rank Newton method. Returns an approximate solution Z such that X  Z Z^T.\n\nSee [1, Alg. 9].\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.checkprojlure-Tuple{Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.checkprojlure","text":"Check if X satisfies the projected positive real Lur'e equation.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.compress_lr-Tuple{Any, Int64}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.compress_lr","text":"compress_lr(Z, r::Int)\n\nCompresses Z via SVD such that it holds Z Z^T  Z_mathrmnew Z_mathrmnew^T, where Z_mathrmnew is the returned matrix.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.compress_lr-Tuple{Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.compress_lr","text":"compress_lr(Z; tol=1e-02 * sqrt(size(Z, 1) * eps(Float64)))\n\nCompresses Z via SVD such that it holds Z Z^T  Z_mathrmnew Z_mathrmnew^T, where Z_mathrmnew is the returned matrix.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.initial_shifts","page":"SpectralFactorMOR","title":"SpectralFactorMOR.initial_shifts","text":"initial_shifts(A,E,B,max_iterations=10,iter=0)\n\nReturns initial shifts for LR-ADI method.\n\nSee [Kür16_Doktorarbeit, pp. 92-95] and https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/lradi.py.\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.irka-Tuple{DescriptorSystems.AbstractDescriptorStateSpace, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.irka","text":"i0irka(\n    sys::Union{DescriptorStateSpace,SparseDescriptorStateSpace},\n    r::Int, irka_options::IRKAOptions\n)\n\nIRKA for index-0 systems.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.irka-Tuple{SemiExplicitIndex1DAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.irka","text":"i1irka(sys::SemiExplicitIndex1DAE, r::Int, irka_options::IRKAOptions)\n\nIRKA for semiexplicit index-1 DAEs.\n\nGugercin, S., Stykel, T., & Wyatt, S. (2013). Model Reduction of Descriptor Systems by Interpolatory Projection Methods. SIAM. https://doi.org/10.1137/130906635\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.irka-Tuple{StaircaseDAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.irka","text":"i1irka(sys::StaircasePHDAE, r::Int, irka_options::IRKAOptions)\n\nIRKA for port-Hamiltonian DAEs in staircase form.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.is_valid-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.is_valid","text":"Assert that sys is in valid staircase form.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.isastable-Tuple{DescriptorSystems.AbstractDescriptorStateSpace}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.isastable","text":"Check if sys is asymptotically stable.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.lrcf-Tuple{Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.lrcf","text":"lrcf(X, trunc_tol)\n\nComputes an approximate low-rank Cholesky-like  factorization of a symmetric positive semi-definite matrix X s.t. X = Z'*Z (up to a prescribed tolerance trunc_tol).\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.lyapc_lradi","page":"SpectralFactorMOR","title":"SpectralFactorMOR.lyapc_lradi","text":"lyapc_lradi(A, E, B, P_l=I; tol=1e-12, max_iterations=150)\n\nLR-ADI method for the projected continuous-time Lyapunov equation [DAEForumIV, Algorithm 7]\n\nEXA^T + AXE^T = - P_l BB^T P_l^T,\nX = P_r X P_r^T.\n\nReturns an approximate solution Z such that\n\nX ≈ Z*Z'.\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.new_shifts","page":"SpectralFactorMOR","title":"SpectralFactorMOR.new_shifts","text":"new_shifts(A,E,V,Z,prev_shifts,subspace_columns=6)\n\nReturns new intermediate shifts for LR-ADI method.\n\nSee [Kür16_Doktorarbeit, pp. 92-95] and https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/lradi.py.\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_c_gramian-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_c_gramian","text":"pr_c_gramian(sys::SemiExplicitIndex1DAE)\n\nComputes the positive real controllability Gramian, i.e., the unique stabilizing solution of\n\nAXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0)^-1 (P_l B - EXC^T)^T = 0quad\nX = P_r X P_r^T\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_c_gramian-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_c_gramian","text":"pr_c_gramian(sys::StaircaseDAE)\n\nComputes the positive real observability Gramian, i.e., the unique stabilizing solution of\n\nAXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0)^-1 (P_l B - EXC^T)^T = 0quad\nX = P_r X P_r^T\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_c_gramian_lr-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_c_gramian_lr","text":"pr_c_gramian_lr(sys::SemiExplicitIndex1DAE)\n\nComputes a low-rank factor of positive real controllability Gramian, i.e., the unique stabilizing solution of\n\nAXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0)^-1 (P_l B - EXC^T)^T = 0quad\nX = P_r X P_r^T\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nReturns an approximate solution Z such that X  Z Z^T.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_o_gramian-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_o_gramian","text":"pr_o_gramian(sys::SemiExplicitIndex1DAE)\n\nComputes positive real observability Gramian, i.e., the unique stabilizing solution of\n\nA^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0)^-1(C P_r-B^TXE) = 0quad\nX = P_l^T Y P_l\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_o_gramian-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_o_gramian","text":"pr_o_gramian(sys::StaircaseDAE)\n\nComputes positive real observability Gramian, i.e., the unique stabilizing solution of\n\nA^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0)^-1(C P_r-B^TXE) = 0quad\nX = P_l^T Y P_l\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_o_gramian_lr-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_o_gramian_lr","text":"pr_o_gramian_lr(sys::SemiExplicitIndex1DAE)\n\nComputes a low-rank factor of the positive real observability Gramian, i.e., the unique stabilizing solution of\n\nA^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0)^-1(C P_r-B^TXE) = 0quad\nX = P_l^T Y P_l\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nReturns an approximate solution Z such that X  Z Z^T.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.prbaltrunc-Tuple{SemiExplicitIndex1DAE, Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.prbaltrunc","text":"prbaltrunc(sys::SemiExplicitIndex1DAE, r::Int)\n\nPositive real balanced truncation for descriptor systems [DAEForumIV, Algorithm 2].\n\nThe parameter r corresponds to the reduced order of the finite part of the system.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.retry_irka-Tuple{Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.retry_irka","text":"Retry IRKA max_retries times and pick the result with the lowest H2-error. syssp is the strictly proper subsystem of the full-order model.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.sfmor-Tuple{SemiExplicitIndex1DAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.sfmor","text":"A solution to the positive real projected Lur'e equation can be supplied via the parameter X. If no X is supplied, the positive real observability Gramian is computed.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.sfmor-Tuple{StaircaseDAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.sfmor","text":"A solution to the positive real projected Lur'e equation can be supplied via the parameter X. If no X is supplied, the positive real observability Gramian is computed.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.splitsys-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.splitsys","text":"splitsys(sys::SemiExplicitIndex1DAE)\n\nReturn infinite and strictly proper subsystems.\n\nsys1 is the infinite subsystem.\nsys2 is the strictly proper subsystem.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.splitsys-Tuple{SpectralFactorMOR.AlmostKroneckerDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.splitsys","text":"splitsys(sys::AlmostKroneckerDAE)\n\nReturn infinite and strictly proper subsystems.\n\nsys1 is the infinite subsystem.\nsys2 is the strictly proper subsystem.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.splitsys-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.splitsys","text":"splitsys(sys::StaircaseDAE)\n\nReturn infinite and strictly proper subsystems.\n\nsys1 is the infinite subsystem.\nsys2 is the strictly proper subsystem.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.toindex0-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.toindex0","text":"toindex0(sys::SemiExplicitIndex1DAE)\n\nTransform sys to a system with index 0.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.tokronecker-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.tokronecker","text":"Transform sys into almost Kronecker form [AchAM2021].\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.truncation-Tuple{Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.truncation","text":"truncation(d, L, trunc_tol)\n\nComputes a rank revealing factorization for a given LDL-decomposition of S = L * diagm(d) * L (up to a prescribed tolerance trunc_tol).\n\nReturns dr,Lr such that Lr * diagm(d) * Lr approx S.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#References","page":"SpectralFactorMOR","title":"References","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"P. Benner and T. Stykel. Model Order Reduction for Differential-Algebraic Equations: A Survey. In: Surveys in Differential-Algebraic Equations IV (Springer International Publishing, 2017); pp. 107–160.\n\n\n\n","category":"page"},{"location":"#Passivity-preserving-model-reduction-for-descriptor-systems-via-spectral-factorization","page":"Home","title":"Passivity-preserving model reduction for descriptor systems via spectral factorization","text":"","category":"section"}]
}
