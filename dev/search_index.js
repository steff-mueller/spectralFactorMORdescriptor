var documenterSearchIndex = {"docs":
[{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"CurrentModule = SpectralFactorMOR","category":"page"},{"location":"SpectralFactorMOR/#SpectralFactorMOR","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"Documentation for the SpectralFactorMOR package.","category":"page"},{"location":"SpectralFactorMOR/#Installation","page":"SpectralFactorMOR","title":"Installation","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"To install the package in your Julia environment, run:","category":"page"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"pkg> add https://github.com/steff-mueller/spectralFactorMORdescriptor.git:src/SpectralFactorMOR","category":"page"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"and load it in your session:","category":"page"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"julia> using SpectralFactorMOR","category":"page"},{"location":"SpectralFactorMOR/#API","page":"SpectralFactorMOR","title":"API","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"","category":"page"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"Modules = [SpectralFactorMOR]","category":"page"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.AlmostKroneckerDAE","page":"SpectralFactorMOR","title":"SpectralFactorMOR.AlmostKroneckerDAE","text":"Represents a system of the form\n\nE = beginbmatrix\n    E_11  0       0  0\n    0       E_22  0  0\n    0       0       0  0\n    0       0       0  0\nendbmatrix quad\nA = beginbmatrix\n    0  0       0  I\n    0  A_22  0  0\n    0  0       I  0\n    -I000\nendbmatrix\n\nwith E_11 and E_22 nonsingular. The system matrices B and C are partitioned accordingly as\n\nB = beginbmatrix\n    B_1\n    B_2\n    B_3\n    B_4\nendbmatrix quad\nC = beginbmatrix\n    C_1  C_2  C_3  C_4\nendbmatrix\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.IRKAOptions","page":"SpectralFactorMOR","title":"SpectralFactorMOR.IRKAOptions","text":"The struct holds options for the IRKA algorithm.\n\nThe conv_tol option sets the convergence tolerance.\nThe max_iterations option sets the maximum number of iterations.\nThe cycle_detection_length controls how many past iterations are considered for cycle detection. The cycle_detection_tol sets the cycle detection tolerance. If cycle_detection_tol is zero, no cycle detection is performed.\nThe s_init_start and s_init_stop control the initial interpolation points. The initial interpolation points are distributed logarithmically between 10^s_init_start and 10^s_init_stop.\nIf randomize_s_init is true, the initial interpolation points are disturbed randomly by randn. The variance is set by randomize_s_var.\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.SemiExplicitIndex1DAE","page":"SpectralFactorMOR","title":"SpectralFactorMOR.SemiExplicitIndex1DAE","text":"Represents a system of the form\n\nE = beginbmatrix\n    E_11  0\n    0  0\nendbmatrix quad\nA = beginbmatrix\n    A_11  A_12\n    A_21  A_22\nendbmatrix\n\nwith E_11 nonsingular. The system matrices B and C are partitioned accordingly as\n\nB = beginbmatrix\n    B_1\n    B_2\nendbmatrix quad\nC = beginbmatrix\n    C_1  C_2\nendbmatrix\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.StaircaseDAE","page":"SpectralFactorMOR","title":"SpectralFactorMOR.StaircaseDAE","text":"Represents a system of the form\n\nE = beginbmatrix\n    E_11  0       0  0\n    0       E_22  0  0\n    0       0       0  0\n    0       0       0  0\nendbmatrixquad\nA = beginbmatrix\n    A_11  A_12  A_13  A_14\n    A_21  A_22  A_23  0\n    A_31  A_32  A_33  0\n    A_41  0       0       0\nendbmatrix\n\nwith E_11 E_22 A_14=-A_41^TA_33 nonsingular.\n\n\n\n\n\n","category":"type"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.arec_lr_nwt","page":"SpectralFactorMOR","title":"SpectralFactorMOR.arec_lr_nwt","text":"arec_lr_nwt(\n    F, E, Q, R, P_r=I, P_l=I; \n    conv_tol=1e-12, max_iterations=20, \n    lyapc_lradi_tol=1e-12, lyapc_lradi_max_iterations=150\n)\n\nSolves the Riccati equation\n\nFXE^T + EXF^T + EXQ^TQXE^T + P_l RR^T P_l^T = 0quad\nX = P_r X P_r^T\n\nusing the low-rank Newton method. Returns an approximate solution Z such that X  Z Z^T.\n\nSee [1, Alg. 9].\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.checkprojlure-Tuple{Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.checkprojlure","text":"checkprojlure(sys, X)\n\nChecks if X satisfies the projected positive real Lur'e equation, i. e., if there exist matrices L and M such that\n\nbeginalign*\n\tbeginbmatrix\n\t\t-A^TXE-E^TXA  P_r^TC^T - E^TXB\n\t\tCP_r - B^TXE  M_0+M_0^T\n\tendbmatrix = beginbmatrix\n\t\tL^T\n        M^T\n\tendbmatrix beginbmatrix\n\t\tL  M\n\tendbmatrix\n\tX=X^T geq 0 X=P_l^TXP_l\nendalign*\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.compress_lr-Tuple{Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.compress_lr","text":"compress_lr(Z, r)\n\nCompresses Z via SVD such that it holds Z Z^T  Z_mathrmnew Z_mathrmnew^T, where Z_mathrmnew is the returned matrix.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.compress_lr-Tuple{Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.compress_lr","text":"compress_lr(Z; tol=1e-02 * sqrt(size(Z, 1) * eps(Float64)))\n\nCompresses Z via SVD such that it holds Z Z^T  Z_mathrmnew Z_mathrmnew^T, where Z_mathrmnew is the returned matrix.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.initial_shifts","page":"SpectralFactorMOR","title":"SpectralFactorMOR.initial_shifts","text":"initial_shifts(A, E, B, max_iterations=10, iter=1)\n\nReturns initial shifts for the LR-ADI method.\n\nSee [2, pp. 92-95] and https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/lradi.py.\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.irka-Tuple{DescriptorSystems.AbstractDescriptorStateSpace, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.irka","text":"irka(\n    sys::AbstractDescriptorStateSpace, r, irka_options::IRKAOptions;\n    P_r=I, P_l=I, Dr=nothing\n)\n\nComputes a reduced-order model for sys using the Iterative rational Krylov Algorithm (IRKA) [3].\n\nThe parameter r corresponds to the dimension of the reduced-order model. If Dr=nothing, the feed-through matrix D_r of the reduced-order model is set to the original feed-through matrix D of the full-order model. For descriptor systems, the spectral projectors used for interpolation may be provided via the parameters P_r and P_l [4, Sec. 3].\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.irka-Tuple{SemiExplicitIndex1DAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.irka","text":"irka(sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions)\n\nComputes a reduced-order model for the semi-explicit index-1 system sys using the Iterative rational Krylov Algorithm (IRKA) [3, 4].\n\nThe parameter r corresponds to the dimension of the reduced-order model.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.irka-Tuple{StaircaseDAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.irka","text":"irka(sys::StaircaseDAE, r, irka_options::IRKAOptions)\n\nComputes a reduced-order model for the staircase system sys using the Iterative rational Krylov Algorithm (IRKA) [3, 4].\n\nThe parameter r corresponds to the dimension of the reduced-order model.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.is_valid-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.is_valid","text":"is_valid(sys::StaircaseDAE)\n\nAsserts that sys is in valid staircase form.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.isastable-Tuple{DescriptorSystems.AbstractDescriptorStateSpace}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.isastable","text":"isastable(sys::AbstractDescriptorStateSpace)\n\nChecks if sys is asymptotically stable.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.ispr-Tuple{DescriptorSystems.AbstractDescriptorStateSpace}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.ispr","text":"ispr(sys::AbstractDescriptorStateSpace) -> (ispr, mineigval, w)\n\nChecks if the system is positive real by sampling the Popov function on the imaginary axis and verifying that the Popov function is positive semi-definite for all sample points.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.lrcf-Tuple{Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.lrcf","text":"lrcf(X, trunc_tol)\n\nComputes an approximate low-rank Cholesky-like  factorization of a symmetric positive semi-definite matrix X s.t. X = Z^T Z (up to a prescribed tolerance trunc_tol).\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.lyapc_lradi","page":"SpectralFactorMOR","title":"SpectralFactorMOR.lyapc_lradi","text":"lyapc_lradi(A, E, B, P_l=I; tol=1e-12, max_iterations=150)\n\nComputes a low-rank solution for the projected continuous-time Lyapunov equation\n\nEXA^T + AXE^T = - P_l BB^T P_l^Tquad\nX = P_r X P_r^T\n\nusing the LR-ADI method [1, Alg. 7]. Returns an approximate solution Z such that X  Z Z^T.\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.new_shifts","page":"SpectralFactorMOR","title":"SpectralFactorMOR.new_shifts","text":"new_shifts(A, E, V, Z, prev_shifts, subspace_columns=6)\n\nReturns new intermediate shifts for the LR-ADI method.\n\nSee [2, pp. 92-95] and https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/lradi.py.\n\n\n\n\n\n","category":"function"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_c_gramian-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_c_gramian","text":"pr_c_gramian(sys::SemiExplicitIndex1DAE)\n\nComputes the positive real controllability Gramian, i.e., the unique stabilizing solution of\n\nAXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0)^-1 (P_l B - EXC^T)^T = 0quad\nX = P_r X P_r^T\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_c_gramian-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_c_gramian","text":"pr_c_gramian(sys::StaircaseDAE)\n\nComputes the positive real observability Gramian, i.e., the unique stabilizing solution of\n\nAXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0)^-1 (P_l B - EXC^T)^T = 0quad\nX = P_r X P_r^T\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_c_gramian_lr-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_c_gramian_lr","text":"pr_c_gramian_lr(sys::SemiExplicitIndex1DAE)\n\nComputes a low-rank factor of positive real controllability Gramian, i.e., the unique stabilizing solution of\n\nAXE^T + EXA^T + (P_l B - EXC^T) (M_0+M_0)^-1 (P_l B - EXC^T)^T = 0quad\nX = P_r X P_r^T\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nReturns an approximate solution Z such that X  Z Z^T.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_o_gramian-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_o_gramian","text":"pr_o_gramian(sys::SemiExplicitIndex1DAE)\n\nComputes positive real observability Gramian, i.e., the unique stabilizing solution of\n\nA^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0)^-1(C P_r-B^TXE) = 0quad\nX = P_l^T Y P_l\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_o_gramian-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_o_gramian","text":"pr_o_gramian(sys::StaircaseDAE)\n\nComputes positive real observability Gramian, i.e., the unique stabilizing solution of\n\nA^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0)^-1(C P_r-B^TXE) = 0quad\nX = P_l^T Y P_l\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nNote: Uses the dense solver MatrixEquations.garec.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.pr_o_gramian_lr-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.pr_o_gramian_lr","text":"pr_o_gramian_lr(sys::SemiExplicitIndex1DAE)\n\nComputes a low-rank factor of the positive real observability Gramian, i.e., the unique stabilizing solution of\n\nA^TXE + E^TXA + (P_r^T C^T-E^TXB)(M_0+M_0)^-1(C P_r-B^TXE) = 0quad\nX = P_l^T Y P_l\n\nwhere P_r, P_l and M_0 are defined by the given system sys.\n\nReturns an approximate solution Z such that X  Z Z^T.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.prbaltrunc-Tuple{SemiExplicitIndex1DAE, Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.prbaltrunc","text":"prbaltrunc(sys::SemiExplicitIndex1DAE, r, Z_prc, Z_pro)\n\nComputes a reduced-order model for the semi-explicit index-1 system sys using positive real balanced truncation [1, Alg. 2].\n\nThe parameter r corresponds to the dimension of the reduced-order model.\n\nThe parameter Z_prc must be a (low-rank) Cholesky factor such that Z_mathrmprc^T Z_mathrmprc is the positive real controllability Gramian.\n\nThe parameter Z_pro must be a (low-rank) Cholesky factor such that Z_mathrmpro^T Z_mathrmpro is the positive real observability Gramian.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.prbaltrunc-Tuple{StaircaseDAE, Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.prbaltrunc","text":"prbaltrunc(sys::StaircaseDAE, r, Z_prc, Z_pro)\n\nComputes a reduced-order model for the staircase system sys using positive real balanced truncation [1, Alg. 2].\n\nThe parameter r corresponds to the dimension of the finite part of the reduced-order model.\n\nThe parameter Z_prc must be a (low-rank) Cholesky factor such that Z_mathrmprc^T Z_mathrmprc is the positive real controllability Gramian.\n\nThe parameter Z_pro must be a (low-rank) Cholesky factor such that Z_mathrmpro^T Z_mathrmpro is the positive real observability Gramian.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.retry_irka-Tuple{Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.retry_irka","text":"retry_irka(run, max_tries, syssp)\n\nRetries IRKA max_tries times and picks the result with the lowest H2-error. The strictly proper subsystem of the full-order model syssp must be supplied  in order to compute the H2-error. The parameter run must be a callable which executes IRKA.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.sfmor-Tuple{SemiExplicitIndex1DAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.sfmor","text":"sfmor(\n    sys::SemiExplicitIndex1DAE, r, irka_options::IRKAOptions;\n    X=nothing, compute_factors = :together, compute_factors_tol=1e-12\n)\n\nExecutes the spectral factor MOR method for a semi-explicit index-1 system.\n\nA solution to the positive real projected Lur'e equation can be supplied via the parameter X. If no X is supplied, the positive real observability Gramian is computed using pr_o_gramian.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.sfmor-Tuple{StaircaseDAE, Any, IRKAOptions}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.sfmor","text":"sfmor(\n    sys::StaircaseDAE, r, irka_options::IRKAOptions;\n    X = nothing, compute_factors = :together, compute_factors_tol=1e-12\n)\n\nExecutes the spectral factor MOR method for a system in staircase form.\n\nA solution to the positive real projected Lur'e equation can be supplied via the parameter X. If no X is supplied, the positive real observability Gramian is computed using pr_o_gramian.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.splitsys-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.splitsys","text":"splitsys(sys::SemiExplicitIndex1DAE) -> (sys1, sys2)\n\nReturns the infinite and strictly proper subsystems of sys.\n\nsys1 is the infinite subsystem.\nsys2 is the strictly proper subsystem.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.splitsys-Tuple{SpectralFactorMOR.AlmostKroneckerDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.splitsys","text":"splitsys(sys::AlmostKroneckerDAE) -> (sys1, sys2)\n\nReturns the infinite and strictly proper subsystems of sys.\n\nsys1 is the infinite subsystem.\nsys2 is the strictly proper subsystem.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.splitsys-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.splitsys","text":"splitsys(sys::StaircaseDAE) -> (sys1, sys2)\n\nReturns the infinite and strictly proper subsystems of sys.\n\nsys1 is the infinite subsystem.\nsys2 is the strictly proper subsystem.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.toindex0-Tuple{SemiExplicitIndex1DAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.toindex0","text":"toindex0(sys::SemiExplicitIndex1DAE)\n\nTransforms sys to a system with index 0. Returns a system of type SemiExplicitIndex1DAE.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.tokronecker-Tuple{StaircaseDAE}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.tokronecker","text":"tokronecker(sys::StaircaseDAE)\n\nTransforms sys into almost Kronecker form [5, Alg. 5]. Returns a system of type AlmostKroneckerDAE.\n\nThe result is cached using the Memoize.jl package.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#SpectralFactorMOR.truncation-Tuple{Any, Any, Any}","page":"SpectralFactorMOR","title":"SpectralFactorMOR.truncation","text":"truncation(d, L, trunc_tol) -> (dr, Lr)\n\nComputes a rank revealing factorization for a given LDL-decomposition of S = L * mathrmdiag(d) * L^T (up to a prescribed tolerance trunc_tol) such that L_r * diag(d_r) * L_r^T approx S.\n\n\n\n\n\n","category":"method"},{"location":"SpectralFactorMOR/#References","page":"SpectralFactorMOR","title":"References","text":"","category":"section"},{"location":"SpectralFactorMOR/","page":"SpectralFactorMOR","title":"SpectralFactorMOR","text":"P. Benner and T. Stykel. Model Order Reduction for Differential-Algebraic Equations: A Survey. In: Surveys in Differential-Algebraic Equations IV (Springer International Publishing, 2017); pp. 107–160.\n\n\n\nP. Kürschner. Efficient low-rank solution of large-scale matrix equations. Ph.D. Thesis, Shaker Verlag Aachen (2016).\n\n\n\nA. C. Antoulas, C. A. Beattie and S. Gugercin. Interpolatory Methods for Model Reduction (Society for Industrial and Applied Mathematics, 2020).\n\n\n\nS. Gugercin, T. Stykel and S. Wyatt. Model Reduction of Descriptor Systems by Interpolatory Projection Methods. SIAM Journal on Scientific Computing 35, B1010–B1033 (2013).\n\n\n\nF. Achleitner, A. Arnold and V. Mehrmann. Hypocoercivity and controllability in linear semi-dissipative Hamiltonian ordinary differential equations and differential-algebraic equations. ZAMM - Journal of Applied Mathematics and Mechanics (2021), arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/zamm.202100171.\n\n\n\n","category":"page"},{"location":"#Passivity-preserving-model-reduction-for-descriptor-systems-via-spectral-factorization","page":"Home","title":"Passivity-preserving model reduction for descriptor systems via spectral factorization","text":"","category":"section"},{"location":"#Citing","page":"Home","title":"Citing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use this project for academic work, please consider citing our publication","category":"page"},{"location":"","page":"Home","title":"Home","text":"TODO","category":"page"},{"location":"#How-to-install","page":"Home","title":"How to install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Clone the project and navigate to the folder:","category":"page"},{"location":"","page":"Home","title":"Home","text":"shell> git clone https://github.com/steff-mueller/spectralFactorMORdescriptor.git\nshell> cd spectralFactorMORdescriptor","category":"page"},{"location":"","page":"Home","title":"Home","text":"Activate and instantiate the Julia environment using the Julia package manager to install the required packages:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> activate .\npkg> instantiate","category":"page"},{"location":"#How-to-reproduce","page":"Home","title":"How to reproduce","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The scripts/ folder contains TOML configuration files for the different experiments. Point the RCL_CONFIG environment variable to the experiment you want to run. Use the scripts/rcl.jl script to run an experiment.  For example, for scripts/rcl_dae2_random_mimo.toml, execute the following commands:","category":"page"},{"location":"","page":"Home","title":"Home","text":"shell> cd spectralFactorMORdescriptor\nshell> export RCL_CONFIG=\"scripts/rcl_dae2_random_mimo.toml\"\nshell> julia --project=. scripts/rcl.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"The experiment results are stored under data/.","category":"page"},{"location":"#Julia-package","page":"Home","title":"Julia package","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The project contains a Julia package under src/SpectralFactorMOR which you can include in your own Julia projects to use the methods. See SpectralFactorMOR.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Distributed under the MIT License. See LICENSE for more information.","category":"page"},{"location":"#Contact","page":"Home","title":"Contact","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Steffen Müller - steffen.mueller@simtech.uni-stuttgart.de\nBenjamin Unger - benjamin.unger@simtech.uni-stuttgart.de","category":"page"}]
}
