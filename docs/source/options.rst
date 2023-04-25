.. include:: options_intro.txt

Boolean options
=========================
.. csv-table::
    :header: "name", "default", "description"
    :widths: 15, 10, 50

    "check_lanczos_residual", "false", "checks the Lanczos residual at the end of the eigenvector computation"
    "continued_fraction", "false", "Uses the continued fraction solver for the Green function instead of the band Lanczos method"
    "dual_basis", "false", "uses the dual basis for wavevector computations"
    "no_degenerate_BL", "false", "forbids band lanczos to proceed when the eigenstates have degenerate energies"
    "nosym", "false", "does not take cluster symmetries into account"
    "one_body_solution", "false", "Only solves the one-body part of the problem, for the Green function"
    "parallel_sectors", "false", "uses openMP to parallelize the computation of the Green function structures across the different sectors (uses more memory)"
    "periodic", "false", "considers the cluster(s) as periodic"
    "periodized_averages", "false", "computes lattice averages using the periodized Green function"
    "print_Hamiltonian", "false", "Prints the Hamiltonian on the screen, if small enough"
    "print_all", "false", "prints dependent parameters as well"
    "print_variances", "false", "prints the variance of the operators in files"
    "strip_anomalous_self", "false", "sets to zero the anomalous part of the self-energy"
    "verb_ED", "false", "prints ED information and progress"
    "verb_Hilbert", "false", "prints progress information on bases and operators in the Hilbert space"
    "verb_integrals", "false", "prints information and progress about integrals"
    "verb_warning", "false", "prints warnings"
    "zero_dim", "false", "sets the spatial dimension to zero, on any model"



Integer-valued options
=========================
.. csv-table::
    :header: "name", "default", "description"
    :widths: 15, 10, 50

    "Davidson_states", "1", "Number of states requested in the Davidson-Liu algorithm"
    "GK_min_regions", "8", "minimum number of regions in the Gauss-Kronrod method"
    "PRIMME_algorithm", "1", "PRIMME algorithm to solve ground state (if qcm_wed was compiled with PRIMME): 1 (PRIMME_DYNAMIC - default), 2 (PRIMME_DEFAULT_MIN_TIME), 3 (PRIMME_DEFAULT_MIN_MATVECS), 4 (PRIMME_Arnoldi), 5 (PRIMME_GD), 6 (PRIMME_GD_plusK), 7 (PRIMME_GD_Olsen_plusK), 8 (PRIMME_JD_Olsen_plusK), 9 (PRIMME_RQI), 10 (PRIMME_JDQR), 11 (PRIMME_JDQMR), 12 (PRIMME_JDQMR_ETol), 13 (PRIMME_STEEPEST_DESCENT), 14 (PRIMME_LOBPCG_OrthoBasis), 15 (PRIMME_LOBPCG_OrthoBasis_Window). See PRIMME documentation for more information"
    "PRIMME_preconditionning", "0", "Choose of preconditionner to solve ground state (if qcm_wed was compiled with PRIMME): 0 (No preconditionning), 1 (Jacobi preconditionner)"
    "cuba2D_mineval", "1024", "minimum number of integrand evaluations in CUBA (2D)"
    "cuba3D_mineval", "16000", "minimum number of integrand evaluations in CUBA (3D)"
    "kgrid_side", "32", "number of wavevectors on the side in a fixed wavevector grid"
    "max_dim_full", "256", "Maximum dimension for using full diagonalization"
    "max_dim_print", "64", "Maximum dimension for printing vectors and matrices"
    "max_iter_BL", "600", "Maximum number of iterations in the band Lanczos procedure"
    "max_iter_CF", "400", "Maximum number of iterations in the continuous fraction Lanczos procedure"
    "max_iter_QN", "60", "maximum number of iterations in the quasi-Newton method"
    "max_iter_lanczos", "600", "Maximum number of iterations in the Lanczos procedure"
    "print_precision", "8", "precision of printed output"
    "seed", "0", "seed of the random number generator"



Real-valued options
=========================
.. csv-table::
    :header: "name", "default", "description"
    :widths: 15, 10, 50

    "Qmatrix_vtol", "1e-10", "minimum value of a Qmatrix contribution"
    "Qmatrix_wtol", "1e-05", "maximum difference in frequencies in Q-matrix"
    "accur_Davidson", "1e-05", "maximum norm of residuals in the Davidson-Liu algorithm"
    "accur_OP", "0.0001", "accuracy of lattice averages"
    "accur_Q_matrix", "1e-05", "tolerance in the normalization of the Q matrix"
    "accur_SEF", "5e-08", "Accuracy of the Potthoff functional"
    "accur_band_lanczos", "1e-12", "energy difference tolerance for stopping the BL process"
    "accur_continued_fraction", "0.01", "value of beta below which the simple Lanczod process stops"
    "accur_deflation", "1e-07", "norm below which a vector is deflated in the band Lanczos method"
    "accur_lanczos", "1e-12", "tolerance of the Ritz residual estimate in the Lanczos method"
    "band_lanczos_minimum_gap", "1e-05", "gap between the lowest two states in BL below which the method fails"
    "cutoff_scale", "1e+12", "high-frequency cutoff in integrals"
    "eta", "0.0001", "value of the imaginary part of the frequency in Chern number/Berry phase computations"
    "large_scale", "20", "high-frequency region for imaginary frequency axis integrals"
    "minimum_weight", "0.01", "minimum weight in the density matrix"
    "small_scale", "0.5", "low-frequency region for imaginary frequency axis integrals"
    "temperature", "0", "Temperature of the system."



Char-valued options
=========================
.. csv-table::
    :header: "name", "default", "description"
    :widths: 15, 10, 50

    "Ground_state_method", "L", "Desired method to compute the ground state: L (Lanczos method - default), P (use external PRIMME eigensolver - need qcm_wed to be compiled with PRIMME)"
    "Hamiltonian_format", "E", "Desired Hamiltonian format: S (CSR matrix), O (individual operators), F (factorized), N (none = on the fly), E (Eigen CSR matrix))"
    "periodization", "G", "periodization scheme: G, S, M, C or N (None)"



