#include <cstring>
#include <fstream>

#include "global_parameter.hpp"

namespace QCM {
  bool is_global_parameter_initialized = false;
  void global_parameter_init()
  {
    if(is_global_parameter_initialized) return;
    is_global_parameter_initialized = true;

    new_global_bool(false, "dual_basis", "uses the dual basis for wavevector computations");
    new_global_bool(false, "periodic","considers the cluster(s) as periodic");
    new_global_bool(false, "periodized_averages","computes lattice averages using the periodized Green function");
    new_global_bool(false, "zero_dim","sets the spatial dimension to zero, on any model");
    new_global_bool(false, "print_all","prints dependent parameters as well");

    new_global_double(0.0001, "eta", "value of the imaginary part of the frequency in Chern number/Berry phase computations");
    new_global_double(0.5,"small_scale", "low-frequency region for imaginary frequency axis integrals");
    new_global_double(1.0e12, "cutoff_scale", "high-frequency cutoff in integrals");
    new_global_double(1e-4, "accur_OP", "accuracy of lattice averages");
    new_global_double(20,"large_scale", "high-frequency region for imaginary frequency axis integrals");
    new_global_double(5e-8,"accur_SEF", "Accuracy of the Potthoff functional");
    
    new_global_int(0,"seed","seed of the random number generator");
    new_global_int(1024,"cuba2D_mineval","minimum number of integrand evaluations in CUBA (2D)");
    new_global_int(16000,"cuba3D_mineval","minimum number of integrand evaluations in CUBA (3D)");
    new_global_int(32,"kgrid_side","number of wavevectors on the side in a fixed wavevector grid");
    new_global_int(60,"max_iter_QN","maximum number of iterations in the quasi-Newton method");
    new_global_int(64,"max_dim_print","Maximum dimension for printing vectors and matrices");
    new_global_int(8, "GK_min_regions","minimum number of regions in the Gauss-Kronrod method");
    new_global_int(9,"print_precision","precision of printed output");

    new_global_char('G', "periodization", "periodization scheme: G, S, M, C or N (None)");

    // ED global parameters

    new_global_bool(false,"check_lanczos_residual","checks the Lanczos residual at the end of the eigenvector computation");
    new_global_bool(false,"no_degenerate_BL","forbids band lanczos to proceed when the eigenstates have degenerate energies");
    new_global_bool(false,"nosym", "does not take cluster symmetries into account");
    new_global_bool(false,"one_body_solution","Only solves the one-body part of the problem, for the Green function");
    new_global_bool(false,"print_Hamiltonian","Prints the Hamiltonian on the screen, if small enough");
    new_global_bool(false,"strip_anomalous_self","sets to zero the anomalous part of the self-energy");
    new_global_bool(false,"continued_fraction","Uses the continued fraction solver for the Green function instead of the band Lanczos method");
    new_global_bool(false,"verb_ED","prints ED information and progress");
    new_global_bool(false,"verb_integrals","prints information and progress about integrals");
    new_global_bool(false,"verb_Hilbert","prints progress information on bases and operators in the Hilbert space");
    new_global_bool(false,"verb_warning","prints warnings");
    new_global_bool(false,"print_variances","prints the variance of the operators in files");
    new_global_bool(false,"parallel_sectors","uses openMP to parallelize the computation of the Green function structures across the different sectors (uses more memory)");
    new_global_bool(false,"merge_states","merges states in the mixed state case");

    new_global_double(1e-12,"accur_band_lanczos","energy difference tolerance for stopping the BL process");
    new_global_double(0.01,"accur_continued_fraction","value of beta below which the simple Lanczos process stops");
    new_global_double(1.0e-5,"accur_Davidson","maximum norm of residuals in the Davidson-Liu algorithm");
    new_global_double(1e-7,"accur_deflation","norm below which a vector is deflated in the band Lanczos method");
    new_global_double(1e-12,"accur_lanczos","tolerance of the Ritz residual estimate in the Lanczos method");
    new_global_double(1.0e-5,"accur_Q_matrix","tolerance in the normalization of the Q matrix");
    new_global_double(1e-5,"band_lanczos_minimum_gap","gap between the lowest two states in BL below which the method fails");
    new_global_double(0.01,"minimum_weight","minimum weight in the density matrix");
    new_global_double(1.0e-5, "Qmatrix_wtol", "maximum difference in frequencies in Q-matrix");
    new_global_double(1.0e-10, "Qmatrix_vtol", "minimum value of a Qmatrix contribution");
    new_global_double(0.0,"temperature", "Temperature of the system.");

    new_global_int(1,"Davidson_states","Number of states requested in the Davidson-Liu algorithm");
    new_global_int(64,"max_dim_print","Maximum dimension for printing vectors and matrices");
    new_global_int(256,"max_dim_full","Maximum dimension for using full diagonalization");
    new_global_int(600,"max_iter_BL","Maximum number of iterations in the band Lanczos procedure");
    new_global_int(400,"max_iter_CF","Maximum number of iterations in the continuous fraction Lanczos procedure");
    new_global_int(1000,"max_iter_lanczos","Maximum number of iterations in the Lanczos procedure");
    new_global_int(0,"seed","seed of the random number generator");
    new_global_int(64,"GF_lookup_depth","depth of the look-up table for the cluster Green function");

    new_global_char('E', "Hamiltonian_format", "Desired Hamiltonian format: S (CSR matrix), O (individual operators), F (factorized), N (none = on the fly), E (Eigen CSR matrix))");
#ifdef WITH_PRIMME
    new_global_char('L', "Ground_state_method", "Desired method to compute the ground state: L (Lanczos method - default), D (Davidson method), P (use external PRIMME eigensolver - need qcm_wed to be compiled with PRIMME)");
    new_global_int(0,"PRIMME_preconditionning","Choose of preconditionner to solve ground state (if qcm_wed was compiled with PRIMME): 0 (No preconditionning), 1 (Jacobi preconditionner)");
    new_global_int(1, "PRIMME_algorithm", "PRIMME algorithm to solve ground state (if qcm_wed was compiled with PRIMME): 1 (PRIMME_DYNAMIC - default), 2 (PRIMME_DEFAULT_MIN_TIME), 3 (PRIMME_DEFAULT_MIN_MATVECS), 4 (PRIMME_Arnoldi), 5 (PRIMME_GD), 6 (PRIMME_GD_plusK), 7 (PRIMME_GD_Olsen_plusK), 8 (PRIMME_JD_Olsen_plusK), 9 (PRIMME_RQI), 10 (PRIMME_JDQR), 11 (PRIMME_JDQMR), 12 (PRIMME_JDQMR_ETol), 13 (PRIMME_STEEPEST_DESCENT), 14 (PRIMME_LOBPCG_OrthoBasis), 15 (PRIMME_LOBPCG_OrthoBasis_Window). See PRIMME documentation for more information");
    new_global_bool(false,"Ground_state_init_last", "Keep ground state in memory for initializing eigenvector in further ground state solving");
#else
    new_global_char('L', "Ground_state_method", "Desired method to compute the ground state: L (Lanczos method - default), D (Davidson method)");
#endif
  }
}

