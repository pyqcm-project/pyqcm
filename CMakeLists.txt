cmake_minimum_required(VERSION 3.18)

# Compiler options
if(APPLE)
    # enforce compiler because Apple system can use a broken compiler
    if((NOT CMAKE_C_COMPILER) AND (NOT CMAKE_CXX_COMPILER))
        message(
      STATUS
        "Enforce Clang and Clang++ compilers on Apple. To disable this behavior, manually set CMake variables -DCMAKE_C_COMPILER=[your_c_compiler] and -DCMAKE_CXX_COMPILER=[your_cxx_compiler]"
    )
        set(CMAKE_C_COMPILER "clang")
        set(CMAKE_CXX_COMPILER "clang++")
    endif()
elseif(UNIX)
    # Nothing
else()
    message(WARNING "Potentially unsupported system, compilation may fail...")
endif()

if(POLICY CMP0135) # From CMake 3.24
    # We don't care about DOWNLOAD_EXTRACT_TIMESTAMP; prefer the new behavior
    cmake_policy(SET CMP0135 NEW)
endif()

# --------------------
# QCM specific options
# --------------------
option(
  BLA_VENDOR
  "BLAS implementation to use. See CMake vendor documentation https://cmake.org/cmake/help/latest/module/FindBLAS.html for more information"
)
option(
  EIGEN_HAMILTONIAN
  "Specify to compile with Eigen format for the Hamiltonian for better performance in the diagonalisation solver on multi-core machine (require Eigen library)"
  ON)
option(
  WITH_PRIMME
  "Whether to use or not the PRIMME library and its eigensolver for finding ground state of the Hamiltonian (needs EIGEN_HAMILTONIAN=1)"
  ON)
option(
  DOWNLOAD_PRIMME
  "Specify to download and compile automatically the PRIMME eigensolver library (needs WITH_PRIMME=ON)"
  ON)
option(
  PRIMME_DIR
  "Specify the path to the PRIMME root directory for linking pyqcm library against if not download automatically"
)
option(
  WITH_GF_OPT_KERNEL
  "Automatically detect host processor AVX2 capabilities (i.e. Intel processors after 2014 and AMD processors after 2015) and compile the optimized kernel (5 times quicker) for Green function evaluation"
  OFF)
option(QCM_DEBUG "Enable extra verbose information for debug purpose" OFF)
option(USE_OPENMP "Enable parallelization using OpenMP" ON)
option(
  GENERIC_BUILD
  "Compile pyqcm with the minimum optimization flag to target generic machine"
  ON)

# Fetch git hash for traceability, not for versioning
execute_process(
    COMMAND git rev-parse --short HEAD
    OUTPUT_VARIABLE QCM_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
if(NOT QCM_GIT_HASH)
    set(QCM_GIT_HASH "unknown")
endif()
message(STATUS "Git hash: ${QCM_GIT_HASH}")

# When built through scikit-build, SKBUILD_PROJECT_VERSION is provided by
# setuptools-scm. For a direct CMake build, derive the version from git tags so
# the artifact is not stamped with the 0.0.0 placeholder.
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    execute_process(
        COMMAND git describe --tags --always
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    # Extract a strict X.Y.Z (CMake's project(VERSION ...) accepts numbers only)
    # from tags like "v2.26.1" or "v2.26.1-5-gc29031a".
    if(GIT_DESCRIBE MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        set(SKBUILD_PROJECT_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
    else()
        set(SKBUILD_PROJECT_VERSION "0.0.0")
        message(WARNING "Could not derive version from git; defaulting to 0.0.0")
    endif()
endif()

message(STATUS "PYQCM version: ${SKBUILD_PROJECT_VERSION}")
project(
  qcm
  LANGUAGES C CXX
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "Quantum Cluster Methods shared library")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Host processor : ${CMAKE_HOST_SYSTEM_PROCESSOR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# ---------------------------------
# Compilation option (optimisation)
# ---------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -xHost -funroll-loops")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ftree-vectorize -funroll-loops")
    include(CheckCXXCompilerFlag)
    if(NOT GENERIC_BUILD)
        check_cxx_compiler_flag("-march=native" NATIVE_ARCH_BUILD)
        if(NATIVE_ARCH_BUILD)
            message(STATUS "Target this machine architecture (enable -march=native)")
            set(CMAKE_CXX_FLAGS_RELEASE "-march=native ${CMAKE_CXX_FLAGS_RELEASE}")
        endif()
    endif()
endif()

if(QCM_DEBUG)
    add_definitions(-DQCM_DEBUG)
endif()

# -------------
# Define target sources (target itself is created below via nanobind_add_module,
# once nanobind has been discovered)
# -------------
set(QCM_SOURCES
  src_qcm/CPT.cpp
  src_qcm/Chern.cpp
  src_qcm/QCM.cpp
  src_qcm/average.cpp
  src_qcm/basis3D.cpp
  src_qcm/lattice_hybrid.cpp
  src_qcm/lattice3D.cpp
  src_qcm/lattice_model.cpp
  src_qcm/lattice_model_instance.cpp
  src_qcm/lattice_operator.cpp
  src_qcm/parameter_set.cpp
  src_qcm/profile.cpp
  src_ed/ED_basis.cpp
  src_ed/Operators/Heisenberg_operator.cpp
  src_ed/Operators/Hund_operator.cpp
  src_ed/Hamiltonian/PRIMME_solver.cpp
  src_ed/Hamiltonian/Lanczos.cpp
  src_ed/binary_state.cpp
  src_ed/continued_fraction.cpp
  src_ed/continued_fraction_set.cpp
  src_ed/matrix_continued_fraction.cpp
  src_ed/Operators/destruction_operator.cpp
  src_ed/model.cpp
  src_ed/model_instance.cpp
  src_ed/model_instance_base.cpp
  src_ed/qcm_ED.cpp
  src_ed/sector.cpp
  src_ed/symmetry_group.cpp
  src_python/common_Py.cpp
  src_python/qcm_lib.cpp
  src_util/ED_global_parameter.cpp
  src_util/global_parameter.cpp
  src_util/hdf5_io.cpp
  src_util/integrate.cpp
  external/cubature/hcubature.c
  external/cubature/pcubature.c
  src_util/matrix.cpp
  src_util/parser.cpp
  src_util/VDVH_kernel.cpp
  src_util/vector_num.cpp)

# Find just Python interpreter and development components
find_package(Python3 3.10
        COMPONENTS
        Interpreter
        COMPONENTS Interpreter Development NumPy
        REQUIRED
)

# -----------------
# Python components
# -----------------
# Find Python3
find_package(
  Python3 3.10
  COMPONENTS
  Interpreter Development NumPy
  REQUIRED
)

# Then handle NumPy separately
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    RESULT_VARIABLE NUMPY_DETECT_RESULT
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    ERROR_VARIABLE NUMPY_DETECT_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Verify NumPy headers exist
if(NOT EXISTS "${NUMPY_INCLUDE_DIR}/numpy/ndarrayobject.h")
    message(FATAL_ERROR "NumPy headers not found in ${NUMPY_INCLUDE_DIR}")
endif()

# Python information
message(STATUS "Detected NumPy include directory: ${NUMPY_INCLUDE_DIR}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libs: ${Python3_LIBRARIES}")

# --------
# nanobind
# --------
# nanobind's config requires the `Python` (not `Python3`) package, with the
# Development.Module component (so libpython is not linked explicitly).
find_package(Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# Locate the nanobind CMake package shipped with the installed Python module.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
message(STATUS "nanobind dir: ${nanobind_ROOT}")

# -------------
# Define target (Python extension module) via nanobind
# -------------
nanobind_add_module(qcm NB_STATIC ${QCM_SOURCES})

# Produce a plain artifact name (qcm.so), imported directly as the `qcm` module.
set_target_properties(
  qcm
  PROPERTIES
  PREFIX ""
  OUTPUT_NAME "qcm"
  SUFFIX ".so")

# ------
# Eigen3 (always required: used in matrix.cpp and optionally for Hamiltonian)
# ------

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(qcm PUBLIC Eigen3::Eigen)

# -------------------------------------
# External Dependencies for Hamiltonian
# -------------------------------------
if(EIGEN_HAMILTONIAN)
  find_package(Eigen3 REQUIRED NO_MODULE)
  target_link_libraries(qcm PUBLIC Eigen3::Eigen)
  message(STATUS "Compile with Eigen Hamiltonian format 'E'")
  add_compile_definitions(EIGEN_HAMILTONIAN)
endif()

# ------------------
# PRIMME Eigensolver
# ------------------
if(WITH_PRIMME)
    if(NOT EIGEN_HAMILTONIAN)
        message(
      FATAL_ERROR
        "PRIMME eigensolver require pyqcm to be build with Eigen Hamiltonian. Add -DEIGEN_HAMILTONIAN=1 to CMake."
    )
    endif()
    if(DOWNLOAD_PRIMME)
        include(ExternalProject)
        set(CMD_MAKE_PRIMME "make")
        list(APPEND CMD_MAKE_PRIMME "lib")
        list(APPEND CMD_MAKE_PRIMME "CC=${CMAKE_C_COMPILER}"
         "CFLAGS=${CMAKE_CXX_FLAGS_RELEASE} -fPIC -DNDEBUG")
        set(PRIMME_LIBRARY ${CMAKE_SOURCE_DIR}/external/primme/lib/libprimme.a)
        # The byproduct lives in a directory that does not exist until PRIMME is
        # downloaded/built. Create it now so the Makefile generator can emit a
        # file-level rule for the byproduct (otherwise the link step fails with
        # "No rule to make target libprimme.a").
        file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/external/primme/lib)
        ExternalProject_Add(
      PRIMME
      PREFIX ${CMAKE_SOURCE_DIR}/external
      SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/primme
      URL https://github.com/primme/primme/archive/refs/tags/v3.2.tar.gz
      URL_HASH MD5=b26968d0ea8aa2e6feefc89f3c863062
      CONFIGURE_COMMAND
      COMMAND ""
      BUILD_COMMAND
      COMMAND ${CMD_MAKE_PRIMME}
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ""
      STEP_TARGETS build
      BUILD_BYPRODUCTS ${PRIMME_LIBRARY})
        # Consume PRIMME through an IMPORTED target so the dependency on the
        # ExternalProject (and thus the byproduct rule) propagates to whatever
        # links against it, guaranteeing PRIMME is built before qcm is linked.
        add_library(primme STATIC IMPORTED GLOBAL)
        set_target_properties(primme PROPERTIES IMPORTED_LOCATION ${PRIMME_LIBRARY})
        # qcm depends on the ExternalProject so PRIMME is built before qcm links.
        add_dependencies(qcm PRIMME)
    else()
        find_library(
      PRIMME_LIBRARY REQUIRED
      NAMES libprimme.a
      HINTS "/usr/local/lib" ${PRIMME_DIR}/lib
            "${CMAKE_SOURCE_DIR}/external/primme/lib")
        add_library(primme STATIC IMPORTED GLOBAL)
        set_target_properties(primme PROPERTIES IMPORTED_LOCATION ${PRIMME_LIBRARY})
    endif()
    get_filename_component(PRIMME_DIR ${PRIMME_LIBRARY} DIRECTORY)
    set(PRIMME_DIR "${PRIMME_DIR}/..")
    add_compile_definitions(WITH_PRIMME)
    message(STATUS "PRIMME_DIR: ${PRIMME_DIR}")
    message(STATUS "PRIMME lib: ${PRIMME_LIBRARY}")
endif()

# -------------
# BLAS / LAPACK
# -------------
include(FindLAPACK REQUIRED)
message(STATUS "LAPACK lib: ${LAPACK_LIBRARIES}")

# ------------
# HDF5
# ------------
find_package(HDF5 REQUIRED COMPONENTS CXX)
target_link_libraries(qcm PRIVATE HDF5::HDF5)
target_include_directories(qcm PRIVATE ${HDF5_INCLUDE_DIRS})

# ------------------
# Cubature library
# ------------------
set(CUBATURE_DIR ${CMAKE_SOURCE_DIR}/external/cubature)
message(STATUS "Cubature dir: ${CUBATURE_DIR}")

# ----------------------------------------------
# Optimized kernel for Green function evaluation
# ----------------------------------------------
if(GENERIC_BUILD)
    set(WITH_GF_OPT_KERNEL OFF)
    message(
    WARNING
      "Generic build cannot use AVX2 optimized kernel for Green function evaluation. Setting WITH_GF_OPT_KERNEL to false"
  )
endif()
if(WITH_GF_OPT_KERNEL)
    if(APPLE)

    else()
        set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external" ${CMAKE_MODULE_PATH})
        include(FindAVX2)
        if(AVX2_FOUND)
            add_compile_definitions(HAVE_AVX2)
        else()
            message(
        WARNING
          "WITH_GF_KERNEL specified but the host processor does not have AVX2 capability: optimized kernel will not be used."
      )
            set(AVX2_FOUND false)
        endif()
        message(
      STATUS
        "Using AVX2 optimized kernel for Green function evaluation: ${AVX2_FOUND}"
    )
    endif()
endif()

# ----------------------------
# Parallelization using OpenMP
# ---------------------------
if(USE_OPENMP)
    if(APPLE)
        execute_process(
      COMMAND brew --prefix libomp
      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")

        set(OpenMP_ROOT "${HOMEBREW_LIBOMP_PREFIX}")
        message(STATUS "openMP library : ${HOMEBREW_LIBOMP_PREFIX}")
    endif()

    find_package(OpenMP REQUIRED)
    target_link_libraries(qcm PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------
# Link dependencies
# -----------------
# NB: do not link ${Python3_LIBRARIES} explicitly; nanobind_add_module handles
# Python linkage (uses Development.Module / -undefined dynamic_lookup).
# Link the PRIMME IMPORTED target (only present when WITH_PRIMME is enabled) so
# its build-order dependency on the ExternalProject is honoured.
target_link_libraries(qcm PUBLIC ${LAPACK_LIBRARIES}
                      $<TARGET_NAME_IF_EXISTS:primme> OpenMP::OpenMP_CXX)

target_include_directories(
  qcm
  PUBLIC src_qcm
         src_ed
         src_python
         src_util
         ${Python3_INCLUDE_DIRS}
         ${NUMPY_INCLUDE_DIR}
         ${CUBATURE_DIR}
         ${PRIMME_DIR}/include)

# ---------------
# Install targets
# ---------------
# Be sure to install in the pyqcm folder for a direct CMake build. CMake always
# defines CMAKE_INSTALL_PREFIX (default /usr/local), so test the dedicated
# "initialized to default" flag instead: only redirect to the source tree when
# the user did not pass an explicit prefix. Under scikit-build the prefix is set
# explicitly, so the staged wheel layout is left untouched.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH
        "Install path prefix" FORCE)
endif()

install(TARGETS qcm DESTINATION pyqcm)

