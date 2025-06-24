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
  DOWNLOAD_CUBA
  "Specify to download and compile automatically the CUBA integration library"
  ON)
option(
  CUBA_DIR
  "If CUBA not downloaded and build by this project, specify the path to CUBA directory for linking Pyqcm against (must contain compiled Cuba library libcuba.a along with the header cuba.h"
)
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
  "Specify the path to the PRIMME root directory for linking qcm_wed library against if not download automatically"
)
option(
  WITH_GF_OPT_KERNEL
  "Automatically detect host processor AVX2 capabilities (i.e. Intel processors after 2014 and AMD processors after 2015) and compile the optimized kernel (5 times quicker) for Green function evaluation"
  OFF)
option(QCM_DEBUG "Enable extra verbose information for debug purpose" OFF)
option(APPLE_LIBOMP_PREFIX
       "Custom path to the OpenMP library libomp on Apple computer")
option(USE_OPENMP "Enable parallelization using OpenMP" ON)
option(
  GENERIC_BUILD
  "Compile qcm_wed with the minimum optimization flag to target generic machine"
  ON)

# set the project name and version
set(GIT_DESCRIBE_CMD "")
execute_process(
  COMMAND git describe --abbrev=0
  COMMAND tr -d 'v'
  OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "QCM_WED version: ${GIT_DESCRIBE_VERSION}")
project(
  qcm
  LANGUAGES C CXX
  VERSION ${GIT_DESCRIBE_VERSION}
  DESCRIPTION "Quantum Cluster Methods shared library")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Host processor : ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "Architecture: ${ARCHITECTURE}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g ")

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
# Define target
# -------------

add_library(
  qcm SHARED
  src_qcm/CPT.cpp
  src_qcm/Chern.cpp
  src_qcm/QCM.cpp
  src_qcm/average.cpp
  src_qcm/basis3D.cpp
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
  src_util/integrate.cpp
  src_util/matrix.cpp
  src_util/parser.cpp
  src_util/VDVH_kernel.cpp
  src_util/vector_num.cpp)

set_target_properties(
  qcm
  PROPERTIES VERSION ${PROJECT_VERSION}
             PREFIX ""
             SUFFIX ".so")

# -----------------
# Python components
# -----------------

# Find Python3
find_package(
  Python3 3.10
  COMPONENTS Interpreter Development NumPy
  REQUIRED)

# Manual NumPy include detection if not found
if(NOT DEFINED Python3_NumPy_INCLUDE_DIRS)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(
    STATUS "Manually detected NumPy include directory: ${NUMPY_INCLUDE_DIR}")
else()
  set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})
endif()

# Python information
message(STATUS "Detected NumPy include directory: ${NUMPY_INCLUDE_DIR}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libs: ${Python3_LIBRARIES}")

# -------------------------------------
# External Dependencies for Hamiltonian
# -------------------------------------

if(EIGEN_HAMILTONIAN)
  find_package(Eigen3 REQUIRED)
  target_include_directories(qcm PUBLIC ${EIGEN3_INCLUDE_DIR})
  message(STATUS "Compile with Eigen Hamiltonian format 'E'")
  message(STATUS "Eigen3 directory: ${EIGEN3_INCLUDE_DIR}")
  add_compile_definitions(EIGEN_HAMILTONIAN)
endif()

# ------------------
# PRIMME Eigensolver
# ------------------

if(WITH_PRIMME)
  if(NOT EIGEN_HAMILTONIAN)
    message(
      FATAL_ERROR
        "PRIMME eigensolver require qcm_wed to be build with Eigen Hamiltonian. Add -DEIGEN_HAMILTONIAN=1 to CMake."
    )
  endif()
  if(DOWNLOAD_PRIMME)
    include(ExternalProject)
    set(CMD_MAKE_PRIMME "make")
    list(APPEND CMD_MAKE_PRIMME "lib")
    list(APPEND CMD_MAKE_PRIMME "CC=${CMAKE_C_COMPILER}"
         "CFLAGS=${CMAKE_CXX_FLAGS_RELEASE} -fPIC -DNDEBUG")
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
      BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/external/primme/lib/libprimme.a)
    add_dependencies(qcm PRIMME)
    set(PRIMME_LIBRARY ${CMAKE_SOURCE_DIR}/external/primme/lib/libprimme.a)
  else()
    find_library(
      PRIMME_LIBRARY REQUIRED
      NAMES libprimme.a
      HINTS "/usr/local/lib" ${PRIMME_DIR}/lib
            "${CMAKE_SOURCE_DIR}/external/primme/lib")
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
# Cuba library
# ------------

if(DOWNLOAD_CUBA)
  include(ExternalProject)
  ExternalProject_Add(
    Cuba
    PREFIX ${CMAKE_SOURCE_DIR}/external
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/cuba
    URL https://feynarts.de/cuba/Cuba-4.2.2.tar.gz
    URL_HASH MD5=a92897343027893ba2ec40dc15558b32
    CONFIGURE_COMMAND
    COMMAND ${CMAKE_SOURCE_DIR}/external/cuba/configure
            "CFLAGS=${CMAKE_CXX_FLAGS_RELEASE} -fPIC" "CC=${CMAKE_C_COMPILER}"
    BUILD_COMMAND
    COMMAND make "lib"
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    STEP_TARGETS build
    BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/external/cuba/libcuba.a)
  add_dependencies(qcm Cuba)
  set(CUBA_LIBRARY ${CMAKE_SOURCE_DIR}/external/cuba/libcuba.a)
else()
  find_library(
    CUBA_LIBRARY
    NAMES libcuba.a
    HINTS "${CUBA_DIR}" "/usr/local/lib" "$ENV{HOME}/lib"
          "${CMAKE_SOURCE_DIR}/external/cuba/")
endif()

get_filename_component(CUBA_DIR ${CUBA_LIBRARY} DIRECTORY)
message(STATUS "CUBA lib: ${CUBA_LIBRARY}")
message(STATUS "CUBA dir: ${CUBA_DIR}")

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
# ----------------------------

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# -----------------
# Link dependencies
# -----------------

target_link_libraries(qcm ${Python3_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${CUBA_LIBRARY} ${PRIMME_LIBRARY} OpenMP::OpenMP_CXX)

target_include_directories(
  qcm
  PUBLIC src_qcm
         src_ed
         src_python
         src_util
         ${Python3_INCLUDE_DIRS}
         ${NUMPY_INCLUDE_DIR}
         ${CUBA_DIR}
         ${PRIMME_DIR}/include)

# ---------------
# Install targets
# ---------------
# Be sure to install in the pyqcm folder
if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
endif()

install(TARGETS qcm DESTINATION pyqcm)
