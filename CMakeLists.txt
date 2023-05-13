cmake_minimum_required(VERSION 3.15)

# set the project name and version
project(qcm LANGUAGES CXX VERSION 1.0 DESCRIPTION "Quantum Cluster Methods shared library")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Host processor : ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "Architecture: ${ARCHITECTURE}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ftree-vectorize")
    set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -ftree-vectorize")
endif()

#------------------
# Define target 
#------------------
add_library(qcm SHARED
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
    src_util/vector_num.cpp
)

set_target_properties(qcm PROPERTIES VERSION ${PROJECT_VERSION} PREFIX "" SUFFIX ".so")

#------------------
# Python components
#------------------
find_package(Python3 3.7 COMPONENTS Interpreter Development NumPy REQUIRED)
message(STATUS "python executable: ${Python3_EXECUTABLE}")
find_package(PythonLibs REQUIRED)
message(STATUS "python include dir: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "python libs: ${PYTHON_LIBRARIES}")
set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})
message(STATUS "numpy dir : ${NUMPY_INCLUDE_DIR}")

if(SKBUILD)
    message(STATUS "this is built with SciKit-build")
else()
    execute_process (COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

#------------------
# External Dependencies for Hamiltonian
#------------------

if (EIGEN_HAMILTONIAN)
    find_package(Eigen3 REQUIRED)
    target_include_directories(qcm PUBLIC ${EIGEN3_INCLUDE_DIR})
    message(STATUS "Compile with Eigen Hamiltonian format 'E'")
    message(STATUS "Eigen3 directory: ${EIGEN3_INCLUDE_DIR}")
    add_compile_definitions(EIGEN_HAMILTONIAN)
endif()

if (PETSC_HAMILTONIAN)
    find_package(PETSc REQUIRED)
    target_link_libraries(qcm ${PETSC_LIBRARIES})
    target_include_directories(qcm ${PETSC_INCLUDE_DIR})
    message(STATUS "PETSc lib: ${PETSC_LIBRARIES}")
    add_compile_definitions(PETSC_HAMILTONIAN)
endif()


#------------------
# PRIMME Eigensolver
#------------------

if (WITH_PRIMME)
    if (NOT EIGEN_HAMILTONIAN)
        message(FATAL_ERROR "PRIMME eigensolver require qcm_wed to be build with Eigen Hamiltonian. Add -DEIGEN_HAMILTONIAN=1 to CMake.")
    endif()
    if (DOWNLOAD_PRIMME)
        include(ExternalProject)
        set(cmd_make_primme "make")
        list(APPEND cmd_make_primme "lib")
        list(APPEND cmd_make_primme "CFLAGS=${CMAKE_CXX_FLAGS_RELEASE} -fPIC -DNDEBUG")
        ExternalProject_Add(PRIMME
            PREFIX ${CMAKE_SOURCE_DIR}/external
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/primme
            URL https://github.com/primme/primme/archive/refs/tags/v3.2.tar.gz
            URL_HASH MD5=b26968d0ea8aa2e6feefc89f3c863062
            CONFIGURE_COMMAND COMMAND ""
            BUILD_COMMAND COMMAND ${cmd_make_primme}
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND ""
            STEP_TARGETS build
            BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/external/primme/lib/libprimme.a
        )
        add_dependencies(qcm PRIMME)
        set(PRIMME_LIBRARY ${CMAKE_SOURCE_DIR}/external/primme/lib/libprimme.a)
    else()
        find_library(
            PRIMME_LIBRARY REQUIRED
            NAMES libprimme.a
            HINTS "/usr/local/lib" ${PRIMME_DIR}/lib "${CMAKE_SOURCE_DIR}/external/primme/lib"
        )
    endif()
    get_filename_component(PRIMME_DIR ${PRIMME_LIBRARY} DIRECTORY)
    set(PRIMME_DIR "${PRIMME_DIR}/..")
    add_compile_definitions(WITH_PRIMME)
    message(STATUS "PRIMME_DIR: ${PRIMME_DIR}")
    message(STATUS "PRIMME lib: ${PRIMME_LIBRARY}")
endif()

#------------------
# BLAS / LAPACK
#------------------

include(FindLAPACK)
message(STATUS "LAPACK lib: ${LAPACK_LIBRARIES}")

#------------------
# Cuba library
#------------------
if (DOWNLOAD_CUBA)
    include(ExternalProject)
    ExternalProject_Add(Cuba
        PREFIX ${CMAKE_SOURCE_DIR}/external
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/cuba
        URL http://www.feynarts.de/cuba/Cuba-4.2.2.tar.gz
        URL_HASH MD5=a92897343027893ba2ec40dc15558b32
        CONFIGURE_COMMAND COMMAND ${CMAKE_SOURCE_DIR}/external/cuba/configure "CFLAGS=-O3 -fPIC -ffast-math -fomit-frame-pointer"
        BUILD_COMMAND COMMAND make "lib"
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
        STEP_TARGETS build
        BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/external/cuba/libcuba.a
    )
    add_dependencies(qcm Cuba)
    set(CUBA_LIBRARY ${CMAKE_SOURCE_DIR}/external/cuba/libcuba.a)
else()
    find_library(CUBA_LIBRARY
        NAMES libcuba.a
        HINTS "${CUBA_DIR}" "/usr/local/lib" "$ENV{HOME}/lib" "${CMAKE_SOURCE_DIR}/external/cuba/"
    )
endif()

get_filename_component(CUBA_DIR ${CUBA_LIBRARY} DIRECTORY)
message(STATUS "CUBA lib: ${CUBA_LIBRARY}")
message(STATUS "CUBA dir: ${CUBA_DIR}")


#------------------
# Link dependencies
#------------------
target_link_libraries(qcm ${PYTHON_LIBRARIES} ${LAPACK_LIBRARIES} ${CUBA_LIBRARY} ${PRIMME_LIBRARY})

target_include_directories(qcm PUBLIC src_qcm src_ed src_python src_util ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR}/numpy ${CUBA_DIR} ${PRIMME_DIR}/include)


#------------------
# Parallelization using OpenMP
#------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qcm OpenMP::OpenMP_CXX)
endif()


# install(TARGETS qcm DESTINATION ${PYTHON_INCLUDE_DIRS}/site-packages)
# install(TARGETS qcm DESTINATION ${CMAKE_SOURCE_DIR}/pyqcm)
install(TARGETS qcm DESTINATION pyqcm)
