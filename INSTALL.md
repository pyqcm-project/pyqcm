# PyQCM installation

## Default installation using pip

This is the easiest installation of PyQCM suitable for all users.
It builds the PyQCM library and downloads runtime dependencies automatically using the default system compiler and libraries.

1. Install CMake (may change depending of your system):
On Ubuntu:
```
sudo apt install cmake
```
On MacOS:
```
brew install cmake
```


2. Open a terminal and clone the Pyqcm Git repository:

```
git clone https://bitbucket.org/dsenechQCM/qcm_wed.git
cd qcm_wed
```

2. Configure Pyqcm to use whether or not optional dependencies (optional, see below):
```
export CMAKE_ARGS="[BUILD_ARG1]=[VALUE1] [BUILD_ARG2]=[VALUE2] ..."
```

3. Perform automatic installation with `pip`:

```
pip install .
```

## Building the documentation

The code's documentation can be built by going to the "docs" directory and typing "./makedoc"
By default this produces a web site within the "html" folder. Open the file "html/index.html".
Alternatively, one can acccess the documentation at

https://qcm-wed.readthedocs.io/en/latest/

## Build options

Additional options for building the code are described in the introduction section of the documentation.

* ``-DDOWNLOAD_CUBA=0/1``: Specify to download and compile automatically the CUBA integration library.
* ``-DCUBA_DIR=[path_to_CUBA_root_dir]``: If CUBA not downloaded from above, specify the path to CUBA directory for linking Pyqcm against (must contain compiled Cuba library ``libcuba.a`` along with the header ``cuba.h``).
* ``-DBLA_VENDOR=[value]``: BLAS implementation to use. See `CMake vendor documentation <https://cmake.org/cmake/help/latest/module/FindBLAS.html?highlight=bla_vendor#blas-lapack-vendors>`_ for more information (recommended: do not specify or ``FlexiBLAS`` on Digital Alliance clusters in Canada). Otherwise install LAPACK/BLAS on your system and cmake will probably find it. For MacOS : will use the Accelerate Framework automatically.
* ``-DEIGEN_HAMILTONIAN=0/1``: Specify to compile with Eigen format for the Hamiltonian for better performance in the diagonalisation solver on multi-core machine (require `Eigen library <https://eigen.tuxfamily.org/index.php?title=Main_Page>`_).
* ``-DWITH_PRIMME=0/1``: Whether to use or not the PRIMME library and its eigensolver for finding ground state of the Hamiltonian (needs ``-DEIGEN_HAMILTONIAN=1``).
* ``-DDOWNLOAD_PRIMME=0/1``: Specify to download and compile automatically the PRIMME eigensolver library (needs ``-DWITH_PRIMME=1``).
* ``-DPRIMME_DIR=[path_to_PRIMME_root_dir]``: Specify the path to the PRIMME root directory for linking qcm_wed library against if not download automatically.


## Installation from source

These instructions allow a finely tunned compilation of the qcm library and dependencies for better performance.
Library and dependencies compilation and installation are no more handled automatically.


### CUBA numerical integration library

1. Download and extract sources:
```
wget http://www.feynarts.de/cuba/Cuba-4.2.2.tar.gz #download
tar -xf Cuba-4.2.2.tar.gz #decompress
```

2. Configure and compile:
```
cd Cuba-4.2.2 #enter extraction CUBA directory
./configure CFLAGS="-O3 -fPIC -ffast-math -fomit-frame-pointer" #configure
make lib #compile
```

### PRIMME eigensolver

1. Download and extract sources:
```
wget https://github.com/primme/primme/archive/refs/tags/v3.2.tar.gz #download
tar -xf primme-3.2.tar.gz #decompress
```

2. Compile:
```
make lib
```

### Eigen linear algebra library

Eigen could be installed on many Linux system by their respective package manager.
For example, for Ubuntu and Debian machine:
```
sudo apt install libeigen3-dev
```

Or, on MacOS with:
```
brew install eigen
```

### Pyqcm

1. Clone PyQCM:

```
git clone https://bitbucket.org/dsenechQCM/qcm_wed.git
```

3. Configure and compile QCM library:

```
cd qcm_wed 
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUBA_DIR=../Cuba-4.2.2 [...] #see build options above
cmake --build .
cp ./qcm* ../pyqcm/.
```

4. Add the source directory (qcm_wed) to your PYTHONPATH:

```
export PYTHONPATH="$HOME/qcm_wed:$PYTHONPATH"
```

## Tips on MacOS

Compiling CUBA on MacOS must be done with the same compiler as the rest, and the ./configure
file of the CUBA distribution may not follow this. Therefore it is recommended to manually set
the following cmake options:

```
-DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=c++
```

## Tips for compilation on Compute Canada server (not up to date)

### Compilation on Intel based cluster with ICC compiler

CUBA configure step (2): 

```
./configure FC=`which ifort` CC=`which icc` CFLAGS="-O3 -fPIC"
```

QCM configure step (3) with CMake:

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUBA_DIR=../CUBA-4.2.2 -DCMAKE_CXX_COMPILER=`which icc` -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INCLUDE_PATH=${MKLLROOT}/lib/intel64_lin/
```

### Compilation with FlexiBLAS

QCM configure step (3) with CMake 3.21.4: 

```
module load cmake/3.21.4 flexiblas/3.0.4 scipy-stack
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUBA_DIR=../CUBA-4.2.2 -DBLA_VENDOR=FlexiBLAS
```


