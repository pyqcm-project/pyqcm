import sys
import os
from setuptools import find_packages
import skbuild

#ne pas importer numpy, et pas de try/except sur skbuild, mais plutot mentionner son installation dans le readme

import os
import subprocess
git_hash = str(subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']))
version = str(subprocess.check_output(['git', 'describe', '--tags']))
git_hash = git_hash[2:-3]
version = version[2:-3]
version = version.rsplit("-")[0]

fout = open("pyqcm/qcm_git_hash.py", "w")
fout.write("git_hash = '{:s}'\n".format(git_hash))
fout.write("version = '{:s}'\n".format(version))
fout.close() 
 


skbuild.setup(
    name="pyqcm",
    version=version[1:],
    description="Quantum cluster methods for the physics of strongly correlated systems",
    author="David Sénéchal",
    license="GPL",
    packages=find_packages(),
    cmake_args=["-DDOWNLOAD_CUBA=1", "-DEIGEN_HAMILTONIAN=0"],
    # package_dir={"": "."},
    # cmake_install_dir=".",
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "scipy"],
    python_requires=">=3.7",
)

