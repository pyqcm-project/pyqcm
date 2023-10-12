import sys
import os
from setuptools import find_packages
import skbuild
import pathlib

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
 
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

skbuild.setup(
    name="pyqcm",
    version=version[1:],
    description="Quantum cluster methods for the physics of strongly correlated systems",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/dsenechQCM/qcm_wed/',
    author="David Sénéchal",
    license="GPL",
    packages=find_packages(),
    # package_dir={"": "."},
    # cmake_install_dir=".",
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "scipy"],
    python_requires=">=3.7",
)

