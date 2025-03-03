from setuptools import setup, find_packages

# install_requires = ["botorch", "dill", "click", "scikit-learn", "scipy", "matplotlib", "numpy"]
install_requires = []
extras_require = {
    "gui": ["PySide6"],
    "multi_processing_smp": ["pathos"],
    "mpi": ["mpi4py"]
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='veropt',
    version='0.6.0',
    packages=find_packages(),
    url='https://github.com/aster-stoustrup/veropt',
    license='OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    author='Aster Stoustrup',
    author_email='aster.stoustrup@gmail.com',
    description='The Versatile Optimiser (VerOpt)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require
)
