import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")

setup(name="Connectome utilities",
      version="0.1.1",
      author="BBP",
      packages=find_packages(),
      install_requires=["numpy>=1.20.0",
                        "h5py>=2.10.0"
                        "pandas>=1.2.4",
                        "tables>=3.6",
                        "scipy>=1.5.0",
                        "tqdm>=4.50.0",
                        "lazy>=1.4",
                        "scikit-learn>=1.0",
                        "h5py>=3.0"],
      description="Utilities for topological characracterization of circuit connectivity.",
      license="GPL",
      url="",
      download_url="",
      include_package_data=False,
      python_requires=">=3.8",
      keywords=('computational neuroscience',
                'computational models',
                'analysis',
                'connectomics'
                'BlueBrainProject'),
      classifiers=['Development Status :: Pre-Alpha',
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   'Environment :: Console',
                   'License :: LGPL',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.8',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering :: Bio-Informatics :: Neuro-Informatics',
                   'Topic :: Utilities'],
      scripts=[])
