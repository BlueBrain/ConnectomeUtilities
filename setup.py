# SPDX-License-Identifier: Apache-2.0
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit("Sorry, Python < 3.8 is not supported.")

setup(name="Connectome utilities",
      version="0.3.1",
      author="Blue Brain Project, EPFL",
      packages=find_packages(),
      install_requires=["numpy>=1.20.0",
                        "h5py>=3.6.0",
                        "pandas>=2.0.0",
                        "tables>=3.6",
                        "scipy>=1.8.0",
                        "tqdm>=4.50.0",
                        "lazy>=1.4",
                        "scikit-learn>=1.0",
                        "libsonata>=0.1.10",
                        "bluepysnap>=1.0.0",
                        "voxcell"],
      description="Complex network representation and analysis layer",
      license="Apache 2.0",
      url="",
      download_url="",
      include_package_data=False,
      python_requires=">=3.8",
      extras_require={
          "Diffusion_mapping": [
              "mapalign @ git+https://github.com/satra/mapalign.git@master"
              ]
      },
      keywords=('computational neuroscience',
                'modeling',
                'analysis',
                'connectomics',
                'circuits',
                'morphology',
                'atlas'),
      classifiers=['Development Status :: Pre-Alpha',
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   'Environment :: Console',
                   'License :: Apache 2.0',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.8',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering :: Bio-Informatics :: Neuro-Informatics',
                   'Topic :: Utilities'],
      scripts=[])
