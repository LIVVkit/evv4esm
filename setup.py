from __future__ import absolute_import, print_function, unicode_literals

import os
import re

from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), 'r') as f:
    long_desc = f.read()
with open(os.path.join(here, 'eve', '__init__.py')) as f:
    init_file = f.read()

setup(
      name='eve',
      version=re.search(r'{}\s*=\s*[(]([^)]*)[)]'.format('__version_info__'),
                        init_file
                        ).group(1).replace(', ', '.'),

      description='Extended verification and validation for earth system models',
      long_description=long_desc,
      long_description_content_type='text/markdown',

      url='https://code.ornl.gov/LIVVkit/eve',

      author='Joseph H. Kennedy',
      author_email='kennedyjh@ornl.gov',

      license='BSD',
      include_package_data=True,

      classifiers=['Development Status :: 5 - Production/Stable',

                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Software Development :: Testing',

                   'License :: OSI Approved :: BSD License',

                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   ],

      install_requires=[
          'six',
          'rpy2',
          'numpy',
          'scipy',
          'pandas',
          'livvkit',
          'netCDF4',
          'matplotlib',
          ],

      packages=[
          'eve'
          ],

      entry_points={'console_scripts': ['livv = livvkit.__main__:main']},

      zip_safe=False,
      )
