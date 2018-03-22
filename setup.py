from setuptools import setup

setup(name='eve',
      version='0.1',
      description='extended verification and validation for earth system models',
      url='https://code.ornl.gov/LIVVkit/eve',
      author='Joseph H. Kennedy',
      author_email='kennedyjh@ornl.gov',
      license='BSD',
      install_requires=[
          'LIVVkit',
          'six',
          'numpy',
          'scipy',
          'pandas',
          'netCDF4',
          'matplotlib'
          ],
      packages=[
          'eve'
          ],
      include_package_data=True,
      zip_safe=False)
