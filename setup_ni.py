import os
import re
from setuptools import setup
from setuptools import Command


def _get_version():
    """Returns the PROTEUS software version from the
    file `src/dswx_sar/version_ni.py`
       Returns
       -------
       version : str
            PROTEUS software version
    """

    version_file = os.path.join('src', 'dswx_sar', 'version_ni.py')

    with open(version_file, 'r') as f:
        text = f.read()

    # Get first match of the version number contained in the version file
    # This regex should match a pattern like: VERSION = '3.2.5', but it
    # allows for varying spaces, number of major/minor versions,
    # and quotation mark styles.
    p = re.search("VERSION[ ]*=[ ]*['\"]\d+([.]\d+)*['\"]", text)

    # Check that the version file contains properly formatted text string
    if p is None:
        raise ValueError(
            f'Version file {version_file} not properly formatted.'
            " It should contain text matching e.g. VERSION = '2.3.4'")

    # Extract just the numeric version number from the string
    p = re.search("\d+([.]\d+)*", p.group(0))

    return p.group(0)


__version__ = version = VERSION = _get_version()


print(f'DSWX-SAR version {version}')


class CleanCommand(Command):
    """Custom clean command to tidy up the project root
    after running `python setup.py install`."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Make sure to remove the .egg-info file
        os.system('rm -vrf .scratch_dir ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./src/*.egg-info')


long_description = open('README.md').read()

package_data_dict = {}

package_data_dict['dswx_sar'] = [
    os.path.join('defaults', 'algorithm_parameter_ni.yaml'),
    os.path.join('defaults', 'dswx_ni.yaml'),
    os.path.join('schemas', 'algorithm_parameter_ni.yaml'),
    os.path.join('schemas', 'dswx_ni.yaml')]

setup(
    name='dswx-ni',
    version=version,
    description='Compute Dynamic Surface Water Extent (DSWx)'
                ' from NISAR (NI) data',
    # Gather all packages located under `src`.
    # (A package is any directory containing an __init__.py file.)
    package_dir={'': 'src'},
    packages=['dswx_sar'],
    package_data=package_data_dict,
    classifiers=['Programming Language :: Python',],
    scripts=['src/dswx_sar/dswx_ni.py',
             'src/dswx_sar/dswx_comparison.py'],
    install_requires=['argparse', 'numpy', 'yamale',
                      'scipy', 'pytest', 'requests',
                      'h5py', 'scikit-image', 'mgrs', 'pyproj',
                      'opencv-python'],
    url='https://github.com/opera-adt/DSWX-SAR',
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
    cmdclass={
        'clean': CleanCommand,
        }
)
