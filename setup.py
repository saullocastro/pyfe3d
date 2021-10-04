from __future__ import absolute_import

import os
import inspect
import subprocess
from setuptools import setup, find_packages
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize


is_released = True
version = '0.2.7'


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info(version, is_released):
    fullversion = version
    if not is_released:
        git_revision = git_version()
        fullversion += '.dev0+' + git_revision[:7]
    return fullversion


def write_version_py(version, is_released, filename='pyfe3d/version.py'):
    fullversion = get_version_info(version, is_released)
    with open("./pyfe3d/version.py", "wb") as f:
        f.write(b'__version__ = "%s"\n' % fullversion.encode())
    return fullversion


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()


#_____________________________________________________________________________

install_requires = [
        "numpy",
        "scipy",
        "alg3dpy",
        ]

#Trove classifiers
CLASSIFIERS = """\

Development Status :: 3 - Alpha
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
License :: OSI Approved :: BSD License
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Operating System :: Microsoft :: Windows
Operating System :: Unix

"""

fullversion = write_version_py(version, is_released)

if os.name == 'nt':
    compile_args = ['/openmp', '/O2']
    link_args = []
else:
    compile_args = ['-fopenmp', '-static', '-static-libgcc', '-static-libstdc++']
    link_args = ['-fopenmp', '-static-libgcc', '-static-libstdc++']

if 'CYTHON_TRACE_NOGIL' in os.environ.keys():
    compile_args = ['-O0']
    link_args = []

include_dirs = [
            np.get_include(),
            ]

extension_kwargs = dict(
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    language='c++',
    )

extensions = [
    Extension('pyfe3d.beamprop',
        sources=[
            './pyfe3d/beamprop.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.shellprop',
        sources=[
            './pyfe3d/shellprop.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.spring',
        sources=[
            './pyfe3d/spring.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.truss',
        sources=[
            './pyfe3d/truss.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.beamlr',
        sources=[
            './pyfe3d/beamlr.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.beamc',
        sources=[
            './pyfe3d/beamc.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.quad4r',
        sources=[
            './pyfe3d/quad4r.pyx',
            ],
        **extension_kwargs),

    ]

ext_modules = cythonize(extensions,
        compiler_directives={'linetrace': True},
        language_level = '3',
        )

data_files = [('', [
        'README.md',
        'AUTHORS',
        'LICENSE',
        ])]

s = setup(
    name = "pyfe3d",
    version = fullversion,
    author = "Saullo G. P. Castro",
    author_email = "S.G.P.Castro@tudelft.nl",
    description = ("Finite elements for 3D problems in Python/Cython"),
    license = "3-Clause BSD",
    keywords = "finite elements structural analysis static buckling vibration dynamics",
    url = "https://github.com/saullocastro/pyfe3d",
    data_files=data_files,
    long_description=read('README.md'),
    long_description_content_type = 'text/markdown',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=install_requires,
    ext_modules = ext_modules,
    include_package_data=True,
    packages=find_packages(),
)

