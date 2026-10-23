r"""Build of the Cython extensions of pyfe3d

The package metadata, including the version, is in ``pyproject.toml``.
This file keeps only what ``pyproject.toml`` cannot express: the Cython
extensions with their compiler flags per platform and the file
``pyfe3d/version.py``, written from the version of ``pyproject.toml``
and read by ``pyfe3d.version`` and ``doc/source/conf.py``.
"""
import platform
import os
import sys

from setuptools import setup, Extension
from Cython.Build import cythonize

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


setupdir = os.path.dirname(os.path.abspath(__file__))


def read_version():
    with open(os.path.join(setupdir, 'pyproject.toml'), 'rb') as f:
        return tomllib.load(f)['project']['version']


# NOTE the year of the copyright of the documentation, see doc/source/conf.py
year = '2026'


def write_version_py(version):
    version_file = os.path.join(setupdir, 'pyfe3d', 'version.py')
    with open(version_file, 'wb') as f:
        f.write(b'__version__ = "%s"\n' % version.encode())
        f.write(b'__year__ = "%s"\n' % year.encode())


write_version_py(read_version())

# NOTE a coverage build, see .github/workflows/coverage.yml, requested with
#      CYTHON_TRACE_NOGIL in the environment or with --define CYTHON_TRACE...
trace = ('CYTHON_TRACE_NOGIL' in os.environ.keys()
         or any('CYTHON_TRACE' in arg for arg in sys.argv))

# NOTE flags for speed. No module uses prange, so OpenMP is not needed. GCC
#      and Clang get -O3 explicitly, because the level inherited from the
#      Python build is not guaranteed, and -fno-math-errno, which lets sqrt()
#      compile to a single instruction and changes no result. MSVC is
#      already at its fastest standard-conforming setting with the /O2 and
#      /GL that setuptools passes. Flags that change floating-point results,
#      such as /fp:fast or -ffast-math, and flags that tie a wheel to the CPU
#      that built it, such as -march=native, are deliberately left out
define_macros = []
if platform.system() == 'Windows':
    compile_args = ['/O2']
    link_args = []
elif platform.system() == 'Linux':
    compile_args = ['-O3', '-fno-math-errno']
    link_args = ['-static-libgcc', '-static-libstdc++']
else: # MAC-OS
    compile_args = ['-O3', '-fno-math-errno']
    link_args = []

if trace:
    # NOTE unoptimized, so that every traced line maps to code. Since Python
    #      3.12 Cython traces through sys.monitoring by default, which the
    #      Cython.Coverage plugin cannot follow, hence the legacy tracing
    if os.name == 'nt': # Windows
        compile_args = ['/Od']
    else: # MAC-OS or Linux
        compile_args = ['-O0']
    link_args = []
    define_macros = [('CYTHON_TRACE_NOGIL', '1'),
                     ('CYTHON_USE_SYS_MONITORING', '0')]

include_dirs = [
            ]

extension_kwargs = dict(
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=define_macros,
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
    Extension('pyfe3d.tria3r',
        sources=[
            './pyfe3d/tria3r.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.tria3dsg',
        sources=[
            './pyfe3d/tria3dsg.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.quad4',
        sources=[
            './pyfe3d/quad4.pyx',
            ],
        **extension_kwargs),
    Extension('pyfe3d.quad4r',
        sources=[
            './pyfe3d/quad4r.pyx',
            ],
        **extension_kwargs),

    ]

def generated_with_other_trace_mode(ext):
    r"""Whether the C++ file generated from a Cython source of ``ext`` was
    generated with, or without, line tracing, the opposite of this build

    cythonize regenerates a C++ file only when its source is newer, so
    switching between a coverage build and a normal one would otherwise
    reuse the C++ file of the other mode without notice.
    """
    for source in ext.sources:
        if not source.endswith('.pyx'):
            continue
        cpp = os.path.splitext(source)[0] + '.cpp'
        if os.path.isfile(cpp):
            with open(cpp, encoding='utf-8', errors='ignore') as f:
                if ('__Pyx_TraceLine(' in f.read()) != trace:
                    return True
    return False

# NOTE line tracing only for a coverage build, since the profiling hooks it
#      generates otherwise stay active in every function call
ext_modules = cythonize(extensions,
        compiler_directives={'linetrace': trace},
        language_level = '3',
        force=any(generated_with_other_trace_mode(ext) for ext in extensions),
        )

setup(
    ext_modules=ext_modules,
)
