import platform
import os
import sys
import inspect
import subprocess
from setuptools import setup, find_packages
from setuptools.extension import Extension

from Cython.Build import cythonize


is_released = True
version = '0.10.0'
year = '2026'


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
    version_file = "./pyfe3d/version.py"
    if os.path.isfile(version_file):
        os.remove(version_file)
    with open(version_file, "wb") as f:
        f.write(b'__version__ = "%s"\n' % fullversion.encode())
        f.write(b'__year__ = "%s"\n' % year.encode())
    return fullversion


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()


#_____________________________________________________________________________

#Trove classifiers
CLASSIFIERS = """\

Development Status :: 4 - Beta
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
Topic :: Software Development
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: POSIX :: BSD
Operating System :: Microsoft :: Windows
Operating System :: Unix
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
Programming Language :: Python :: 3.13
Programming Language :: Python :: 3.14
License :: OSI Approved :: BSD License

"""

fullversion = write_version_py(version, is_released)

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

data_files = [('', [
        'README.md',
        'AUTHORS',
        'LICENSE',
        ])]

package_data = {
        'pyfe3d': ['*.pxd', '*.pyx'],
        '': ['tests/*.*'],
        }

keywords = [
            'finite elements',
            'structural analysis',
            'structural optimization',
            'static analysis',
            'buckling',
            'vibration',
            'panel flutter',
            'structural dynamics',
            'implicit time integration',
            'explicit time integration',
            ]

s = setup(
    name = "pyfe3d",
    version = fullversion,
    author = "Saullo G. P. Castro",
    author_email = "S.G.P.Castro@tudelft.nl",
    description = ("General-purpose finite element solver for structural analysis and optimization based on Python and Cython"),
    long_description = read('README.md'),
    long_description_content_type = 'text/markdown',
    license = "3-Clause BSD",
    keywords = keywords,
    url = "https://github.com/saullocastro/pyfe3d",
    package_data = package_data,
    data_files = data_files,
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    ext_modules = ext_modules,
    include_package_data = True,
    packages = find_packages(),
)

