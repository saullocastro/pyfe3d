General-purpose finite element solver - pyfe3d
==============================================

The ``pyfe3d`` module is a general-purpose finite element solver for structural
analysis and optimization based on Python and Cython. The main principles
guiding the development of ``pyfe3d`` are: simplicity, efficiency and
compatibility. The aimed level of compatibility allows one to run this solver
in any platform, including the Google Colab environment.


Code repository
---------------

https://github.com/saullocastro/pyfe3d


Citing this library
-------------------


Saullo G. P. Castro. (2025). General-purpose finite element solver based on Python and Cython (Version 0.6.2). Zenodo. DOI: https://doi.org/10.5281/zenodo.6573489.


Tutorials
---------

.. toctree::
    :maxdepth: 2

    tutorials.rst


Usage Examples
--------------
.. toctree::
    :maxdepth: 2

    ex_linear_static.rst
    ex_linear_buckling_plate.rst
    ex_linear_buckling_cylinder_torsion.rst
    ex_natural_frequency_cylinder.rst


Topics
-------

.. toctree::
    :maxdepth: 1

    fe_overview.rst
    properties.rst
    repository_structure.rst


Installing pyfe3d
-----------------

First, you should try to install from the distributed binaries by simply
doing::

    python -m pip install pyfe3d

If a distribution could not be found, you can try to install from the source
code using::

    python -m pip install .

Another alternative is the following::

    python -m pip install -r requirements.txt
    python setup.py install

If none of the above alternatives worked for you, this link shares some
information on how to set up a C compiler on different operating systems: 

https://cython2.readthedocs.io/en/latest/src/quickstart/install.html



License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1

AUTHORS
-------

.. literalinclude:: ../../AUTHORS
    :encoding: latin-1


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

