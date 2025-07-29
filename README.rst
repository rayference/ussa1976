USSA1976
========

**WARNING**

This project is unmaintained starting from July 29th 2025. Its features are
however entirely transferred to the `Joseki <https://github.com/rayference/joseki>`_
library. See `this pull request <https://github.com/rayference/joseki/pull/372>`_
for more information.

*The U.S. Standard Atmosphere 1976 model.*

|PyPI| |Python Version| |License|

|Read the Docs| |Tests| |Codecov|

|pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/ussa1976.svg
   :target: https://pypi.org/project/ussa1976/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/ussa1976
   :target: https://pypi.org/project/ussa1976
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/ussa1976
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/ussa1976/latest.svg?label=Read%20the%20Docs
   :target: https://ussa1976.readthedocs.io/
   :alt: Read the documentation at https://ussa1976.readthedocs.io/
.. |Tests| image:: https://github.com/nollety/ussa1976/workflows/Tests/badge.svg
   :target: https://github.com/nollety/ussa1976/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/nollety/ussa1976/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/nollety/ussa1976
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

This package implements the atmosphere thermophysical model provided by the
National Aeronautics and Space Administration technical report NASA-TM-X-74335
published in 1976 and entitled *U.S. Standard Atmosphere, 1976*.

Features
--------

* Run the U.S. Standard Atmosphere 1976 model on your custom altitude grid
* Compute all 14 atmospheric variables of the model as a function of altitude:
   * air temperature
   * air pressure
   * number density (of individual species)
   * air number density
   * air density
   * air molar volume
   * air pressure scale height
   * air particles mean speed
   * air particles mean free path
   * air particles mean collision frequency
   * speed of sound in air
   * air dynamic viscosity
   * air kinematic viscosity
   * air thermal conductivity coefficient
* Results stored in `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_
  format
* Command-line interface
* Python interface


Requirements
------------

* Python 3.8+


Installation
------------

You can install *USSA1976* via pip_ from PyPI_:

.. code:: console

   $ pip install ussa1976


Usage
-----

* For the Command-line interface, please see the
  `Command-line Reference <Usage_>`_ for details.
* For the Python interface, refer to the `User Guide <_user_guide>`_.

Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `MIT license`_,
*USSA1976* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT license: https://opensource.org/licenses/MIT
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/nollety/ussa1976/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://ussa1976.readthedocs.io/en/latest/usage.html
