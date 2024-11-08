=========
repurpose
=========

|ci| |cov| |pip| |doc|

.. |ci| image:: https://github.com/TUW-GEO/repurpose/actions/workflows/build.yml/badge.svg?branch=master
   :target: https://github.com/TUW-GEO/repurpose/actions

.. |cov| image:: https://coveralls.io/repos/github/TUW-GEO/repurpose/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/repurpose?branch=master

.. |pip| image:: https://badge.fury.io/py/repurpose.svg
    :target: http://badge.fury.io/py/repurpose

.. |doc| image:: https://readthedocs.org/projects/repurpose/badge/?version=latest
   :target: http://repurpose.readthedocs.org/


This package provides routines for the conversion of image formats to time
series and vice versa. It works best with the readers and writers
supported by `pynetcf <https://github.com/TUW-GEO/pynetcf>`_.
The main use case is for data that is sampled irregularly in
space or time. If you have data that is sampled in regular intervals then there
are alternatives to this package which might be better for your use case. See
`Alternatives`_ for more detail.

The readers and writers have to conform to the API specifications of the base
classes defined in `pygeobase <https://github.com/TUW-GEO/pygeobase>`_ to work
without adpation.

Installation
============

This package requires `python>=3.9` and depends on the following libraries that
should be installed with `conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_
or `mamba <https://github.com/conda-forge/miniforge>`_

.. code::

    conda install -c conda-forge numpy netCDF4 pyresample

Afterwards you can install this package and all remaining dependencies via:

.. code::

    pip install repurpose


On macOS if you get `ImportError: Pykdtree failed to import its C extension`,
then it might be necessary to install the pykdtree package from conda-forge

.. code::

    conda install -c conda-forge pykdtree

Optional Dependencies
---------------------
Some packages are only needed to run unit tests and build docs of this package.
They can be installed via ``pip install repurpose[testing]`` and/or
``pip install repurpose[docs]``.

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.593577.svg
   :target: https://doi.org/10.5281/zenodo.593577

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Modules
=======

It includes the main modules:

- ``img2ts`` for image/swath to time series conversion, including support for
  spatial resampling.
- ``ts2img`` for time series to image conversion, including support for temporal
  resampling.
- ``resample`` for spatial resampling of (regular or irregular) gridded data to different resolutions.
- ``process`` contains a framework for parallel processing, error handling and logging based on `joblib <https://github.com/joblib/joblib>`_

Alternatives
============

If you have data that can be represented as a 3D datacube then these projects
might be better suited to your needs.

- `Climate Data Operators (CDO)
  <https://code.zmaw.de/projects/cdo/embedded/index.html>`_ can work with
  several input formats, stack them and change the chunking to allow time series
  optimized access. It assumes regular sampling in space and time as far as we
  know.
- `netCDF Operators (NCO) <http://nco.sourceforge.net/#Definition>`_ are similar
  to CDO with a stronger focus on netCDF.
- `xarray <https://docs.xarray.dev/en/stable/>`_ can read, restructure, write
  netcdf data as datacubes and apply functions across dimensions.

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the repurpose repository to your account
- make a new feature branch from the repurpose master branch
- Add your feature
- Please include tests for your contributions in one of the test directories.
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch
