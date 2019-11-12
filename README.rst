=========
repurpose
=========

.. image:: https://travis-ci.org/TUW-GEO/repurpose.svg?branch=master
    :target: https://travis-ci.org/TUW-GEO/repurpose

.. image:: https://coveralls.io/repos/github/TUW-GEO/repurpose/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/repurpose?branch=master

.. image:: https://badge.fury.io/py/repurpose.svg
    :target: http://badge.fury.io/py/repurpose

.. image:: https://readthedocs.org/projects/repurpose/badge/?version=latest
   :target: http://repurpose.readthedocs.org/


This package provides routines for the conversion of image formats to time
series and vice versa. It is part of the `poets project
<http://tuw-geo.github.io/poets/>`_ and works best with the readers and writers
supported there. The main use case is for data that is sampled irregularly in
space or time. If you have data that is sampled in regular intervals then there
are alternatives to this package which might be better for your use case. See
`Alternatives`_ for more detail.

The readers and writers have to conform to the API specifications of the base
classes defined in `pygeobase <https://github.com/TUW-GEO/pygeobase>`_ to work
without adpation.

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.593577.svg
   :target: https://doi.org/10.5281/zenodo.593577

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.593577 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
============

This package should be installable through pip:

.. code::

    pip install repurpose

Modules
=======

It includes two main modules:

- ``img2ts`` for image/swath to time series conversion, including support for
  spatial resampling.
- ``ts2img`` for time series to image conversion, including support for temporal
  resampling. This module is very experimental at the moment.
- ``resample`` for spatial resampling of (regular or irregular) gridded data to different resolutions.

Alternatives
============

If you have data that can be represented as a 3D datacube then these projects
might be better suited to your needs.

- `PyReshaper <https://github.com/NCAR/PyReshaper>`_ is a package that works
  with NetCDF input and output and converts time slices into a time series
  representation.
- `Climate Data Operators (CDO)
  <https://code.zmaw.de/projects/cdo/embedded/index.html>`_ can work with
  several input formats, stack them and change the chunking to allow time series
  optimized access. It assumes regular sampling in space and time as far as we
  know.
- `netCDF Operators (NCO) <http://nco.sourceforge.net/#Definition>`_ are similar
  to CDO with a stronger focus on netCDF.

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

Development setup
-----------------

For Development we recommend a ``conda`` environment

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the repurpose repository to your account
- make a new feature branch from the repurpose master branch
- Add your feature
- Please include tests for your contributions in one of the test directories.
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch

Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
