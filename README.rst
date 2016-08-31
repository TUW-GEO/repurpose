=========
repurpose
=========

This package provides routines for the conversion of image formats to time
series and vice versa. It is part of the `poets° project
<http://tuw-geo.github.io/poets/>`_ and works best with the readers and writers
supported there. The main use case is for data that is sampled irregularly in
space or time. If you have data that is sampled in regular intervals then there
are alternatives to this package which might be better for your use case. See
`Alternatives`_ for more detail.

The readers and writers have to conform to the API specifications of the base
classes defined in `pygeobase <https://github.com/TUW-GEO/pygeobase>`_ to work
without adpation.

Contents
========

It includes two main modules:

- ``img2ts`` for image/swath to time series conversion, including support for
  spatial resampling. The functionality is already used 
- ``ts2img`` for time series to image conversion, including support for temporal
  resampling. This module is very experimental at the moment.

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

Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
