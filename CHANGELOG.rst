=========
Changelog
=========

Unreleased changes in master branch
===================================
-

Version 0.11
============

- Use joblib for parallel processing framework, improved logging
- Added option to parallelize Img2Ts process
- Fix bug where a wrong grid attribute was used.

Version 0.10
============

- Ts2Img module was rebuilt. Allows conversion of time series with NN lookup.
- Added example notebook for converting ASCAT time series into regularly gridded images.
- Added a simple parallelization framework, with logging and error handling.
- Added the option to pass custom pre- and post-processing functions to ts2img.

Version 0.9
===========

- Update for new pyscaffold
- Fixed bug where resampling failed when a BasicGrid was passed instead of a CellGrid

Version 0.8
===========

- Update pyscaffold package structure (pyscaffold 3)
- Drop py2 support
- Add pypi deployment to travis.

Version 0.7
===========

- Add resample functions (from pytesmo)

Version 0.6
===========

- Update setup.cfg

Version 0.5
===========

- Update readme
- Update pyscaffold version in setup.py because of compatibility issues with setuptools 39

Version 0.4
===========

- Enable compression by default.

Version 0.3
===========

- Enable image to timeseries conversion if missing images are encountered.

Version 0.2
===========

- First public version
- Rename to repurpose
- Improve test coverage

Version 0.1
===========

- initial version supporting image to time series conversion
- draft for time series to image conversion
