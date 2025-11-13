=========
Changelog
=========

Unreleased changes in master branch
===================================
- Changed a test where the dim order was not time, lat, lon (`PR #44 <https://github.com/TUW-GEO/repurpose/pull/44>`_)

Version 0.13.2
==============
- Change the write non orthogonal function in the img2ts.py file so it discards
  non existent time stamps from the timeseries (`PR #42 <https://github.com/TUW-GEO/repurpose/pull/42>`_)

Version 0.13.1
==============
- Img2Ts now logs only errors and warnings
- Handled an edge case where Img2Ts failed when a whole set of images was missing

Version 0.13
============
- PynetCF time series type (OrthoMultiTs or IndexedRaggedTs) are now stored in
  the global `timeSeries_format` attribute.
- `time_coverage_end` global attribute added to each time series file, this is
  updated when appending data to an existing file.
- Some utility function were added that are often required in our conversion
  packages

Version 0.12
============
- Updates to repurpose (img2ts) for performant conversion (for non-orthogonal data)
- img2ts logging was improved
- The method for parallelization was updated to allow different backends
- A check was implemented to repeatedly try to append to a (temporarily) unavailable file
- Input grid for img2ts is now used from the input dataset if not specified by the user. This allows e.g. conversion of swath data.

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
