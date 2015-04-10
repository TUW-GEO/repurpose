# imgtsimg

This package provides routines for the conversion of image formats to time series and vice versa.

The readers and writers have to conform to the API specifications of the base
classes that are currently in the pytesmo.io.dataset_base module. Those will be
moved into a separate package with improved documentation ASAP.

# About literal programming.

The package has two sets of source files. The folders `org-files` and `imgtsimg`
have the same subfolder structure. The org files contain all the code including
documentation and the thinking behind the program structure. All the editing is
done in the .org files. The relevant python code is then tangled to the .py
files to be usable as a standard python package.

If this does not work we can always revert back to regular Python files.


