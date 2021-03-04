pyGEEMs
=======

|PyPi Cheese Shop| |Documentation| |Build Status| |Code Quality| |Test Coverage| |License| |DOI|

A Python library for Geotechnical Earthquake Engineering Models.

Information on the installation and usage can be found in the documentation_.

.. _documentation: https://pygeems.readthedocs.io/

Features
--------

Dynamic properties:
    - Wang andd Stokoe (2018)
    - Wangit, Dejong, and Shantz (2012)
Ground motion:
    - Rezaeian et al. (2014) damping scaling factor
    - Rathje et al. (2005) mean, average, and predominant periods
    - Abrahamson et al. (2016) Arias intensity
Slope displacement methods:
    - Rathje & Saygili (2008)
    - Rathje & Antonakos (2011)
    - Watson-Lamprey & Abrahamson (2006)
    - Bray & Travasarou (2007) model for crustal earthquakes
    - Bray, Travasarou, and Macedo (2018) model for subduction earthquakes

When authors provide test cases, those test cases are implemented or test cases are found.

Contributing
------------

If you want to see a model added, add it and supporting test cases. The code is
formatted using black_ and the docstrings are formatted to the `numpy
standard`_.

.. _black: https://black.readthedocs.io/en/stable/
.. _`numpy standard`: https://numpydoc.readthedocs.io/en/latest/format.html

Citation
--------

Please cite this software using the DOI_.

.. _DOI: https://zenodo.org/badge/latestdoi/5086299

.. |PyPi Cheese Shop| image:: https://img.shields.io/pypi/v/pygeems.svg
   :target: https://img.shields.io/pypi/v/pygeems.svg
.. |Documentation| image:: https://readthedocs.org/projects/pygeems/badge/?version=latest
    :target: https://pygeems.readthedocs.io/?badge=latest
.. |Build Status| image:: https://travis-ci.org/arkottke/pygeems.svg?branch=master
   :target: https://travis-ci.org/arkottke/pygeems
.. |Code Quality| image:: https://api.codacy.com/project/badge/Grade/ca4491ec1be44c239be7730c2b4021a6
   :target: https://www.codacy.com/manual/arkottke/pygeems
.. |Test Coverage| image:: https://api.codacy.com/project/badge/Coverage/ca4491ec1be44c239be7730c2b4021a6
   :target: https://www.codacy.com/manual/arkottke/pygeems
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. |DOI| image:: https://zenodo.org/badge/154161889.svg
   :target: https://zenodo.org/badge/latestdoi/154161889
