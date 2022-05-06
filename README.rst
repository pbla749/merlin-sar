Merlin SAR for despeckling Synthetic Aperture Radar (SAR) images
------------------------------

Based on the work of Emanuele Dalsasso, post-doctoral researcher at CNAM and Telecom Paris.

Speckle fluctuations seriously limit the interpretability of synthetic aperture radar (SAR) images. This package provides a despeckling function that can highly improve the quality and interpretability of SAR images.

The package contains both test and train parts, wether you wish to despeckle a single pic (test) or urse our model to build ou improve your own.

To know more about the researcher's work : https://arxiv.org/abs/2110.13148

To get a test function using Tensorflow's framework : https://gitlab.telecom-paris.fr/ring/MERLIN/-/blob/master/README.md


.. image:: https://img.shields.io/pypi/v/merlin.svg
        :target: https://pypi.python.org/pypi/merlin

.. image:: https://img.shields.io/travis/audreyr/merlin.svg
        :target: https://travis-ci.com/audreyr/merlin

.. image:: https://readthedocs.org/projects/merlin/badge/?version=latest
        :target: https://merlin.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Installation
------------

Install merlin by running::

    pip install -i https://test.pypi.org/simple/ merlinsar==0.2.9


Initialization for merlin package.
----------------------------------
::

    import merlinsar


* Free software: MIT license

Authors
-------

* Emanuele Dalsasso : Post-doctoral researcher at CNAM and Telecom Paris

* Youcef Kemiche : Hi! PARIS machine learning engineer

* Pierre Blanchard : Hi! PARIS machine learning engineer


Use cases
------------
The package offers you 3 different methods for despeckling your SAR images: the fullsize method, the coordinates based method and the crop method.

* I have a high-resolution SAR image and I want to apply the despeckling function to the whole of it::

    from merlinsar.test.spotlight import despeckle
    despeckle(image_path,destination_directory,model_weights_path=model_weights_path)

* I have a high-resolution SAR image but I only want to apply the despeckling function to a specific area for which I know the coordinates::

    from merlinsar.test.spotlight import despeckle_from_coordinates
    despeckle_from_coordinates(image_path,coordinates_dict,destination_directory,model_weights_path=model_weights_path)

* I have a high-resolution SAR image but I want to apply the despeckling function to an area I want to select with a crop::

    from merlinsar.test.spotlight import despeckle_from_crop
    despeckle_from_crop(image_path,destination_directory,model_weights_path=model_weights_path)





Dependencies
------------



Contribute
----------

- Source Code: https://github.com/hi-paris/merlin

FAQ
---

Please contact us at engineer@hi-paris.fr
