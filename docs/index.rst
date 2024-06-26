.. fusiondls documentation master file, created by
   sphinx-quickstart on Tue Jun 25 17:05:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fusiondls's documentation!
=====================================

This is a version of the DLS model (Lipschultz 2016) based on the work
of Cyd Cowley (Cowley 2022) and Ryoko Osawa, as used in the STEP
report "STEP: Impact of magnetic topology on detachment performance
and sensitivity".

There are example notebooks provided which go through some basic and
some more advanced use cases of the code. This version of the DLS
features:

- A thermal front of finite width
- An efficient bisection algorithm solver
- Cooling curves for neon, argon and nitrogen valid up to 300eV
- Ability to include radiation above the X-point
- Radial heat source above the X-point
- Example B field profiles for three STEP configurations

To get started, check out the notebook ``Example 1 - basics``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Examples

   examples/*

.. toctree::
   :maxdepth: 2
   :caption: Reference

   API Reference <api>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
