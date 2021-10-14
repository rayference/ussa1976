.. _user_guide:

User Guide
==========

This page presents the Python interface of ``ussa1976``.

For details on how to use the command-line interface to ``ussa1976``, refer
to :ref:`usage page <usage>`.

Getting started
---------------

Make a U.S. Standard Atmosphere data set with the default altitude meshes:

.. code-block:: python

   import ussa1976

   ds = ussa1976.make()
