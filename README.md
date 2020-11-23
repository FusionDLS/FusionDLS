# DLS-model
A collection of models for the location sensitivity of a plasma detachment front.

The best way to run the current code is to run the Lengyel-Reinke-Formulation.py file. 
This file is set to compare the results from the simple thermal front model to the Lengyel-Reinke version of the model.
To apply the model to a specific grid, simply change the gridFile variable to the location of the balance.nc file that you want to use.
If analysis on the outer target is desired, ensure that the Type variable in unpackConfiguration is set to 'outer'. (or 'inner' for inner)

Required packages for this model are:

Scipy
Numpy
Matplotlib


