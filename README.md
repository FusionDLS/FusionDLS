# FusionDLS - The Detachment Location Sensitivity Model

This is a version of the DLS model (Lipschultz 2016) based on the work of Cyd Cowley (Cowley 2022) and Ryoko Osawa, as used in the STEP report "STEP: Impact of magnetic topology on detachment performance and sensitivity".

There are example notebooks provided which go through some basic and some more advanced use cases of the code. This version of the DLS features:
- A thermal front of finite width
- An efficient bisection algorithm solver
- Cooling curves for neon, argon and nitrogen valid up to 300eV
- Ability to include radiation above the X-point
- Radial heat source above the X-point
- Example B field profiles for three STEP configurations

You can find the previous version of the DLS model along with an early MATLAB build in the branch "old".

To get started, check out the notebook Example 1 - basics.

Before finalising changes to the model, please ensure the physics is working correctly by running the Analytic_Benchmark.ipynb notebook!