2025_03_26:
This run directory was created as part of the non-thermal beam stopping development for the following purpose:

Create f4d sloshing ion beam distribution using a 100 keV beam so that we can use such distribution
in our investigation of the nonthermal beam stopping power compared to the thermal case.

To create such sloshing on beam distribution requires that we run cql3dm in time-dependent mode to evolve the distribution function to a sloshing ion distribution driven by NBI at the desired energy, which in this case is 100 keV at 45 degrees. We also want to explore the result at other beam angles.

We have developed a MATLAB script which takes the FIDASIM HDF5 distribution function file and computes the equivalent maxwellian distribution and computes the rate coefficient integral and associated beam stopping power, mean free path and absorbed power for both the non-thermal (sloshing ion) and thermal distribution functions.

We are interested in looking at this calculation with a 100 keV sloshing ion distribution.
