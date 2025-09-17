2025-09-11
To run a coupled  case with a vacuum vessel described with analytic surfaces, we only need the following files:
- vacuum_vessel.nml
- runid_config.nml
- runid_cql_config.nml
- EQDSK file
- cqlinput file


test_1
2025-09-11:
This run failed at some point. We were able to get decent results in the 12 ms range. but later the electron temperature starts to peak at some rho = 0.5 which doesnt make sense and reaches values in excess of 24 keV.
This leads to issues with rate coefficients for H_e reactions.
We are keeping these files for reference only and to compare with other tests
Here we are exchanging f4d files between codes. We are sampling the nonthermal distribution for CX sampling

test_2:
Repeat but up to 50 ms and recording outputs to .ps file more often

test_3:
2025-09-11:
Same as above but we are using only NDMC (Seed neutrals). disabling DCX and HALO by setting:
- FIDASIM: calc_sink = 0
- CQL3D: frsink_cx = 'enabled' (CQL3Ds own internal sink operator)
