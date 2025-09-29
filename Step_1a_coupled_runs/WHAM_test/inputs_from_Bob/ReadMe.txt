2025-09-11
cqlinput_iaea2.2_f4d_fidasim
This was created from the base input cqlinoput but we added the namelist variables
needed to run coupled with FIDASIM:
- f4d output
- in frsetup, to read FIDASIM birth point files and to compute sink from FIDASIM data

We then copy this file as "cqlinput" into the run directory so that CQL3DM can find it
