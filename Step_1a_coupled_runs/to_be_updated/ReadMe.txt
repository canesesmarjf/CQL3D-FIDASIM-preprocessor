2024-12-19:
############
These scripts need to be updated and tested. for example, the script to run in perlmutter needs to be made into something like "Launch_CQL3D.sh" or have a flag which selects between the different ways of running:
- local computer debug
- local computer normal mpi run
- perlmutter debug run in interactive
- perlmutter normal run in interactive
- perlmutter normal run in batch mode (srun)

Thus far we only have a local computer run mode
