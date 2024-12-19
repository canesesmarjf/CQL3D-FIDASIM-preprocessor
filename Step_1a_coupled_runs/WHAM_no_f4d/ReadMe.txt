In this directory, we hold the input data required for to preprocess FIDASIM input files.

The FIDASIM input files are in HDF5 format.

The files in this directory form a consistent dataset from WHAM simulations performed by Yuri Petrov using cql3dm in perlmutter.

Yuri has provided the directory in perlmutter where this data is located
- /pscratch/sd/y/ypetrov/cql3dm/wk1

The self consistent data consists of"
1- cqlinput file: contains NBI geometry data, potential profiles, number of species
2- EQDSK file: contains the magnetic field data and flux surface information
3- standard .nc output file: Contains plasma profiles, potential, etc
4- f4d .nc outputfile: Contains the 4D resolved full ditribution function produced by CQL3D runs

There are two f4d files in this dataset:

WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120_f4d_001.nc
and
WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120_f4d_002.nc

"001" means species #1 (D)
"002" means species #2 (e)
