#!/bin/bash

# ================================================
# ============ IMPORTANT !!! =====================
# ================================================

# This script terminates the execution of CQL3D
# To be used when executable is ran in the background.
# Has only been tested for single node operation in local machine.
# Has not been tested in perlmutter and may not be advisable to use this in NERSC

# ================================================
# ============ IMPORTANT !!! =====================
# ================================================


EXECUTABLE="xcql3dm_mpi.gfortran64"

# Terminate all instances of the executable:
pkill -f $EXECUTABLE