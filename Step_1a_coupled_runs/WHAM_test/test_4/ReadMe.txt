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

test_4: same as above but with DCX and HALOSx4, we are using FIDASIM calc_sink = 1

Entering subroutine f4dwrite
Output f4d dist function for general species k=           1

 f4dwrite: Species k=           1   umx/vnorm=   2.6146281379312870E-002
MIN/MAX of f4d   0.00000000       2.73414547E+21

Entering subroutine f4dwrite
Output f4d dist function for general species k=           2

 f4dwrite: Species k=           2   umx/vnorm=   1.0000000000000000
MIN/MAX of f4d   0.00000000       1.48011617E+16
inside netcdfrw2.f...
netcdfrw2 initialize, kopt=           1 n=          50
9
10
TDCHIEF: tcpu_urfchief=     0.000    tcpu_netcdf=     0.000    tcpu_impavnc=     5.854
tcpu_diagnostics 1 and 2:     0.000     2.667
Finished Time Step ======>   50     tcpu(sec.)==    47.297


PGPLOT CLOSED at time step n=          50
tdwritef_43: For checkup SUM(reden),SUM(energy)=   402778138052095.19        554.11872847143025
tdwritef_44: For checkup SUM(totcurz),SUM(rovs)=   2.3292200537619509E-007   0.0000000000000000
tdtloop: >>> END OF CQL3D PROGRAM <<<
tdtloop: Execution time (seconds)   2466.37842
a_cqlp: rank, Exec.time tarray(2)-tarray(1)    0  2466.359
MPI Full time =   4832.8182629999992

real    80m32.942s
user    871m32.959s
sys     5m6.813s
(FIDASIM_env) jfcm@Tanooki:~/Repos/CQL3D-FIDASIM-preprocessor/Step_1a_coupled_runs$ 
