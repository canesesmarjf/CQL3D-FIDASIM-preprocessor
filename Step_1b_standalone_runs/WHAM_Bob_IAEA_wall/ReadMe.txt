scenario_1:
==============
T_wall = 3e-3
p_specular = 0.3
p_absorb = 0.3
Ta_surf
throat1
throat2
pump
standard beam grid size

scenario_2:
==============
Same as above but increased beam grid size
from zmax of 21 cm to 40 cm and ymax of 21 cm to 40 cm
and zmin = -zmax, ymin = -ymax
x dimensions remained the same

scenario_3:
=============
In all previous runs, we had made mistakes in how we compute the halo neutrals.
The error was in the primate variables in the OMP process and we were having a race conditions of various variables.
This now has been fixed and the halo generations always converges produding correct results.
This data set under scenario 3 represents a dataset with correct results.
The simulation converges with 4 to 5 generations.
What needs to be done next is to move the pump and surface reflection definitions to theuser interface

!! Define neutral gas temperature at the wall:
T_wall = 3e-3 !! [keV]
vT = sqrt(T_wall/(v2_to_E_per_amu*thermal_mass(1))) !! [cm/s]

!! Wall reaction probabilities:
p_specular = 0.7 !! Specular reflection
p_absorb = 0.03 !! Absorbed by the wall
p_thermal = 1 - p_absorb - p_specular !! Thermal scattering from wall

!! Define pump aperture:
pump%xmax = beam_grid%box%xmax
pump%xmin = beam_grid%box%xmin
pump%ymax = -15
pump%ymin = +15
pump%zmin = -25 - 15
pump%zmax = -25 + 15
call pad_aabb(pump,+beam_grid%dr*1E-2)

!! Define vacuum chamber mirror throats:
throat1%xmax = +5
throat1%xmin = -5
throat1%ymax = +5
throat1%ymin = -5
throat1%zmax =  beam_grid%box%zmax
throat1%zmin =  beam_grid%box%zmax
call pad_aabb(throat1,+beam_grid%dr*1E-2)

throat2%xmax = +5
throat2%xmin = -5
throat2%ymax = +5
throat2%ymin = -5
throat2%zmax = beam_grid%box%zmin
throat2%zmin = beam_grid%box%zmin
call pad_aabb(throat2,+beam_grid%dr*1E-2)

!! Define a Ta surface
Ta_surf%xmax = beam_grid%box%xmin
Ta_surf%xmin = beam_grid%box%xmin
Ta_surf%ymax = +30
Ta_surf%ymin = -30
Ta_surf%zmax = +30
Ta_surf%zmin = -30
call pad_aabb(Ta_surf,+beam_grid%dr*1E-2)

Hence:

p_specular = 0.7
p_absorb = 0.03
Ta surface

scenario_4:
=============
same as 3 but increased the wall absorption probability from 0.03 to 0.3. Everything else being the same.
Generations converged in 3 steps. Massive drop in edge neutral density

p_specular = 0.7
p_absorb = 0.3
Ta surface

scenario_5:
============
Same as 4 but reduce the wall absorption probability from 0.3 to 0.15.
Four generations needed and edge neutral density is 1/3 of scenario_3

p_specular = 0.7
p_absorb = 0.15
Ta surface

scenario_6:
===========
Repeat scenario 3 with low wall absorption (model saturated wall) and also remove Ta surface

p_specular = 0.7
p_absorb = 0.03
No Ta surface

scenario_7:
=============
same as 6 but reduce p_specular from 0.7 to 0.07, thus, mostly thermal emission (90%)

p_specular = 0.07
p_absorb = 0.03
No Ta surface
