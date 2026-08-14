#!/bin/bash
mur=0.0
for mus in 0.0 0.3; do 
#for mus in 0.0 0.03 0.1 0.3; do 
    wallmus=$mus
    wallmur=$mur
    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus $wallmus -v wallmur $wallmur
    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus 0 -v wallmur 0
done

#mur=0.03
#for mus in 0.03 0.1 0.3; do 
#    wallmus=$mus
#    wallmur=$mur
#    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus $wallmus -v wallmur $wallmur
#    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus 0 -v wallmur 0
#done
#mur=0.1
#for mus in 0.1 0.3; do 
#    wallmus=$mus
#    wallmur=$mur
#    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus $wallmus -v wallmur $wallmur
#    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus 0 -v wallmur 0
#done
mur=0.3
for mus in 0.3; do 
    wallmus=$mus
    wallmur=$mur
    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus $wallmus -v wallmur $wallmur
    ~/software/aps_lammps/build_bpmheat_Aug24mpi/lmp -in in.pour.2dhopper -v mus $mus -v mur $mur -v wallmus 0 -v wallmur 0
done
