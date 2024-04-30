# TBG_Electrochem
This repository contains all the data and code to construct the figures presented in the paper “Microscopic origin of twist dependent electron transfer rate in bilayer graphene”

The directory "FreeEnergy_ReorgEnergy" contains energy gap data from simulations and codes to produce the figures related to Marcus curves and reorganization energy. The script "plotFigMarcusReorg.py" will generate the plot presented in the main text for the free energy surface of the ferrous/ferric couple. The other two python scripts will generate the free energy surfaces for K/K+ and Cl-/Cl shown in the Supplementary Material.
  
The directory "LAMMPS_input_example" contains a LAMMPS input file and an associated data file that can be run to generate a trajectory for the Fe(II) ion. An up-to-date version of LAMMPS that includes the ELECTRODE package is required to run it

The directory called "Rate_Angle" will compute and plot the angle-dependent rate derived from the two model density of states considered in this work (linear and lorentzian)
