# GS-PDEs and surface RDEs
This repository contains the FEniCSx implementation of a coupled geometric surface and surface reaction-diffusion model for finite element simulations of cell and nuclear mechanics during confined cell migration. The model describes the evolution of the cell and nuclear surfaces and incorporates molecular dynamics on the evolving nuclear envelope.
# Features
- Implements the surface Finite Element Method (FEM) to simulate cell and nuclear motion under confinement.
- Solves a surface reaction--diffusion system on the evolving nuclear envelope to model the dynamics of molecular species involved.
- Couples biochemical dynamics to the mechanical properties of the nuclear envelope through state-dependent material parameters.
- Simulates cell translocation through a confined microchannel under an externally applied pressure difference.
- Supports spatially heterogeneous initial conditions for the biochemical species.
- Generates simulation data suitable for post-processing and visualization in ParaView.
# Prerequisites
FEniCSx and Python installed.
ParaView (recommended) for visualization of simulation results.
# Running the Simulation
Open the main script file: cell_nucleus.py.
Configure the model parameters (e.g., material properties, geometry, and time-stepping parameters).
Run the script. The simulation results will be saved in the specified output directory.
# Visualizing Results
Open the generated output files in ParaView to visualize and analyze the deformation and motion of the cell and nucleus during translocation.
# Acknowledgements
This work is based on research conducted by Francesca Ballatore, Silvia Comunian, Anotida Madzvamuse, Slimane Ait-Si-Ali, and Rachele Allena.
For more details, please refer to the associated publication.

