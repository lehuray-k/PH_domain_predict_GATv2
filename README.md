Welcome to the code base for the manuscript "Rapid Prediction of Lipid Interaction Sites on Pleckstrin Homology Domains Using Graph Neural Networks and Molecular Dynamics Simulations". This repository contains model parameters, a script for making predictions on any PH domain structure as well as scripts for training and cross-validation.

Requirements: python 3, tensorflow 2.13.0, keras 2.13, numpy, pandas, mdtraj, scikit-learn and biopython

Scripts have been tested on macOS (with M2 processor), debian and centOS. We recommend using miniconda to install tensorflow and keras.

predict
#-----------------------------------------------------#
INSTRUCTIONS FOR PREPARING PDB FILES FOR PREDICTION
- 'Clean' PDB by removing all lines that do not begin with "ATOM"
- IMPORTANT - If the structure contains multiple alternative locations "altLocs" for some atoms (indicated in column 17 of PDB file) these MUST to be modified/removed so there is only one location per atom.
- For best performance PDB file should contain only 1 PH domain, in 1 chain and with no missing residues
- The structure should ideally begin 1-2 residues before B1 strand and terminate 1-2 residues after C-terminal alpha helix
- Avoid leaving LYS/ARG/HIS/GLU/ASP at C or N termini, as the loose charges at the termini may affect prediction
- Modify 'user defined settings below' to make a list of the location of the pdb files you want to make predictions for
#-----------------------------------------------------#
