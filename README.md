Welcome to the code base for the manuscript "Rapid Prediction of Lipid Interaction Sites on Pleckstrin Homology Domains Using Graph Neural Networks and Molecular Dynamics Simulations". This repository contains model parameters, a script for making predictions on any PH domain structure as well as scripts for training and cross-validation.

Requirements: python 3, tensorflow 2.13.0, keras 2.13, numpy, pandas, mdtraj, scikit-learn and biopython

Scripts have been tested on macOS (with M2 processor), debian and centOS. We recommend using miniconda to install tensorflow and keras.

#-----------------------------------------------------#

USAGE

predict_on_structure.py enables users to make predictions on pdb files of PH domains. The predicted normalized frequency of contacts with phosphoinositide headgroups will be printed, plotted using matplotlib and optionally also saved into the betafactor column of a new PDB file, allowing visualization on the structure.

To make predictions, prepare PDB files following the instructions below, and then edit the files variable on line 27 of predict_on_structure.py to list the locations of the PDB files you wish to make predictions for. Some example structure files have been already prepared and are available in the 

files = ["tutorial_examples/testing_predict_on_structure/1mai_clean.pdb","tutorial_examples/testing_predict_on_structure/5c79_clean.pdb","tutorial_examples/testing_predict_on_structure/7yis_clean.pdb","tutorial_examples/testing_predict_on_structure/1h6h_clean.pdb"] # list of PDB file to make predictions for

#-----------------------------------------------------#

INSTRUCTIONS FOR PREPARING PDB FILES FOR PREDICTION

- 'Clean' PDB by removing all lines that do not begin with "ATOM"
- IMPORTANT - If the structure contains multiple alternative locations "altLocs" for some atoms (indicated in column 17 of PDB file) these MUST to be modified/removed so there is only one location per atom.
- For best performance PDB file should contain only 1 PH domain, in 1 chain and with no missing residues
- The structure should ideally begin 1-2 residues before B1 strand and terminate 1-2 residues after C-terminal alpha helix
- Avoid leaving LYS/ARG/HIS/GLU/ASP at C or N termini, as the loose charges at the termini may affect prediction
- Modify 'user defined settings below' to make a list of the location of the pdb files you want to make predictions for
#-----------------------------------------------------#
