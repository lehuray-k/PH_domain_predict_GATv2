Welcome to the codebase for the manuscript "Rapid Prediction of Lipid Interaction Sites on Pleckstrin Homology Domains Using Graph Neural Networks and Molecular Dynamics Simulations". This repository contains a prediction script for PH domain pdb structures, tutorial example data, model parameters, all training data as well as scripts for training and cross-validation.

Requirements: python 3, tensorflow 2.13.0, keras 2.13, numpy, pandas, mdtraj, scikit-learn, matplotlib and biopython

Scripts have been tested on macOS (with M2 processor), Debian 12 and centOS 7. We recommend using miniconda to install tensorflow 2.13.0 and keras 2.13 (they have also been successfully tested with TF 2.12.0 but earlier versions may cause issues). Prediction does not require the use of a GPU and should be computationally fast on all modern consumer processers.

#-----------------------------------------------------#

USAGE

predict_on_structure.py enables users to make predictions on pdb files of PH domains. The predicted normalized frequency of contacts with phosphoinositide headgroups will be printed, plotted using matplotlib and optionally also saved into the betafactor column of a new PDB file, allowing visualization on the structure in programs such as VMD (Visual Molecular Dynamics). Please download the ph_domain_data folder and keep this in the same directory as the predict_on_structure.py script.

To make predictions, we have prepared four tutorial examples found in ph_domain_data/tutorial_examples/. predict_on_structure.py is already set up to run inferences on these strutrues. The predictions will be printed to the terminal, displayed as a matplotlib graph (optional, enabled by default) and new pdb files will be saved in the original folder, with the predictions stored in the beta column (optional, enabled by default). To perform predictions on different PH domain structures, simply update the 'files' variable on line 27 of predict_on_structure.py to list the path to the PDB files you wish to make predictions for. Please also follow the instructions below to prepare PDB files for prediction.

files = ["tutorial_examples/testing_predict_on_structure/1mai_clean.pdb","tutorial_examples/testing_predict_on_structure/5c79_clean.pdb","tutorial_examples/testing_predict_on_structure/7yis_clean.pdb","tutorial_examples/testing_predict_on_structure/1h6h_clean.pdb"] # list of PDB file to make predictions for

On a linux system, if all dependencies have been installed, the simplest way to run the script is to download the repository from github either manually or using git

git clone https://github.com/lehuray-k/PH_domain_predict_GATv2.git

Then use python (must be python3) to run predictions

python predict_on_structure.py

#-----------------------------------------------------#

INSTRUCTIONS FOR PREPARING PDB FILES FOR PREDICTION

- 'Clean' PDB by removing all lines that do not begin with "ATOM"
- IMPORTANT - If the structure contains multiple alternative locations "altLocs" for some atoms (indicated in column 17 of PDB file) these MUST to be modified/removed so there is only one location per atom.
- For best performance PDB file should contain only 1 PH domain, in 1 chain and with no missing residues
- The structure should ideally begin 1-2 residues before B1 strand and terminate 1-2 residues after C-terminal alpha helix
- Avoid leaving LYS/ARG/HIS/GLU/ASP at C or N termini, as the disordered charges at the termini may affect prediction
- Modify the files variable below the comment 'user defined settings below' to make a list of the location of the pdb files you want to make predictions for

