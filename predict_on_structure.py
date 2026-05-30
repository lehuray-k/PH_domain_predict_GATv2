import os

import tensorflow as tf

from MultiHeadGraphAttention import MultiHeadGraphAttention_v2
from protein_representation import ProteinRepresentation

# disable GPU for this script by default as model is fast on CPU,
# and GPU may not be available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
#-----------------------------------------------------#
INSTRUCTIONS FOR PREPARING PDB FILES
- 'Clean up' PDB by removing all lines that do not begin with "ATOM"
- IMPORTANT - If the structure contains multiple alternative locations "altLocs"
    for some atoms (indicated in column 17 of PDB file) these MUST
    be modified/removed so there is only one location per atom.
- For best performance PDB file should contain only 1 PH domain, in 1 chain
    and with no missing residues
- The structure should ideally begin 1-2 residues before B1 strand
    and terminate 1-2 residues after C-terminal alpha helix
- Avoid leaving LYS/ARG/HIS/GLU/ASP at C or N termini,
    as the loose charges at the termini may affect prediction
- Modify 'user defined settings below' to make a list of the location
    of the pdb files you want to make predictions for
#-----------------------------------------------------#
"""
# USER DEFINED SETTINGS

files = [
    "ph_domain_data/tutorial_examples/1mai_clean.pdb",
    "ph_domain_data/tutorial_examples/5c79_clean.pdb",
    "ph_domain_data/tutorial_examples/7yis_clean.pdb",
    "ph_domain_data/tutorial_examples/1h6h_clean.pdb",
]  # list of PDB file to make predictions for

# location of trained model parameters
model_weights_file = (
    "ph_domain_data/models/GATv2model_2023-06-10_01_10foldCV_seed907_fold7.h5"
)

# whether to write a new PDB file with predicted contacts in the beta/temp factor column
write_new_structure_with_predicted_contacts = True

# whether to make matplotlib plot of predictions
plot_predicted_contacts = True
# -----------------------------------------------------#

if plot_predicted_contacts:
    import matplotlib.pyplot as plt

# load model
print("\nLoading model")
model = tf.keras.models.load_model(
    model_weights_file,
    custom_objects={"MultiHeadGraphAttention_v2": MultiHeadGraphAttention_v2},
    compile=False,
)
print(model.summary())

for file in files:
    processed_structure = ProteinRepresentation(file)

    processed_structure.predict_phosphoinositide_contacts(model)

    if write_new_structure_with_predicted_contacts:
        processed_structure.output_prediction_to_new_pdb_file()

    if plot_predicted_contacts:
        print("Plotting data in matplotlib")
        fig, ax = plt.subplots()
        plt.suptitle(
            processed_structure.pdb_file.split("/")[-1],
            y=0.95,
            weight="heavy",
            font="arial",
        )
        ax.plot(processed_structure.prediction, c="#364B9A", label="Prediction", lw=0.7)
        ax.set_facecolor("#F7F7F7")
        ax.set_xlim(0, len(processed_structure.prediction))
        ax.set_ylim(0, 1)
        # uncomment below to turn off residue number labels
        # ax.set_xticks([])
        ax.set_xlabel("Residue", weight="book")
        ax.set_ylabel("Predicted normalized frequency of contacts", weight="book")
        plt.show()
