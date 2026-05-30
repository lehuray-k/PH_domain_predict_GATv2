import numpy as np
import tensorflow as tf

from protein_ML_utils import (
    DSSP_threestate_simplified,
    beta_factor_of_c_alpha_atoms_biopandas,
    charge_neighbourhood_from_distance_matrix,
    compute_distance_matrix_and_inter_residue_unit_vectors,
    modify_beta_factor_in_pdb,
    one_hot_AA_encoding,
    sequence_from_pdb_mdtraj,
    shrake_rupley_solvent_accessibility,
)


class ProteinRepresentation:
    """
    Class implementing the protein representation for graph neural network and inference
    """

    def __init__(self, pdb_file):
        """_summary_

        Parameters
        ----------
        pdb_file : str
            Path to PDB file
        """
        self.pdb_file = pdb_file
        self.node_features = None
        self.distance_matrix = None
        self.inter_residue_unit_vectors = None
        self.edges_list = None
        self.true_y = (
            None  # ground truth contacts values extracted from PDB beta factor column
        )
        self.preprocess_structure()

    def preprocess_structure(self):
        """
        Prepare protein representation for graph neural network, by
          calculating node features, generating edge list and calculating edge features.
        In this implementation the protein is treated as a fully-connected graph
          where all amino acids are connected to each other
        Node features:
            Amino acid identity (one hot encoding)
            Simplified secondary structure class (one hot encoding)
            Solvent accessible surface area
            Total charge within 8 angstroms
        Edge features:
            Alpha carbon distance (nm) from amino acid i to amino acid j
            Unit vector in direction from amino acid i to amino acid j (alpha carbons)
        """
        print(f"\nProcessing input features from {self.pdb_file}\n")
        self.sequence = sequence_from_pdb_mdtraj(self.pdb_file)
        self.one_hot = one_hot_AA_encoding(self.sequence)
        true_beta_factor = beta_factor_of_c_alpha_atoms_biopandas(self.pdb_file)
        self.seq_len = len(self.sequence)
        self.one_hot = one_hot_AA_encoding(self.sequence)
        self.shrake_rupley_sa = shrake_rupley_solvent_accessibility(self.pdb_file)
        self.DSSP = DSSP_threestate_simplified(self.pdb_file)
        self.distance_matrix, self.inter_residue_unit_vectors = (
            compute_distance_matrix_and_inter_residue_unit_vectors(self.pdb_file)
        )
        self.neighbouring_charges = charge_neighbourhood_from_distance_matrix(
            self.sequence, 0.8, distance_matrix=self.distance_matrix
        )
        # Concatenate node features
        self.node_features = np.concatenate(
            (self.one_hot, self.DSSP, self.shrake_rupley_sa, self.neighbouring_charges),
            axis=1,
        )
        # Generate list of edges
        # Distance cutoff for determining neighbour status.
        #  Set to 200 = essentially inf. distance;
        #  allows fully-connected graph of protein structure
        distance_cutoff = 200
        protein_neighbourhood_list = []
        for AA1 in range(0, np.shape(self.distance_matrix)[0]):
            for AA2 in range(0, np.shape(self.distance_matrix)[1]):
                if (
                    self.distance_matrix[AA1][AA2] > 0
                ):  # residue will not have edge to itself
                    if self.distance_matrix[AA1][AA2] <= distance_cutoff:
                        protein_neighbourhood_list.append([int(AA1), int(AA2)])
        # Prepare edge features
        edge_features = []
        for AA1 in range(0, np.shape(self.distance_matrix)[0]):
            AA2_edge_features_list = []
            for AA2 in range(0, np.shape(self.distance_matrix)[0]):
                AA2_edge_features_list.append(
                    [
                        self.distance_matrix[AA1][AA2],
                        self.inter_residue_unit_vectors[AA1][AA2][0],
                        self.inter_residue_unit_vectors[AA1][AA2][1],
                        self.inter_residue_unit_vectors[AA1][AA2][2],
                    ]
                )
            edge_features.append(AA2_edge_features_list)
        # Set up input tensors
        self.node_features = tf.expand_dims(tf.ragged.constant(self.node_features), 0)
        self.edges_list = tf.expand_dims(
            tf.ragged.constant(protein_neighbourhood_list), 0
        )
        self.edge_features = tf.expand_dims(tf.ragged.constant(edge_features), 0)

    def predict_phosphoinositide_contacts(self, model):
        """
        Run trained model inference on protein representation
          to obtain prpedicted phosphoinositide contacts

        Parameters
        ----------
        model :  tf.keras.Model
            Trained TF model

        Yields
        ------
        self.prediction
            Predicted phosphoinositide normalized contact frequency for each amino acid
        """
        print(f"\nRunning prediction for {self.pdb_file}\n")
        self.prediction = model.predict(
            [self.node_features, self.edges_list, self.edge_features]
        )
        self.prediction = tf.squeeze(self.prediction, axis=[-1])
        self.prediction = tf.squeeze(self.prediction, axis=[0])
        self.prediction = tf.divide(
            self.prediction, tf.reduce_max(self.prediction)
        )  # normalization to yield normalized contacts
        print(
            "Predicted normalized frequency of contacts:\n"
            + str(self.prediction.numpy())
        )

    def output_prediction_to_new_pdb_file(self, alternative_new_file_name=None):
        """_summary_

        Parameters
        ----------
        alternative_new_file_name : str, optional
            Optional path to write new PDB file name, by default None and will write
              new file with _GATv2-PIPcontacts-prediction appended to input PDB file new
        """
        if alternative_new_file_name is None:
            new_file_name = self.pdb_file.replace(
                ".pdb", "_GATv2-PIPcontacts-prediction.pdb"
            )
        else:
            new_file_name = alternative_new_file_name
        print(f"Writing predicted contacts to file {new_file_name}")
        modify_beta_factor_in_pdb(
            self.pdb_file, new_file_name, self.prediction.numpy().tolist()
        )