import numpy as np
import warnings
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
# disable GPU for this script as model is fast on CPU, and GPU can introduce additional compatibility issues 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import mdtraj
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder

"""
#-----------------------------------------------------#
INSTRUCTIONS FOR PREPARING PDB FILES
- 'Clean up' PDB by removing all lines that do not begin with "ATOM"
- IMPORTANT - If the structure contains multiple alternative locations "altLocs" for some atoms (indicated in column 17 of PDB file) these MUST to be modified/removed so there is only one location per atom.
- For best performance PDB file should contain only 1 PH domain, in 1 chain and with no missing residues
- The structure should ideally begin 1-2 residues before B1 strand and terminate 1-2 residues after C-terminal alpha helix
- Avoid leaving LYS/ARG/HIS/GLU/ASP at C or N termini, as the loose charges at the termini may affect prediction
- Modify 'user defined settings below' to make a list of the location of the pdb files you want to make predictions for
#-----------------------------------------------------#
"""
# USER DEFINED SETTINGS

files = ["ph_domain_data/tutorial_examples/1mai_clean.pdb","ph_domain_data/tutorial_examples/5c79_clean.pdb","ph_domain_data/tutorial_examples/7yis_clean.pdb","ph_domain_data/tutorial_examples/1h6h_clean.pdb"] # list of PDB file to make predictions for

model_weights_file = "ph_domain_data/models/GATv2model_2023-06-10_01_10foldCV_seed907_fold7.h5" # location of trained model parameters

write_new_structure_with_predicted_contacts = True # whether or not to write a new PDB file with predicted contacts in the beta/temp factor column
plot_predicted_contacts = True # whether or not to make matplotlib plot of predictions
#-----------------------------------------------------#

if plot_predicted_contacts == True:
    import matplotlib.pyplot as plt

# Define functions for extracting features from structure

def sequence_from_pdb_mdtraj(structure_file):
    """
    Uses MDTraj to obtain amino acid sequence from PDB structure file

    Parameters
    ----------
    structure_file : str
        Path to PDB structure

    Returns
    -------
    str
        Amino acid sequence in one-letter format encoding
    """
    traj = mdtraj.load(structure_file)
    # Get the topological information
    topology = traj.topology
    # Extract the sequence
    sequence_mdtraj = topology.to_fasta()
    sequence_mdtraj = ''.join(sequence_mdtraj)
    return sequence_mdtraj

def one_hot_AA_encoding(sequence):
    """
    Returns one hot encoding matrix in the form
            A R N D C Q E G H I L K M F P S T W Y V
    pos 1   1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pos 2   0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    pos 3   0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
    pos 4   0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    etc.

    Parameters
    ----------
    sequence : str
        Amino acid sequence in one-letter format

    Returns
    -------
    numpy.ndarray
        Amino acid sequence as one-hot encoded matrix
    """
    masterkey_sequence = "ARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYV"
    sequence_length = len(sequence)
    matrix = np.zeros((sequence_length,20))
    for AA in range(0,len(sequence)):
        matrix[AA,masterkey_sequence.index(sequence[AA])] = 1
    one_hot_encoding_matrix = matrix
    return one_hot_encoding_matrix

def beta_factor_of_c_alpha_atoms_biopandas(structure_file):
    """Uses biopandas library to obtain the beta factor value of each amino acid alpha-carbon from a protein structure PDB file

    Parameters
    ----------
    structure_file : str
        Path to PDB file

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, 1) representing the beta factor value for each amino acid alpha-carbon 
    """
    parser = PDBParser()
    structure = parser.get_structure("structure_file", structure_file)
    ca_bfactor_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:  # Ensure the residue has a Cα atom
                    ca_atom = residue["CA"]
                    bfactor = ca_atom.get_bfactor()
                    ca_bfactor_list.append(bfactor)
    ca_bfactor_array = np.array(ca_bfactor_list)
    return ca_bfactor_array

def shrake_rupley_solvent_accessibility(structure,mode="residue"):
    """
    Calculates shrake-rupley solvent accessibility for each amino acid in a protein structure using MDTraj.

    Parameters
    ----------
    structure : str
        PDB file path
    mode : str, optional
        Mode option for MDTraj shake_rupley method, by default "residue"

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, 1) representing the calculated shrake-rupley solvent accessibility for each amino acid
    """
    structure = mdtraj.load(structure)
    shake_rupley_sa = mdtraj.shrake_rupley(structure,mode=mode).transpose()
    return shake_rupley_sa

def DSSP_threestate_simplified(structure):
    """
    Calculates simplified DSSP secdonary structure class labels for each amino acid in a protein structure using MDTraj, and returns this in one-hot encoded format.
    
    Parameters
    ----------
    structure : str
        PDB file path

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, 3) representing the assigned three-state DSSP secondary structure label for each amino acid as a one-hot vector.
        In simplified DSSP annotation helices are H, strands are E and coiled/unstructured elements are C. In this function the annotations are encoded as a one-hot vector, e.g.:
                    H E C
            pos 1   1 0 0
            pos 2   1 0 0
            pos 3   0 0 1
            pos 4   0 1 0
            etc.
    """
    structure = mdtraj.load(structure)
    DSSP = mdtraj.compute_dssp(structure,simplified=True)
    # create the OneHotEncoder object
    encoder = OneHotEncoder(categories=[['H', 'E', 'C']])
    # reshape the input array to a 2D matrix
    arr_reshaped = np.reshape(DSSP, (-1, 1))
    # fit the encoder to the reshaped array and transform it to a one-hot encoded array
    one_hot_encoded = encoder.fit_transform(arr_reshaped).toarray()
    return one_hot_encoded

def compute_distance_matrix_and_inter_residue_unit_vectors(PDB_file):
    """
    Computes the pairwise distance (in nanometers) and the unit vectors between all pairs of amino acids in a protein structure using MDTraj and the alpha-carbon positions.

    Parameters
    ----------
    PDB_file : str
        Path to PDB file

    Returns
    -------
    numpy.ndarray: distance_matrix
        Array of shape (n_amino_acids, n_amino_acids) containing the pairwise distance (in nanometers) between the alpha-carbon positions of all amino acids in the protein structure.

    numpy.ndarray: unit_vector_matrix
        Array of shape (n_amino_acids, n_amino_acids, 3) containing the pairwise unit vectors between the alpha-carbon positions of all amino acids in the protein structure.
    """
    structure = mdtraj.load(PDB_file)
    residue_indices = list(residue.index for residue in structure.topology.residues)  # Convert generator to a list
    num_residues = len(residue_indices)
    ca_atom_indices_dict = {}
    for residue_index in residue_indices:
        #Get the CA atom indices for the current residue index
        for atom in structure.topology.atoms:
            if (atom.name == "CA"):
                if (atom.residue.index == residue_index):
                    ca_atom_indices_dict[residue_index] = atom.index
    pairs = np.zeros((num_residues**2,2))
    count = 0
    for x in ca_atom_indices_dict.keys():
        for y in ca_atom_indices_dict.keys():
            pairs[count,0] = ca_atom_indices_dict[x]
            pairs[count,1] = ca_atom_indices_dict[y]
            count = count+1
    displacement_vectors = mdtraj.compute_displacements(structure, pairs, periodic=False)
    displacement_vectors = displacement_vectors.reshape(num_residues, num_residues,3)
    distance_matrix = np.sqrt((displacement_vectors**2).sum(axis=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Ignore RuntimeWarnings in the case of division by zero, which arises on diagonals of matrix. This will be handled by setting the NaNs to 0
        unit_vector_matrix = np.divide(displacement_vectors,np.expand_dims(distance_matrix,-1))
        unit_vector_matrix[np.isnan(unit_vector_matrix)] = 0
    return distance_matrix,unit_vector_matrix

def charge_neighbourhood_from_distance_matrix(sequence, distance_cutoff, distance_matrix):
    """Computes total charge within a radial distance of each amino acid, using simplified charge assignments (K, R, H = +1 charge; D, E = -1 charge).

    Parameters
    ----------
    sequence : str or array-like
        Amino acid sequence encoded using one-letter code
    distance_cutoff : float
        Distance cutoff from the alpha carbon in nanometers, charges will be counted for all amino acids within this distance
    distance_matrix : numpy.ndarray
        Array of shape (n_amino_acids, n_amino_acids) containing the pairwise unit vectors between the alpha-carbon positions of all amino acids in the protein structure.

    Returns
    -------
    _type_
        _description_
    """
    # Map amino acids to their charges
    amino_acid_charge_map = {'K': 1, 'R': 1, 'H': 1, 'D': -1, 'E': -1}

    # Convert sequence to corresponding charges
    charge_sequence = np.array([amino_acid_charge_map.get(aa, 0) for aa in sequence])

    # Create a boolean matrix where distances are less than the cutoff
    within_cutoff = distance_matrix <= distance_cutoff
    # Compute the neighbourhood charge matrix
    neighbourhood_charge_matrix = within_cutoff.dot(charge_sequence).reshape(-1, 1)

    return neighbourhood_charge_matrix

def pdb_to_pandas(pdb_file):
    """
    Imports a protein structure PDB file as a pandas array

    Parameters
    ----------
    pdb_file : str
        Path to PDB file

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe representation of the PDB file, where each row contains the data for a residue, with the following columns:
            ['ATOM', 'Atom serial number', 'Atom name', 'Alternate location indicator', 'Residue name', 'Chain identifier',
            'Residue sequence number', 'Code for insertions of residues','X coordinate', 'Y coordinate', 'Z coordinate',
            'Occupancy', 'Temperature factor', 'Segment identifier', 'Element symbol', 'Charge']
    """
    file = open(pdb_file,"r")
    lines = file.readlines()
    file.close()
    atomlines = []
    for item in lines:
        if item[:4] == "ATOM":
            atomlines.append(item)
    # Define column names
    columns = ['ATOM', 'Atom serial number', 'Atom name', 'Alternate location indicator', 'Residue name', 'Chain identifier', 'Residue sequence number', 'Code for insertions of residues', 'X coordinate', 'Y coordinate', 'Z coordinate', 'Occupancy', 'Temperature factor', 'Segment identifier','Element symbol','Charge']

    # Initialize an empty list to store parsed data
    parsed_data = []

    # Loop through each data string in the input list
    for data_string in atomlines:
        # Extract data using string slicing
        data = [data_string[0:4].strip(),
                int(data_string[6:11]),
                data_string[12:16].strip(),
                data_string[16],
                data_string[17:20].strip(),
                data_string[21],
                int(data_string[22:26]),
                data_string[26],
                float(data_string[30:38].strip()),
                float(data_string[38:46].strip()),
                float(data_string[46:54].strip()),
                float(data_string[54:60].strip()),
                float(data_string[60:66].strip()),
                data_string[72:76].strip(),
                data_string[76:78].strip(),
                data_string[78:80].strip()]

        # Append the extracted data to the parsed_data list
        parsed_data.append(data)

    # Create DataFrame from parsed_data
    df = pd.DataFrame(parsed_data, columns=columns)
    return df

def pandas_to_pdb(pandas_dataframe,pdb_file_to_write):
    """
    Writes a pandas representation of a protein structure to a PDB file format.

    Parameters
    ----------
    pandas_dataframe : pandas.core.frame.DataFrame
        Dataframe representation of the PDB file, where each row contains the data for a residue, with the following columns:
            ['ATOM', 'Atom serial number', 'Atom name', 'Alternate location indicator', 'Residue name', 'Chain identifier',
            'Residue sequence number', 'Code for insertions of residues','X coordinate', 'Y coordinate', 'Z coordinate',
            'Occupancy', 'Temperature factor', 'Segment identifier', 'Element symbol', 'Charge']
    pdb_file_to_write : Path to PDB file to be written.
        str
    """
    lines = []
    df = pandas_dataframe
    for row in df.iterrows():
        data = row[1]
        line = []
        for x in range(0,80):
            line.append(' ')
        line[0:4] = "{:<4s}".format(str(data['ATOM']))
        line[7:12] = "{:>4s}".format(str(data['Atom serial number']))
        line[13:17] = "{:<4s}".format(str(data['Atom name']))
        line[16] = "{:1s}".format(str(data['Alternate location indicator']))
        line[17:20] = "{:>3s}".format(str(data['Residue name']))
        line[21] = "{:1s}".format(str(data['Chain identifier']))
        line[22:26] = "{:>4d}".format(data['Residue sequence number'])
        line[27] = "{:1s}".format(str(data['Code for insertions of residues']))
        line[30:38] = "{:>8.3f}".format(data['X coordinate'])
        line[38:46] = "{:>8.3f}".format(data['Y coordinate'])
        line[46:54] = "{:>8.3f}".format(data['Z coordinate'])
        line[54:60] = "{:>6.2f}".format(data['Occupancy'])
        line[60:66] = "{:>6.2f}".format(data['Temperature factor'])
        lines.append(''.join(line))
    file = open(pdb_file_to_write,"w")
    for item in lines:
        file.writelines(item+"\n")
    file.close()

def modify_beta_factor_in_pdb(pdb_file,pdb_file_to_write,new_beta_factor_list):
    """
    Reads the values of the beta-factor values of a PDB file, sets these to new values provided in new_beta_factor_list and writes a new PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to pdb file to load
    pdb_file_to_write : str
        Path for pdb file to be written, set the same as pdb_file for overwrite behaviour
    new_beta_factor_list : list
        List of new beta factor values. Must be provided for the full sequence in order.
    """
    pdb_df = pdb_to_pandas(pdb_file)
    residue_list = pdb_df['Residue sequence number'].unique()
    for x in range(0,len(residue_list)):
        pdb_df.loc[pdb_df['Residue sequence number'] == residue_list[x], 'Temperature factor'] = new_beta_factor_list[x]
    pandas_to_pdb(pdb_df,pdb_file_to_write)

# Define custom model architecture

class GraphAttention_v2(layers.Layer):
    """
    Tensorflow Keras layer implementation of graph attention head using the GATv2 mechanism from the paper:
    
    'How Attentive are Graph Attention Networks?', Shaked Brody, Uri Alon, Eran Yahav, arXiv preprint arXiv:2105.14491 (2021)

    This implementation has been adapted to include edge features 

    -------
    Mathematical summary

    The output for the GATv2 layer at node_i (Output_i) is calculated as follows:

        Let h_a be node features of the node_a

        Let e_a,b be features of the edge from node_a to node_b

        Let W_* represent parameters of the network
        
        Calculate the attention coefficient c_(i,j) of a node j connected to node_i
            c_(i,j) = W_attention•LeakyReLu([W_nodes•h_i || W_neighbour-nodes•h_j || W_edges•e_(i,j)])
                where • denotes matric multiplication and || denotes concatenation

        Calculate the attention score for a node j connected to node_i
            a_(i,j) = SOFTMAX_over_all_j( c_(i,j) )

        Calculate Output at node_i
            Output_i = SIGMOID( SUM_over_all_j[a_(i,j) * (W_neighbour•h_j)] )
                where * denotes element-wise multiplication and • denotes matrix multiplication
                Note that the final SIGMOID non-linearity is not implemented in this class, so that the output can be concatenated and/or averaged in a multi-head strategy and then the non-linearity applied
    -------
    """
    def __init__(
        self,
        units,
        kernel_regularizer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        units : int
            Number of hidden units for node parameter matrices (edge and attention parameter matrices are scale accordingly)
        kernel_regularizer :  tf.keras.Regularizer , optional
            Optional regularizer, by default None
        """
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.GlorotUniform(seed=134631644)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        # Initialise parameters
        # Node parameter matrix
        self.kernel_left = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_left",
        )
        # Node neighbour parameter matrix
        self.kernel_right = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_right",
        )
        # Attention parameter matrix
        self.kernel_attention = self.add_weight(
            shape=((self.units * 2) + input_shape[2][-1], 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        # Edge parameter matrix
        self.kernel_edge_features_attention = self.add_weight(
            shape=(input_shape[2][-1],input_shape[2][-1]),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_edge_features_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges, edge_features = inputs
        node_states = tf.squeeze(node_states,axis=0)
        # If edges is a ragged tensor then convert to normal tensor. We need this because we can't index into the ragged dimension of ragged tensors
        if isinstance(edges,tf.RaggedTensor) == True:
            edges = edges.to_tensor()
        edges = tf.squeeze(edges,axis=0)
        if isinstance(edge_features,tf.RaggedTensor) == True:
            edge_features = edge_features.to_tensor()
        edge_features = tf.squeeze(edge_features,axis=0)
        # Linearly transform node states
        node_states_transformed_left = tf.matmul(node_states, self.kernel_left)
        node_states_transformed_right = tf.matmul(node_states, self.kernel_right)
        # (1) Compute pair-wise attention co-efficients
        node_states_expanded = tf.concat((tf.gather(node_states_transformed_left, edges[:,0]),tf.gather(node_states_transformed_right, edges[:,1])),axis=-1)
        node_states_expanded = tf.nn.leaky_relu(node_states_expanded)
        edge_features_gathered = tf.gather_nd(edge_features,edges)
        edge_features_transformed = tf.matmul(edge_features_gathered, self.kernel_edge_features_attention)
        node_states_expanded = tf.concat((node_states_expanded,edge_features_transformed),axis = -1)
        attention_scores = tf.matmul(node_states_expanded, self.kernel_attention)
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Softmax and normalize to get attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate to calculate output
        node_states_neighbors = tf.gather(node_states_transformed_right, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        out = tf.expand_dims(out, 0)
        return out

class MultiHeadGraphAttention_v2(layers.Layer):
    """
    TF Keras layer which aggregates multiple graph attention heads, either by concatenation or averaging.
    Performs non-linearity (ReLU) on the aggregated attention head outputs
    """
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        """
        Parameters
        ----------
        units : int
            Hidden units dimension size for node parameters (other parameters are scaled accordingly)
        num_heads : int, optional
            Number of graph attention heads, by default 8
        merge_type : str["concat" or "average"], optional
            Optionally specificy method of aggregating graph attention head outputs, either "concat" or "average", by default "concat"
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention_v2(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices, edge_features = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices, edge_features])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

# Define class for processing protein structures and making predictions
class protein_representation:
    """
    Class implementing the representation of the protein for the graph neural network and inference
    """
    def __init__(self,pdb_file):
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
        self.true_y = None # ground truth contacts values extracted from PDB beta factor column
        self.preprocess_structure()

    def preprocess_structure(self):
        """
        Prepare protein representation for graph neural network, by calculating node features, generating edge list and calculating edge features.
        In this implementation the protein is treated as a fully-connected graph where all amino acids are connected to each other
        Node features:
            Amino acid identity (one hot encoding)
            Simplified secondary structure class (one hot encoding)
            Solvent accessible surface area
            Total charge within 8 angstroms
        Edge features:
            Alpha carbon distance (nm) from amino acid i to amino acid j
            Unit vector in direction from amino acid i to amino acid j (alpha carbons)
        """
        print("\nProcessing input features from %s\n"%(self.pdb_file))
        self.sequence = sequence_from_pdb_mdtraj(self.pdb_file)
        self.one_hot = one_hot_AA_encoding(self.sequence)
        true_beta_factor = beta_factor_of_c_alpha_atoms_biopandas(self.pdb_file)
        self.seq_len = len(self.sequence)
        self.one_hot = one_hot_AA_encoding(self.sequence)
        self.shrake_rupley_sa = shrake_rupley_solvent_accessibility(self.pdb_file)
        self.DSSP = DSSP_threestate_simplified(self.pdb_file)
        self.distance_matrix,self.inter_residue_unit_vectors = compute_distance_matrix_and_inter_residue_unit_vectors(self.pdb_file)
        self.neighbouring_charges = charge_neighbourhood_from_distance_matrix(self.sequence,0.8,distance_matrix = self.distance_matrix)
        # Concatenate node features
        self.node_features = np.concatenate((self.one_hot,self.DSSP,self.shrake_rupley_sa,self.neighbouring_charges),axis=1)
        # Generate list of edges
        # Distance cutoff for determining neighbour status. Set to 200 = essentially inf. distance; allows fully-connected graph of protein structure
        distance_cutoff = 200
        protein_neighbourhood_list = []
        for AA1 in range(0,np.shape(self.distance_matrix)[0]):
            for AA2 in range(0,np.shape(self.distance_matrix)[1]):
                    if self.distance_matrix[AA1][AA2] > 0: # residue will not have edge to itself
                        if self.distance_matrix[AA1][AA2] <= distance_cutoff:
                                protein_neighbourhood_list.append([int(AA1),int(AA2)])
        # Prepare edge features
        edge_features = []
        for AA1 in range(0,np.shape(self.distance_matrix)[0]):
            AA2_edge_features_list = []
            for AA2 in range(0,np.shape(self.distance_matrix)[0]):
                    AA2_edge_features_list.append([self.distance_matrix[AA1][AA2],self.inter_residue_unit_vectors[AA1][AA2][0],self.inter_residue_unit_vectors[AA1][AA2][1],self.inter_residue_unit_vectors[AA1][AA2][2]])
            edge_features.append(AA2_edge_features_list)
        # Set up input tensors
        self.node_features = tf.expand_dims(tf.ragged.constant(self.node_features),0)
        self.edges_list = tf.expand_dims(tf.ragged.constant(protein_neighbourhood_list),0)
        self.edge_features = tf.expand_dims(tf.ragged.constant(edge_features),0)

    def predict_phosphoinositide_contacts(self,model):
        """
        Run trained model inference on protein representation to obtain prpedicted phosphoinositide contacts

        Parameters
        ----------
        model :  tf.keras.Model 
            Trained TF model

        Yields
        ------
        self.prediction
            Predicted phosphoinositide normalized contact frequency for each amino acid
        """
        print("\nRunning prediction for %s\n"%(self.pdb_file))
        self.prediction = model.predict([self.node_features,self.edges_list,self.edge_features])
        self.prediction = tf.squeeze(self.prediction,axis=[-1])
        self.prediction = tf.squeeze(self.prediction,axis=[0])
        self.prediction = tf.divide(self.prediction,tf.reduce_max(self.prediction)) # normalization to yield normalized contacts
        print("Predicted normalized frequency of contacts:\n"+str(self.prediction.numpy()))

    def output_prediction_to_new_pdb_file(self,alternative_new_file_name=None):
        """_summary_

        Parameters
        ----------
        alternative_new_file_name : str, optional
            Optional path to write new PDB file name, by default None and will write new file with _GATv2-PIPcontacts-prediction appended to input PDB file new
        """
        if alternative_new_file_name == None:
            new_file_name = self.pdb_file.replace(".pdb","_GATv2-PIPcontacts-prediction.pdb")
        else:
            new_file_name = alternative_new_file_name
        print("Writing predicted contacts to file %s"%(new_file_name))
        modify_beta_factor_in_pdb(self.pdb_file,new_file_name,self.prediction.numpy().tolist())

# load model
print("\nLoading model")
model = tf.keras.models.load_model(model_weights_file,custom_objects={'MultiHeadGraphAttention_v2': MultiHeadGraphAttention_v2},compile=False)
print(model.summary())

for file in files:
    processed_structure = protein_representation(file)

    processed_structure.predict_phosphoinositide_contacts(model)

    if write_new_structure_with_predicted_contacts == True:
        processed_structure.output_prediction_to_new_pdb_file()

    if plot_predicted_contacts == True:
        print("Plotting data in matplotlib")
        fig,ax = plt.subplots()
        plt.suptitle(processed_structure.pdb_file.split("/")[-1],y=0.95,weight='heavy',font='arial')
        ax.plot(processed_structure.prediction,c='#364B9A',label='Prediction',lw=0.7)
        ax.set_facecolor('#F7F7F7')
        ax.set_xlim(0,len(processed_structure.prediction))
        ax.set_ylim(0,1)
        # uncomment below to turn off residue number labels
        #ax.set_xticks([])
        ax.set_xlabel("Residue",weight='book')
        ax.set_ylabel("Predicted normalized frequency of contacts",weight='book')
        plt.show()
