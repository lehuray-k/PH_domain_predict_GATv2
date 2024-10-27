import MDAnalysis as mda
import mdtraj
import numpy as np
import os
import h5py
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import copy

np.set_printoptions(threshold=np.inf)

def amino_acid_code_converter_three_to_one(three_letter_code):
    """Transforms a three letter amino acid code to one letter amino acid code. Only handles one amino acid code at a time.

    Parameters
    ----------
    three_letter_code : str
        Three letter amino acid code, for a sigle amino acid, in capitals, e.g. "GLY".
        Also accepts alternative amino acid codes MSE and CME.

    Returns
    -------
    str
        One letter amino acid code

    Raises
    ------
    Exception
        Raises an exception if the input three letter code is not recognised
    """
    three_letter_code_list = ["GLY","ALA","LEU","MET","PHE","TRP","LYS","GLN","GLU","SER","PRO","VAL","ILE","CYS",
                                "TYR","HIS","ARG","ASN","ASP","THR","MSE","CME"]
    one_letter_code_list = ["G","A","L","M","F","W","K","Q","E","S","P","V","I","C","Y","H","R","N","D","T","M","C"]
    one_letter_code = ""
    # correct for anomaly where 3-letter code in PDB is for example AHIS or BLYS
    if len(three_letter_code) == 4:
        three_letter_code = three_letter_code[1:]
    for x in range(0,len(three_letter_code_list)):
        if three_letter_code == three_letter_code_list[x]:
            one_letter_code = str(one_letter_code_list[x])
    if one_letter_code in one_letter_code_list:
        return one_letter_code
    else:
        raise Exception("INVALID THREE LETTER CODE INPUT: %s"%(three_letter_code))

def extract_resids_sequence_and_beta_factor_from_pdb(PDB_file):
    """
    Takes PDB file and returns numpy 2d array of shape (n_residues, 3) whose elements are [resnum,resname,beta_factor] for each residue
    The beta factor value for the alpha-carbon is selected
    Uses MDAnalysis library.

    :return: data_array

    Parameters
    ----------
    PDB_file : str
        Path to PDB file

    Returns
    -------
    numpy.ndarray
        2d array of of shape (n_residues, 3).
        The elements of first dimension are the residues
        The elements of the second dimension are [resnum (Str), resname (Str), beta_factor (Str)] for the residue

    Raises
    ------
    Exception
        Raises exception when a mismatch in residue ids is detected
    """
    structure = mda.Universe(PDB_file)
    res_ids = []
    beta_factors = []
    data_array = []
    for residue in structure.residues:
        res_ids.append(residue.resid)
    for atom in structure.select_atoms("name CA"):
        if atom.resid in res_ids:
            data_array.append([int(atom.resid),amino_acid_code_converter_three_to_one(atom.resname),float(atom.tempfactor)])
        else:
            raise Exception("Atom resid not in residue list")
    data_array = np.array(data_array)
    # select only unique rows in the array, this corrects for an issue in MDAnalysis with rare PDBs where there are alternate coordinates for some atoms leading to duplicate residues in MDAnalysis
    unique_rows, indices = np.unique(data_array, axis=0, return_index=True)
    data_array = unique_rows[np.argsort(indices)]
    return data_array

def one_hot_AA_encoding(sequence):
    """
    Returns one hot encoding matrix in the form
            A R N D C Q E G H I L K M F P S T W Y V
    pos 1   1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pos 2   0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    pos 3   0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
    pos 4   0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    etc.
    :param sequence: STR amino acid sequence in one-letter format
    :return: np.array one_hot_AA_encoding_matrix
    """
    masterkey_sequence = "ARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYV"
    sequence_length = len(sequence)
    matrix = np.zeros((sequence_length,20))
    for AA in range(0,len(sequence)):
        matrix[AA,masterkey_sequence.index(sequence[AA])] = 1
    one_hot_encoding_matrix = matrix
    return(one_hot_encoding_matrix)

def charge_encoding(sequence):
    """Encodes charges of amino acid sequences using simple naive model where K and R are assigned +1 charge, H +0.8 and D and E -1, returning one charge value per residue in the sequence

    Parameters
    ----------
    sequence : str or array-like
        Amino acid sequence encoded using one-letter code

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, 1) representing the assigned charges for each amino acid
    """
    sequence_length = len(sequence)
    matrix = np.zeros((sequence_length,1))
    for AA in range(0,len(sequence)):
        if sequence[AA] == "R":
            matrix[AA,0] = 1
        elif sequence[AA] == "K":
            matrix[AA,0] = 1
        elif sequence[AA] == "H":
            matrix[AA,0] = 0.8
        elif sequence[AA] == "D":
            matrix[AA,0] = -1
        elif sequence[AA] == "E":
            matrix[AA,0] = -1
    return matrix

def one_hot_charge_encoding(sequence):
    """Encodes charges of amino acid sequences using simple naive model where K and R are assigned +1 charge, H +0.8 and D and E -1, returning 3 values per residue in the sequence (correspoding to postively charged, neutral or negatively charged)

    Parameters
    ----------
    sequence : str or array-like
        Amino acid sequence encoded using one-letter code

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, 3) representing the assigned charges for each amino acid. Similar to one-hot encoding, the elements of the second dimension correspond to positive charge, neutral charge and negative charge.
    """
    sequence_length = len(sequence)
    matrix = np.zeros((sequence_length,3))
    for AA in range(0,len(sequence)):
        if sequence[AA] == "R":
            matrix[AA,0] = 1
        elif sequence[AA] == "K":
            matrix[AA,0] = 1
        elif sequence[AA] == "H":
            matrix[AA,0] = 0.8
        elif sequence[AA] == "D":
            matrix[AA,2] = 1
        elif sequence[AA] == "E":
            matrix[AA,2] = 1
        else:
            matrix[AA,1] = 1
    return matrix

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
    print(DSSP)
    # create the OneHotEncoder object
    encoder = OneHotEncoder(categories=[['H', 'E', 'C']])
    # reshape the input array to a 2D matrix
    arr_reshaped = np.reshape(DSSP, (-1, 1))
    # fit the encoder to the reshaped array and transform it to a one-hot encoded array
    one_hot_encoded = encoder.fit_transform(arr_reshaped).toarray()
    return one_hot_encoded

def compute_residue_pairwise_distance_matrix(structure,pad_to=None):
    """
    Computes pairwise distance (in nanometers) between all amino acids in protein structure using MDTraj and the alpha-carbon positions.

    Parameters
    ----------
    structure : str
        PDB file path
    pad_to : int, optional
        Optionally uses zero-padding to pad the returned matrix to a sequence length set by pad_to, by default None.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, n_amino_acids) containing the pairwise distance (in nanometers) between the alpha-carbon positions of all amino acids in the protein structure.
    """
    structure = mdtraj.load(structure)
    residue_index_list = []
    for residue in structure.topology.residues:
        residue_index_list.append(residue.index)
    pairs = np.zeros((len(residue_index_list)**2,2))
    count = 0
    for x in range(0,len(residue_index_list)):
        for y in range(0,len(residue_index_list)):
            pairs[count] = [x,y]
            count = count+1
    distances, pairs = mdtraj.compute_contacts(structure,contacts=pairs,scheme='ca')
    distance_matrix = np.zeros((len(residue_index_list),len(residue_index_list)))
    count = 0
    for x in range(0,len(residue_index_list)):
        for y in range(0,len(residue_index_list)):
            distance_matrix[x][y] = distances[0][count]
            count = count+1
    if pad_to != None:
        pad_width = ((0, pad_to-np.shape(distance_matrix)[0]), (0, pad_to-np.shape(distance_matrix)[0]))
        distance_matrix = np.pad(distance_matrix, pad_width, mode='constant', constant_values=0)
    return distance_matrix

def compute_inter_residue_unit_vectors(PDB_file,pad_to=None):
    """
    Computes pairwise unit vectors between all pairs of amino acids in protein structure using MDTraj and the alpha-carbon positions.

    Parameters
    ----------
    structure : str
        PDB file path
    pad_to : int, optional
        Optionally uses zero-padding to pad the returned matrix to a sequence length set by pad_to, by default None.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, n_amino_acids, 3) containing the pairwise unit vectors between the alpha-carbon positions of all amino acids in the protein structure.

    Raises
    ------
    Exception
        Raises exception when a mismatch in residue ids is detected
    """
    structure = mda.Universe(PDB_file)
    res_ids = []
    position_array = []
    for residue in structure.residues:
        res_ids.append(residue.resid)
    for atom in structure.select_atoms("name CA"):
        if atom.resid in res_ids:
            position_array.append(atom.position)
        else:
            raise Exception("Atom resid not in residue list")
    unit_vector_matrix = np.zeros((len(res_ids),len(res_ids),3))
    for AA1 in range(0,len(position_array)):
        for AA2 in range(0,len(position_array)):
            distance = np.linalg.norm(position_array[AA2]-position_array[AA1])
            vector = np.array(position_array[AA2])-np.array(position_array[AA1])
            if distance == 0:
                unit_vector = [0,0,0]
            else:
                unit_vector = vector/distance
            unit_vector_matrix[AA1][AA2][0] = unit_vector[0]
            unit_vector_matrix[AA1][AA2][1] = unit_vector[1]
            unit_vector_matrix[AA1][AA2][2] = unit_vector[2]
    return unit_vector_matrix

def charge_neighbourhood_from_distance_matrix(sequence,distance_cutoff,distance_matrix = None,protein_index_in_matrix=None):
    """Computes total charge within a radial distance of each amino acid, using simplified charge assignments (K, R, H = +1 charge; D, E = -1 charge).

    Parameters
    ----------
    sequence : str or array-like
        Amino acid sequence encoded using one-letter code
    distance_cutoff : float
        Distance cutoff from the alpha carbon in nanometers, charges will be counted for all amino acids within this distance
    distance_matrix : numpy.ndarray, optional
        Optional, by default None, in which case the distance matrix be calculated.
        Array containing the pairwise unit vectors between the alpha-carbon positions of all amino acids in the protein structure.
    protein_index_in_matrix : int, optional
        If provided distance matrix contains addition dimension for multiple proteins, specifies the index for the protein of interest, by default None

    Returns
    -------
    numpy.ndarray
        Array of shape (n_amino_acids, 1) representing the total charge within the cutoff distance neighbourhood of each amino acid.
    """
    if distance_matrix is None:
        distance_matrix_file = open("/home/kyle/membrane_machine_learning/DCRNN/ph_domain_data/preprocessed_data/distance_matrices_dim100xNonexNone_06Oct23.pkl","rb")
        distance_matrix = pkl.load(distance_matrix_file)
        distance_matrix_file.close()
        if protein_index_in_matrix is None:
            pass
        else:
            distance_matrix = distance_matrix[protein_index_in_matrix]
    else:
        distance_matrix = distance_matrix
    neighbourhood_charge_matrix = np.zeros((len(sequence),1))
    for x in range(0,len(sequence)):
        for y in range(0,len(sequence)):
            if distance_matrix[x][y] > distance_cutoff:
                continue
            else:
                if sequence[y] == 'K':
                    neighbourhood_charge_matrix[x] = neighbourhood_charge_matrix[x]+1
                elif sequence[y] == 'R':
                    neighbourhood_charge_matrix[x] = neighbourhood_charge_matrix[x]+1
                elif sequence[y] == 'H':
                    neighbourhood_charge_matrix[x] = neighbourhood_charge_matrix[x]+1
                elif sequence[y] == 'D':
                    neighbourhood_charge_matrix[x] = neighbourhood_charge_matrix[x]-1
                elif sequence[y] == 'E':
                    neighbourhood_charge_matrix[x] = neighbourhood_charge_matrix[x]-1
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
    pdb_file_to_write : str
        Path to PDB file to be written.
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
    """Reads the values of the beta-factor values of a PDB file, sets these to new values provided in new_beta_factor_list and writes a new PDB file.

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