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
    """takes as input a three letter amino acid code and outputs the equivalent one letter amino acid code"""
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
    takes PDB file and returns numpy 2d array whose elements are [resnum,resname,beta_factor]
    beta factor is taken from the Ca
    :param PDB_file: str path to PDB file
    :return: data_array
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
    unique_rows, indices = np.unique(data_array, axis=0, return_index=True)
    data_array = unique_rows[np.argsort(indices)]
    # select only unique rows in the array, this corrects for an issue in MDAnalysis with rare PDBs where there a alternate coordinates for some atoms leading to duplicate residues in MDAnalysis
    #data_array = np.vstack({tuple(row) for row in data_array})
    return(data_array)

def hydrophobicity_from_sequence(sequence):
    hydrophobicity_dict = {'A':41,
                           'R':-14,
                           'N':-28,
                           'D':-55,
                           'C':49,
                           'Q':-10,
                           'E':-31,
                           'G':0,
                           'H':8,
                           'I':99,
                           'L':97,
                           'K':-23,
                           'M':74,
                           'F':100,
                           'P':-46,
                           'S':-5,
                           'T':13,
                           'W':97,
                           'Y':63,
                           'V':76}
    hydrophobicity_array = np.zeros((len(sequence),1))
    for x in range(0,len(sequence)):
        hydrophobicity_array[x] = hydrophobicity_dict[sequence[x]]/100
    return hydrophobicity_array

def generate_psiblast_pssm(sequence):
    """
    Runs psiblast to generate position specific scoring matrix (pssm) from sequence alignments
    :param sequence: amino acid sequence (str)
    :return: pssm as numpy array
    """
    # write temporary FASTA file for sequence
    fasta_file = open("/tmp/"+sequence+".fasta","w")
    fasta_file.write(sequence)
    fasta_file.close()
    # generate temporary pssm file
    os.system("psiblast -db swissprot -query /tmp/%s.fasta -num_iterations 3 -out_ascii_pssm /tmp/%s.pssm"%(sequence,sequence))
    pssm_file_lines = open("/tmp/"+sequence+".pssm","r").readlines()
    # filter pssm lines to only those which have a number (corresponding the last digit of resnumbered in position 4.
    # this filters out lines which don't contain the pssm data
    filtered_pssm_file_lines = []
    for a in range(0,len(pssm_file_lines)):
        try:
            if pssm_file_lines[a][4].isdigit():
                filtered_pssm_file_lines.append(pssm_file_lines[a].split(' '))
        except:
            pass
    for a in range(0,len(filtered_pssm_file_lines)):
        while '' in filtered_pssm_file_lines[a]:
            filtered_pssm_file_lines[a].remove('')
        filtered_pssm_file_lines[a] = filtered_pssm_file_lines[a][2:22]
    filtered_pssm_file_lines = np.array(filtered_pssm_file_lines,dtype=float)
    pssm = filtered_pssm_file_lines
    return(pssm)

def RaptorX_predictproperty(sequence):
    """
    Runs RaptorX predict property and returns matrices of three-state secondary structure prediction (SS3) and solvent accessibility (SA)
    :param sequence: amino acid sequence (str)
    :return: ss3_array, SA_array
    """
    # write temporary FASTA file for sequence
    fasta_file = open("/tmp/"+sequence+".fasta","w")
    fasta_file.write(sequence)
    fasta_file.close()
    # run RaptorX, generating temporary RaptorX prediction files
    os.system("sh /usr/not-backed-up-2/kyle/ML/literature_code/Predict_Property/Predict_Property.sh -i /tmp/%s.fasta -o /tmp/%s"%(sequence,sequence))
    # parse three-state secondary structure prediction SS3
    ss3_array = []
    ss3_file_lines = open("/tmp/%s/%s.ss3"%(sequence,sequence)).readlines()
    ss3_file_lines = ss3_file_lines[2:]
    for a in range(0,len(ss3_file_lines)):
        ss3_file_lines[a] = ss3_file_lines[a].split(" ")
        while "" in ss3_file_lines[a]:
            ss3_file_lines[a].remove("")
        # the line is now a list with elements [resid,amino_acid,letter_of_predicted_SS,probH,probE,probC,newline character]
        # store the probabilities in the array
        ss3_array.append([ss3_file_lines[a][3],ss3_file_lines[a][4],ss3_file_lines[a][5]])
    ss3_array = np.array(ss3_array,dtype=float)
    # parse three-state solvent accessibility SA
    SA_array = []
    SA_file_lines = open("/tmp/%s/%s.acc"%(sequence,sequence)).readlines()
    SA_file_lines = SA_file_lines[3:]
    for a in range(0,len(SA_file_lines)):
        SA_file_lines[a] = SA_file_lines[a].split(" ")
        while "" in SA_file_lines[a]:
            SA_file_lines[a].remove("")
        # the line is now a list with elements [resid,amino_acid,letter_of_predicted_SS,probH,probE,probC,newline character]
        # store the probabilities in the array
        SA_array.append([SA_file_lines[a][3],SA_file_lines[a][4],SA_file_lines[a][5]])
    SA_array = np.array(SA_array,dtype=float)
    return(ss3_array,SA_array)

def one_hot_AA_encoding(sequence):
    """
    Returns one hot encoding matrix in the form
            A R N D C Q E G H I L K M F P S T W Y V
    pos 1
    pos 2
    pos 3
    pos 4
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
    return(matrix)

def one_hot_charge_encoding(sequence):
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
    return(matrix)

def shrake_rupley_solvent_accessibility(structure,mode="residue"):
    structure = mdtraj.load(structure)
    shake_rupley_sa = mdtraj.shrake_rupley(structure,mode=mode).transpose()
    return(shake_rupley_sa)

def DSSP_threestate_simplified(structure):
    structure = mdtraj.load(structure)
    DSSP = mdtraj.compute_dssp(structure,simplified=True)
    # create the OneHotEncoder object
    encoder = OneHotEncoder(categories=[['H', 'E', 'C']])
    # reshape the input array to a 2D matrix
    arr_reshaped = np.reshape(DSSP, (-1, 1))
    # fit the encoder to the reshaped array and transform it to a one-hot encoded array
    one_hot_encoded = encoder.fit_transform(arr_reshaped).toarray()
    return(one_hot_encoded)

def compute_residue_pairwise_distance_matrix(structure,pad_to=None):
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
    return(distance_matrix)

def compute_inter_residue_unit_vectors(PDB_file,pad_to=None):
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
    return(unit_vector_matrix)

def charge_neighbourhood_from_distance_matrix(sequence,distance_cutoff,distance_matrix = None,protein_index_in_matrix=None):
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
    return(neighbourhood_charge_matrix)

def pdb_to_pandas(pdb_file):
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
    return(df)

def pandas_to_pdb(pandas_dataframe,pdb_file_to_write):
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
    pdb_df = pdb_to_pandas(pdb_file)
    residue_list = pdb_df['Residue sequence number'].unique()
    for x in range(0,len(residue_list)):
        pdb_df.loc[pdb_df['Residue sequence number'] == residue_list[x], 'Temperature factor'] = new_beta_factor_list[x]
    pandas_to_pdb(pdb_df,pdb_file_to_write)