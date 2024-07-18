
import os
from Bio import SeqIO, PDB
from Bio.PDB.Polypeptide import protein_letters_3to1

# -----------------------------------------------------------------------------
# Sequence input from FASTA file(s) -------------------------------------------
# -----------------------------------------------------------------------------

def seq_input_from_fasta(fasta_file_path, use_names = True):
    '''
    This part takes as input a fasta file with the IDs and sequences of each 
    protein that potentially forms part of the complex.
    FASTA format:
        
        >Protein_ID1|Protein_name1
        MASCPTTDGVL
        >Protein_ID2|Protein_name2
        MASCPTTSCLSTAS
        ...
    
    Returns:
        prot_IDs, prot_names, prot_seqs, prot_lens, prot_N
        (list)    (list)      (list)     (list)     (int)
    '''
    
    # Initialize empty lists to store features of each protein
    prot_IDs = []
    prot_names = []
    prot_seqs = []
    prot_lens = []
    
    # Parse the FASTA file and extract information from header (record.id)
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        prot_IDs.append(str(record.id).split("|")[0])       # ID
        prot_names.append(str(record.id).split("|")[1])     # Name
        prot_seqs.append(str(record.seq))                   # Sequence
        prot_lens.append(len(record.seq))                    # Length
    
    # Calculate the number of proteins
    prot_N = len(prot_IDs)
        
    # Progress
    print(f"INITIALIZING: extracting data from {fasta_file_path}")
        
    return prot_IDs, prot_names, prot_seqs, prot_lens, prot_N

# -----------------------------------------------------------------------------
# Extract the sequences from AF2 PDB file(s) ----------------------------------
# -----------------------------------------------------------------------------

def extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers = None):
    '''
    This parts extract the sequence of each PDB files in the above folders 
    (AF2-2mers and AF2-Nmers) and stores it in memory as a nested dict.
    The resulting dictionary will have the following format:
        
        {path_to_AF2_prediction_1:
         {"A":
              {sequence: MASCPTTDGVL,
               length: 11},
          "B":
              {sequence: MASCPTTSCLSTAS,
               length: 15}},
         path_to_AF2_prediction_2:
         {"A": ...}
        }
    
    '''
    
    if AF2_Nmers != None:
        folders_to_search = [AF2_2mers, AF2_Nmers]
    else:
        folders_to_search = [AF2_2mers]
    
    # List to store all PDB files
    all_pdb_files = []
    
    def find_pdb_files(root_folder):
        pdb_files = []
        
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                # Select only one PDB file for each prediction (unrelaxed and rank 1)
                if filename.endswith(".pdb") and "unrelaxed" in filename and "rank_001" in filename:
                    pdb_files.append(os.path.join(foldername, filename))
        
        return pdb_files
    
    # Extracts the sequence of each chain of a PDB and returns a dictionary with
    # keys as chain letters 
    def extract_sequence_from_PDB_atoms(pdb_file):
        structure = PDB.PDBParser(QUIET=True).get_structure("protein", pdb_file)
        model = structure[0]  # Assuming there is only one model in the structure
    
        sequences = {}
        for chain in model:
            chain_id = chain.id
            sequence = ""
            for residue in chain:
                if PDB.is_aa(residue):
                    sequence += protein_letters_3to1[residue.get_resname()]
            sequences[chain_id] = sequence
    
        return sequences
    
    # Dict to store AF2_prediction folder, chains, sequences and lengths (nested dicts)
    all_pdb_data = {}
    
    # Find all PDB files in all folders
    print("Finding all rank1 PDB files in AF2 prediction folders...")
    for folder in folders_to_search:
        pdb_files_in_folder = find_pdb_files(folder)
        all_pdb_files.extend(pdb_files_in_folder)
    print(f"   - Number of rank1 PDB files found: {len(all_pdb_files)}")
    
    # Extract the sequence from each PDB and each file chain and save it as a dict
    # with AF2 model folder as key
    print("Extacting protein sequences of each PDB chain...")
    for pdb_file_path in all_pdb_files:
        sequences = extract_sequence_from_PDB_atoms(pdb_file_path)
        model_folder = os.path.split(pdb_file_path)[0]
    
        # Save the sequence, chain ID, and length into a nested dict
        for chain_id, sequence in sequences.items():
            # Create or update the outer dictionary for the model folder
            model_data = all_pdb_data.setdefault(model_folder, {})
            
            # Create or update the inner dictionary for the chain ID
            model_data[chain_id] = {
                "sequence": sequence,
                "length": len(sequence)
            }
            
    return all_pdb_data

# -----------------------------------------------------------------------------
# Merge sequences from FASTA file with PDB data ------------------------------
# -----------------------------------------------------------------------------

def merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs):
    '''
    This part combines the data extracted from the PDBs and the data extracted
    from the FASTA file. Modifies all_pdb_data dict
    '''
    
    # Merge sequences from FASTA file with PDB data
    for model_folder, chain_data in all_pdb_data.items():
        for chain_id, data in chain_data.items():
            sequence = data["sequence"]
    
            # Check if the sequence matches any sequence from the FASTA file
            for i, fasta_sequence in enumerate(prot_seqs):
                if sequence == fasta_sequence:
                    # Add protein_ID to the existing dictionary
                    data["protein_ID"] = prot_IDs[i]
    
    # # Print the updated all_pdb_data dictionary (debug)
    # print(json.dumps(all_pdb_data, indent=4))
