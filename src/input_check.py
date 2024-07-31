
import os
import json
import sys
from Bio import SeqIO, PDB
from Bio.PDB.Polypeptide import protein_letters_3to1
from logging import Logger

from utils.logger_setup import configure_logger

# -----------------------------------------------------------------------------
# Sequence input from FASTA file(s) -------------------------------------------
# -----------------------------------------------------------------------------

def seq_input_from_fasta(fasta_file_path: str, use_names: bool = True, logger: Logger | None = None):
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
    if logger is None:
        logger = configure_logger()(__name__)
    
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
        prot_lens.append(len(record.seq))                   # Length
    
    # Calculate the number of proteins
    prot_N = len(prot_IDs)
        
    # Progress
    logger.info(f"INITIALIZING: extracting data from {fasta_file_path}")

    # Use names?
    if use_names:
        prot_IDs_backup = prot_IDs
        prot_IDs = prot_names
        prot_names = prot_IDs_backup
        
    return prot_IDs, prot_names, prot_seqs, prot_lens, prot_N

# -----------------------------------------------------------------------------
# Extract the sequences from AF2 PDB file(s) ----------------------------------
# -----------------------------------------------------------------------------

# Extracts the sequence of each chain of a PDB and returns a dictionary with
# keys as chain letters 
def extract_sequence_from_PDB_atoms(pdb_file: str):
    """Takes in a PDB file path and returns a list of all the protein sequences,
    one for each protein chain.

    Args:
        pdb_file (str): path to a PDB file

    Returns:
        list: list of protein sequences in the PDB file
    """    
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


def remove_duplicate_predictions(all_pdb_data: dict, logger: Logger | None = None):
    '''
    Analyzes all_pdb_data to search for predictions that are equivalent (e.g., when
    the order of the proteins where switched). It modifies it by removing the last 
    encountered duplicates. Gives a warning for each encountered duplicate.
    '''
    if logger is None:
        logger = configure_logger()(__name__)
    
    seen_sequences = set()

    for path in list(all_pdb_data.keys()):
        sequences = [data['sequence'] for data in all_pdb_data[path].values()]
        sorted_sequences = tuple(sorted(sequences))
        
        if sorted_sequences in seen_sequences:
            logger.warning(f"Duplicate prediction found and removed: {path}")
            del all_pdb_data[path]
        else:
            seen_sequences.add(sorted_sequences)
    

def extract_seqs_from_AF2_PDBs(AF2_2mers: str, AF2_Nmers: str = None, logger: Logger | None = None):
    '''Extract the sequence of each PDB files in the above folders 
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
    if logger is None:
        logger = configure_logger()(__name__)
    
    if AF2_Nmers is not None:
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
    
    
    # Dict to store AF2_prediction folder, chains, sequences and lengths (nested dicts)
    all_pdb_data = {}
    
    # Find all PDB files in all folders
    logger.info("Finding all rank1 PDB files in AF2 prediction folders...")
    for folder in folders_to_search:
        pdb_files_in_folder = find_pdb_files(folder)
        all_pdb_files.extend(pdb_files_in_folder)
    logger.info(f"   - Number of rank1 PDB files found: {len(all_pdb_files)}")
    
    # Extract the sequence from each PDB and each file chain and save it as a dict
    # with AF2 model folder as key
    logger.info("Extracting protein sequences of each PDB chain...")
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
    
    remove_duplicate_predictions(all_pdb_data, logger = logger)

    return all_pdb_data

# -----------------------------------------------------------------------------
# Merge sequences from FASTA file with PDB data ------------------------------
# -----------------------------------------------------------------------------

def get_unique_pdb_sequences(all_pdb_data: dict):
    '''
    Takes in all_pdb_data dict and returns the unique amino acid sequences 
    in all the PDB files.
    '''
    # Get unique sequences in PDB files
    PDB_sequences = list(set([chain_data["sequence"] for prediction in all_pdb_data.values() for chain_data in prediction.values()]))
    return PDB_sequences


# Check for errors
def compare_sequences(prot_seqs: list, PDB_sequences: list, logger: Logger | None = None):

    if logger is None:
        logger = configure_logger()(__name__)

    # Progress
    logger.info("Detected proteins:")
    logger.info(f"    - FASTA file    : {len(prot_seqs)}")
    logger.info(f"    - PDB files     : {len(PDB_sequences)}")

    # Check if Nº of sequences are not equal
    if len(prot_seqs) != len(PDB_sequences):

        logger.error("Unique Nº of sequences in prot_seqs: %d", len(prot_seqs))
        logger.error("Unique Nº of sequences in PDB_sequences: %d", len(PDB_sequences))
        
        logger.error("Unique sequences in FASTA: %s", prot_seqs)
        logger.error("Unique sequences in PDBs: %s", PDB_sequences)

        raise ValueError("Number of detected unique sequences in PDB files are not equal than those on FASTA file")

    # Check the correspondence between sequences in FASTA and PDBs
    if sorted(prot_seqs) != sorted(PDB_sequences):

        logger.error("Sorted unique sequences in FASTA: %s", sorted(prot_seqs))
        logger.error("Sorted unique sequences in PDBs: %s", sorted(PDB_sequences))

        logger.error("MultimerMapper is unable to manage protein domains separately")
        logger.error("This will be revised in a future version")

        raise ValueError("At least one detected unique sequence is different than those on FASTA file")
    

def merge_fasta_with_PDB_data(all_pdb_data: dict, prot_IDs: list, prot_names: list,
                               prot_seqs: list, prot_lens: list, prot_N: int, use_names: bool,
                               logger: Logger | None = None):
    '''
    This part combines the data extracted from the PDBs and the data extracted
    from the FASTA file. Modifies all_pdb_data dict
    '''

    if logger is None:
        logger = configure_logger()(__name__)
    
    # Get unique sequences in PDB files
    PDB_sequences = get_unique_pdb_sequences(all_pdb_data)

    # Check if there is any inconsistency with the provided data
    compare_sequences(prot_seqs, PDB_sequences, logger = logger)

    # Merge sequences from FASTA file with PDB data
    for model_folder, chain_data in all_pdb_data.items():
        for chain_id, data in chain_data.items():
            sequence = data["sequence"]
    
            # Check if the sequence matches any sequence from the FASTA file
            for i, fasta_sequence in enumerate(prot_seqs):
                if sequence == fasta_sequence:
                    # Add protein_ID to the existing dictionary
                    data["protein_ID"] = prot_IDs[i]

    logger.info("")
    for i in range(prot_N):
        logger.info(f"Protein number: {i+1}")
        logger.info(f"    ID      : {prot_IDs[i]}")
        logger.info(f"    Name    : {prot_names[i]}")
        logger.info(f"    Seq     : {prot_seqs[i]}")
        logger.info(f"    L       : {prot_lens[i]}")
    logger.info("")



def check_graph_resolution_preset(graph_resolution_preset: str,
                                  prot_IDs: list,
                                  logger: Logger):
    

    if not os.path.isfile(os.path.expanduser(graph_resolution_preset)):
        logger.warning(f"File not found: {graph_resolution_preset}")
        choice = input("Do you want to provide a new file path? (1)\n Skip domain detection algorithm using preset? (2)\n Exit? (3)\n Enter 1, 2 or 3: ")
        
        if choice == '1':
            graph_resolution_preset = input("Please provide the new file location: ")
            logger.warning(f"New file location: {graph_resolution_preset}")
            graph_resolution_preset = check_graph_resolution_preset(graph_resolution_preset, prot_IDs, logger)
            return graph_resolution_preset
            
        elif choice == '2':
            logger.warning("Skipping domain detection using graph_resolution_preset.")
            graph_resolution_preset = None
            return graph_resolution_preset
        
        elif choice == '3':
            logger.info("Exiting MultimerMapper.")
            sys.exit()

        else:
            logger.warning("Invalid choice. Skipping domain detection using graph_resolution_preset.")
            graph_resolution_preset = None
            return graph_resolution_preset
    
    # Read the json file
    with open(graph_resolution_preset, 'r') as json_file:
        graph_resolution_preset_dict = json.load(json_file)

    # Check if all proteins have a preset value
    for protein_ID in prot_IDs:
        if protein_ID not in graph_resolution_preset_dict:
            logger.error(f'The protein ID {protein_ID} have no resolution preset in {graph_resolution_preset}.')
            logger.error( 'Please provide a proper graph resolution preset file or recompute it.')
            logger.error( "Exiting MultimerMapper.")
            sys.exit()

    return graph_resolution_preset


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# For debugging
def main():

    import argparse

    parser = argparse.ArgumentParser(
        description='Process sequences from FASTA and PDB files.')
    parser.add_argument(
        'fasta_file_path', type=str,
        help='Path to the input FASTA file')
    parser.add_argument(
        'AF2_2mers', type=str,
        help='Path to the directory containing AF2 2mers PDB files')
    parser.add_argument(
        '--AF2_Nmers', type=str, default=None, 
        help='Path to the directory containing AF2 Nmers PDB files (optional)')
    parser.add_argument('--use_names', action='store_true',
                        help='Use protein names instead of IDs')
    
    args = parser.parse_args()
    
    # Use names?
    use_names = args.use_names

    # FASTA file
    fasta_file_path = args.fasta_file_path
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
        fasta_file_path, use_names)
    
    # Use names?
    if args.use_names:
        prot_IDs_backup = prot_IDs
        prot_IDs = prot_names
        prot_names = prot_IDs_backup
    
    # PDB files
    #all_pdb_data = extract_seqs_from_AF2_PDBs(args.AF2_2mers, args.AF2_Nmers)
    all_pdb_data = extract_seqs_from_AF2_PDBs(args.AF2_2mers)

    # Combine the data from both 
    merge_fasta_with_PDB_data(all_pdb_data = all_pdb_data,
                              prot_IDs = prot_IDs,
                              prot_names = prot_names, 
                              prot_seqs = prot_seqs,
                              prot_lens = prot_lens,
                              prot_N = prot_N,
                              use_names = use_names)


if __name__ == "__main__":
    main()