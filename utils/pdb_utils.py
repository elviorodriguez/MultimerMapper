
import os
import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain
from Bio.SeqUtils import seq1
from Bio import PDB
from scipy.spatial.transform import Rotation

from src.coordinate_analyzer import plot_traj_metadata

def get_chain_sequence(chain: Chain):
    
    chain_seq = seq1(''.join(residue.resname for residue in chain))
    
    return chain_seq

def get_domain_atoms(chain, start, end):
    return [atom for atom in chain.get_atoms() if atom.name == 'CA' and start <= atom.get_parent().id[1] <= end]


def get_domain_atoms_for_pdockq(chain, start, end):
    domain_atoms = [atom for atom in chain.get_atoms() if atom.name == 'CB' or (atom.name == 'CA' and atom.get_parent().resname == 'GLY')]
    domain_atoms = [atom for atom in domain_atoms if start <= atom.get_parent().id[1] <= end]
    
    coords = np.array([atom.coord for atom in domain_atoms])
    plddt = np.array([atom.bfactor for atom in domain_atoms])
    
    return coords, plddt


# Computes 3D distance
def calculate_distance(coord1, coord2):
    """
    Calculates and returns the Euclidean distance between two 3D coordinates.
    
    Parameters:
        - coord1 (list/array): xyz coordinates 1.
        - coord2 (list/array): xyz coordinates 2.
    
    Returns
        - distance (float): Euclidean distance between coord1 and coord2
    """
    return np.sqrt(np.sum((np.array(coord2) - np.array(coord1))**2))


# ----------- Helper classes for selecting chains and domains -----------------

class ChainSelect(PDB.Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        return chain.id == self.chain.id
    
class DomainSelect(PDB.Select):
    def __init__(self, chain, start, end):
        self.chain = chain
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        return (residue.get_parent().id == self.chain.id and 
                self.start <= residue.id[1] <= self.end)
    
# ---------------------------------------------------------------------------



def kabsch_rmsd(P, Q):
    """
    Calculates the RMSD between two sets of points using the Kabsch algorithm.
    """
    P = P - np.mean(P, axis=0)
    Q = Q - np.mean(Q, axis=0)

    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)
    
    return np.sqrt(np.sum(np.square(P - np.dot(Q, R.T))) / len(P))




# # Helper fx to save trajectory file and metadata
# def save_pairwise_trajectory(sorted_indices, protein_ID, filename_suffix, protein_trajectory_folder,
                    
#                              RMSDs, mean_pLDDTs, CM_dist,

#                              aligned_chains, all_chain_types, all_chain_info, domain_start=None, domain_end=None):
    
#     trajectory_file = os.path.join(protein_trajectory_folder, f'{protein_ID}_{filename_suffix}_traj.pdb')
    
#     df_cols = ['Traj_N', 'Type', 'Are_chains', 'Rank', 'Model', 'RMSD', 'pLDDT', 'CM_dist']
#     trajectory_df = pd.DataFrame(columns=df_cols)
    
#     io = PDB.PDBIO()
#     with open(trajectory_file, 'w') as f1:
#         for i, idx in enumerate(sorted_indices):
#             chain = aligned_chains[idx]
#             chain_type = all_chain_types[idx]
#             chain_info = all_chain_info[idx]
#             model_name = f"MODEL_{i+1}_{chain_type}_{chain_info[0]}_{'-'.join(map(str, chain_info[1]))}_{chain_info[2]}"
            
#             model_data = pd.DataFrame({
#                 "Traj_N"  : [i+1], 
#                 "Type"    : [chain_type],
#                 "Is_chain": [chain_info[0]],
#                 "Rank"    : [chain_info[2]],
#                 "Model"   : ['__vs__'.join(map(str, chain_info[1]))],
#                 "RMSD"    : [RMSDs[idx]],
#                 "pLDDT"   : [mean_pLDDTs[idx]],
#                 "CM_dist" : [CM_dist[idx]]
#                 })
#             trajectory_df = pd.concat([trajectory_df, model_data], ignore_index=True)
            
#             io.set_structure(chain.parent)
#             if domain_start is not None and domain_end is not None:
#                 io.save(f1, select=DomainSelect(chain, domain_start, domain_end), write_end=False)
#             else:
#                 io.save(f1, select=ChainSelect(chain), write_end=False)
#             f1.write(f"ENDMDL\nTITLE     {model_name}\n")
    
#     # Generate some plots
#     plot_traj_metadata(metadata_list = trajectory_df['RMSD'],
#                        metadata_type = "RMSD",
#                        protein_trajectory_folder = protein_trajectory_folder,
#                        protein_ID = protein_ID, filename_suffix = filename_suffix)
#     plot_traj_metadata(metadata_list = trajectory_df['pLDDT'],
#                        metadata_type = "Mean pLDDT",
#                        protein_trajectory_folder = protein_trajectory_folder,
#                        protein_ID = protein_ID, filename_suffix = filename_suffix)
#     plot_traj_metadata(metadata_list = trajectory_df['ROG'],
#                        metadata_type = "CM_dist",
#                        protein_trajectory_folder = protein_trajectory_folder,
#                        protein_ID = protein_ID, filename_suffix = filename_suffix)
    
#     trajectory_df_file = os.path.join(protein_trajectory_folder, f'{protein_ID}_{filename_suffix}_traj.tsv')
#     trajectory_df.to_csv(trajectory_df_file, sep="\t", index=False)
    
#     return trajectory_file



import matplotlib.pyplot as plt

class DomainSelect2(PDB.Select):
    def __init__(self, chain_id, start, end):
        self.chain_id = chain_id
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        return (residue.parent.id == self.chain_id and 
                self.start <= residue.id[1] <= self.end)
    

def plot_traj_metadata2(metadata_list, metadata_type, pairwise_traj_dir, P1_ID, P2_ID, filename_suffix):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metadata_list) + 1), metadata_list)
    plt.title(f'{metadata_type} vs Trajectory Number')
    plt.xlabel('Trajectory Number')
    plt.ylabel(metadata_type)
    plt.savefig(os.path.join(pairwise_traj_dir, f'{P1_ID}_{P2_ID}_{filename_suffix}_{metadata_type.replace(" ", "_")}.png'))
    plt.close()

def save_pairwise_trajectory(sorted_indices, P1_ID, P2_ID, filename_suffix, pairwise_traj_dir,
                             pairwise_domains_traj_dict, P1_dom_start, P1_dom_end, P2_dom_start, P2_dom_end):
    
    trajectory_file = os.path.join(pairwise_traj_dir, f'{P1_ID}_{P2_ID}_{filename_suffix}_traj.pdb')
    df_cols = ['Traj_N', 'Type', 'Model', 'Rank', 'RMSD', 'pDockQ', 'Mean_pLDDT', 'CM_dist']
    trajectory_df = pd.DataFrame(columns=df_cols)

    io = PDB.PDBIO()
    with open(trajectory_file, 'w') as f1:
        for i, idx in enumerate(sorted_indices):
            model = pairwise_domains_traj_dict['full_pdb_model'][idx]
            model_type = pairwise_domains_traj_dict['type'][idx]
            model_info = pairwise_domains_traj_dict['model_proteins'][idx]
            model_rank = pairwise_domains_traj_dict['rank'][idx]
            model_name = f"MODEL_{i+1}_{model_type}_{'-'.join(map(str, model_info))}_{model_rank}"

            model_data = pd.DataFrame({
                "Traj_N"    : [i+1],
                "Type"      : [model_type],
                "Model"     : ['__vs__'.join(map(str, model_info))],
                "Rank"      : [model_rank],
                "RMSD"      : [pairwise_domains_traj_dict['CM_dist'][idx]],  # Using CM_dist as RMSD for now
                "pDockQ"    : [pairwise_domains_traj_dict['full_model_pdockq'][idx]],
                "Mean_pLDDT": [pairwise_domains_traj_dict['domains_mean_plddt'][idx]],
                "CM_dist"   : [pairwise_domains_traj_dict['CM_dist'][idx]]
            })
            trajectory_df = pd.concat([trajectory_df, model_data], ignore_index=True)

            io.set_structure(model)
            io.save(f1, select=DomainSelect2('A', P1_dom_start, P1_dom_end), write_end=False)
            io.save(f1, select=DomainSelect2('B', P2_dom_start, P2_dom_end), write_end=False)
            f1.write(f"ENDMDL\nTITLE     {model_name}\n")

    # Generate plots
    plot_traj_metadata2(metadata_list=trajectory_df['RMSD'], metadata_type="RMSD",
                       pairwise_traj_dir=pairwise_traj_dir, P1_ID=P1_ID, P2_ID=P2_ID, filename_suffix=filename_suffix)
    plot_traj_metadata2(metadata_list=trajectory_df['Mean_pLDDT'], metadata_type="Mean pLDDT",
                       pairwise_traj_dir=pairwise_traj_dir, P1_ID=P1_ID, P2_ID=P2_ID, filename_suffix=filename_suffix)
    plot_traj_metadata2(metadata_list=trajectory_df['CM_dist'], metadata_type="CM_dist",
                       pairwise_traj_dir=pairwise_traj_dir, P1_ID=P1_ID, P2_ID=P2_ID, filename_suffix=filename_suffix)

    trajectory_df_file = os.path.join(pairwise_traj_dir, f'{P1_ID}_{P2_ID}_{filename_suffix}_traj.tsv')
    trajectory_df.to_csv(trajectory_df_file, sep="\t", index=False)

    return trajectory_file