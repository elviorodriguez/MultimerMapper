import pandas as pd
import numpy as np
from Bio.PDB import Chain, Superimposer
from Bio.PDB.Polypeptide import protein_letters_3to1

def add_domain_RMSD_against_reference(graph, domains_df, sliced_PAE_and_pLDDTs,
                                        pairwise_2mers_df, pairwise_Nmers_df,
                                        domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff):
    
    hydrogens = ('H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 
                    'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 
                    'HZ3', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HZ', 'HD1',
                    'HE1', 'HD11', 'HD12', 'HD13', 'HG', 'HG1', 'HD21', 'HD22', 'HD23',
                    'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HE21', 'HE22',
                    'HE2', 'HH', 'HH2')
    
    def create_model_chain_from_residues(residue_list, model_id=0, chain_id='A'):

        # Create a Biopython Chain
        chain = Chain.Chain(chain_id)

        # Add atoms to the chain
        for residue in residue_list:
            chain.add(residue)
            
        return chain

    def calculate_rmsd(chain1, chain2, trimming_RMSD_plddt_cutoff):
        # Make sure both chains have the same number of atoms
        if len(chain1) != len(chain2):
            raise ValueError("Both chains must have the same number of atoms.")

        # Initialize the Superimposer
        superimposer = Superimposer()

        # Extract atom objects from the chains (remove H atoms)
        atoms1 = [atom for atom in list(chain1.get_atoms()) if atom.id not in hydrogens]
        atoms2 = [atom for atom in list(chain2.get_atoms()) if atom.id not in hydrogens]
        
        # Check equal length
        if len(atoms1) != len(atoms2):
            raise ValueError("Something went wrong after H removal: len(atoms1) != len(atoms2)")
        
        # Get indexes with lower than trimming_RMSD_plddt_cutoff atoms in the reference 
        indices_to_remove = [i for i, atom in enumerate(atoms1) if atom.bfactor is not None and atom.bfactor < domain_RMSD_plddt_cutoff]
        
        # Remove the atoms
        for i in sorted(indices_to_remove, reverse=True):
            del atoms1[i]
            del atoms2[i]
            
        # Check equal length after removal
        if len(atoms1) != len(atoms2):
            raise ValueError("Something went wrong after less than pLDDT_cutoff atoms removal: len(atoms1) != len(atoms2)")

        # Set the atoms to the Superimposer
        superimposer.set_atoms(atoms1, atoms2)

        # Calculate RMSD
        rmsd = superimposer.rms

        return rmsd
    
    def get_graph_protein_pairs(graph):
        graph_pairs = []
        
        for edge in graph.es:
            prot1 = edge.source_vertex["name"]
            prot2 = edge.target_vertex["name"]
            
            graph_pairs.append((prot1,prot2))
            graph_pairs.append((prot2,prot1))
            
        return graph_pairs
    
    print("Computing domain RMSD against reference and adding it to combined graph.")
    
    # Get all pairs in the graph
    graph_pairs = get_graph_protein_pairs(graph)
    
    # Work protein by protein
    for vertex in graph.vs:
        
        protein_ID = vertex["name"]
        ref_structure = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"]
        ref_residues = list(ref_structure.get_residues())
        
        # Add sub_domains_df to vertex
        vertex["domains_df"] = domains_df.query(f'Protein_ID == "{protein_ID}"').filter(["Domain", "Start", "End", "Mean_pLDDT"])
        
        # Initialize dataframes to store RMSD
        columns = ["Domain","Model","Chain", "Mean_pLDDT", "RMSD"]
        vertex["RMSD_df"] = pd.DataFrame(columns = columns)
        
        print(f"   - Computing RMSD for {protein_ID}...")
        
        # Work domain by domain
        for D, domain in domains_df.query(f'Protein_ID == "{protein_ID}"').iterrows():
            
            
            # Do not compute RMSD for disordered domains
            if domain["Mean_pLDDT"] < domain_RMSD_plddt_cutoff:
                continue
            
            # Start and end indexes for the domain
            start = domain["Start"] - 1
            end = domain["End"] - 1
            domain_num = domain["Domain"]
            
            # Create a reference chain for the domain (comparisons are made against it)
            ref_domain_chain = create_model_chain_from_residues(ref_residues[start:end])
            
            # Compute RMSD for 2-mers models that are part of interactions (use only rank 1)
            for M, model in pairwise_2mers_df.query(f'(protein1 == "{protein_ID}" | protein2 == "{protein_ID}") & rank == 1').iterrows():
                
                prot1 = str(model["protein1"])
                prot2 = str(model["protein2"])
                
                model_proteins = (prot1, prot2)
                
                # If the model does not represents an interaction, jump to the next one
                if (prot1, prot2) not in graph_pairs:
                    continue
                
                # Work chain by chain in the model
                for query_chain in model["model"].get_chains():
                    query_chain_ID = query_chain.id
                    query_chain_seq = "".join([protein_letters_3to1[res.get_resname()] for res in query_chain.get_residues()])
                    
                    # Compute RMSD only if sequence match
                    if query_chain_seq == sliced_PAE_and_pLDDTs[protein_ID]["sequence"]:
                        
                        query_domain_residues = list(query_chain.get_residues())
                        query_domain_chain = create_model_chain_from_residues(query_domain_residues[start:end])
                        query_domain_mean_pLDDT = np.mean([list(res.get_atoms())[0].get_bfactor() for res in query_domain_chain.get_residues()])
                        query_domain_RMSD = calculate_rmsd(ref_domain_chain, query_domain_chain, domain_RMSD_plddt_cutoff)
                        
                        query_domain_RMSD_data = pd.DataFrame({
                            "Domain": [domain_num],
                            "Model": [model_proteins],
                            "Chain": [query_chain_ID],
                            "Mean_pLDDT": [round(query_domain_mean_pLDDT, 1)],
                            "RMSD": [round(query_domain_RMSD, 2)] 
                            })
                        
                        vertex["RMSD_df"] = pd.concat([vertex["RMSD_df"], query_domain_RMSD_data], ignore_index = True)
            
            
            # Compute RMSD for N-mers models that are part of interactions (use only rank 1)
            for M, model in pairwise_Nmers_df.query(f'(protein1 == "{protein_ID}" | protein2 == "{protein_ID}") & rank == 1').iterrows():
                
                prot1 = model["protein1"]
                prot2 = model["protein2"]
                
                model_proteins = tuple(model["proteins_in_model"])
                
                # If the model does not represents an interaction, jump to the next one
                if (prot1, prot2) not in graph_pairs:
                    continue
                
                # Work chain by chain in the model
                for query_chain in model["model"].get_chains():
                    query_chain_ID = query_chain.id
                    query_chain_seq = "".join([protein_letters_3to1[res.get_resname()] for res in query_chain.get_residues()])
                    
                    # Compute RMSD only if sequence match
                    if query_chain_seq == sliced_PAE_and_pLDDTs[protein_ID]["sequence"]:
                        
                        query_domain_residues = list(query_chain.get_residues())
                        query_domain_chain = create_model_chain_from_residues(query_domain_residues[start:end])
                        query_domain_mean_pLDDT = np.mean([list(res.get_atoms())[0].get_bfactor() for res in query_domain_chain.get_residues()])
                        query_domain_RMSD = calculate_rmsd(ref_domain_chain, query_domain_chain, domain_RMSD_plddt_cutoff)
                        
                        query_domain_RMSD_data = pd.DataFrame({
                            "Domain": [domain_num],
                            "Model": [model_proteins],
                            "Chain": [query_chain_ID],
                            "Mean_pLDDT": [round(query_domain_mean_pLDDT, 1)],
                            "RMSD": [round(query_domain_RMSD, 2)]
                            })
                        
                        vertex["RMSD_df"] = pd.concat([vertex["RMSD_df"], query_domain_RMSD_data], ignore_index = True)

    # remove duplicates
    for vertex in graph.vs:
        vertex["RMSD_df"] = vertex["RMSD_df"].drop_duplicates().reset_index(drop = True)

