
import os
import pandas as pd

from tests.convergency.convergency_test import run_complex

######################################################################
########################### Configurations ###########################
######################################################################

# (FPR, PAE_cutoff, N_models_cutoff)
cutoffs_list = [

    # ------------- Original cutoffs (DiscobaMultimer 20r) -------------

    # # FPR = 0.01
    # (0.01,  1.86, 1),
    # (0.01,  1.89, 2),
    # (0.01,  2.70, 3),
    # (0.01,  4.21, 4),
    # (0.01, 10.50, 5),
    # # FPR = 0.05
    # (0.05,  3.09, 1),
    # (0.05,  4.42, 2),
    # (0.05,  7.18, 3),
    # (0.05, 10.44, 4),
    # (0.05, 12.98, 5)

    # ------------- New cutoff (AF2m + AF3 average) -------------

    # FPR = 0.05 (AF2m)
    (0.01,  1.30, 1),
    (0.01,  1.30, 2),
    (0.01,  1.50, 3),
    (0.01,  1.56, 4),
    (0.01,  1.56, 5),
    (0.02,  1.86, 1),
    (0.02,  1.89, 2),
    (0.02,  2.70, 3),
    (0.02,  4.21, 4),
    (0.02, 10.50, 5),
    (0.05,  2.84, 1),
    (0.05,  4.34, 2),
    (0.05,  6.59, 3),
    (0.05,  9.82, 4),
    (0.05, 12.11, 5),
    (0.1 ,  3.83, 1),
    (0.1 ,  7.11, 1),
    (0.1 , 10.28, 1),
    (0.1 , 11.58, 1),
    (0.1 , 14.03, 1)

    # # FPR = 0.05 (AF2m)
    # (0.05, 9.81, 4)
]

# Homooligomers files locations
homooligomers_benchmark_dir = "/home/elvio/Desktop/homooligomeric_states_benchmark"

homooligomers_ids = ['1A7K.4','1R26.1','2P7U.1','2RM5.1','3H4C.1','4BAS.1','5AB5.2',
                     '5EUC.4','5F8V.2','6J9Q.2','7MK0.6','7S2H.2','7TUR.2','7V4J.5',
                     '7WHR.6','7WP3.3','8AXJ.1','8I40.4','8J6E.2','8SF3.1']

homo_complexes_dict = {}
for id in homooligomers_ids:
    homo_complexes_dict[id] = {
        "fasta_file" : f'{homooligomers_benchmark_dir}/fastas/{id.split(".")[0]}.fasta',
        "AF2_2mers"  : f'{homooligomers_benchmark_dir}/AF2_2mers/{id}',
        "AF2_Nmers"  : f'{homooligomers_benchmark_dir}/AF2_Nmers/{id}'
    }

# Files location
hetero_complexes_dict = {
    "3MZL": {
        "fasta_file" : '/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_3MZL/proteins_mm.fasta',
        "AF2_2mers"  : '/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_3MZL/2-mers',
        "AF2_Nmers"  : '/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_3MZL/N-mers'
    },
    "Complex_2": {
        "fasta_file" : None,
        "AF2_2mers"  : None,
        "AF2_Nmers"  : None
    },
}

# Output directory for benchmark
benchmark_dir = "/home/elvio/Desktop/stoichiometry_benchmark"
os.makedirs(benchmark_dir, exist_ok=True)

######################################################################
############################# HELPER FXs #############################
######################################################################

def run_multiple_complexes(complexes_dict, cutoffs_list, benchmark_dir):

    complexes_conv_stoichs_dict = {}

    for comp in complexes_dict.keys():

        conv_stoichs_dict = run_complex(
            fasta_file  = complexes_dict[comp]['fasta_file'],
            AF2_2mers   = complexes_dict[comp]['AF2_2mers'],
            AF2_Nmers   = complexes_dict[comp]['AF2_Nmers'],
            out_path    = f'{benchmark_dir}/{comp}',
            cutoff_list = cutoffs_list
        )

        complexes_conv_stoichs_dict[comp] = conv_stoichs_dict

    return complexes_conv_stoichs_dict


def convert_convergent_stoichs_dict_to_df(complexes_conv_stoichs_dict):
    rows = []
    
    for subject in complexes_conv_stoichs_dict:
        comp_id, expected_stoich = subject.split(".")
        
        for cutoff_key in complexes_conv_stoichs_dict[subject]:
            FPR, N_models = [k.split("=")[1] for k in cutoff_key.split("_")]
            detected = complexes_conv_stoichs_dict[subject][cutoff_key]["convergent_stoichiometries"]
            
            if detected:
                detected_stoich = len(min(detected, key=len))
            else:
                detected_stoich = 1
    
            rows.append({
                "Entities": 1,
                "Subject": subject,
                "Expected": expected_stoich,
                "FPR": FPR,
                "N_models": N_models,
                "Detected": detected_stoich
            })

    stoich_bench_df = pd.DataFrame(rows, columns=['Entities', 'Subject', 'Expected', 'FPR', 'N_models', 'Detected'])
    
    return stoich_bench_df


######################################################################
############################## Running ###############################
######################################################################

# # TESTS
# comp = "8I40.4"
# # comp = "1R26.1"

# conv_stoichs_dict = run_complex(
#     fasta_file  = homo_complexes_dict[comp]['fasta_file'],
#     AF2_2mers   = homo_complexes_dict[comp]['AF2_2mers'],
#     AF2_Nmers   = homo_complexes_dict[comp]['AF2_Nmers'],
#     out_path    = f'{benchmark_dir}/{comp}/test',
#     cutoff_list = cutoffs_list
# )

complexes_conv_stoichs_dict = run_multiple_complexes(homo_complexes_dict, cutoffs_list, benchmark_dir)

stoich_bench_df = convert_convergent_stoichs_dict_to_df(complexes_conv_stoichs_dict)

            

