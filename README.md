![image](https://github.com/user-attachments/assets/a71fc1ea-eaaf-44db-baa3-3c78d16de612)

# What is MultimerMapper?
It is a computational tool for the integration, analysis and visualization of AlphaFold interaction landscapes.

# What can I do with MultimerMapper?


# Installation

```
# Clone the repo
git clone

# Go to repo dir
cd MultimerMapper

# Create MultimerMapper environment
conda env create -f env.yml
```

# Run test to verify it works


```
# Activate MultimerMapper env to run the pipeline
conda activate MultimerMapper

# Only 2-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --out_path tests/output_2mers --use_names --overwrite

# Both 2-mers and N-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --AF2_Nmers tests/EAF6_EPL1_PHD1/N-mers --out_path tests/output_Nmers --use_names --overwrite
```


# Using manual_domains.tsv
```
# Only 2-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --out_path tests/output_2mers --use_names --overwrite --manual_domains tests/EAF6_EPL1_PHD1/manual_domains.tsv

# Both 2-mers and N-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --AF2_Nmers tests/EAF6_EPL1_PHD1/N-mers --out_path tests/output_Nmers --use_names --overwrite --manual_domains tests/EAF6_EPL1_PHD1/manual_domains.tsv
```
