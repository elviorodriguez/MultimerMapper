![image](https://github.com/user-attachments/assets/a71fc1ea-eaaf-44db-baa3-3c78d16de612)

# What is MultimerMapper?
It is a computational tool for the integration, analysis and visualization of AlphaFold interaction landscapes.

# What can I do with MultimerMapper?


# Installation
MultimerMapper requires Anaconda/Miniconda (Miniconda installation guide: https://docs.anaconda.com/miniconda/miniconda-install) to create an isolated environment for the software.
To install it, clone the repo in the directory you want, cd to the repo and install the environment using conda:
```sh
# Clone the repo
git clone https://github.com/elviorodriguez/MultimerMapper.git

# Go to repo dir
cd MultimerMapper

# Create "MultimerMapper" environment
conda env create -f environment.yml
```
Every time you want to use MultimerMapper, activate the environment with the following command:
```sh
# Activate MultimerMapper env to run the pipeline
conda activate MultimerMapper
```

# Verify installation
You can verify the installation as follows.

```sh
# Activate MultimerMapper env to run the pipeline
conda activate MultimerMapper

# Take a look at the usage (this must give no errors)
python multimer_mapper.py -h
```


```
# Only 2-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --out_path tests/output_2mers --use_names --overwrite

# Both 2-mers and N-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --AF2_Nmers tests/EAF6_EPL1_PHD1/N-mers --out_path tests/output_Nmers --use_names --overwrite
```


## Using manual_domains.tsv
```sh
# Only 2-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --out_path tests/output_2mers --use_names --overwrite --manual_domains tests/EAF6_EPL1_PHD1/manual_domains.tsv

# Both 2-mers and N-mers
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --AF2_Nmers tests/EAF6_EPL1_PHD1/N-mers --out_path tests/output_Nmers --use_names --overwrite --manual_domains tests/EAF6_EPL1_PHD1/manual_domains.tsv
```
