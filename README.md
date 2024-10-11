![image](https://github.com/user-attachments/assets/a71fc1ea-eaaf-44db-baa3-3c78d16de612)

# What is MultimerMapper?
It is a computational tool for the integration, analysis and visualization of AlphaFold interaction landscapes. It is presented as an innovative tool designed to help researchers understand and visualize large protein complexes easily using protein structure prediction.

It is still under active development... So, keep an eye on it.

Here's how it works:
 - **Start with protein sequences**: Input a list of protein sequences you want to study. Typically, they will come from an interactomic experiment.
 - **Generate predictions**: MultimerMapper guides you to create the most informative set of protein structure predictions using AF2-multimer (AF3 soon!).
 - **Analyze interactions**: The tool examines how proteins interact in different combinations (2-mers, 3-mers, etc.) and compares these interactions.
 - **Classify interactions**: MultimerMapper categorizes protein-protein interactions as: static (always present), dynamic positive (only activated in certain modeling contexts), dynamic negative (inhibited in different modeling context).
 - **Visualize results**: The tool creates 2D and 3D visualizations to help you answer: How proteins assemble into larger complexes? Which interactions are stable or changing? What are the potential stoichiometry of the complexes?
 - **Iterate and refine**: MultimerMapper suggests new combinations to predict, helping you explore the "interaction landscape" until you have a complete picture.
 - **Gain insights**: Use the visualizations and analyses to infer the most likely structures and dynamics of your protein complexes.
By combining predictions and smart analysis, MultimerMapper helps bridge the gap between individual protein structures and complex molecular assemblies, offering new perspectives on how proteins work together in cells.

# What can I do with MultimerMapper?
- Automatic/semi-automatic/manual domain detection
- Protein-protein interaction (PPI) prediction
- PPI dynamics capturing
- Protein dynamics capturing
- Interactive PPI graph (2D) generation
- Residue-residue contact (RRC) detection
- Interactive RRC graph (3D) generation (BETA)
- Interaction surface mapping (BETA)
- Metrics clustering analysis
- Pseudo-molecular dynamics trajectories generation (RMSD trajectories)
- Partner enrichment
- Interface with combinatorial assembler
- Much more comming!

# How does it works?
The software can perform several tasks by analyzing AF2-multimer (AF3 comming soon) predictions ensembles of different combinations of proteins separated in two different directories:

    ```2-mers```: contains all possible pairwise interaction predictions of the set of proteins (homo-2-mers and hetero-2-mers)
    ```N-mers (optional)```: contains diifferent combinations (3-mers, 4-mers, etc.) interaction predictions of the set of proteins that 
    
It maps all protein-protein interactions present in the predictions

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

## Add multimer_mapper alias (optional)
You can add an alias to run MultimerMapper by adding the following to your ```.bashrc``` file:
```sh
# Replace <user_name> with your user_name and <path_to_MM> with the repository path
alias multimer_mapper="python /home/<user_name>/<path_to_MM>/multimer_mapper.py"
```
Restart the shell and you will be able to call MultimerMapper using ```multimer_mapper``` as a shell command.
```
# Display multimer_mapper help message
multimer_mapper -h
```

# Verify installation
You can verify the installation as follows:

```sh
# Activate MultimerMapper env to run the pipeline
conda activate MultimerMapper

# Take a look at the usage (this must give no errors)
python multimer_mapper.py -h
```
There is a testing dataset composed of three trypanosomatid proteins (EAF6, EPL1 and PHD1) with all possible 2-mers and N-mers combinations that reached convergence. First run the pipeline only with 2-mers, and take a look at the output.

```sh
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


# Exploring pLDDT clusters, RMSF and RMSD trajectories

## 
Let's look at the example pLDDT clusters of PHD1 (tests/expected_output/RMSF_and_plddt_clusters/PHD1/PHD1-pLDDT_clusters.png):

![PHD1-pLDDT_clusters](https://github.com/user-attachments/assets/63f7abc0-68d2-4532-b74b-9c4b89e51607)

We can see 2 clusters. The main difference between them is in the domain 6. If we open the metadata TSV file

## Visualization of RMSD trajectories
In ChimeraX, open the sample weighted RMSD trajectory located in tests/expected_output/monomer_trajectories/PHD1/PHD1_domain_6
```sh
# Re-align the models to the lowest RMSD model (it will always be #1.1)
mm #1.2-1000 to #1.1

# Apply AlphaFold pLDDT color scheme on the bfactor
color bfactor palette alphafold

# Open the slider on the model #1 series
mseries slider #1
```
You can play/stop the slider, slow/accelerate it down using the buttons on the bottom-right slider. Once you find a good pose, you can save a video with the red record button:


https://github.com/user-attachments/assets/064b9c0c-820a-49c7-94f6-560877e95440

# Do you want to combine MultimerMapper with your own pipelines programatically?
Have a look at devs secction (for developers). There you will find explanaitions of MultimerMapper's main functionalities output data structures and their meaning.
