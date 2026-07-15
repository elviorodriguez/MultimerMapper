![image](https://github.com/elviorodriguez/MultimerMapper/blob/main/report/multimermapper_logo.png?raw=true)

# What is MultimerMapper?
It is a computational tool designed for the generation, integration, analysis, and visualization of AlphaFold interaction landscapes based on the novel concept of "context-dependent protein structure prediction" and the implementation of a stoichiometric space exploration algorithm. It is presented as an innovative tool to help researchers understand and visualize how protein complexes behave under different modeling conditions and infer their underlying stoichiometries.

MANUSCRIPT: https://www.biorxiv.org/content/10.1101/2025.10.24.684463v1

TUTORIAL (BiotrAIn - Rosario 2026): follow the colab example [![Colab Example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elviorodriguez/BiotrAIn_MultimerMapper/blob/main/MM_notebook_BiotrAIn.ipynb), along with the course [GUIDE](https://github.com/elviorodriguez/BiotrAIn_MultimerMapper/tree/main/course_material)

Here's how it works:
 - **Start with protein sequences**: Input a list of protein sequences you want to study. Typically, they will come from an interactomic experiment.
 - **Generate predictions**: MultimerMapper guides you to create the most informative set of protein structure predictions using AF2-multimer (and now AF3 too!).
 - **Analyze interactions**: The tool examines how proteins interact in different combinations (2-mers, 3-mers, etc.) and compares these interactions.
 - **Classify interactions**: MultimerMapper categorizes protein-protein interactions as: *static* (always present), *dynamic positive* (only activated in certain modeling contexts), *dynamic negative* (inhibited in different modeling context).
 - **Visualize results**: The tool creates 2D and 3D visualizations to help you answer:
    - How proteins assemble into larger complexes?
    - Which interactions are stable or changing?
    - What are the potential stoichiometries of the underlying complexes?
 - **Iterate and refine**: At each iteration, MultimerMapper suggests new combinations to predict, helping you explore the "interaction landscape" until you have a complete picture.
 - **Gain insights**: Use the visualizations and analyses to infer the most likely structures and dynamics of your protein complexes.
By combining predictions and smart analysis, MultimerMapper helps bridge the gap between individual protein structures and complex molecular assemblies, offering new perspectives on how proteins work together in cells.

# What can I do with MultimerMapper?
- Automatic/semi-automatic/manual domain detection
- Protein-protein interaction (PPI) prediction
- PPI dynamics capturing
- Protein dynamics capturing
- Interactive PPI graph (2D) generation
- Residue-residue contact (RRC) detection
- RRC dynamics capturing
- Interactive RRC graph (3D) generation
- Interaction surface mapping
- Metrics clustering analysis
- Multivalency detection
- Contact matrix clustering analysis (Detects different binding modes)
- Stoichiometry inference (Homo-oligomerig/Binary and overall complex stoichiometries)
- Pseudo-molecular dynamics trajectories generation (RMSD pseudo-trajectories)
- Bias in Partner Distribution (BPD) analysis
- Partner enrichment
- Stoichiometric Space Exploration Algorithm
- Much more coming! (DNA, RNA, PTMs, small molecules and ions)

# How does it works?
The software can perform several tasks by analyzing AF2-multimer or AF3 prediction ensembles of different combinations of proteins separated in two different directories:

    2-mers: contains all possible pairwise interaction predictions of the set of proteins (homo-2-mers and hetero-2-mers)
    N-mers: contains different combinations (3-mers, 4-mers, etc.) interaction predictions of the set of proteins 
    
It maps all PPIs and RRCs present in the predictions by decomposing the models into their pairwise sub-components and captures how they change depending on the modelling context (which other proteins were present in the models and in which quantity). This information is converted into PPI and RRC graph representations inside MultimerMapper that you can visually explore in interactive HTML plots. The 2-mer predictions are fixed for each system, but N-mers are suggested by a stoichiometric space exploration algorithm that explores protein combinations based on interaction evidence. Each N-mer is clasified as stable or unstable based, and only stable complexes (parents) can create new derived stoichiometries (children). In general, the biggest stable complexes at each branch of the stoichiometric space match the expected stoichiometries of the underlying complexes. You can visualize this information as 3D Stoichiometric Space Plots.

# Have a look at different interactive outputs
   - [interactive PPI 2D plot](https://elviorodriguez.github.io/MultimerMapper/CPX-2369_2D_graph.html)
   - [interactive RRC 3D plot (Plotly)](https://elviorodriguez.github.io/MultimerMapper/CPX-2369_3D_graph_Plotly.html)
   - [interactive RRC 3D plot (py3Dmol)](https://elviorodriguez.github.io/MultimerMapper/CPX-2369_3D_graph_py3Dmol.html)
   - [interactive stoichiometric space plot](https://elviorodriguez.github.io/MultimerMapper/CPX-2369_stoich_space.html)

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

## Add multimer_mapper alias (Linux)
You can add an alias to run MultimerMapper by adding the following to your ```.bashrc``` file:
```sh
# Replace <user_name> with your user_name and <path_to_MM> with the repository path
alias multimer_mapper="python /home/<user_name>/<path_to_MM>/multimer_mapper.py"
```
Restart the shell and you will be able to call MultimerMapper using ```multimer_mapper``` as a shell command.
```sh
# Display multimer_mapper help message
multimer_mapper -h
```

## Add multimer_mapper alias (Mac)
You can add an alias to run MultimerMapper by adding the following to your ```.zshrc``` file:
```zsh
# Replace <user_name> with your user_name and <path_to_MM> with the repository path
alias multimer_mapper="python /Users/<user_name>/<path_to_MM>/multimer_mapper.py"
```
Restart the shell and you will be able to call MultimerMapper using ```multimer_mapper``` as a shell command.
```zsh
# Display multimer_mapper help message
multimer_mapper -h
```

# Verify installation
You can verify the installation as follows:

```sh
# Remember to activate MultimerMapper env to run the pipeline
conda activate MultimerMapper

# Take a look at the usage (this must give no errors)
python multimer_mapper.py -h

# If you have created the alias replace "python multimer_mapper.py" with the alias
multimer_mapper -h
```
There is a testing dataset composed of three trypanosomatid proteins (EAF6, EPL1 and PHD1) with all possible 2-mers and N-mers combinations that reached convergence. First run the pipeline only with 2-mers, and take a look at the output located at ```tests/output_2mers``` or ```tests/output_Nmers```, depending on which you run.

```sh
# Only 2-mers
python multimer_mapper.py --AF_2mers tests/EAF6_EPL1_PHD1/2-mers --out_path tests/output_2mers tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta

# Both 2-mers and N-mers
python multimer_mapper.py --AF_2mers tests/EAF6_EPL1_PHD1/2-mers --AF_Nmers tests/EAF6_EPL1_PHD1/N-mers --out_path tests/output_Nmers tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta
```

## Using manual_domains.tsv
We highly recommend to use the semi-automatic domain detection algorithm inside MultimerMapper to get the best results, as the pipeline relies on proper definition of compact domains to perform RMSD trajectories and capture conformational changes. However, if you know the exact start and end positions of the globular domains of your proteins, you can use a ```manuals_domains.tsv``` file to define them:
```sh
# Only 2-mers
python multimer_mapper.py --AF_2mers tests/EAF6_EPL1_PHD1/2-mers --manual_domains tests/EAF6_EPL1_PHD1/manual_domains.tsv --out_path tests/output_2mers tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta 

# Both 2-mers and N-mers
python multimer_mapper.py --AF_2mers tests/EAF6_EPL1_PHD1/2-mers --AF_Nmers tests/EAF6_EPL1_PHD1/N-mers --manual_domains tests/EAF6_EPL1_PHD1/manual_domains.tsv --out_path tests/output_Nmers tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta 
```
Note that you need to define the span of both disordered loops and globular domains. Have a look at the example file to know its format.

## Visualization of Interactive 2D PPI graphs
One of the main outputs of MultimerMapper is the interactive 2D PPI graph. You can find it inside the output folder (tests/expected_output/2D_graph.html). It represents proteins as nodes and distinct interaction modes between proteins as edges:

![image](https://github.com/user-attachments/assets/4629b19c-ad78-4d20-9e73-e9803650ce1a)

The color of the nodes stands for the dynamic classification of the protein (Static, Dynamic Negative, Dynamic Positive). Edge colors represents the classification of the interactions (Static, Dynamic Negative, Dynamic Positive) and the shape of the edge represents the intensity of the classification (solid, dash, dot). In cases in which there is no N-mers or 2-mers data, edges and proteins will be colored in an orange tone.

You can display more information about the proteins in different modelling contexts by clicking above nodes. The hovertext gives you information about detected domain spans and metrics variation (RMSD against reference and mean pLDDT) depending on the modelling context.

You can display more information about the PPIs in different modelling contexts by clicking above edges. The hovertext gives you information about detected interactions metrics in 2-mers and N-mers (pTM, ipTM, pDockQ, N_models that surpassed cutoffs), which depend on the modelling context.

## Visualization of Interactive 3D RRC graphs
Two RRC graphs are generated. One uses py3Dmol as rendering software (tests/expected_output/3D_graph_py3Dmol.html) and the other uses Plotly (tests/expected_output/3D_graph_plotly.html). Both graphs shows RRCs between proteins, their classification and the surface residue centroids involved in different interactions. However, py3Dmol gives less interactivity but better depth perspective and nicer protein backbone visualizations; while Plotly is much more interactive but gives less depth awareness. Use the one you like the most.

### py3Dmol:
![image](https://github.com/user-attachments/assets/d39026ff-63b6-48dd-b493-34539c6f4f5b)

### Plotly:
![image](https://github.com/user-attachments/assets/411e2f81-c69d-425b-aeda-f84e915a9468)

# Exploring pLDDT clusters, RMSF and RMSD trajectories

## 
Let's look at the example pLDDT clusters of PHD1 (tests/expected_output/RMSF_and_plddt_clusters/PHD1/PHD1-pLDDT_clusters.png):

![PHD1-pLDDT_clusters](https://github.com/user-attachments/assets/63f7abc0-68d2-4532-b74b-9c4b89e51607)

We can see 2 clusters. The main difference between them is in the domain 6. If we open the metadata TSV file

## Visualization of RMSD pseudo-trajectories in ChimeraX
MultimerMapper now has its own RMSD pseudo-trajectory visualizer incorporated, but you can also see it in ChimeraX. Inside ChimeraX, open the sample weighted RMSD trajectory located in tests/expected_output/monomer_trajectories/PHD1/PHD1_domain_6

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

# How to use AF3 predictions
Now AF3 predictions can be converted into a format compatible with MultimerMapper. You need to use the utility script ```utils/af3_compatibility.py``` to convert a single prediction or an entire directory containing multiple AF3 zipped predictions:

```bash
# Process a single ZIP file (with single or multiple predictions inside)
python utils/af3_compatibility.py af3_prediction.zip

# Process a directory containing multiple ZIP files
python utils/af3_compatibility.py /path/to/zip_files/ --out_dir /path/to/output/
```

If a tree of prediction sets is present in the directory (subdirectories), each prediction set gets its own subdirectory in the output folder, making it easy to organize results from multiple AF3 runs (e.g., those from each N-mer iteration of a system).


# Reference
•	Rodríguez Araya E; Serra E. Context-Dependent Protein Structure Prediction Analysis and Stoichiometry Inference with MultimerMapper. bioRxiv 2025. DOI: https://doi.org/10.1101/2025.10.24.684463.
