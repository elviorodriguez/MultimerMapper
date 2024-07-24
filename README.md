# MultimerMapper

# Run test to verify it works


```
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
