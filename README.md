# MultimerMapper

# Run test to verify it works

First without N-mers
```
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --out_path tests/output_2mers --use_names --overwrite
```

And then including N-mers
```
python multimer_mapper.py tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta tests/EAF6_EPL1_PHD1/2-mers --AF2_Nmers tests/EAF6_EPL1_PHD1/N-mers --out_path tests/output_Nmers --use_names --overwrite
```