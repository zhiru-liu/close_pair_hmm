# Bacteroides fragilis CP-HMM Test

This folder mirrors `examples/example1` for `Bacteroides_fragilis_54507`.
It uses the published LiuGood2024 SNV helper from:

`/Users/Device6/Documents/Research/bgoodlab/LiuGood2024_data`

The SNV catalog species and CP-HMM prior species are both
`Bacteroides_fragilis_54507`. The prior is generated into this folder so this
test does not reuse the Tsimane-project `Bacteroides_fragilis.csv` prior.

Generate the species-specific prior used by this test:

```bash
/Users/Device6/miniforge3/envs/dNdS_310/bin/python Bf_test/generate_prior.py --num-samples 5000 --save-samples
```

The prior is written to `Bf_test/priors/Bacteroides_fragilis_54507.csv`.

Run a smoke test:

```bash
/Users/Device6/miniforge3/envs/dNdS_310/bin/python Bf_test/infer_Bf.py --max-pairs 1
```

Run the full published close-pair set:

```bash
/Users/Device6/miniforge3/envs/dNdS_310/bin/python Bf_test/infer_Bf.py
```

Outputs are written to `Bf_test/results/`, including a copy of the published
ground-truth rows for this species and simple event-count / interval-overlap
comparison tables.
