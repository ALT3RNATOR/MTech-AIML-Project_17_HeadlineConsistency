# Data

`sample_data.csv` is a tiny synthetic schema example only; it is not used to compute project metrics. The full cleaned FNC-1 source CSVs are not bundled. Place them in the path configured as `source_dir` in the first section of `notebooks/experiment.ipynb`.

The completed challenge-review CSV is a local input and is ignored by Git because it contains news text. For a fresh checkout, supply the manually reviewed file at `data/challenge_headline_review_completed.csv`. The notebook generates fit/validation/official-test splits and challenge candidates.
