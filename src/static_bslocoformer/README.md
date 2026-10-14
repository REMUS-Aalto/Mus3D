# Static BS-Locoformer (VDBGPO baseline)

Six-output BS-Locoformer (the "Static BS-Locoformer" row). `bandit_split.py` and
`bslocoformer.py` are the band-split encoder/decoder and Locoformer from
spectral_feature_compression (Saijo & Bando). Trained with the query
BS-Locoformer's trainer:

    CONFIG=BSLocoformer_static sbatch train_bslocoformer_separator.sh
