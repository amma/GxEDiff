# GxEDiff

Code for *GxEDiff: Graph-Based Diffusion for Average Yield Prediction of Genotypes across Environments*
(A. Morshedian and M. Domaratzki, AAAI 2027 AgriAI Workshop).

GxEDiff completes the sparse genotype × environment yield table with a denoising diffusion model whose denoiser is a
graph neural network. Genotypes (SNP markers) and environments (LSTM summary of the season's weather) are the nodes;
measured yields, the 10 closest relatives and the 10 most similar environments are the edges. Empty cells are filled in
50 rounds with self-conditioning, and the prediction is the mean of 100 filled tables from each of 10 trained models.

## Data

- G2F: Genomes to Fields 2024 G×E prediction competition release (Chen et al. 2026,
  https://doi.org/10.1186/s13104-026-07629-5). Place it in `raw/G2F_2024/` with its `Training_data/` and
  `Testing_data/` folders (trait, metadata, weather, `5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt` and
  `7_Testing_Observed_Values.csv`).
- MaizeGEP (Zhang et al. 2026, https://doi.org/10.1093/gpbjnl/qzaf140), downloaded from http://user.ebreed.cn:9992/scb/.
  Place `normalized_PHE.xlsx`, `extracted/ID_gen_260_12233.txt` and `extracted/normalized_ENV.xlsx` in `raw/MaizeGEP/`.

## Scripts

Inputs
- `prepare_g2f.py`: yield cells (replicate-adjusted hybrid means per site-year), 2,425 markers, kinship neighbours,
  weekly weather and crop-stage covariates -> `data/g2f.pkl`
- `prepare_maizegep.py`: the same for MaizeGEP (12,232 markers, county-years) -> `data/maizegep.pkl`

Model
- `common.py`: environment LSTM autoencoder, weather neighbours, environment means, noise schedule, hiding schemes, metrics
- `denoiser.py`: heterogeneous graph transformer and cell MLP
- `gxediff.py` (GPU): training and filling for one test design and seed; `--gnn-only` and `--no-self-cond` give the ablations

Baselines
- `gblup.py` (GPU): GBLUP and GBLUP + W (environmental covariates)
- `card.py` (GPU): CARD, diffusion without the graph

Results
- `evaluate.py`: Table 2 (r̄ and ρ), spread over trained models, forward tests, depth and self-conditioning
- `figures.py`: Figures 1-4
- `run_all.sh`: every run of the paper, then the tables and figures
- `results/`, `figures/`: the outputs of `evaluate.py` and `figures.py` for the runs reported in the paper

## Test designs

| Flag | Design |
|---|---|
| (default) | G2F: train 2014-2023, predict the official 2024 yields (seen and unseen hybrids) |
| `--year 2018/2020/2022` | G2F forward test: train on all earlier years |
| `--year Y --train-from F` | the same with a six-year window (2022 from 2016, 2024 from 2018) |
| `--loyo Y` | G2F leave-one-year-out within 2014-2023 |
| `--data maizegep` | MaizeGEP: the 70 varieties never grown in 2017-2020 are held out |

Example: `python3 gxediff.py --seed 0` writes `runs/g2f2024/gxediff_s0/test_samples.npz` with the true yields and the
100 filled values of every test cell. `--smoke` runs a tiny version to check the installation.

Figure 3c of the paper was produced with an earlier version of the model; the optional block in `run_all.sh` runs the
same leave-one-year-out test with the final model.

## Requirements

Python 3.10 with the packages in `requirements.txt` (torch 2.5.1 with CUDA, torch_geometric 2.6.1).
A GxEDiff run takes about 10 minutes and a CARD run about 35 minutes on one NVIDIA RTX 6000 Ada GPU.
