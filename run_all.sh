#!/usr/bin/env bash
# All experiments of the paper, one after another on one GPU (about 10 min per GxEDiff run, 35 min per CARD run).
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6
SEEDS10="0 1 2 3 4 5 6 7 8 9"

python3 prepare_g2f.py --raw raw/G2F_2024
python3 prepare_maizegep.py --raw raw/MaizeGEP

# G2F 2024 test: Table 2, Figures 2 and 4
python3 gblup.py
for s in $SEEDS10; do python3 gxediff.py --seed $s; done
for s in $SEEDS10; do python3 card.py --seed $s; done
for s in 0 1 2; do python3 gxediff.py --gnn-only --seed $s; done
for s in 0 1 2; do python3 gxediff.py --no-self-cond --seed $s; done
for s in 0 1 2; do python3 gxediff.py --layers 3 --seed $s; done
for s in 0 1 2 3 4 5; do python3 gxediff.py --layers 15 --seed $s; done

# forward tests: Figure 3
for y in 2018 2020 2022; do
  python3 gblup.py --year $y
  for s in 0 1 2; do
    python3 gxediff.py --year $y --seed $s
    python3 gxediff.py --year $y --gnn-only --seed $s
  done
done
for yf in 2022:2016 2024:2018; do
  y=${yf%:*}; f=${yf#*:}
  python3 gblup.py --year $y --train-from $f
  for s in 0 1 2; do
    python3 gxediff.py --year $y --train-from $f --seed $s
    python3 gxediff.py --year $y --train-from $f --gnn-only --seed $s
  done
done

# MaizeGEP unseen varieties: Table 2
python3 gblup.py --data maizegep
for s in $SEEDS10; do
  python3 gxediff.py --data maizegep --seed $s
  python3 gxediff.py --data maizegep --gnn-only --seed $s
  python3 card.py --data maizegep --seed $s
done

# optional, leave-one-year-out on G2F 2014-2023 (Figure 3c)
# for y in $(seq 2014 2023); do
#   python3 gblup.py --loyo $y
#   for s in 0 1 2; do python3 gxediff.py --loyo $y --seed $s; python3 gxediff.py --loyo $y --gnn-only --seed $s; done
# done

python3 evaluate.py
python3 figures.py
