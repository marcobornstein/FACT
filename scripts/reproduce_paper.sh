#!/usr/bin/env bash
# Rerun every experiment whose logs are committed in output/.
#
# Results go to $OUT (default: output-reproduced) so the committed logs stay untouched; plot them with
#   python plot.py --results-dir output-reproduced --figures-dir figures-reproduced
# Set MPIRUN to use a different launcher, called as: $MPIRUN -n <agents> python train.py ...
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-output-reproduced}
MPIRUN=${MPIRUN:-mpirun}

run() {
    local agents=$1
    shift
    "$MPIRUN" -n "$agents" python train.py --output-dir "$OUT" "$@"
}

# CIFAR-10 and MNIST (Figures 1-3, 5, 6). Run r of the iid setting used seed r; runs 1, 2, 3 of the
# non-iid settings used seeds 4, 2, 3.
for dataset in cifar10 mnist; do
    fed_optimizer=""
    if [ "$dataset" = cifar10 ]; then
        fed_optimizer="--fed-optimizer adam"
    fi
    for run in 1 2 3; do
        # shellcheck disable=SC2086
        run 16 --dataset "$dataset" --seed "$run" $fed_optimizer \
            --name "fact-random-sandwich-uniform-cost-run$run"
    done
    for alpha in 0.6 0.3; do
        for run in 1 2 3; do
            seed=$run
            if [ "$run" = 1 ]; then
                seed=4
            fi
            # shellcheck disable=SC2086
            run 16 --dataset "$dataset" --seed "$seed" --non-iid "$alpha" $fed_optimizer \
                --name "fact-random-sandwich-uniform-cost-noniid-$alpha-run$run"
        done
    done
done

# HAM10000 (Figures 4, 7).
run 10 --dataset ham10000 --seed 2024 --name fact-sandwich-uniform-cost-run1
run 10 --dataset ham10000 --seed 2024 --name fact-sandwich-uniform-cost-run2
run 10 --dataset ham10000 --seed 2025 --name fact-sandwich-uniform-cost-run3

# HAM10000 with 1, 4 or 8 free riders training on 25% of their optimal data (robustness experiment).
for free_riders in 1 4 8; do
    run 10 --dataset ham10000 --seed 2024 --nonuniform-cost --batch-size 64 --free-riders "$free_riders" \
        --name "fact-robustness-$free_riders-nonuniform-cost-iid-run1"
done
