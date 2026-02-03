#!/bin/bash

# --- Run experiments on the Deep Blending dataset ---
# Datasets: ["drjohnson" "playroom"]

# DECLARE ARRAYS
declare -a dsets_arr=("drjohnson" "playroom")
# Corresponding resolutions for each dataset
declare -a resolution_arr=(1 1)

# FIXED HYPERPARAMETERS
enc_biggest_mlp_dim=128
shared_encoder="hashgrid"
xyz_lr=0.001
hybrid_lr=0.0005
threshold=0.0005

# OTHER FIXED PARAMETERS
operator="mul"
type="both"
mlp_hidden_layer_size="3"
quantum_layers=4
shared_lr=0.001
iterations=30000

# GENERIC PATHS (Relative to current directory)
# Assumes structure:
#   ./train_experiments.py
#   ./data/db/<dataset>
#   ./results/db

TRAIN_SCRIPT="train_experiments.py"
DATA_ROOT="/shared/sets/datasets/gaussian_splatting/db"
EXP_DIR="./results/db"

# Ensure experiment directory exists
mkdir -p "$EXP_DIR"

# LOOP OVER ALL DATASETS (using array indices)
for i in "${!dsets_arr[@]}"; do
    
    # GET CURRENT CONFIGURATION
    dset=${dsets_arr[$i]}
    resolution=${resolution_arr[$i]}

    echo "======================================="
    echo "Starting experiment for dataset: $dset (Resolution: $resolution)"

    # SET EXPERIMENT NAME AND PATHS
    exp_name="hybrid_no-hyper_enc-${enc_biggest_mlp_dim}_shared-${shared_encoder}_xyz-${xyz_lr}_hybrid-${hybrid_lr}_dset-${dset}_type-${type}_operator-${operator}"
    
    dset_path="${DATA_ROOT}/${dset}"
    model_out_path="${EXP_DIR}/${exp_name}"

    # Print job information (Sanitized)
    echo "enc_biggest_mlp_dim: $enc_biggest_mlp_dim"
    echo "shared_encoder: $shared_encoder"
    echo "xyz_lr: $xyz_lr"
    echo "hybrid_lr: $hybrid_lr"
    echo "densify_grad_threshold: $threshold"
    echo "model_output_path: $model_out_path"
    echo "---------------------------------------"

    # RUN TRAINING SCRIPT
    python "$TRAIN_SCRIPT" \
        -s "$dset_path" \
        -m "$model_out_path" \
        -r "$resolution" \
        --eval \
        --vdgs_type "$type" \
        --vdgs_operator "$operator" \
        --quantum_layers "$quantum_layers" \
        --perform_quantum \
        --perform_mlp \
        --no_hyper \
        --xyz_lr_init "$xyz_lr" \
        --xyz_lr_final "$xyz_lr" \
        --hybrid_lr_init "$hybrid_lr" \
        --hybrid_lr_final "$hybrid_lr" \
        --shared_encoder "$shared_encoder" \
        --enc_biggest_mlp_dim "$enc_biggest_mlp_dim" \
        --shared_lr_init "$shared_lr" \
        --shared_lr_final "$shared_lr" \
        --iterations "$iterations" \
        --save_iterations "$iterations" \
        --densify_grad_threshold "$threshold"

    echo "=== Completed: $dset ==="
    echo ""
done

echo "All experiments finished."
