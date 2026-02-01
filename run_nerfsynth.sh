#!/bin/bash

# --- Run experiments on the full NeRF synthetic dataset ---
# Datasets: ["drums" "chair" "ficus" "hotdog" "mic" "ship" "lego" "materials"] (8 datasets)

# DECLARE ARRAY FOR DATASETS
declare -a dsets_arr=("drums" "chair" "ficus" "hotdog" "mic" "ship" "lego" "materials")

# FIXED HYPERPARAMETERS
hyper_nr_blocks=4
hyper_hidden_size=192
hyper_lr_init=0.00005
hyper_lr_final=$hyper_lr_init

# OTHER FIXED PARAMETERS
operator="mul"
type="both"
mlp_hidden_layer_size="3"
quantum_layers=4
iterations=30000

# GENERIC PATHS (Relative to current directory)
# Assumes structure:
#   ./train_experiments.py
#   ./data/nerf_synth_mipnerf/<dataset>
#   ./results/<dataset>

TRAIN_SCRIPT="train_experiments.py"
DATA_ROOT="./data/nerf_synth_mipnerf"
EXP_DIR="./results/nerfsynth"

# Ensure experiment directory exists
mkdir -p "$EXP_DIR"

# LOOP OVER ALL DATASETS
for dset in "${dsets_arr[@]}"; do
    
    echo "======================================="
    echo "Starting experiment for dataset: $dset"
    
    # Construct paths for current dataset
    dset_path="${DATA_ROOT}/${dset}"
    exp_name="${dset}"
    model_out_path="${EXP_DIR}/${exp_name}"

    # Print job information (Sanitized)
    echo "hyper_nr_blocks: $hyper_nr_blocks"
    echo "hyper_hidden_size: $hyper_hidden_size"
    echo "hyper_lr_init: $hyper_lr_init"
    echo "model_output_path: $model_out_path"
    echo "---------------------------------------"

    # RUN TRAINING SCRIPT
    python "$TRAIN_SCRIPT" \
        -s "$dset_path" \
        -m "$model_out_path" \
        --eval \
        --vdgs_type "$type" \
        --vdgs_operator "$operator" \
        --quantum_layers "$quantum_layers" \
        --perform_quantum \
        --perform_mlp \
        --hyper_encoder "spherical" \
        --mlp_hidden_layer_sizes "${mlp_hidden_layer_size}" \
        --hyper_nr_blocks "$hyper_nr_blocks" \
        --hyper_hidden_size "$hyper_hidden_size" \
        --hyper_lr_init "$hyper_lr_init" \
        --hyper_lr_final "$hyper_lr_final" \
        --shared_encoder "none" \
        --iterations "$iterations" \
        --save_iterations "$iterations"

    echo "=== Completed: $dset ==="
    echo ""
done

echo "All experiments finished."