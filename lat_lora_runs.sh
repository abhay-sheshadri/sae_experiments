#!/bin/bash

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate cb

# Define arrays for the parameter variations
pgd_layers_options=("embedding" "4,8,12,16,20" "8" "12")
attack_seq_options=("input" "total")
epsilon_options=(1.0 3.0)

# Function to create and submit Slurm job
create_and_submit_job() {
    local pgd_layers=$1
    local attack_seq=$2
    local epsilon=$3
    local job_name="${pgd_layers//,/_}_${attack_seq}_eps${epsilon}"
    local script_name="scripts/${job_name}.sh"

    # Create Slurm script
    cat << EOF > "$script_name"
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=jupyter_logs/${job_name}_%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# #SBATCH --partition=grayswan
#SBATCH --account=cais
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate cb

python lora_train_model.py \
    --use_sft \
    --attack_seq="$attack_seq" \
    --adversary_loss="output" \
    --pgd_iterations=32 \
    --pgd_layers="$pgd_layers" \
    --num_steps=150 \
    --eval_pretrained_probes \
    --epsilon="$epsilon"
EOF

    # Submit the job
    sbatch "$script_name"
    echo "Submitted job: $job_name"
}

# Loop through all combinations and submit jobs
for pgd_layers in "${pgd_layers_options[@]}"; do
    for attack_seq in "${attack_seq_options[@]}"; do
        for epsilon in "${epsilon_options[@]}"; do
            create_and_submit_job "$pgd_layers" "$attack_seq" "$epsilon"
        done
    done
done

echo "All jobs submitted"