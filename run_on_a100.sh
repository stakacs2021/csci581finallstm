#!/bin/bash
#SBATCH --job-name=lstm_stock_pred
#SBATCH --output=lstm_%j.out
#SBATCH --error=lstm_%j.err
#SBATCH --time=04:00:00          # 4 hours (adjust as needed)
#SBATCH --partition=gpu           # GPU partition (adjust to your cluster's partition name)
#SBATCH --gres=gpu:a100:1         # Request 1 A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # CPUs for data loading
#SBATCH --mem=32G                  # Memory (adjust as needed)

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load necessary modules (adjust for your cluster)
# Uncomment and modify based on your cluster's module system
# module load python/3.12
# module load cuda/11.8  # or appropriate CUDA version
# module load cudnn/8.6  # if available

# Activate virtual environment
# Option 1: If you've created venv on the cluster
source 581lstm/bin/activate

# Option 2: If using conda
# conda activate lstm_env

# Option 3: If using system Python with pip install --user
# export PATH=$HOME/.local/bin:$PATH

# Verify GPU is available
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Verify TensorFlow sees the GPU
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
echo ""

# Run the training script
echo "=========================================="
echo "Starting LSTM Training"
echo "=========================================="
python train_lstm.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

