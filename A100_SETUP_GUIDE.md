# Running LSTM Training on A100 GPU (cscigpu Cluster)

This guide explains how to run your LSTM stock prediction model on a machine with an A100 GPU, specifically the cscigpu cluster.

---

## Method 1: SLURM Job Submission (Recommended for Long Training)

### Step 1: Transfer Files to Cluster

```bash
# From your local machine, transfer files to cscigpu
scp -r /home/sn0wstacks/Documents/Github/CSUCF25/csci581/csci581finallstm username@cscigpu.cs.ucf.edu:~/csci581finallstm/

# Or use rsync (better for updates)
rsync -avz --exclude '581lstm' --exclude '__pycache__' \
  /home/sn0wstacks/Documents/Github/CSUCF25/csci581/csci581finallstm/ \
  username@cscigpu.cs.ucf.edu:~/csci581finallstm/
```

### Step 2: SSH into Cluster

```bash
ssh username@cscigpu.cs.ucf.edu
cd ~/csci581finallstm
```

### Step 3: Set Up Environment on Cluster

**Option A: Create Virtual Environment on Cluster**

```bash
# Load Python module (adjust version as needed)
module load python/3.12  # or python/3.11, etc.

# Create virtual environment
python -m venv 581lstm

# Activate it
source 581lstm/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: Use Conda (if available)**

```bash
module load conda  # if available
conda create -n lstm_env python=3.12
conda activate lstm_env
pip install -r requirements.txt
```

**Option C: Use System Python with --user**

```bash
pip install --user -r requirements.txt
```

### Step 4: Modify SLURM Script

Edit `run_on_a100.sh` to match your cluster's configuration:

```bash
# Check available partitions
sinfo

# Check GPU availability
sinfo -o "%P %G"  # Shows partitions and GPU types

# Modify these lines in run_on_a100.sh:
#SBATCH --partition=gpu           # Change to your GPU partition name
#SBATCH --gres=gpu:a100:1         # May need to change to gpu:1 or gpu:a100:1
```

### Step 5: Submit Job

```bash
# Make script executable
chmod +x run_on_a100.sh

# Submit job
sbatch run_on_a100.sh

# Check job status
squeue -u $USER

# View output (while running)
tail -f lstm_<jobid>.out

# View errors
tail -f lstm_<jobid>.err
```

### Step 6: Retrieve Results

```bash
# After job completes, download results
scp username@cscigpu.cs.ucf.edu:~/csci581finallstm/*.png ./
scp username@cscigpu.cs.ucf.edu:~/csci581finallstm/best_lstm_model.keras ./
scp username@cscigpu.cs.ucf.edu:~/csci581finallstm/lstm_*.out ./
```

---

## Method 2: Interactive Session (For Testing/Development)

### Step 1: Request Interactive GPU Session

```bash
# Request interactive session with A100
srun --partition=gpu --gres=gpu:a100:1 --time=02:00:00 --pty bash

# Or with more resources
srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

### Step 2: Activate Environment and Run

```bash
# Once in interactive session
cd ~/csci581finallstm
source 581lstm/bin/activate

# Verify GPU
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run training
python train_lstm.py
```

---

## Method 3: Direct SSH (If You Have Direct Access)

If you have direct SSH access to a node with A100:

```bash
# SSH to specific node (if known)
ssh username@gpu-node.cs.ucf.edu

# Or use srun to get a node
srun --pty bash

# Then activate environment and run
cd ~/csci581finallstm
source 581lstm/bin/activate
python train_lstm.py
```

---

## Verifying GPU Usage

### Check GPU is Available

```bash
# Check GPU hardware
nvidia-smi

# Check TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Monitor GPU During Training

In a separate terminal (or while job is running):

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or continuous monitoring
nvidia-smi -l 1  # Updates every 1 second
```

### Expected Output

You should see:
- GPU name: A100-SXM4-40GB (or similar)
- GPU utilization: 50-100% during training
- Memory usage: Varies based on batch size
- Temperature: Normal operating range

---

## Troubleshooting

### Issue: "No GPU found"

**Solutions:**
1. Verify you're on a GPU node: `hostname` (should be a GPU node)
2. Check SLURM allocation: `echo $SLURM_JOB_GPUS` or `echo $CUDA_VISIBLE_DEVICES`
3. Verify CUDA is available: `nvcc --version`
4. Check TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"`

### Issue: "Out of Memory (OOM)"

**Solutions:**
1. Reduce batch size in `train_lstm.py`: Change `batch_size=32` to `batch_size=16` or `8`
2. Reduce LSTM units: Change `LSTM(64)` to `LSTM(32)`
3. Request more GPU memory (if available): `--gres=gpu:a100:1` (some clusters have different A100 sizes)

### Issue: "Module not found" or Import Errors

**Solutions:**
1. Ensure virtual environment is activated: `which python` should show venv path
2. Reinstall packages: `pip install -r requirements.txt --force-reinstall`
3. Check Python version matches: `python --version` (should be 3.12)

### Issue: "Mixed precision errors"

**Solutions:**
1. If you get dtype errors, you can disable mixed precision:
   ```python
   # Comment out this line in train_lstm.py:
   # tf.keras.mixed_precision.set_global_policy('mixed_float16')
   ```
2. Or ensure you have TensorFlow 2.10+ with proper CUDA support

### Issue: SLURM Partition/Queue Issues

**Solutions:**
1. Check available partitions: `sinfo`
2. Check your account/group: `groups`
3. Contact cluster admin if you don't have GPU access
4. Try different partition names: `gpu`, `gpu-a100`, `a100`, etc.

---

## Optimizing for A100

### Batch Size Tuning

A100 has 40GB memory, so you can use larger batches:

```python
# In train_lstm.py, try:
batch_size=64  # or 128 if memory allows
```

### Mixed Precision

The script already enables mixed precision (float16), which is optimal for A100:

```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Multi-GPU (If Available)

If you have access to multiple A100s:

```python
# Add to train_lstm.py after GPU config:
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = models.Sequential([...])
    model.compile(...)
```

Then request multiple GPUs in SLURM:
```bash
#SBATCH --gres=gpu:a100:2  # Request 2 A100s
```

---

## Expected Performance

On A100 GPU, you should see:
- **Training time**: ~2-5 minutes for 50 epochs (depending on data size)
- **GPU utilization**: 60-90% during training
- **Memory usage**: 2-8 GB GPU memory (plenty of room on 40GB A100)
- **Speedup**: 10-50x faster than CPU training

---

## Quick Reference Commands

```bash
# Submit job
sbatch run_on_a100.sh

# Check job status
squeue -u $USER

# Cancel job
scancel <jobid>

# View output
cat lstm_<jobid>.out

# Interactive session
srun --partition=gpu --gres=gpu:a100:1 --time=02:00:00 --pty bash

# Monitor GPU
watch -n 1 nvidia-smi
```

---

## Cluster-Specific Notes

**For cscigpu cluster specifically:**
- Partition name may be `gpu` or `gpu-a100`
- May need to specify account: `#SBATCH --account=your_account`
- May have time limits (check with `sinfo -p gpu`)
- May need to load specific modules for CUDA/TensorFlow

**Contact cluster admin if:**
- You don't have access to GPU partition
- You're unsure about partition names
- You need help with module loading
- You encounter permission errors

---

**Last Updated**: December 1, 2025

