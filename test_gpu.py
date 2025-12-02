"""
Quick GPU Test Script
Run this to verify GPU is available and TensorFlow can use it
"""

import tensorflow as tf
import sys

print("=" * 60)
print("GPU Test for A100")
print("=" * 60)

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check if built with CUDA
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List physical devices
print("\nPhysical Devices:")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"  - {device}")

# List GPUs specifically
print("\nGPU Devices:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        try:
            # Get GPU details
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Details: {details}")
        except:
            pass
        
        # Check memory info
        try:
            memory_info = tf.config.experimental.get_memory_info(gpu.name)
            print(f"    Current memory: {memory_info['current'] / 1024**3:.2f} GB")
            print(f"    Peak memory: {memory_info['peak'] / 1024**3:.2f} GB")
        except:
            pass
else:
    print("  No GPUs found!")
    print("\n⚠️  WARNING: No GPU detected. Training will be slow on CPU.")
    sys.exit(1)

# Test GPU computation
print("\n" + "=" * 60)
print("Testing GPU Computation")
print("=" * 60)

try:
    with tf.device('/GPU:0'):
        # Create a simple tensor and perform operation
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"✓ GPU computation successful!")
        print(f"  Result: {c.numpy()}")
except Exception as e:
    print(f"✗ GPU computation failed: {e}")
    sys.exit(1)

# Check mixed precision
print("\n" + "=" * 60)
print("Mixed Precision Check")
print("=" * 60)
try:
    policy = tf.keras.mixed_precision.global_policy()
    print(f"Current policy: {policy.name}")
    if policy.name == 'mixed_float16':
        print("✓ Mixed precision (float16) enabled - optimal for A100")
    else:
        print(f"  Current policy: {policy.name}")
except Exception as e:
    print(f"Could not check mixed precision: {e}")

print("\n" + "=" * 60)
print("✓ GPU Test Complete - Ready for Training!")
print("=" * 60)

