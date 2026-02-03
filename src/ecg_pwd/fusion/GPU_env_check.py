#!/usr/bin/env python
"""Environment diagnostic script to verify TensorFlow GPU visibility and basic runtime configuration."""

import sys
import tensorflow as tf
import numpy as np

print("="*80)
print("GPU/CPU CONFIGURATION CHECK")
print("="*80)

# 1. TensorFlow version
print(f"\n1. TensorFlow Version: {tf.__version__}")

# 2. Check if TensorFlow was built with CUDA support
print(f"\n2. Built with CUDA: {tf.test.is_built_with_cuda()}")

# 3. List all physical devices
print("\n3. Available Physical Devices:")
print("-"*80)
devices = tf.config.list_physical_devices()
if devices:
    for device in devices:
        print(f"   • {device}")
else:
    print("   ⚠ No devices found!")

# 4. Specifically check for GPUs
print("\n4. GPU Devices:")
print("-"*80)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"     GPU {i}: {gpu.name}")
        # Get GPU details if possible
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"             Device Name: {details.get('device_name', 'Unknown')}")
                print(f"             Compute Capability: {details.get('compute_capability', 'Unknown')}")
        except:
            pass
else:
    print("   ❌ No GPUs detected!")
    print("   → TensorFlow will use CPU for training")

# 5. Check logical devices (what TensorFlow will actually use)
print("\n5. Logical Devices (What TensorFlow Uses):")
print("-"*80)
logical_devices = tf.config.list_logical_devices()
for device in logical_devices:
    print(f"   • {device}")

# 6. Check GPU memory configuration
print("\n6. GPU Memory Configuration:")
print("-"*80)
if gpus:
    for i, gpu in enumerate(gpus):
        try:
            # Check if memory growth is enabled
            memory_growth = tf.config.experimental.get_memory_growth(gpu)
            print(f"   GPU {i}:")
            print(f"     Memory Growth Enabled: {memory_growth}")
        except Exception as e:
            print(f"   GPU {i}: Could not get memory info ({e})")
else:
    print("   No GPUs to configure")

# 7. Quick computation test
print("\n7. Computation Test:")
print("-"*80)
print("   Running a simple matrix multiplication...")

# Create tensors
with tf.device('/CPU:0'):
    cpu_a = tf.random.normal([1000, 1000])
    cpu_b = tf.random.normal([1000, 1000])

if gpus:
    with tf.device('/GPU:0'):
        gpu_a = tf.random.normal([1000, 1000])
        gpu_b = tf.random.normal([1000, 1000])

# Time CPU computation
import time
start = time.time()
with tf.device('/CPU:0'):
    cpu_result = tf.matmul(cpu_a, cpu_b)
cpu_time = time.time() - start
print(f"   CPU computation time: {cpu_time:.4f} seconds")

# Time GPU computation if available
if gpus:
    start = time.time()
    with tf.device('/GPU:0'):
        gpu_result = tf.matmul(gpu_a, gpu_b)
    gpu_time = time.time() - start
    print(f"   GPU computation time: {gpu_time:.4f} seconds")
    print(f"   → GPU is {cpu_time/gpu_time:.1f}x faster than CPU")
else:
    print("   (GPU test skipped - no GPU available)")

# 8. Check CUDA and cuDNN versions
print("\n8. CUDA Configuration:")
print("-"*80)
try:
    from tensorflow.python.platform import build_info
    print(f"   CUDA Version: {build_info.build_info['cuda_version']}")
    print(f"   cuDNN Version: {build_info.build_info['cudnn_version']}")
except:
    print("   Could not retrieve CUDA/cuDNN versions")

# 9. Check environment variables
print("\n9. Relevant Environment Variables:")
print("-"*80)
import os
env_vars = ['CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_CPP_MIN_LOG_LEVEL']
for var in env_vars:
    value = os.environ.get(var, 'Not set')
    print(f"   {var}: {value}")

# 10. What will happen during training?
print("\n" + "="*80)
print("SUMMARY: What will your training use?")
print("="*80)

if gpus:
    print("✓ GPU TRAINING")
    print(f"  → TensorFlow will automatically use GPU: {gpus[0].name}")
    print(f"  → Your model.fit() will run on GPU by default")
    print(f"  → No code changes needed - GPU usage is automatic")
else:
    print("❌ CPU TRAINING")
    print("  → TensorFlow will use CPU for training")
    print("  → This will be SIGNIFICANTLY slower than GPU")
    print("\n  To enable GPU:")
    print("  1. Make sure you have an NVIDIA GPU")
    print("  2. Install CUDA Toolkit")
    print("  3. Install cuDNN")
    print("  4. Install tensorflow-gpu or tensorflow>=2.0 with GPU support")
    print("  5. Verify with: nvidia-smi")

print("\n" + "="*80)
print("HOW YOUR CODE CURRENTLY HANDLES GPU")
print("="*80)

print("""
Your current code (train_wavenet_flexible_loss.py):
  • Does NOT explicitly specify device (GPU/CPU)
  • Relies on TensorFlow's automatic device placement
  • TensorFlow automatically uses GPU if available
  • If no GPU available, falls back to CPU

This is the STANDARD approach and works well in most cases.

When you call model.fit():
  - TensorFlow checks for available GPUs
  - If GPU found → uses GPU automatically
  - If no GPU → uses CPU automatically

No code changes needed for GPU usage!
""")

# 11. Recommendations
print("="*80)
print("RECOMMENDATIONS")
print("="*80)

if gpus:
    print("""
✓ You have GPU available - great!
  
Optional optimizations you can add to your code:

1. Enable memory growth (prevents TensorFlow from allocating all GPU memory):
   
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)

2. Explicitly set visible devices (useful if you have multiple GPUs):
   
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0

3. Add device logging to see where operations run:
   
   tf.debugging.set_log_device_placement(True)

But for most cases, your current code is fine!
""")
else:
    print("""
❌ No GPU detected - you're training on CPU

This means:
  • Training will be 10-100x slower than GPU
  • Large models may take days instead of hours
  • Not recommended for production training

To fix:
  1. Check if you have an NVIDIA GPU:
     nvidia-smi
  
  2. If no GPU: Consider using:
     • Google Colab (free GPU)
     • AWS/GCP/Azure GPU instances
     • Local machine with GPU
  
  3. If GPU exists but not detected:
     • Check CUDA installation
     • Check TensorFlow GPU installation
     • Check CUDA_VISIBLE_DEVICES
""")

print("="*80)