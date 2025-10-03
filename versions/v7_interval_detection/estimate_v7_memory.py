"""
Estimate V7 memory usage for RTX 5090 optimization
"""

import torch

def estimate_memory():
    """Estimate memory usage for V7 model"""

    print("="*60)
    print("V7 Memory Estimation for RTX 5090 (32GB VRAM)")
    print("="*60)

    # Model parameters
    input_dim = 284  # 142 coords + 71 speed + 71 accel
    hidden_dim = 256
    num_actions = 4
    num_agents = 4
    num_anchors = 5  # anchor_scales
    sequence_length = 1000

    # Calculate model size
    print("\n1. Model Parameters:")

    # Conv layers
    conv_params = (
        input_dim * 128 * 3 +  # Conv1
        128 * 256 * 3          # Conv2
    )
    print(f"  Conv layers: ~{conv_params:,} params")

    # LSTM
    lstm_params = (
        (256 + hidden_dim) * hidden_dim * 4 +  # Layer 1
        (hidden_dim * 2 + hidden_dim) * hidden_dim * 4  # Layer 2
    ) * 2  # Bidirectional
    print(f"  BiLSTM: ~{lstm_params:,} params")

    # Detection heads
    head_params = (
        (hidden_dim * 2 * 128 + 128 * num_anchors * num_actions) +  # Action
        (hidden_dim * 2 * 128 + 128 * num_anchors * num_agents) * 2 +  # Agent + Target
        (hidden_dim * 2 * 128 + 128 * num_anchors * 2) +  # Boundary
        (hidden_dim * 2 * 128 + 128 * num_anchors)  # Objectness
    )
    print(f"  Detection heads: ~{head_params:,} params")

    total_params = conv_params + lstm_params + head_params
    print(f"\n  Total params: ~{total_params:,}")

    # Model memory (FP32)
    model_memory_fp32 = total_params * 4 / 1024**3  # GB
    model_memory_fp16 = total_params * 2 / 1024**3  # GB
    print(f"  Model memory (FP32): {model_memory_fp32:.2f} GB")
    print(f"  Model memory (FP16): {model_memory_fp16:.2f} GB")

    # Activation memory per sample
    print("\n2. Activation Memory (per sample):")

    # Input
    input_memory = sequence_length * input_dim * 4 / 1024**2  # MB
    print(f"  Input: {input_memory:.2f} MB")

    # After Conv (downsampled by 4)
    conv_seq_len = sequence_length // 4
    conv_memory = conv_seq_len * 256 * 4 / 1024**2  # MB
    print(f"  Conv output: {conv_memory:.2f} MB")

    # LSTM hidden states
    lstm_memory = conv_seq_len * hidden_dim * 2 * 4 / 1024**2  # MB
    print(f"  LSTM output: {lstm_memory:.2f} MB")

    # Anchors and predictions
    num_anchor_positions = sequence_length // 4  # Stride 4
    total_anchors = num_anchor_positions * num_anchors
    anchor_memory = total_anchors * (
        num_actions + num_agents * 2 + 2 + 1  # action, agent, target, boundary, objectness
    ) * 4 / 1024**2  # MB
    print(f"  Anchors & predictions ({total_anchors:,} anchors): {anchor_memory:.2f} MB")

    total_activation_per_sample = input_memory + conv_memory + lstm_memory + anchor_memory
    print(f"\n  Total per sample: {total_activation_per_sample:.2f} MB")

    # Estimate max batch size
    print("\n3. Batch Size Estimation:")

    available_vram = 32  # GB for RTX 5090
    overhead = 2  # GB for system overhead

    # FP32
    usable_vram_fp32 = (available_vram - overhead - model_memory_fp32) * 1024  # MB
    max_batch_fp32 = int(usable_vram_fp32 / (total_activation_per_sample * 2))  # *2 for gradients
    print(f"  FP32 mode:")
    print(f"    Available for activations: {usable_vram_fp32:.0f} MB")
    print(f"    Max batch size: ~{max_batch_fp32}")

    # FP16
    usable_vram_fp16 = (available_vram - overhead - model_memory_fp16) * 1024  # MB
    max_batch_fp16 = int(usable_vram_fp16 / (total_activation_per_sample))  # FP16 halves memory
    print(f"  FP16 mode (mixed precision):")
    print(f"    Available for activations: {usable_vram_fp16:.0f} MB")
    print(f"    Max batch size: ~{max_batch_fp16}")

    # Recommendations
    print("\n4. Recommendations:")

    safe_batch_fp32 = max(8, max_batch_fp32 // 2)
    safe_batch_fp16 = max(16, max_batch_fp16 // 2)

    print(f"  FP32 (conservative): batch_size = {safe_batch_fp32}")
    print(f"  FP16 (recommended): batch_size = {safe_batch_fp16} ⭐")

    # With gradient accumulation
    print("\n5. With Gradient Accumulation:")

    target_effective_batch = 64
    for batch in [16, 24, 32, 48]:
        accum_steps = target_effective_batch // batch
        if accum_steps >= 1:
            print(f"  batch={batch}, accum_steps={accum_steps} → effective_batch={batch * accum_steps}")

    print("\n6. Suggested Configuration (FP16):")
    print(f"  batch_size: 32-48")
    print(f"  gradient_accumulation_steps: 1-2")
    print(f"  mixed_precision: true")
    print(f"  effective_batch_size: 32-96")

    print("\n" + "="*60)
    print("✓ Use batch_size=32 with FP16 for optimal 5090 utilization")
    print("✓ Monitor VRAM usage, can increase to 48 if stable")
    print("="*60)


if __name__ == '__main__':
    estimate_memory()
