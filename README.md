# muon_sr

This package provides PyTorch optimizers and utilities for `bfloat16` training. It adds stochastic rounding (SR) to the Muon and AdamW optimizers. For performance, most components are also implemented with fused Triton kernels.

## Installation

This package requires PyTorch and Triton.

```bash
pip install torch triton
```

**Note:** The Triton-accelerated features require a Linux environment with an NVIDIA GPU and a properly configured CUDA toolkit. The pure PyTorch components are platform-agnostic.

## What's Inside

#### Optimizers

*   `MuonSR`: The Muon optimizer with stochastic rounding for `bfloat16` parameters.
*   `TritonMuonSR`: A faster version of `MuonSR` using fused Triton kernels. It's intended for matrices (`ndim >= 2`).
*   `TritonAdamW`: An AdamW implementation with stochastic rounding, written as a single fused Triton kernel.
*   `MuonSRWithAuxAdam` & `TritonMuonSRWithAuxAdam`: "Hybrid" optimizers that handle both Muon and AdamW parameters. This is useful if you want to optimize weight matrices with Muon and other parameters (like biases or embeddings) with AdamW.

#### Gradient Accumulation

*   `SRAccumulator` & `TritonSRAccumulator`: These provide hooks to perform gradient accumulation with stochastic rounding. The Triton version is faster.

The main idea is that stochastic rounding can help with numerical stability during `bfloat16` training. It's applied during parameter updates, momentum calculations, and gradient accumulation.

---

**References:**

This package references implementations from:
*   [KellerJordan/muon](https://github.com/KellerJordan/muon)
*   [lodestone-rock/torchastic](https://github.com/lodestone-rock/torchastic)

