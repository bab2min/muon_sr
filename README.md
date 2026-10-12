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


## 8-bit AdamW states + BF16 stochastic rounding

```python
from muon_sr import TritonAdamW8bit

model = model.cuda().to(torch.bfloat16)
optimizer = TritonAdamW8bit.for_model(
    model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0.01, block_size=256, min_8bit_size=4096,
)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Large, non-embedding tensors use bitsandbytes-style signed/unsigned dynamic
codebooks: both moments are stored as uint8 codes, with separate FP32 block
absmax values. Codebooks match bitsandbytes exactly. The quantizer follows its
pivot/midpoint rules and first-moment sign correction. Weight decay follows the
8-bit bitsandbytes order (after the Adam step). Unlike its BF16 update path,
this implementation keeps the complete weight update in FP32 registers until
the final stochastic-rounded BF16 store. It does not maintain FP32 master weights.

`for_model()` puts embedding weights (including tied heads) and tensors smaller
than 4096 elements in `optim_bits=16` groups. These groups store **BF16 moments
with stochastic rounding**, not the FP32 fallback used by bitsandbytes.
All groups store weights as BF16 with SR. Arithmetic is still FP32 inside the
kernel. A plain `model.parameters()` iterator cannot identify embeddings; use
`for_model()` or explicit groups:

```python
optimizer = TritonAdamW8bit([
    {"params": hidden_params, "optim_bits": 8},
    {"params": embedding_and_small_params, "optim_bits": 16},
], lr=1e-3)
```

Separate groups preserve exclusions before ZeRO flattening. The size threshold
alone applies to the tensors received by the optimizer, which may already be
flattened by a framework. `optim_bits=32` is not supported by this optimizer.

8-bit moment storage is `2*N + 8*ceil(N/block_size)` bytes per tensor, plus the
shared two 256-entry FP32 codebooks. At block size 256 this is about 2.03125
bytes/element for large tensors. BF16 moment groups use 4 bytes/element.
Weights use another 2 bytes/element. Model replicas, gradients, activations,
and framework buffers are excluded from these figures.

Requires contiguous CUDA BF16 parameters and dense contiguous gradients.
CPU offload, centralization, AMSGrad, percentile clipping, paging and arbitrary
bitsandbytes checkpoint imports are not implemented. Nonfinite gradient elements
leave weights unchanged and reset their moments, following the blockwise
bitsandbytes path. Finite all-zero blocks are handled explicitly.

Checkpoint loading preserves uint8 moments, BF16 fallback moments and FP32
scales/codebooks. Resume with the same parameter order and partitioning for
reproducible SR. Changing block size/state precision after initialization is
not supported. The earlier linear/square-root prototype checkpoints are rejected.
DeepSpeed's default FP32 masters are incompatible; BF16 flat partitions require
separate distributed step/checkpoint validation. This is not a claim of bitwise
or convergence equivalence to bitsandbytes.

Codebook generation and quantizer search are adapted from bitsandbytes.