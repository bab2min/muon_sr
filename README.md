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

## Training with `TritonMuonSRWithAuxAdam`

The model is kept in `bfloat16`. There are no FP32 master weights; stochastic rounding is applied whenever
weights or optimizer states are written back to BF16.

### 1. Split the parameters into a Muon group and an AdamW group

Muon is meant for the hidden weight matrices. Embeddings, the output head, norms and biases go to the AdamW group.
Every group needs a `use_muon` key.

```python
import torch
from muon_sr import TritonMuonSRWithAuxAdam

model = model.cuda().to(torch.bfloat16)      # all trainable parameters must be BF16

muon_params, adam_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim >= 2 and not any(k in name for k in ("embed", "lm_head", "head")):
        muon_params.append(p)                # hidden 2D matrices (attention / MLP projections)
    else:
        adam_params.append(p)                # embeddings, output head, norms, biases

optimizer = TritonMuonSRWithAuxAdam([
    dict(params=muon_params, use_muon=True,  lr=0.02, momentum=0.95, weight_decay=0.01),
    dict(params=adam_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
], batch_ns=True)
```

Defaults when a key is omitted:

| group | defaults |
|---|---|
| `use_muon=True` | `lr=0.02`, `momentum=0.95`, `nesterov=True`, `ns_steps=5`, `weight_decay=0` |
| `use_muon=False` | `lr=3e-4`, `betas=(0.9, 0.95)`, `eps=1e-10`, `weight_decay=0` |

### 2. Training loop

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

for step in range(total_steps):
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

PyTorch LR schedulers scale each group from its own initial `lr`, so the Muon and AdamW learning rates keep their ratio.

### 3. Gradient accumulation

This step is optional. Native accumulation sums micro-batch gradients in BF16 with round-to-nearest. With a few
micro-batches this trained as well as SR accumulation in `examples.py` (variants 5 vs 6 and 7 vs 8 below), so the
accumulator matters mostly when many micro-batches are summed. To accumulate with stochastic rounding, register the
accumulator hooks once and move the accumulated buffer into `.grad` before each optimizer step:

```python
from muon_sr import TritonSRAccumulator

TritonSRAccumulator.assign_hooks(model)

for step in range(total_steps):
    for _ in range(grad_accum):
        (loss_fn(model(x), y) / grad_accum).backward()
    TritonSRAccumulator.reassign_grad_buffer(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### 4. Hugging Face `Trainer`

Pass the optimizer through `optimizers=` and let the Trainer build the scheduler. Load the model in BF16:

```python
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
optimizer = TritonMuonSRWithAuxAdam(groups, batch_ns=True)    # groups as in step 1
trainer = Trainer(model=model, args=TrainingArguments(..., bf16=True), optimizers=(optimizer, None), ...)
```

### Caveats

* **BF16 only.** `step()` asserts that every parameter, including the AdamW group, is `torch.bfloat16`.
  Do not combine it with FP32 parameters under autocast (AMP).
* **Muon expects 2D matrices.** Newton-Schulz orthogonalizes over the last two dimensions and treats leading
  dimensions as a batch. A 4D convolution weight would be orthogonalized per `(kh, kw)` slice, so reshape such
  weights to 2D yourself or put them in the AdamW group. Embeddings and the output head should use AdamW.
* **Learning rates differ in scale.** The Muon `lr` (around `0.02`) is not comparable to an AdamW `lr`. Each Muon update is
  orthogonalized and multiplied by `sqrt(max(1, rows / cols))`.
* **Weight decay is decoupled.** Weights are multiplied by `1 - lr * weight_decay` before the update, using the
  group's own `lr`. With the larger Muon `lr`, the same `weight_decay` decays much faster than in the AdamW group.
* **Full, unsharded matrices.** Newton-Schulz needs each whole weight matrix. Optimizer sharding that flattens or
  splits parameters (FSDP, DeepSpeed ZeRO) is not supported.
* **Sparse gradients are not supported.**
* **`batch_ns=True`** stacks the momentum of parameters with the same shape and runs Newton-Schulz once per shape (in
  chunks of up to 2^28 elements). This is faster but allocates the stacked momentum up front. If only some parameters of a
  shape have gradients in a step, that shape falls back to per-parameter updates.
* **Reproducible SR noise.** Muon and AdamW derive their rounding seeds from the step and the parameter index, so all
  data-parallel ranks apply the same noise. Keep the same parameter order and grouping when resuming. `state_dict()` and
  `load_state_dict()` work with `batch_ns=True`; the stacked buffers are rebuilt on the next step.
* `TritonSRAccumulator` draws its seeds from Python's `random` module.

### Example results (`examples.py`)

`examples.py` trains a small byte-level GPT (8 layers, d_model 384, vocab 16384) for 500 steps
(batch 32 × seq 128, grad accum 4) from identical weights and batches. The reference Muon
(`pip install git+https://github.com/KellerJordan/Muon`) is needed for variants 1-3.

```bash
python examples.py corpus.txt
```

Run with `muon_sr` 0.1.2 and PyTorch 2.10 on A100-SXM4-80GB, using the first 20 MB of a Korean Wikipedia text dump as the corpus.

| # | Variant | Final loss | Mean of last 50 | Final vs FP32 | Time | Peak MB |
|---|---|---|---|---|---|---|
| 1 | FP32 Muon (reference) | 1.3143 | 1.3169 | +0.0000 | 87.3s | 2357 |
| 2 | AMP BF16 Muon | 1.3153 | 1.3173 | +0.0010 | 35.8s | 1813 |
| 3 | Pure BF16 Muon, round-to-nearest | 1.3340 | 1.3367 | +0.0197 | 30.8s | 1288 |
| 4 | `MuonSRWithAuxAdam` + `SRAccumulator` | 1.3105 | 1.3168 | -0.0037 | 53.1s | 1285 |
| 5 | `TritonMuonSRWithAuxAdam` + `TritonSRAccumulator` | 1.3145 | 1.3173 | +0.0002 | 39.6s | 1287 |
| 6 | `TritonMuonSRWithAuxAdam`, no accumulator | 1.3164 | 1.3180 | +0.0021 | 31.6s | 1286 |
| 7 | `TritonMuonSRWithAuxAdam(batch_ns=True)` + `TritonSRAccumulator` | 1.3145 | 1.3172 | +0.0002 | 30.6s | 1285 |
| 8 | `TritonMuonSRWithAuxAdam(batch_ns=True)`, no accumulator | 1.3164 | 1.3173 | +0.0021 | 24.5s | 1285 |

Time is wall-clock for 500 steps.

* With stochastic rounding (4-8), pure BF16 training reaches the FP32 and AMP loss (mean of the last 50 steps 1.3168-1.3180).
  Pure BF16 without it (3) ends about 0.02 higher.
* The SR variants use about 1.29 GB, against 1.81 GB for AMP and 2.36 GB for FP32.
* The Triton SR optimizer (5) is 1.34x faster than the PyTorch one (4). `batch_ns=True` (7) is a further 1.29x faster,
  and with no accumulator as well (8) it is 1.62x faster than (5), the fastest of all variants.
* With 4 micro-batches, dropping the SR accumulator changes the mean of the last 50 steps by at most 0.0007
  (5 vs 6, 7 vs 8).

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