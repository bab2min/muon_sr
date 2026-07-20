"""
Comparison of five Muon training configurations on a byte-level language model (vocab=256):

  1. FP32       SingleDeviceMuon  — no quantization, upper-bound reference
  2. AMP BF16   SingleDeviceMuon  — FP32 params + BF16 forward (torch.autocast)
  3. BF16       SingleDeviceMuon  — pure BF16, round-to-nearest, no stochastic rounding
  4. BF16 + SR  MuonSR    — stochastic rounding, PyTorch implementation
  5. BF16 + SR  TritonMuonSR        — stochastic rounding, fused Triton kernels

All five runs start from identical weights and see identical batches.
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from muon import SingleDeviceMuonWithAuxAdam
from muon_sr import (
    StochasticAccumulator,
    MuonSRWithAuxAdam,
    TritonAccumulator,
    TritonMuonSRWithAuxAdam,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(x.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    def __init__(self, vocab_size=256, d_model=384, n_heads=6, n_layers=8, max_seq_len=256):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks    = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        T = idx.size(1)
        x = self.embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_corpus(path: str) -> torch.Tensor:
    """
    Load corpus bytes. Tiles the data until it is at least 2 MB so there
    is enough variety for training.
    """
    with open(path, "rb") as f:
        raw = f.read()

    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()

    min_bytes = 2 * 1024 * 1024  # tile up to ~2 MB
    if data.numel() < min_bytes:
        repeats = (min_bytes + data.numel() - 1) // data.numel()
        data = data.repeat(repeats)

    return data


def make_dataset(
    data: torch.Tensor,
    total_steps: int,
    grad_accum: int,
    batch_size: int,
    seq_len: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-sample all (x, y) pairs so both runs see identical batches.
    y is x shifted by one position (next-byte prediction).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    n_batches = total_steps * grad_accum
    max_start = data.numel() - seq_len - 1

    starts = torch.randint(0, max_start, (n_batches, batch_size), generator=rng)

    # Build index tensors: [n_batches, batch_size, seq_len]
    offsets = torch.arange(seq_len)
    xs = data[(starts.unsqueeze(-1) + offsets)]          # [..., seq_len]
    ys = data[(starts.unsqueeze(-1) + offsets + 1)]      # next byte

    return xs.to(device), ys.to(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_param_groups(model):
    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        is_embedding_or_head = any(k in name for k in ("embed", "head", "pos_embed"))
        is_norm_or_bias      = any(k in name for k in ("ln", "bias"))
        if param.ndim >= 2 and not is_embedding_or_head and not is_norm_or_bias:
            muon_params.append(param)
        else:
            adam_params.append(param)
    return [
        dict(params=muon_params, lr=0.02, momentum=0.95, weight_decay=0.01, use_muon=True),
        dict(params=adam_params, lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1, use_muon=False),
    ]


def run(label, model, optimizer, xs, ys, grad_accum, total_steps, log_every,
        accumulator_cls=None, autocast=False):
    """
    accumulator_cls: StochasticAccumulator / TritonAccumulator for BF16 SR grad accum.
                     None for native PyTorch accumulation (FP32, AMP, naive BF16).
    autocast:        True for AMP — wraps forward+loss in torch.autocast("cuda", dtype=bfloat16).
                     Parameters remain FP32; gradients are scaled back to FP32 automatically.
    """
    if accumulator_cls is not None:
        accumulator_cls.assign_hooks(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    loss_log = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(total_steps):
        step_loss = 0.0
        for micro in range(grad_accum):
            idx = step * grad_accum + micro
            x, y = xs[idx], ys[idx]
            if autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss / grad_accum).backward()
            step_loss += loss.item() / grad_accum

        if accumulator_cls is not None:
            accumulator_cls.reassign_grad_buffer(model)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss_log.append(step_loss)

        if (step + 1) % log_every == 0:
            avg = sum(loss_log[-log_every:]) / log_every
            lr  = scheduler.get_last_lr()[0]
            print(f"  [{label}] step {step+1:3d}/{total_steps}  loss {avg:.4f}  lr {lr:.2e}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return loss_log, elapsed, peak_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(corpus_path: str):
    device      = "cuda"
    dtype       = torch.bfloat16
    vocab_size  = 256
    batch_size  = 32
    seq_len     = 128
    grad_accum  = 4
    total_steps = 1000
    log_every   = 20
    seed        = 42

    torch.manual_seed(seed)

    data = build_corpus(corpus_path)
    print(f"Corpus size : {data.numel():,} bytes")

    xs, ys = make_dataset(data, total_steps, grad_accum, batch_size, seq_len, device, seed)

    base_model = SmallGPT(vocab_size=vocab_size).to(device=device, dtype=dtype)
    print(f"Parameters  : {sum(p.numel() for p in base_model.parameters()):,}\n")

    def fp32_model():
        m = SmallGPT(vocab_size=vocab_size).to(device=device, dtype=torch.float32)
        for p_dst, p_src in zip(m.parameters(), base_model.parameters()):
            p_dst.data.copy_(p_src.data.float())
        return m

    # ---- 1. FP32 baseline ---------------------------------------------------
    print("=== 1. FP32 SingleDeviceMuon (no quantization, baseline) ===")
    model_fp32 = fp32_model()
    optim_fp32 = SingleDeviceMuonWithAuxAdam(make_param_groups(model_fp32))
    losses_fp32, time_fp32, mem_fp32 = run(
        "FP32   ", model_fp32, optim_fp32,
        xs, ys, grad_accum, total_steps, log_every,
    )

    # ---- 2. AMP BF16 (FP32 params + BF16 forward) --------------------------
    print("\n=== 2. AMP BF16 SingleDeviceMuon (FP32 params + BF16 autocast) ===")
    model_amp = fp32_model()
    optim_amp = SingleDeviceMuonWithAuxAdam(make_param_groups(model_amp))
    losses_amp, time_amp, mem_amp = run(
        "AMP    ", model_amp, optim_amp,
        xs, ys, grad_accum, total_steps, log_every,
        autocast=True,
    )

    # ---- 3. Pure BF16 naive (round-to-nearest) ------------------------------
    print("\n=== 3. BF16 SingleDeviceMuon (round-to-nearest, no stochastic rounding) ===")
    model_naive = copy.deepcopy(base_model)
    optim_naive = SingleDeviceMuonWithAuxAdam(make_param_groups(model_naive))
    losses_naive, time_naive, mem_naive = run(
        "Naive  ", model_naive, optim_naive,
        xs, ys, grad_accum, total_steps, log_every,
    )

    # ---- 4. BF16 + SR (PyTorch) ---------------------------------------------
    print("\n=== 4. MuonSRWithAuxAdam (PyTorch + SR) ===")
    model_pt = copy.deepcopy(base_model)
    optim_pt = MuonSRWithAuxAdam(make_param_groups(model_pt))
    losses_pt, time_pt, mem_pt = run(
        "SR-PT  ", model_pt, optim_pt,
        xs, ys, grad_accum, total_steps, log_every,
        accumulator_cls=StochasticAccumulator,
    )

    # ---- 5. BF16 + SR (Triton + TritonAccumulator) --------------------------
    print("\n=== 5. TritonMuonSRWithAuxAdam + TritonAccumulator ===")
    model_tr = copy.deepcopy(base_model)
    optim_tr = TritonMuonSRWithAuxAdam(make_param_groups(model_tr))
    losses_tr, time_tr, mem_tr = run(
        "SR-Tri ", model_tr, optim_tr,
        xs, ys, grad_accum, total_steps, log_every,
        accumulator_cls=TritonAccumulator,
    )

    # ---- 6. BF16 + SR (Triton only, no accumulator) -------------------------
    print("\n=== 6. TritonMuonSRWithAuxAdam (no accumulator) ===")
    model_tr_no_acc = copy.deepcopy(base_model)
    optim_tr_no_acc = TritonMuonSRWithAuxAdam(make_param_groups(model_tr_no_acc))
    losses_tr_no_acc, time_tr_no_acc, mem_tr_no_acc = run(
        "SR-Tri-noAcc", model_tr_no_acc, optim_tr_no_acc,
        xs, ys, grad_accum, total_steps, log_every,
        accumulator_cls=None,
    )

    # ---- Summary ------------------------------------------------------------
    results = [
        ("FP32 Muon          ", losses_fp32,       time_fp32,       mem_fp32,       "(baseline)"),
        ("AMP BF16 Muon      ", losses_amp,        time_amp,        mem_amp,        ""),
        ("NaiveBF16 Muon     ", losses_naive,      time_naive,      mem_naive,      ""),
        ("MuonSR  PT         ", losses_pt,         time_pt,         mem_pt,         ""),
        ("MuonSR  Tri        ", losses_tr,         time_tr,         mem_tr,         ""),
        ("MuonSR  Tri noAcc  ", losses_tr_no_acc,  time_tr_no_acc,  mem_tr_no_acc,  ""),
    ]
    print("\n=== Summary ===")
    print(f"  {'Variant':<26} {'Final Loss':>10}  {'Time':>7}  {'Peak MB':>9}  {'vs FP32 loss':>13}")
    print(f"  {'-'*72}")
    for name, losses, elapsed, peak_mb, note in results:
        diff = abs(losses[-1] - losses_fp32[-1])
        print(f"  {name:<26} {losses[-1]:>10.4f}  {elapsed:>6.1f}s  {peak_mb:>8.0f}M  {diff:>+13.4f}  {note}")

    print(f"\n  Triton speedup over PyTorch SR    : {time_pt / time_tr:.2f}x")
    print(f"  Accumulator overhead (Triton)     : {time_tr / time_tr_no_acc:.2f}x")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to a corpus file as the first argument.")
    main(corpus_path=sys.argv[1])
