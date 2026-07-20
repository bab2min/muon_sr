import random
from collections import defaultdict

import torch
from torch.optim import Optimizer

import triton
import triton.language as tl

from .muon_sr import _newton_schulz, _batched_ns
from .triton_adamw import _adamw_step


@triton.jit
def _muon_prepare_kernel(
    g_ptr,
    buf_ptr,
    out_ptr,
    beta,
    nesterov: tl.constexpr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g   = tl.load(g_ptr  + offsets, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offsets, mask=mask).to(tl.float32)

    new_buf = buf + (1.0 - beta) * (g - buf)

    if nesterov:
        update = g + beta * (new_buf - g)
    else:
        update = new_buf

    rand_buf = tl.randint(seed, offsets).to(tl.int32)
    buf_i = new_buf.to(dtype=tl.int32, bitcast=True)
    buf_i = (buf_i + (rand_buf & 0xFFFF)) & -65536
    buf_rounded = buf_i.to(dtype=tl.float32, bitcast=True)

    tl.store(buf_ptr + offsets, buf_rounded.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offsets, update, mask=mask)


@triton.jit
def _muon_apply_kernel(
    p_ptr,
    update_ptr,
    lr,
    weight_decay,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p      = tl.load(p_ptr      + offsets, mask=mask).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask)

    p = p * (1.0 - lr * weight_decay)
    p = p - lr * update

    rand_p = tl.randint(seed, offsets).to(tl.int32)
    p_i = p.to(dtype=tl.int32, bitcast=True)
    p_i = (p_i + (rand_p & 0xFFFF)) & -65536
    p_rounded = p_i.to(dtype=tl.float32, bitcast=True)

    tl.store(p_ptr + offsets, p_rounded.to(tl.bfloat16), mask=mask)


def _run_prepare(p, grad, buf, beta, nesterov):
    """Momentum + Nesterov (Triton). Writes buf in-place (BF16 SR), returns FP32 pre-NS update."""
    n = p.numel()
    BLOCK_SIZE = 1024
    update = torch.empty(n, dtype=torch.float32, device=p.device)
    _muon_prepare_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        grad.view(-1), buf.view(-1), update,
        beta, nesterov, n, random.getrandbits(32), BLOCK_SIZE=BLOCK_SIZE,
    )
    return update


def _run_apply(p, update_ns, lr, weight_decay):
    """Weight decay + param update (Triton), stochastic round p in-place (BF16)."""
    n = p.numel()
    BLOCK_SIZE = 1024
    _muon_apply_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        p.view(-1), update_ns.reshape(-1),
        lr, weight_decay, n, random.getrandbits(32), BLOCK_SIZE=BLOCK_SIZE,
    )


def _triton_muon_step(p, grad, buf, beta, nesterov, ns_steps, lr, weight_decay):
    """Single-param Triton Muon step (no batching)."""
    update = _run_prepare(p, grad, buf, beta, nesterov)
    update_2d = update.view(p.size(0), -1) if p.ndim > 2 else update.view(p.shape)
    update_ns = _newton_schulz(update_2d, steps=ns_steps)
    update_ns.mul_(max(1.0, update_ns.size(-2) / update_ns.size(-1)) ** 0.5)
    _run_apply(p, update_ns.reshape(p.shape), lr, weight_decay)


def _triton_muon_group_step(params, grads, bufs, beta, nesterov, ns_steps, lr, weight_decay):
    """
    Batched Triton Muon step: same-shape params share a single prepare + NS + apply.

    Phase 1 (Triton): prepare each param — individual kernel launches
    Phase 2 (PyTorch): _batched_ns groups by shape — one NS call per unique shape
    Phase 3 (Triton): apply each param — individual kernel launches
    """
    updates = [_run_prepare(p, g, buf, beta, nesterov)
               for p, g, buf in zip(params, grads, bufs)]
    updates_ns = _batched_ns(params, updates, ns_steps)
    for p, update_ns in zip(params, updates_ns):
        _run_apply(p, update_ns, lr, weight_decay)


class TritonMuonSR(Optimizer):
    r"""
    Drop-in replacement for SingleDeviceMuon with fused Triton kernels and BF16 stochastic rounding.

    Only for hidden weight matrices (ndim >= 2). For embeddings, heads, biases and gains,
    use a separate optimizer (e.g. TritonAdamW).

    Arguments:
        params: ndim >= 2 parameters to optimize
        lr (float): learning rate (default: 0.02)
        weight_decay (float): weight decay (default: 0)
        momentum (float): momentum coefficient (default: 0.95)
        nesterov (bool): Nesterov momentum (default: True)
        ns_steps (int): Newton-Schulz iterations (default: 5)
        batch_ns (bool): group same-shape params into a single NS call (default: False)
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95,
                 nesterov=True, ns_steps=5, batch_ns=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, batch_ns=batch_ns)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            weight_decay = group["weight_decay"]
            beta         = group["momentum"]
            nesterov     = group["nesterov"]
            ns_steps     = group["ns_steps"]
            batch_ns     = group["batch_ns"]

            if batch_ns:
                active, grads, bufs = [], [], []
                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("TritonMuonSR does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["step"] += 1
                    active.append(p)
                    grads.append(p.grad)
                    bufs.append(state["momentum_buffer"])

                if active:
                    _triton_muon_group_step(active, grads, bufs, beta, nesterov, ns_steps, lr, weight_decay)

            else:
                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("TritonMuonSR does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["step"] += 1
                    _triton_muon_step(p, p.grad, state["momentum_buffer"],
                                      beta, nesterov, ns_steps, lr, weight_decay)

        return loss


class TritonMuonSRWithAuxAdam(Optimizer):
    r"""
    Drop-in replacement for SingleDeviceMuonWithAuxAdam with fused Triton kernels and BF16 stochastic rounding.

    Handles both Muon params (use_muon=True) and AdamW params (use_muon=False) in a single
    optimizer, matching the original SingleDeviceMuonWithAuxAdam interface exactly.

    Usage:
        hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2]
        embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        head_params = [model.lm_head.weight]

        param_groups = [
            dict(params=hidden_matrix_params, lr=0.02, momentum=0.95, weight_decay=0.01, use_muon=True),
            dict(params=head_params,   lr=0.22,  betas=(0.8, 0.95), eps=1e-10, weight_decay=0, use_muon=False),
            dict(params=embed_params,  lr=0.6,   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, use_muon=False),
            dict(params=scalar_params, lr=0.04,  betas=(0.8, 0.95), eps=1e-10, weight_decay=0, use_muon=False),
        ]
        optimizer = TritonMuonSRWithAuxAdam(param_groups, batch_ns=True)
    """

    def __init__(self, param_groups, batch_ns=False):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        self.batch_ns = batch_ns
        super().__init__(param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr           = group["lr"]
                weight_decay = group["weight_decay"]
                beta         = group["momentum"]
                nesterov     = group["nesterov"]
                ns_steps     = group["ns_steps"]

                if self.batch_ns:
                    active, grads, bufs = [], [], []
                    for p in group["params"]:
                        assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            raise RuntimeError("TritonMuonSRWithAuxAdam does not support sparse gradients")
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["step"] += 1
                        active.append(p)
                        grads.append(p.grad)
                        bufs.append(state["momentum_buffer"])

                    if active:
                        _triton_muon_group_step(active, grads, bufs, beta, nesterov, ns_steps, lr, weight_decay)

                else:
                    for p in group["params"]:
                        assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            raise RuntimeError("TritonMuonSRWithAuxAdam does not support sparse gradients")
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["step"] += 1
                        _triton_muon_step(p, p.grad, state["momentum_buffer"],
                                          beta, nesterov, ns_steps, lr, weight_decay)

            else:
                lr           = group["lr"]
                beta1, beta2 = group["betas"]
                eps          = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("TritonMuonSRWithAuxAdam does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["ema"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["ema_squared"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["step"] += 1
                    _adamw_step(
                        p=p, g=p.grad,
                        ema=state["ema"], ema_sq=state["ema_squared"],
                        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, step=state["step"],
                    )

        return loss
