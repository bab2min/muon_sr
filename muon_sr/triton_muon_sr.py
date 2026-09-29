import random

import torch
from torch.optim import Optimizer

import triton
import triton.language as tl

from .muon_sr import _newton_schulz, _init_stacked_bufs, _StackedBufsMixin
from .triton_adamw import _adamw_step


def _restore_global_step(state, groups):
    """Recover `_global_step` of batch_ns groups from the per-param steps recorded in `state`."""
    return max((state[p].get("step", 0) for g in groups for p in g["params"]), default=0)


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


def _deterministic_seed(step, param_idx, salt=0):
    """Deterministic seed from step + param_idx so all DDP ranks use the same SR noise."""
    return (step * 2654435761 ^ param_idx * 1234567891 ^ salt) & 0xFFFFFFFF


def _run_prepare(p, grad, buf, beta, nesterov, step, param_idx):
    """Momentum + Nesterov (Triton). Writes buf in-place (BF16 SR), returns FP32 pre-NS update."""
    n = p.numel()
    BLOCK_SIZE = 1024
    update = torch.empty(n, dtype=torch.float32, device=p.device)
    _muon_prepare_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        grad.view(-1), buf.view(-1), update,
        beta, nesterov, n, _deterministic_seed(step, param_idx, salt=0), BLOCK_SIZE=BLOCK_SIZE,
    )
    return update


def _run_apply(p, update_ns, lr, weight_decay, step, param_idx):
    """Weight decay + param update (Triton), stochastic round p in-place (BF16)."""
    n = p.numel()
    BLOCK_SIZE = 1024
    _muon_apply_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        p.view(-1), update_ns.reshape(-1),
        lr, weight_decay, n, _deterministic_seed(step, param_idx, salt=1), BLOCK_SIZE=BLOCK_SIZE,
    )


def _triton_muon_step(p, grad, buf, beta, nesterov, ns_steps, lr, weight_decay, step, param_idx):
    """Single-param Triton Muon step (no batching)."""
    update = _run_prepare(p, grad, buf, beta, nesterov, step, param_idx)
    update_ns = _newton_schulz(update.view(p.shape), steps=ns_steps)
    update_ns.mul_(max(1.0, update_ns.size(-2) / update_ns.size(-1)) ** 0.5)
    _run_apply(p, update_ns, lr, weight_decay, step, param_idx)


def _triton_muon_batch_step(shape_groups, stacked_bufs, buf_indices,
                            beta, nesterov, ns_steps, lr, weight_decay, step):
    """
    Batched Triton Muon step using pre-allocated stacked BF16 momentum buffers.
    FP32 workspace is allocated transiently per shape group and freed after use.

    Phase 1 (Triton): prepare on the full stacked buf/grad — 1 kernel launch per unique shape
    Phase 2 (PyTorch): Newton-Schulz on stacked update    — 1 NS call per unique shape
    Phase 3 (Triton): apply on stacked params             — 1 kernel launch per unique shape
    """
    BLOCK_SIZE = 1024
    for shape_idx, (shape, grp_params) in enumerate(shape_groups.items()):
        active = [p for p in grp_params if p.grad is not None]
        if not active:
            continue

        stacked_buf = stacked_bufs[shape]   # BF16 [N, *shape]
        Na          = len(active)
        all_active  = Na == len(grp_params)

        if all_active:
            # Transient FP32 workspace — allocated here, freed at end of block
            work = torch.empty(Na, *shape, dtype=torch.float32, device=active[0].device)
            for p in active:
                work[buf_indices[id(p)]].copy_(p.grad)

            n = stacked_buf.numel()
            # Phase 1: prepare entire stack in one kernel
            _muon_prepare_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
                work.view(-1), stacked_buf.view(-1), work.view(-1),
                beta, nesterov, n, _deterministic_seed(step, shape_idx, salt=0), BLOCK_SIZE=BLOCK_SIZE,
            )

            # Phase 2: NS operates on last two dims — pass stacked tensor directly
            update_ns = _newton_schulz(work, steps=ns_steps)
            rows, cols = work.size(-2), work.size(-1)
            update_ns.mul_(max(1.0, rows / cols) ** 0.5)
            del work  # free FP32 workspace before phase 3

            # Phase 3: apply per-param using update_ns[i] views — no extra allocation
            for p in active:
                param_idx = shape_idx * 10000 + buf_indices[id(p)]
                _run_apply(p, update_ns[buf_indices[id(p)]], lr, weight_decay, step, param_idx)

        else:
            for p in active:
                param_idx = shape_idx * 10000 + buf_indices[id(p)]
                _triton_muon_step(p, p.grad, stacked_bufs[shape][buf_indices[id(p)]],
                                  beta, nesterov, ns_steps, lr, weight_decay, step, param_idx)


class TritonMuonSR(_StackedBufsMixin, Optimizer):
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
        batch_ns (bool): pre-allocate stacked buffers and batch NS by shape (default: False)
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95,
                 nesterov=True, ns_steps=5, batch_ns=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, batch_ns=batch_ns)
        super().__init__(params, defaults)
        self._batch_state = {}

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

            if group["batch_ns"]:
                gid = id(group)
                if gid not in self._batch_state:
                    device = group["params"][0].device
                    self._batch_state[gid] = _init_stacked_bufs(group["params"], device, self.state)
                shape_groups, stacked_bufs, buf_indices = self._batch_state[gid]

                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is not None and p.grad.is_sparse:
                        raise RuntimeError("TritonMuonSR does not support sparse gradients")

                if not hasattr(self, '_global_step'):
                    self._global_step = _restore_global_step(
                        self.state, [g for g in self.param_groups if g["batch_ns"]])
                self._global_step += 1
                # Recorded per param so that the SR seeds continue after load_state_dict().
                for p in group["params"]:
                    self.state[p]["step"] = self._global_step
                _triton_muon_batch_step(shape_groups, stacked_bufs, buf_indices,
                                        beta, nesterov, ns_steps, lr, weight_decay,
                                        self._global_step)

            else:
                for param_idx, p in enumerate(group["params"]):
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
                                      beta, nesterov, ns_steps, lr, weight_decay,
                                      state["step"], param_idx)

        return loss


class TritonMuonSRWithAuxAdam(_StackedBufsMixin, Optimizer):
    r"""
    Drop-in replacement for SingleDeviceMuonWithAuxAdam with fused Triton kernels and BF16 stochastic rounding.

    Handles both Muon params (use_muon=True) and AdamW params (use_muon=False) in a single
    optimizer, matching the original SingleDeviceMuonWithAuxAdam interface exactly.

    Arguments:
        param_groups: list of param group dicts with 'use_muon' flag
        batch_ns (bool): pre-allocate stacked buffers and batch NS by shape (default: False)

    Usage:
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
        self._batch_state = {}
        super().__init__(param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        param_offset = 0
        for group in self.param_groups:
            group_offset = param_offset
            param_offset += len(group["params"])
            if group["use_muon"]:
                lr           = group["lr"]
                weight_decay = group["weight_decay"]
                beta         = group["momentum"]
                nesterov     = group["nesterov"]
                ns_steps     = group["ns_steps"]

                if group["params"]:
                    if self.batch_ns:
                        gid = id(group)
                        if gid not in self._batch_state:
                            device = group["params"][0].device
                            self._batch_state[gid] = _init_stacked_bufs(group["params"], device, self.state)
                        shape_groups, stacked_bufs, buf_indices = self._batch_state[gid]

                        for p in group["params"]:
                            assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                            if p.grad is not None and p.grad.is_sparse:
                                raise RuntimeError("TritonMuonSRWithAuxAdam does not support sparse gradients")

                        if not hasattr(self, '_global_step'):
                            self._global_step = _restore_global_step(
                                self.state, [g for g in self.param_groups if g["use_muon"]])
                        self._global_step += 1
                        # Recorded per param so that the SR seeds continue after load_state_dict().
                        for p in group["params"]:
                            self.state[p]["step"] = self._global_step
                        _triton_muon_batch_step(shape_groups, stacked_bufs, buf_indices,
                                                beta, nesterov, ns_steps, lr, weight_decay,
                                                self._global_step)

                    else:
                        for param_idx, p in enumerate(group["params"]):
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
                                              beta, nesterov, ns_steps, lr, weight_decay,
                                              state["step"], param_idx)

            else:
                lr           = group["lr"]
                beta1, beta2 = group["betas"]
                eps          = group["eps"]
                weight_decay = group["weight_decay"]

                for i, p in enumerate(group["params"]):
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
                        param_idx=group_offset + i,
                    )

        return loss
