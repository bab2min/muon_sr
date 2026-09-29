import random
import torch
from torch.optim import Optimizer

import triton
import triton.language as tl


@triton.jit
def _adamw_kernel(
    p_ptr,
    g_ptr,
    ema_ptr,
    ema_sq_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction,
    bias_correction_sqrt,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load: p in bf16, rest in bf16 (upcast to fp32 inline)
    p_bf16 = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)
    ema = tl.load(ema_ptr + offsets, mask=mask).to(tl.float32)
    ema_sq = tl.load(ema_sq_ptr + offsets, mask=mask).to(tl.float32)

    # adam moment updates
    ema = beta1 * ema + (1.0 - beta1) * g
    ema_sq = beta2 * ema_sq + (1.0 - beta2) * g * g

    # bias-corrected step size and denominator
    step_size = lr / bias_correction
    denom = tl.sqrt(ema_sq) / bias_correction_sqrt + eps

    # decoupled weight decay with plain lr (standard AdamW)
    p_bf16 = p_bf16 * (1.0 - lr * weight_decay)

    # parameter update
    p_bf16 = p_bf16 - step_size * ema / denom

    # stochastic rounding: p and moments → bf16
    # Philox RNG: produce one 32-bit random int per element (keep as int32, no float conversion)
    rand_ema    = tl.randint(seed,     offsets).to(tl.int32)
    rand_ema_sq = tl.randint(seed + 1, offsets).to(tl.int32)
    rand_p      = tl.randint(seed + 2, offsets).to(tl.int32)

    # reinterpret float32 bits as int32, add low-16 noise, mask, reinterpret back
    ema_i = ema.to(dtype=tl.int32, bitcast=True)
    ema_i = (ema_i + (rand_ema & 0xFFFF)) & -65536
    ema_rounded = ema_i.to(dtype=tl.float32, bitcast=True)

    ema_sq_i = ema_sq.to(dtype=tl.int32, bitcast=True)
    ema_sq_i = (ema_sq_i + (rand_ema_sq & 0xFFFF)) & -65536
    ema_sq_rounded = ema_sq_i.to(dtype=tl.float32, bitcast=True)

    p_i = p_bf16.to(dtype=tl.int32, bitcast=True)
    p_i = (p_i + (rand_p & 0xFFFF)) & -65536
    p_rounded = p_i.to(dtype=tl.float32, bitcast=True)

    # store back as bf16
    tl.store(ema_ptr + offsets, ema_rounded.to(tl.bfloat16), mask=mask)
    tl.store(ema_sq_ptr + offsets, ema_sq_rounded.to(tl.bfloat16), mask=mask)
    tl.store(p_ptr + offsets, p_rounded.to(tl.bfloat16), mask=mask)


def _adamw_step(
    p: torch.Tensor,
    g: torch.Tensor,
    ema: torch.Tensor,
    ema_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    param_idx: int,
):
    bias_correction = 1.0 - beta1 ** step
    bias_correction_sqrt = (1.0 - beta2 ** step) ** 0.5

    n = p.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # different seed per (step, tensor). The parameter index is used instead of the storage
    # address so that the noise is the same across processes, e.g. when resuming from a checkpoint.
    seed = (step * 2654435761 ^ param_idx * 1234567891) & 0xFFFFFFFF

    _adamw_kernel[grid](
        p,
        g,
        ema,
        ema_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction,
        bias_correction_sqrt,
        n,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )


class TritonAdamW(Optimizer):
    r"""
    AdamW with stochastic rounding implemented as a single fused Triton kernel.

    Arguments:
        params: parameters to optimize
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float]): EMA coefficients (default: (0.9, 0.999))
        eps (float): denominator epsilon (default: 1e-8)
        weight_decay (float): decoupled weight decay (default: 0)
        centralization (float): gradient centralization strength (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        param_offset = 0
        for group in self.param_groups:
            group_offset = param_offset
            param_offset += len(group["params"])
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]

            for i, p in enumerate(group["params"]):
                assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TritonAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state["ema_squared"] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                state["step"] += 1

                # centralization: cross-element reduction stays outside the kernel
                if centralization != 0:
                    grad = grad.to(torch.float32)
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )
                    grad = grad.to(torch.bfloat16)

                _adamw_step(
                    p=p,
                    g=grad,
                    ema=state["ema"],
                    ema_sq=state["ema_squared"],
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    step=state["step"],
                    param_idx=group_offset + i,
                )

        return loss


@triton.jit
def _grad_accum_kernel(
    acc_ptr,
    grad_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    acc  = tl.load(acc_ptr  + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)

    acc = acc + grad

    # stochastic round accumulated sum → bf16
    rand = tl.randint(seed, offsets).to(tl.int32)
    acc_i = acc.to(dtype=tl.int32, bitcast=True)
    acc_i = (acc_i + (rand & 0xFFFF)) & -65536
    acc_rounded = acc_i.to(dtype=tl.float32, bitcast=True)

    tl.store(acc_ptr + offsets, acc_rounded.to(tl.bfloat16), mask=mask)


class TritonSRAccumulator:
    """
    Drop-in replacement for SRAccumulator using a fused Triton kernel.

    Fuses: BF16 load(acc) + FP32 load(grad) → FP32 add → stochastic round → BF16 store
    into a single kernel launch per parameter per microbatch, eliminating temporary tensors.

    Usage is identical to SRAccumulator:

        TritonSRAccumulator.assign_hooks(model)

        for step in range(total_steps):
            for _ in range(grad_accum):
                loss = model(x)
                loss.backward()
            TritonSRAccumulator.reassign_grad_buffer(model)
            optimizer.step()
            optimizer.zero_grad()
    """

    @staticmethod
    def _accum_hook(p):
        n = p.grad.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        seed = random.getrandbits(32)

        if hasattr(p, "acc_grad"):
            _grad_accum_kernel[grid](p.acc_grad, p.grad, n, seed, BLOCK_SIZE=BLOCK_SIZE)
            del p.grad
        else:
            p.acc_grad = torch.zeros_like(p.data, dtype=torch.bfloat16)
            _grad_accum_kernel[grid](p.acc_grad, p.grad, n, seed, BLOCK_SIZE=BLOCK_SIZE)
            del p.grad

    @staticmethod
    def reassign_grad_buffer(model):
        for _, p in model.named_parameters():
            if p.requires_grad and hasattr(p, "acc_grad"):
                p.grad = p.acc_grad
                del p.acc_grad

    @staticmethod
    def assign_hooks(model):
        hooks = []
        for _, p in model.named_parameters():
            if p.requires_grad:
                hooks.append(p.register_post_accumulate_grad_hook(TritonSRAccumulator._accum_hook))
        return hooks
