import torch
from torch.optim import Optimizer
from .stochastic_optim import copy_stochastic_


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization. Supports batched (ndim >= 2)."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(torch.float32)


def _muon_step_fp32(grad, buf, p_fp32, beta, nesterov, ns_steps, lr, weight_decay):
    """Shared Muon update logic (FP32). Modifies buf and p_fp32 in-place."""
    buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(buf, beta) if nesterov else buf.clone()

    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = _newton_schulz(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

    if weight_decay != 0:
        p_fp32.mul_(1 - lr * weight_decay)
    p_fp32.add_(update.reshape(original_shape), alpha=-lr)


class MuonSR(Optimizer):
    r"""
    Drop-in replacement for SingleDeviceMuon with BF16 parameters and stochastic rounding.

    Only for hidden weight matrices (ndim >= 2). For embeddings, heads, biases and gains,
    use a separate optimizer (e.g. TritonAdamW).

    Arguments:
        params: ndim >= 2 parameters to optimize
        lr (float): learning rate (default: 0.02)
        weight_decay (float): weight decay (default: 0)
        momentum (float): momentum coefficient (default: 0.95)
        nesterov (bool): Nesterov momentum (default: True)
        ns_steps (int): Newton-Schulz iterations (default: 5)
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MuonSR does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)

                grad = p.grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                buf = state["momentum_buffer"].to(torch.float32)

                _muon_step_fp32(grad, buf, p_fp32, beta, nesterov, ns_steps, lr, weight_decay)

                copy_stochastic_(state["momentum_buffer"], buf)
                copy_stochastic_(p, p_fp32)

        return loss


class MuonSRWithAuxAdam(Optimizer):
    r"""
    Drop-in replacement for SingleDeviceMuonWithAuxAdam with BF16 parameters and stochastic rounding.

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
        optimizer = MuonSRWithAuxAdam(param_groups)
    """

    def __init__(self, param_groups):
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
        super().__init__(param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                beta = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]

                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("MuonSRWithAuxAdam does not support sparse gradients")

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)

                    grad = p.grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    buf = state["momentum_buffer"].to(torch.float32)

                    _muon_step_fp32(grad, buf, p_fp32, beta, nesterov, ns_steps, lr, weight_decay)

                    copy_stochastic_(state["momentum_buffer"], buf)
                    copy_stochastic_(p, p_fp32)

            else:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("MuonSRWithAuxAdam does not support sparse gradients")

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["ema"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["ema_squared"] = torch.zeros_like(p, dtype=torch.bfloat16)

                    state["step"] += 1
                    grad = p.grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema = state["ema"].to(torch.float32)
                    ema_sq = state["ema_squared"].to(torch.float32)

                    bias_correction = 1 - beta1 ** state["step"]
                    bias_correction_sqrt = (1 - beta2 ** state["step"]) ** 0.5
                    step_size = lr / bias_correction

                    ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                    ema_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (ema_sq.sqrt() / bias_correction_sqrt).add_(eps)

                    if weight_decay != 0:
                        p_fp32.mul_(1 - lr * weight_decay)
                    p_fp32.addcdiv_(ema, denom, value=-step_size)

                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_sq)
                    copy_stochastic_(p, p_fp32)

        return loss
