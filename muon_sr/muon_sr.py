import torch
from torch.optim import Optimizer
from collections import defaultdict

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))



class SRAccumulator:
    """
    # init your model
    your_fancy_model = YourFancyModel(*your_model_args)

    # apply stochastic grad accumulator hooks
    SRAccumulator.assign_hooks(your_fancy_model)

    # training
    while True:
        loss = your_fancy_model.loss(*your_model_input)
        for _ in range(grad_accum_length):
            loss.backward()

        # apply grad buffer back
        SRAccumulator.reassign_grad_buffer(your_fancy_model)

        optimizer.step()
        optimizer.zero_grad()
    """

    @staticmethod
    def stochastic_grad_accum(p):
        # hack by adding attributes to "grad"
        if hasattr(p, "acc_grad"):
            acc_grad_fp32 = p.acc_grad.clone().to(torch.float32)
            # acc_grad_fp32 += fp_32_grad
            # upcast the gradient and then add it to p.grad
            acc_grad_fp32.add_(p.grad.to(torch.float32))
            copy_stochastic_(p.acc_grad, acc_grad_fp32)
            del acc_grad_fp32
            del p.grad
        else:
            p.acc_grad = p.grad.clone().to(torch.bfloat16)
            del p.grad

    @staticmethod
    def reassign_grad_buffer(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.grad = p.acc_grad
                del p.acc_grad

    @staticmethod
    def assign_hooks(model):
        hooks = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                hook = p.register_post_accumulate_grad_hook(
                    SRAccumulator.stochastic_grad_accum
                )
                hooks.append(hook)
        return hooks

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
    """Single-param Muon update (FP32). Modifies buf and p_fp32 in-place."""
    buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(buf, beta) if nesterov else buf.clone()
    update = _newton_schulz(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    if weight_decay != 0:
        p_fp32.mul_(1 - lr * weight_decay)
    p_fp32.add_(update, alpha=-lr)


def _init_stacked_bufs(params, device):
    """
    Pre-allocate stacked BF16 momentum buffers grouped by original shape.

    Only momentum (BF16) is pre-allocated. FP32 workspace is allocated transiently
    during each step and freed immediately after, keeping persistent memory minimal.

    Returns:
      shape_groups : {shape: [param, ...]}
      stacked_bufs : {shape: BF16 Tensor[N, *shape]}  — momentum, modified in-place
      buf_indices  : {id(p): int}  — index of p within its shape group
    """
    shape_groups = defaultdict(list)
    for p in params:
        shape_groups[p.shape].append(p)

    stacked_bufs = {}
    buf_indices  = {}

    for shape, grp in shape_groups.items():
        N = len(grp)
        stacked_bufs[shape] = torch.zeros(N, *shape, dtype=torch.bfloat16, device=device)
        for i, p in enumerate(grp):
            buf_indices[id(p)] = i

    return dict(shape_groups), stacked_bufs, buf_indices


def _muon_batch_step(shape_groups, stacked_bufs, buf_indices,
                     beta, nesterov, ns_steps, lr, weight_decay):
    """
    Batched PyTorch Muon step using pre-allocated stacked BF16 momentum buffers.
    FP32 workspace is allocated transiently per shape group and freed after use.
    All params of the same original shape are processed with a single NS call.
    """
    for shape, grp_params in shape_groups.items():
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
                work[buf_indices[id(p)]].copy_(p.grad.to(torch.float32))

            buf_fp32 = stacked_buf.to(torch.float32)
            buf_fp32.lerp_(work, 1 - beta)

            if nesterov:
                work.lerp_(buf_fp32, beta)
            else:
                work.copy_(buf_fp32)

            update_ns = _newton_schulz(work, steps=ns_steps)
            rows, cols = work.size(-2), work.size(-1)
            update_ns.mul_(max(1.0, rows / cols) ** 0.5)
            del work  # free FP32 workspace before per-param apply

            for p in active:
                i      = buf_indices[id(p)]
                p_fp32 = p.clone().to(torch.float32)
                if weight_decay != 0:
                    p_fp32.mul_(1 - lr * weight_decay)
                p_fp32.add_(update_ns[i], alpha=-lr)
                copy_stochastic_(stacked_buf[i], buf_fp32[i])
                copy_stochastic_(p, p_fp32)

        else:
            idx     = [buf_indices[id(p)] for p in active]
            sub_buf = stacked_buf[idx].clone()
            work    = torch.empty(Na, *shape, dtype=torch.float32, device=active[0].device)

            for j, p in enumerate(active):
                work[j].copy_(p.grad.to(torch.float32))

            buf_fp32 = sub_buf.to(torch.float32)
            buf_fp32.lerp_(work, 1 - beta)

            if nesterov:
                work.lerp_(buf_fp32, beta)
            else:
                work.copy_(buf_fp32)

            update_ns = _newton_schulz(work, steps=ns_steps)
            rows, cols = work.size(-2), work.size(-1)
            update_ns.mul_(max(1.0, rows / cols) ** 0.5)

            for j, p in enumerate(active):
                p_fp32 = p.clone().to(torch.float32)
                if weight_decay != 0:
                    p_fp32.mul_(1 - lr * weight_decay)
                p_fp32.add_(update_ns[j], alpha=-lr)
                copy_stochastic_(stacked_buf[idx[j]], buf_fp32[j])
                copy_stochastic_(p, p_fp32)


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
        batch_ns (bool): pre-allocate stacked momentum buffers and batch NS by shape (default: False)
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
                    self._batch_state[gid] = _init_stacked_bufs(group["params"], device)
                shape_groups, stacked_bufs, buf_indices = self._batch_state[gid]

                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is not None and p.grad.is_sparse:
                        raise RuntimeError("MuonSR does not support sparse gradients")

                _muon_batch_step(shape_groups, stacked_bufs, buf_indices,
                                 beta, nesterov, ns_steps, lr, weight_decay)

            else:
                for p in group["params"]:
                    assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("MuonSR does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    grad   = p.grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    buf    = state["momentum_buffer"].to(torch.float32)
                    _muon_step_fp32(grad, buf, p_fp32, beta, nesterov, ns_steps, lr, weight_decay)
                    copy_stochastic_(state["momentum_buffer"], buf)
                    copy_stochastic_(p, p_fp32)

        return loss


class MuonSRWithAuxAdam(Optimizer):
    r"""
    Drop-in replacement for SingleDeviceMuonWithAuxAdam with BF16 parameters and stochastic rounding.

    Handles both Muon params (use_muon=True) and AdamW params (use_muon=False) in a single
    optimizer, matching the original SingleDeviceMuonWithAuxAdam interface exactly.

    Arguments:
        param_groups: list of param group dicts with 'use_muon' flag
        batch_ns (bool): pre-allocate stacked momentum buffers and batch NS by shape (default: False)

    Usage:
        optimizer = MuonSRWithAuxAdam(param_groups, batch_ns=True)
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

        for group in self.param_groups:
            if group["use_muon"]:
                lr           = group["lr"]
                weight_decay = group["weight_decay"]
                beta         = group["momentum"]
                nesterov     = group["nesterov"]
                ns_steps     = group["ns_steps"]

                if self.batch_ns:
                    gid = id(group)
                    if gid not in self._batch_state:
                        device = group["params"][0].device
                        self._batch_state[gid] = _init_stacked_bufs(group["params"], device)
                    shape_groups, stacked_bufs, buf_indices = self._batch_state[gid]

                    for p in group["params"]:
                        assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                        if p.grad is not None and p.grad.is_sparse:
                            raise RuntimeError("MuonSRWithAuxAdam does not support sparse gradients")

                    _muon_batch_step(shape_groups, stacked_bufs, buf_indices,
                                     beta, nesterov, ns_steps, lr, weight_decay)

                else:
                    for p in group["params"]:
                        assert p.dtype == torch.bfloat16, "only bfloat16 is supported."
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            raise RuntimeError("MuonSRWithAuxAdam does not support sparse gradients")
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        grad   = p.grad.to(torch.float32)
                        p_fp32 = p.clone().to(torch.float32)
                        buf    = state["momentum_buffer"].to(torch.float32)
                        _muon_step_fp32(grad, buf, p_fp32, beta, nesterov, ns_steps, lr, weight_decay)
                        copy_stochastic_(state["momentum_buffer"], buf)
                        copy_stochastic_(p, p_fp32)

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
                        raise RuntimeError("MuonSRWithAuxAdam does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["ema"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["ema_squared"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["step"] += 1
                    grad   = p.grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema    = state["ema"].to(torch.float32)
                    ema_sq = state["ema_squared"].to(torch.float32)
                    bias_correction      = 1 - beta1 ** state["step"]
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
