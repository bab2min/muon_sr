"""bitsandbytes-style blockwise dynamic AdamW with BF16 weight SR.

Dynamic maps and quantizer search adapted from bitsandbytes (MIT).
See LICENSE.bitsandbytes. Weights are rounded only after the FP32 update.
"""
import math

import torch
from torch.optim import Optimizer
import triton
import triton.language as tl


def create_dynamic_map(signed=True):
    """The 256-entry bitsandbytes dynamic/unsigned-dynamic codebook."""
    data = []
    for i in range(7):
        boundaries = torch.linspace(0.1, 1, 2 ** (i if signed else i + 1) + 1,
                                    dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2
        data.extend((10 ** (-6 + i) * means).tolist())
        if signed:
            data.extend((-10 ** (-6 + i) * means).tolist())
    return torch.tensor(sorted(data + [0., 1.]), dtype=torch.float32)


@triton.jit
def _quantize(x, Q, SIGNED: tl.constexpr):
    # Same pivot/midpoint tie rules as bitsandbytes quantize_2D.
    pivot = tl.full(x.shape, 127, tl.int32)
    lower_pivot = tl.full(x.shape, 0, tl.int32)
    upper_pivot = tl.full(x.shape, 255, tl.int32)
    lower = tl.full(x.shape, -1. if SIGNED else 0., tl.float32)
    upper = tl.full(x.shape, 1., tl.float32)
    val = tl.load(Q + pivot)
    for shift in tl.static_range(7):
        delta = 64 >> shift
        right = x > val
        lower_pivot = tl.where(right, pivot, lower_pivot)
        upper_pivot = tl.where(right, upper_pivot, pivot)
        lower = tl.where(right, val, lower)
        upper = tl.where(right, upper, val)
        pivot += tl.where(right, delta, -delta)
        val = tl.load(Q + pivot)
    result = tl.where(x > val,
        tl.where(x > (upper + val) * .5, upper_pivot, pivot),
        tl.where(x < (lower + val) * .5, lower_pivot, pivot))
    if SIGNED:
        decoded = tl.load(Q + result)
        # Preserve the sign of tiny negative moments that would round to zero.
        mismatch = (decoded < 0) != (x < 0)
        result += tl.where(mismatch, tl.where(x > 0, 1, -1), 0)
    return result.to(tl.uint8)


@triton.jit
def _adamw_8bit_kernel(P, G, M, V, MS, VS, Q1, Q2, N, LR, B1, B2, EPS, WD,
                       BC1, BC2, SEED, BLOCK: tl.constexpr, QUANTIZED: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    p = tl.load(P + offsets, valid, other=0).to(tl.float32)
    g = tl.load(G + offsets, valid, other=0).to(tl.float32)
    if QUANTIZED:
        m_code = tl.load(M + offsets, valid, other=127).to(tl.int32)
        v_code = tl.load(V + offsets, valid, other=0).to(tl.int32)
        m = tl.load(Q1 + m_code) * tl.load(MS + block)
        v = tl.load(Q2 + v_code) * tl.load(VS + block)
    else:
        m = tl.load(M + offsets, valid, other=0).to(tl.float32)
        v = tl.load(V + offsets, valid, other=0).to(tl.float32)
    finite = (tl.abs(g) < float('inf'))
    m = tl.where(finite, B1 * m + (1 - B1) * g, 0.)
    v = tl.where(finite, B2 * v + (1 - B2) * g * g, 0.)
    # bitsandbytes 8-bit AdamW applies decoupled decay after the Adam step.
    updated = (p - (LR * tl.sqrt(BC2) / BC1) * m / (tl.sqrt(v) + tl.sqrt(BC2) * EPS)) * (1 - LR * WD)
    bits = updated.to(tl.int32, bitcast=True)
    noise = tl.randint(SEED, offsets).to(tl.int32) & 0xFFFF
    rounded = ((bits + noise) & -65536).to(tl.float32, bitcast=True)
    tl.store(P + offsets, tl.where(finite, rounded, p).to(tl.bfloat16), valid)
    if QUANTIZED:
        m_max = tl.max(tl.where(valid, tl.abs(m), 0), axis=0)
        v_max = tl.max(tl.where(valid, tl.abs(v), 0), axis=0)
        m_code = _quantize(m / tl.where(m_max > 0, m_max, 1.), Q1, True)
        v_code = _quantize(v / tl.where(v_max > 0, v_max, 1.), Q2, False)
        tl.store(M + offsets, m_code, valid)
        tl.store(V + offsets, v_code, valid)
        tl.store(MS + block, m_max)
        tl.store(VS + block, v_max)
    else:
        # Excluded tensors keep BF16 moments, with SR on both moment stores.
        m_bits = m.to(tl.int32, bitcast=True)
        v_bits = v.to(tl.int32, bitcast=True)
        m_noise = tl.randint(SEED + 1, offsets).to(tl.int32) & 0xFFFF
        v_noise = tl.randint(SEED + 2, offsets).to(tl.int32) & 0xFFFF
        m_sr = ((m_bits + m_noise) & -65536).to(tl.float32, bitcast=True)
        v_sr = ((v_bits + v_noise) & -65536).to(tl.float32, bitcast=True)
        tl.store(M + offsets, m_sr.to(tl.bfloat16), valid)
        tl.store(V + offsets, v_sr.to(tl.bfloat16), valid)


class TritonAdamW8bit(Optimizer):
    """Dynamic-codebook 8-bit moments, BF16-SR fallback, BF16 weight SR.

    Default blocks contain 256 elements. Tensors below min_8bit_size=4096 use
    BF16 moments with SR. Set group optim_bits=16 to exclude embeddings/other tensors;
    for_model() creates these groups automatically before any ZeRO flattening.
    CUDA, dense contiguous tensors only. No CPU offload, AMSGrad, or clipping.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0., block_size=256, min_8bit_size=4096, optim_bits=8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                    block_size=block_size, min_8bit_size=min_8bit_size,
                                    optim_bits=optim_bits))
        self._maps = {}
        for group in self.param_groups:
            self._validate_group(group)

    @classmethod
    def for_model(cls, model, min_8bit_size=4096, **kwargs):
        """Protect embedding (including tied heads) and small tensor states.

        A bare model.parameters() iterator contains no module-type information;
        use this helper or explicit optim_bits=16 groups to protect embeddings.
        """
        if 'optim_bits' in kwargs:
            raise ValueError('for_model selects optim_bits per group')
        embeddings = {id(p) for mod in model.modules() if isinstance(mod, torch.nn.Embedding)
                      for p in mod.parameters()}
        groups = {8: [], 16: []}
        for p in model.parameters():
            if p.requires_grad:
                bits = 16 if id(p) in embeddings or p.numel() < min_8bit_size else 8
                groups[bits].append(p)
        return cls([dict(params=params, optim_bits=bits) for bits, params in groups.items() if params],
                   min_8bit_size=min_8bit_size, **kwargs)

    @staticmethod
    def _validate_group(g):
        for key in ('lr', 'eps', 'weight_decay'):
            v = g[key]
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{key} must be a finite nonnegative scalar')
        if g['eps'] == 0 or len(g['betas']) != 2 or any(not 0 <= b < 1 for b in g['betas']):
            raise ValueError('eps must be positive and betas must be in [0,1)')
        block = g['block_size']
        if not isinstance(block, int) or block < 32 or block > 4096 or block & (block-1):
            raise ValueError('block_size must be a power of two between 32 and 4096')
        if g['optim_bits'] not in (8,16) or not isinstance(g['min_8bit_size'], int) or g['min_8bit_size'] < 0:
            raise ValueError('optim_bits must be 8/16; min_8bit_size must be nonnegative')

    def _init_state(self, p, group):
        bits = 16 if p.numel() < group['min_8bit_size'] else group['optim_bits']
        state = dict(step=0, format_version=2, block_size=group['block_size'], optim_bits=bits)
        dtype = torch.uint8 if bits == 8 else torch.bfloat16
        state.update(state1=torch.zeros_like(p, dtype=dtype), state2=torch.zeros_like(p, dtype=dtype))
        if bits == 8:
            if p.device not in self._maps:
                self._maps[p.device] = (create_dynamic_map(True).to(p.device), create_dynamic_map(False).to(p.device))
            state['qmap1'], state['qmap2'] = self._maps[p.device]
            blocks = triton.cdiv(p.numel(), group['block_size'])
            state['absmax1'] = torch.zeros(blocks, device=p.device, dtype=torch.float32)
            state['absmax2'] = torch.zeros_like(state['absmax1'])
            state['state1'].fill_(127)  # signed codebook's exact zero
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = 0
        for group in self.param_groups:
            self._validate_group(group)
            b1, b2 = group['betas']
            for p in group['params']:
                param_index = index
                index += 1
                if p.grad is None:
                    continue
                if p.dtype != torch.bfloat16 or not p.is_cuda:
                    raise ValueError('TritonAdamW8bit requires CUDA BF16 parameters')
                g = p.grad
                if g.is_sparse or not p.is_contiguous() or not g.is_contiguous():
                    raise ValueError('Parameters/gradients must be contiguous and dense')
                if not p.numel():
                    continue
                state = self.state[p]
                if not state:
                    state.update(self._init_state(p, group))
                expected_bits = 16 if p.numel() < group['min_8bit_size'] else group['optim_bits']
                if state['block_size'] != group['block_size'] or state['optim_bits'] != expected_bits:
                    raise ValueError('Cannot change quantization configuration after state initialization')
                state['step'] += 1
                n = state['step']
                seed = (n * 2654435761 ^ param_index * 1234567891) & 0xFFFFFFFF
                _adamw_8bit_kernel[(triton.cdiv(p.numel(), group['block_size']),)](
                    p, g, state['state1'], state['state2'],
                    state.get('absmax1', p), state.get('absmax2', p),
                    state.get('qmap1', p), state.get('qmap2', p), p.numel(),
                    group['lr'], b1, b2, group['eps'], group['weight_decay'],
                    1-b1**n, 1-b2**n, seed, BLOCK=group['block_size'],
                    QUANTIZED=state['optim_bits'] == 8)
        return loss

    def load_state_dict(self, state_dict):
        # Preserve uint8 codes and FP32 scales rather than casting them to p.dtype.
        groups = state_dict['param_groups']
        if len(groups) != len(self.param_groups) or any(
                len(a['params']) != len(b['params']) for a,b in zip(groups,self.param_groups)):
            raise ValueError('Checkpoint parameter groups do not match optimizer')
        mapping = dict(zip([i for g in groups for i in g['params']],
                           [p for g in self.param_groups for p in g['params']]))
        restored = {}
        keys = {'state1','state2','qmap1','qmap2','absmax1','absmax2'}
        for group in groups:
            self._validate_group(group)
            for ident in group['params']:
                state = state_dict['state'].get(ident, {})
                if not state:
                    continue
                p = mapping[ident]
                bits = 16 if p.numel() < group['min_8bit_size'] else group['optim_bits']
                if state.get('format_version') != 2 or state.get('block_size') != group['block_size'] or state.get('optim_bits') != bits:
                    raise ValueError('Incompatible state format/quantization configuration')
                specs = {key:(p.shape, torch.uint8 if bits == 8 else torch.bfloat16) for key in ('state1','state2')}
                if bits == 8:
                    specs.update({key:((256,),torch.float32) for key in ('qmap1','qmap2')})
                    specs.update({key:((triton.cdiv(p.numel(),group['block_size']),),torch.float32) for key in ('absmax1','absmax2')})
                restored[ident] = {}
                for key,(shape,dtype) in specs.items():
                    value = state[key]
                    if value.dtype != dtype or tuple(value.shape) != tuple(shape):
                        raise ValueError(f'Invalid checkpoint state: {key}')
                    restored[ident][key] = value.to(device=p.device,dtype=dtype).clone()
        stripped = dict(state_dict, state={i:{k:v for k,v in state.items() if k not in keys}
                                         for i,state in state_dict['state'].items()})
        super().load_state_dict(stripped)
        for ident,values in restored.items():
            self.state[mapping[ident]].update(values)
