"""
Sync-free grouped GEMM for unquantized (FP16/BF16-as-fp16) MoE experts at small row counts, the shape
of batched decode: B jobs give B tokens and B * top_k (token, expert) assignments over E experts.

The dense per-expert path costs about ten launches per active expert and a host sync per layer (at
E = 512 that is about 5,100 launches, 80 ms per layer, launch bound). This path runs the layer as a
fixed, small number of launches with no host readback, so it is also CUDA-graph capturable:

    layout  assignments sorted by expert and padded per expert to a BM-row block (all on the device,
            fixed-size buffers: at most A + min(E, A) * (BM - 1) padded rows)
    gate+up one Triton grouped GEMM over the blocks (each program reads its expert's weights once),
            activation through the same ext kernel as the loop
    down    one Triton grouped GEMM, output scaled by the routing weight
    reduce  per-token sum of its top_k rows in ascending expert order

Numerics against the loop: the same fp16 GEMM inputs, fp32 accumulation, the loop's output dtype per
projection (gate/up: interm dtype, down: fp32), the same activation call, the routing weight applied
to the fp32 down output, and the loop's order of fp32 additions per token (ascending expert id, from
zero). Only the summation order inside each dot product (K loop tiling) differs from cuBLAS, so the
results agree to fp32 rounding in the GEMM and, through the fp16 rounding of gate/up, to the odd
1-ulp fp16 difference in an intermediate.

Experts that receive no row are never read. At B = 128 and E = 512 about 470 experts are touched
(roughly all of them), so a layer is bound by reading its expert weights once.
"""
from __future__ import annotations
import os
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

MOE_DECODE = os.environ.get("EXL3_MOE_DECODE", "0") == "1"
# Largest row count (tokens) served by this path; beyond it the dense loop or grouped path runs
MOE_DECODE_MAX_ROWS = int(os.environ.get("EXL3_MOE_DECODE_MAX_ROWS", 4096))
# Runtime switch for A/B comparisons in one process (util/trace_equiv.py): the weights are stacked for
# the path either way, and with the switch off the dense loop runs on the same stacked views
RUNTIME_ON = [True]
# Verification only (util/trace_equiv.py): run the block in float32 with no intermediate rounding, as the
# yardstick both the loop and the Triton path are measured against
GOLD = [False]
# CUDA graphs of the whole expert block (EXL3_MOE_DECODE_GRAPH=1): the block is about 40 launches and
# 0.8 ms of host time per layer, which becomes the step time once the GPU is fast enough. Row counts are
# padded up to a bucket (padding rows route to expert 0 with weight 0 and are dropped), one graph per
# (layer, bucket), all sharing one memory pool. Off by default.
MOE_DECODE_GRAPH = os.environ.get("EXL3_MOE_DECODE_GRAPH", "0") == "1"
GRAPH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512)
_graph_pool = []


if HAVE_TRITON:

    @triton.jit
    def _grouped_gemm_kernel(
        a_ptr, w1_ptr, w2_ptr, o1_ptr, o2_ptr, rw_ptr, sorted_ptr, eblk_ptr,
        N, K, num_assign, top_k, E,
        stride_am, stride_we, stride_wk, stride_om,
        A_BY_TOKEN: tl.constexpr, DUAL: tl.constexpr, OUT_F32: tl.constexpr, MUL_RW: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        e = tl.load(eblk_ptr + pid_m)
        if e >= E:
            return
        offs_m = pid_m * BM + tl.arange(0, BM)
        rows = tl.load(sorted_ptr + offs_m).to(tl.int64)          # assignment ids; num_assign pads
        valid = rows < num_assign
        if A_BY_TOKEN:
            a_row = tl.where(valid, rows // top_k, 0)
        else:
            a_row = tl.where(valid, rows, 0)
        offs_n = pid_n * BN + tl.arange(0, BN)
        nmask = offs_n < N
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :]
        w_off = e.to(tl.int64) * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]
        w1_ptrs = w1_ptr + w_off
        acc1 = tl.zeros((BM, BN), dtype = tl.float32)
        if DUAL:
            w2_ptrs = w2_ptr + w_off
            acc2 = tl.zeros((BM, BN), dtype = tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            kmask = offs_k < K - k * BK
            a = tl.load(a_ptrs, mask = valid[:, None] & kmask[None, :], other = 0.0)
            b1 = tl.load(w1_ptrs, mask = kmask[:, None] & nmask[None, :], other = 0.0)
            acc1 = tl.dot(a, b1, acc1)
            if DUAL:
                b2 = tl.load(w2_ptrs, mask = kmask[:, None] & nmask[None, :], other = 0.0)
                acc2 = tl.dot(a, b2, acc2)
                w2_ptrs += BK * stride_wk
            a_ptrs += BK
            w1_ptrs += BK * stride_wk
        if MUL_RW:
            rw = tl.load(rw_ptr + rows, mask = valid, other = 0.0).to(tl.float32)
            acc1 = acc1 * rw[:, None]
        o_off = rows[:, None] * stride_om + offs_n[None, :]
        omask = valid[:, None] & nmask[None, :]
        if OUT_F32:
            tl.store(o1_ptr + o_off, acc1, mask = omask)
        else:
            tl.store(o1_ptr + o_off, acc1.to(tl.float16), mask = omask)
        if DUAL:
            if OUT_F32:
                tl.store(o2_ptr + o_off, acc2, mask = omask)
            else:
                tl.store(o2_ptr + o_off, acc2.to(tl.float16), mask = omask)

    @triton.jit
    def _reduce_kernel(d_ptr, order_ptr, out_ptr, H, TOP_K: tl.constexpr, BH: tl.constexpr):
        t = tl.program_id(0)
        offs = tl.program_id(1) * BH + tl.arange(0, BH)
        mask = offs < H
        acc = tl.zeros((BH,), dtype = tl.float32)
        for k in tl.static_range(TOP_K):
            slot = tl.load(order_ptr + t * TOP_K + k).to(tl.int64)
            acc += tl.load(d_ptr + (t * TOP_K + slot) * H + offs, mask = mask, other = 0.0)
        tl.store(out_ptr + t * H + offs, acc, mask = mask)


def pick_block_m(num_assign: int, num_experts: int) -> int:
    """Row block per expert: small for decode (few rows per expert), larger as the average grows"""
    avg = num_assign / max(num_experts, 1)
    if avg <= 6: return 16
    if avg <= 14: return 32
    return 64


def build_layout(flat_expert: torch.Tensor, num_experts: int, bm: int):
    """Device-side expert-sorted, block-padded layout with fixed-size outputs.
    Returns (sorted_ids[P] int32, expert_of_block[P // bm] int32): row r of block b holds the
    assignment index sorted_ids[b * bm + r] (num_assign marks padding), expert_of_block[b] is the
    block's expert (num_experts marks unused blocks)"""
    A = flat_expert.numel()
    E = num_experts
    dev = flat_expert.device
    P = A + min(E, A) * (bm - 1)
    P = (P + bm - 1) // bm * bm
    order = flat_expert.argsort()
    # scatter_add_, not bincount: bincount reads its maximum back to the host (a sync, and not capturable)
    counts = torch.zeros((E,), dtype = torch.long, device = dev).scatter_add_(0, flat_expert, torch.ones_like(flat_expert))
    nblk = (counts + (bm - 1)) // bm
    cum_blk = nblk.cumsum(0)
    first = counts.cumsum(0) - counts
    exp_sorted = flat_expert[order]
    rank = torch.arange(A, device = dev) - first[exp_sorted]
    pos = (cum_blk - nblk)[exp_sorted] * bm + rank
    sorted_ids = torch.full((P,), A, dtype = torch.int32, device = dev)
    sorted_ids[pos] = order.to(torch.int32)
    eblock = torch.searchsorted(cum_blk, torch.arange(P // bm, device = dev), right = True).to(torch.int32)
    return sorted_ids, eblock


def _tiles(n: int, k: int, bm: int):
    bn = 128 if n >= 2048 else 64
    bk = 64 if k % 64 == 0 else 32
    nw = 4 if bm <= 32 else 8
    return bn, bk, nw


def fp32_reference(y, selected, rweights, wg, wu, wd):
    """The routed sum in float32 end to end (gated silu experts), experts added in ascending order. Host-synced
    and slow: for verification only"""
    T, top_k = selected.shape
    E = wu.shape[0]
    flat = selected.reshape(-1)
    tok = torch.arange(T, device = y.device).repeat_interleave(top_k)
    order = flat.argsort()
    tok, w = tok[order], rweights.reshape(-1)[order].float()
    counts = torch.zeros((E,), dtype = torch.long, device = y.device).scatter_add_(0, flat, torch.ones_like(flat)).tolist()
    out = torch.zeros((T, wd.shape[2]), dtype = torch.float, device = y.device)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    start = 0
    for e in range(E):
        c = counts[e]
        if c == 0:
            continue
        t = tok[start:start + c]
        x = y.index_select(0, t).float()
        d = (torch.nn.functional.silu(x @ wg[e].float()) * (x @ wu[e].float())) @ wd[e].float()
        out.index_add_(0, t, d * w[start:start + c].unsqueeze(1))
        start += c
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return out


def fp16_moe_decode(y, selected, rweights, wg, wu, wd, activation, act_limit, interm_dtype, out_dtype_up,
                    out_dtype_gate, out_dtype_down, gated: bool, gateless_act=None):
    """y (T, K) half; selected (T, top_k) expert ids; rweights (T, top_k) half; w* stacked (E, in, out)
    half. Returns the routed sum (T, K_out) float32"""
    if GOLD[0]:
        return fp32_reference(y, selected, rweights, wg, wu, wd)
    T, top_k = selected.shape
    A = T * top_k
    E, K, I = wu.shape
    H = wd.shape[2]
    dev = y.device
    flat = selected.reshape(-1)
    bm = pick_block_m(A, E)
    sorted_ids, eblock = build_layout(flat, E, bm)
    nblocks = eblock.shape[0]
    rw = rweights.reshape(-1)
    y = y.contiguous()

    up_f32 = out_dtype_up == torch.float
    u = torch.empty((A, I), dtype = torch.float if up_f32 else torch.half, device = dev)
    bn, bk, nw = _tiles(I, K, bm)
    if gated:
        g = torch.empty((A, I), dtype = torch.float if out_dtype_gate == torch.float else torch.half, device = dev)
        assert (out_dtype_gate == torch.float) == up_f32
        _grouped_gemm_kernel[(triton.cdiv(I, bn), nblocks)](
            y, wg, wu, g, u, rw, sorted_ids, eblock,
            I, K, A, top_k, E,
            y.stride(0), wu.stride(0), wu.stride(1), u.stride(0),
            A_BY_TOKEN = True, DUAL = True, OUT_F32 = up_f32, MUL_RW = False,
            BM = bm, BN = bn, BK = bk, num_warps = nw, num_stages = 3)
        a = u if interm_dtype == torch.half else torch.empty_like(u, dtype = torch.half)
        activation(g, u, a, act_limit)
        del g
    else:
        _grouped_gemm_kernel[(triton.cdiv(I, bn), nblocks)](
            y, wu, wu, u, u, rw, sorted_ids, eblock,
            I, K, A, top_k, E,
            y.stride(0), wu.stride(0), wu.stride(1), u.stride(0),
            A_BY_TOKEN = True, DUAL = False, OUT_F32 = up_f32, MUL_RW = False,
            BM = bm, BN = bn, BK = bk, num_warps = nw, num_stages = 3)
        a = gateless_act(u)
        if a.dtype != torch.half:
            a = a.half()
    del u

    down_f32 = out_dtype_down == torch.float
    d = torch.empty((A, H), dtype = torch.float if down_f32 else torch.half, device = dev)
    bn, bk, nw = _tiles(H, I, bm)
    _grouped_gemm_kernel[(triton.cdiv(H, bn), nblocks)](
        a, wd, wd, d, d, rw, sorted_ids, eblock,
        H, I, A, top_k, E,
        a.stride(0), wd.stride(0), wd.stride(1), d.stride(0),
        A_BY_TOKEN = False, DUAL = False, OUT_F32 = down_f32, MUL_RW = True,
        BM = bm, BN = bn, BK = bk, num_warps = nw, num_stages = 3)
    del a

    # Ascending expert id per token, the dense loop's accumulation order
    order = selected.argsort(dim = 1).to(torch.int32).contiguous()
    out = torch.empty((T, H), dtype = torch.float, device = dev)
    bh = 256
    _reduce_kernel[(T, triton.cdiv(H, bh))](d, order, out, H, TOP_K = top_k, BH = bh, num_warps = 4)
    return out


class GraphSet:
    """CUDA graphs of fp16_moe_decode for one module, keyed by padded row count"""
    def __init__(self):
        self.graphs = {}

    def clear(self):
        self.graphs.clear()

    def run(self, y, selected, rweights, *args, **kw):
        T = y.shape[0]
        Tb = next((b for b in GRAPH_BUCKETS if b >= T), None)
        if Tb is None or torch.cuda.is_current_stream_capturing():
            return fp16_moe_decode(y, selected, rweights, *args, **kw)
        ent = self.graphs.get(Tb)
        if ent is None:
            if not _graph_pool:
                _graph_pool.append(torch.cuda.graph_pool_handle())
            sy = torch.zeros((Tb, y.shape[1]), dtype = y.dtype, device = y.device)
            ssel = torch.zeros((Tb, selected.shape[1]), dtype = selected.dtype, device = y.device)
            srw = torch.zeros((Tb, rweights.shape[1]), dtype = rweights.dtype, device = y.device)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                      # compile and warm the allocator outside the capture
                fp16_moe_decode(sy, ssel, srw, *args, **kw)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool = _graph_pool[0]):
                out = fp16_moe_decode(sy, ssel, srw, *args, **kw)
            ent = self.graphs[Tb] = (g, sy, ssel, srw, out)
        g, sy, ssel, srw, out = ent
        sy[:T].copy_(y)
        ssel[:T].copy_(selected)
        srw[:T].copy_(rweights)
        if T < Tb:
            ssel[T:].zero_()
            srw[T:].zero_()
        g.replay()
        return out[:T]
