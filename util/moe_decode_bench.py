"""One unquantized MoE layer at decode row counts: the dense per-expert loop (replica of the path in
block_sparse_mlp.py) against the Triton decode path (exllamav3/modules/moe_fp16_decode.py).

Prints one `MOEDEC {json}` line per row count: time, kernel count, relative error against an fp32
reference for both paths, the difference between them, and whether the decode path can be captured
in a CUDA graph (and the replay time).

    python util/moe_decode_bench.py [--experts 512 --hidden 2560 --inter 640 --topk 10 --tokens 1 8 32 128 512 2048]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EXL3_MOE_DECODE", "1")
import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.moe_fp16_decode import fp16_moe_decode, HAVE_TRITON


def route(tokens, E, topk, skew, gen, dev):
    bias = skew * torch.randn(E, generator = gen, device = dev)
    logits = torch.randn(tokens, E, generator = gen, device = dev) + bias
    w, sel = torch.topk(logits.softmax(-1), topk, dim = -1)
    return sel, (w / w.sum(-1, keepdim = True)).half()


def silu_mul(g, u, a, limit):
    ext.silu_mul(g, u, a, limit)


def loop_ref(y, sel, w, wg, wu, wd, E, fp32 = False):
    """The dense loop: host readback of the counts, per active expert index_select, three GEMMs,
    activation, routing weight, fp32 index_add_ (experts in ascending order)"""
    T, K = sel.shape
    flat_e = sel.reshape(-1)
    flat_tok = torch.arange(T, device = y.device).repeat_interleave(K)
    order = flat_e.argsort()
    ts, ws = flat_tok[order], w.reshape(-1)[order]
    counts = torch.bincount(flat_e, minlength = E + 1).tolist()
    out = torch.zeros((T, wd.shape[2]), dtype = torch.float, device = y.device)
    start = 0
    for e in range(E):
        c = counts[e]
        if c == 0:
            continue
        top = ts[start:start + c]
        ww = ws[start:start + c].unsqueeze(1)
        start += c
        xc = y.index_select(0, top)
        if fp32:
            xf = xc.float()
            d = (F.silu(xf @ wg[e].float()) * (xf @ wu[e].float())) @ wd[e].float()
            d *= ww.float()
        else:
            u = torch.empty((c, wu.shape[2]), dtype = torch.half, device = y.device)
            torch.matmul(xc, wu[e], out = u)
            g = torch.empty_like(u)
            torch.matmul(xc, wg[e], out = g)
            ext.silu_mul(g, u, u, 0.0)
            d = torch.empty((c, wd.shape[2]), dtype = torch.float, device = y.device)
            ext.hgemm(u, wd[e], d)
            d.mul_(ww)
        out.index_add_(0, top, d)
    return out


def timed(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    enq = time.perf_counter() - t0
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, enq / iters * 1e3


def kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA")


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main(args):
    dev = torch.device("cuda")
    print("MOEDEC_ENV " + json.dumps(dict(torch = torch.__version__, device = torch.cuda.get_device_name(),
                                          triton = HAVE_TRITON and __import__("triton").__version__)), flush = True)
    gen = torch.Generator(device = dev); gen.manual_seed(0)
    E, H, I, K = args.experts, args.hidden, args.inter, args.topk
    sg = (torch.randn((E, H, I), generator = gen, device = dev) * H ** -0.5).half()
    su = (torch.randn((E, H, I), generator = gen, device = dev) * H ** -0.5).half()
    sd = (torch.randn((E, I, H), generator = gen, device = dev) * I ** -0.5).half()
    for skew in args.skew:
        for T in args.tokens:
            y = torch.randn((T, H), generator = gen, device = dev).half()
            sel, w = route(T, E, K, skew, gen, dev)
            res = dict(tokens = T, skew = skew, assignments = T * K, active = int(torch.bincount(sel.reshape(-1), minlength = E).gt(0).sum()))
            opt = lambda: fp16_moe_decode(y, sel, w, sg, su, sd, silu_mul, 0.0, torch.half, torch.half, torch.half,
                                          torch.float, True)
            ref = loop_ref(y, sel, w, sg, su, sd, E, fp32 = True)
            lp = loop_ref(y, sel, w, sg, su, sd, E)
            o = opt()
            o2 = opt()
            res.update(err_loop_vs_fp32 = rel(lp, ref), err_opt_vs_fp32 = rel(o, ref), err_opt_vs_loop = rel(o, lp),
                       maxabs_opt_vs_loop = (o - lp).abs().max().item(), opt_deterministic = bool(torch.equal(o, o2)))
            res["loop_ms"], res["loop_enq_ms"] = timed(lambda: loop_ref(y, sel, w, sg, su, sd, E), args.iters)
            res["opt_ms"], res["opt_enq_ms"] = timed(opt, args.iters * 5)
            res["speedup"] = res["loop_ms"] / res["opt_ms"]
            res["opt_kernels"] = kernels(opt)
            # Weight bytes read if every touched expert is read once: 3 * H * I * 2 per expert
            res["opt_GBps_if_touched_once"] = res["active"] * 3 * H * I * 2 / (res["opt_ms"] * 1e-3) / 1e9
            try:
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    opt()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    go = opt()
                g.replay(); torch.cuda.synchronize()
                res["graph_equal"] = bool(torch.equal(go, o))
                t0 = time.perf_counter()
                for _ in range(args.iters * 5):
                    g.replay()
                torch.cuda.synchronize()
                res["graph_ms"] = (time.perf_counter() - t0) / (args.iters * 5) * 1e3
            except Exception as ex:
                res["graph_error"] = f"{type(ex).__name__}: {ex}"[:200]
            print("MOEDEC " + json.dumps(res), flush = True)
            del y, sel, w, ref, lp, o, o2
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type = int, default = 512)
    ap.add_argument("--hidden", type = int, default = 2560)
    ap.add_argument("--inter", type = int, default = 640)
    ap.add_argument("--topk", type = int, default = 10)
    ap.add_argument("--tokens", type = int, nargs = "+", default = [1, 8, 32, 128, 256, 1024, 2048])
    ap.add_argument("--skew", type = float, nargs = "+", default = [0.0, 1.0])
    ap.add_argument("--iters", type = int, default = 5)
    main(ap.parse_args())
