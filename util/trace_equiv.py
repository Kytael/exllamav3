"""
Per-token logit KL between MoE expert paths, on the same inputs.

A recorded trace (the JSON sc_trace.py writes) is replayed through the Generator teacher-forced: every
job's sampler returns the recorded next token instead of sampling, so all passes run identical batches
(same prompts, same tokens, same step schedule) and differ only in the MoE expert path:

    base  the dense per-expert loop (the BF16 sampling path as it stands)
    grp   the fork's padded-bmm grouped path (EXL3_MOE_GROUPED=1), an already-accepted reordering; the control
    opt   the Triton decode path (EXL3_MOE_DECODE=1)
    gold  the experts in float32 with no intermediate rounding: the yardstick all three are measured against

For every pair the KL(P || Q) of the logits each job's sampler sees is computed per token (float64 on the
device), with max |delta logit| and top-1 agreement. Step 0 of a job is its prompt's last position (prefill
batch); steps >= 1 are batched decode.

    EXL3_MOE_GROUPED=1 EXL3_MOE_DECODE=1 python util/trace_equiv.py -m <model_dir> --trace trace.json --jobs 32 --steps 48

Prints `TKL pair=<P>|<Q> {json}` lines and a verdict. Exit code 1 if mean KL(base || opt) > --max_mean_kl.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EXL3_MOE_DECODE", "1")
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import Sampler
from exllamav3.modules import BlockSparseMLP
from exllamav3.modules import moe_fp16_decode as mfd


class Forced(Sampler):
    """Returns the recorded token at each step and keeps the logits it was shown"""
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.step = 0
        self.rows = []
    def forward(self, logits, sequence_ids = None, rand_u32 = None, tokenizer = None, logit_mask = None, **kw):
        self.rows.append(logits.reshape(-1).to("cpu"))
        t = self.tokens[min(self.step, len(self.tokens) - 1)]
        self.step += 1
        return torch.tensor([[t]], dtype = torch.long, device = logits.device)


def walk(m):
    yield m
    for c in getattr(m, "modules", None) or []:
        yield from walk(c)


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer, _, _, _ = model_init.init(args)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 2048)
    moes = [m for m in walk(model) if isinstance(m, BlockSparseMLP)]
    print(f"TKL_INFO moe layers {len(moes)}: activation {set(m.activation_fn for m in moes)}, decode_ok {[m.decode_ok for m in moes]}, "
          f"grouped backend {[m.grouped_backend for m in moes]}")
    backends = [m.grouped_backend for m in moes]
    rows = json.load(open(args.trace))["rows"]
    rows = [r for r in rows if len(r["response_ids"]) >= 4][:args.jobs]
    print(f"TKL_INFO jobs {len(rows)} steps<= {args.steps}")

    def setmode(mode):
        for m, b in zip(moes, backends):
            m.grouped_backend = b if mode == "grp" else None
        mfd.RUNTIME_ON[0] = mode in ("opt", "gold")
        mfd.GOLD[0] = mode == "gold"

    def run(mode):
        setmode(mode)
        samplers = []
        for i, r in enumerate(rows):
            forced = r["response_ids"][:args.steps]
            s = Forced(forced)
            samplers.append(s)
            gen.enqueue(Job(input_ids = torch.tensor([r["input_ids"]], dtype = torch.long),
                            max_new_tokens = len(forced), stop_conditions = [], sampler = s, identifier = i))
        while gen.num_remaining_jobs():
            gen.iterate()
        return samplers

    modes = [m for m in args.modes.split(",") if m]
    passes = {}
    for m in modes:
        if m == "grp" and not any(backends):
            print("TKL_INFO grp unavailable (start with EXL3_MOE_GROUPED=1)"); continue
        passes[m] = run(m)
    setmode("opt")

    def compare(pa, pb):
        kl_p, kl_d, dmax, agree, worst = [], [], 0.0, 0, []
        for j, (sa, sb) in enumerate(zip(passes[pa], passes[pb])):
            assert len(sa.rows) == len(sb.rows), (len(sa.rows), len(sb.rows))
            for step, (la, lb) in enumerate(zip(sa.rows, sb.rows)):
                la = la.to("cuda").double(); lb = lb.to("cuda").double()
                lpa = torch.log_softmax(la, -1); lpb = torch.log_softmax(lb, -1)
                kl = (lpa.exp() * (lpa - lpb)).sum().item()
                d = (la - lb).abs().max().item()
                dmax = max(dmax, d)
                agree += int(la.argmax().item() == lb.argmax().item())
                (kl_p if step == 0 else kl_d).append(kl)
                worst.append((kl, j, step, d))
        def stats(v):
            v = sorted(v)
            return dict(n = len(v), mean = sum(v) / max(len(v), 1), median = v[len(v) // 2] if v else 0.0,
                        p99 = v[int(0.99 * len(v))] if v else 0.0, max = v[-1] if v else 0.0)
        allv = kl_p + kl_d
        worst.sort(reverse = True)
        return dict(prefill_last_token = stats(kl_p), decode = stats(kl_d), all = stats(allv),
                    max_abs_logit_delta = dmax, top1_agree = f"{agree}/{len(allv)}",
                    worst5 = [dict(kl = w[0], job = w[1], step = w[2], max_abs_delta = w[3]) for w in worst[:5]])

    results = {}
    order = [m for m in ("gold", "base", "grp", "opt") if m in passes]
    for i, pa in enumerate(order):
        for pb in order[i + 1:]:
            results[(pa, pb)] = compare(pa, pb)
            print(f"TKL pair={pa}|{pb} " + json.dumps(results[(pa, pb)]))
    key = ("base", "opt")
    if key in results:
        mean = results[key]["all"]["mean"]
        ok = mean <= args.max_mean_kl
        print(f"TKL_VERDICT mean KL(base,opt) {mean:.3e} (limit {args.max_mean_kl:.0e}) max {results[key]['all']['max']:.3e}: "
              f"{'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, default_cache_size = 131072, add_draft_model_args = True, default_autosplit_max_batch_size = 64)
    parser.add_argument("--trace", type = str, required = True, help = "Trace JSON from sc_trace.py")
    parser.add_argument("--jobs", type = int, default = 32)
    parser.add_argument("--steps", type = int, default = 48)
    parser.add_argument("--modes", type = str, default = "base,grp,opt,gold")
    parser.add_argument("--max_mean_kl", type = float, default = 1e-5)
    main(parser.parse_args())
