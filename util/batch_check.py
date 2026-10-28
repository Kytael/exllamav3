"""
Check that a model's modules give the same result for rows forwarded together as for rows forwarded
one at a time, the way sc_measure's reference pass runs them (no cache, position 0), and time both.
A streaming measurement engine can batch rows across experiments only if this holds.

    python util/batch_check.py -m /models/slice -r 8 -l 2048

Prints one `RESULT {json}` line: per-module max relative difference between batched and single-row
outputs, and per-module time for both ways.
"""

import argparse
import json
import time

import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.measures import compute_kl_div


@torch.inference_mode()
def main(args):
    device = torch.device("cuda", args.device)
    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(args.length)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    model.load(device = device)
    mods = model.modules

    gen = torch.Generator().manual_seed(0)
    vocab = tokenizer.actual_vocab_size
    ids = torch.randint(0, vocab, (args.rows, args.length), generator = gen)

    def run(x, params, i):
        # The head only sees the compared tokens: full-length logits for every row would not fit
        if i == len(mods) - 1 and x.is_floating_point():
            x = x[:, : args.head_tokens].contiguous()
        x = mods[i].prepare_for_device(x, params)
        torch.cuda.synchronize()
        t = time.perf_counter()
        y = mods[i].forward(x, params)
        torch.cuda.synchronize()
        return y, time.perf_counter() - t

    # Warm-up: first calls pay for kernel loading and library initialization
    x, params = ids[:1], {"input_ids": ids[:1]}
    for i in range(len(mods)):
        x, _ = run(x, params, i)
    x, params = ids, {"input_ids": ids}
    for i in range(len(mods)):
        x, _ = run(x, params, i)
    del x

    # One row at a time, keeping every module's output (the last module only for its first chunk)
    single, t_single = [[] for _ in mods], [0.0] * len(mods)
    for r in range(args.rows):
        params = {"input_ids": ids[r : r + 1]}
        x = ids[r : r + 1]
        for i in range(len(mods)):
            x, dt = run(x, params, i)
            t_single[i] += dt
            single[i].append(x.float().cpu() if x.is_floating_point() else x.cpu())

    # All rows together
    params = {"input_ids": ids}
    x = ids
    diffs, t_batch, err = [], [0.0] * len(mods), None
    for i in range(len(mods)):
        try:
            x, t_batch[i] = run(x, params, i)
        except Exception as e:
            err = f"module {i} ({mods[i].key}): {type(e).__name__}: {e}"[:400]
            break
        out = x
        ref = torch.cat(single[i], dim = 0)
        if out.is_floating_point():
            out = out.float().cpu()
            d = ((out - ref).norm() / ref.norm().clamp(min = 1e-30)).item()
            m = (out - ref).abs().max().item()
        else:
            d = m = float((out.cpu() != ref).sum())
        row = dict(module = i, key = mods[i].key, rel = d, maxabs = m,
                   single_ms = round(t_single[i] * 1e3, 2), batch_ms = round(t_batch[i] * 1e3, 2))
        if i == len(mods) - 1 and out.is_floating_point():
            # What a measurement sees: mean per-token KL(single || batched) over the compared logits
            v = min(vocab, ref.shape[-1])
            kl = compute_kl_div(out.reshape(-1, out.shape[-1]).to(device), ref.reshape(-1, ref.shape[-1]).to(device), v)
            row["kl_mean"] = kl.mean().item()
            row["kl_max"] = kl.max().item()
        diffs.append(row)

    res = dict(rows = args.rows, length = args.length, error = err, modules = diffs,
               single_total_ms = round(sum(t_single) * 1e3, 1),
               batch_total_ms = round(sum(t_batch) * 1e3, 1) if err is None else None)
    print("RESULT " + json.dumps(res), flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required = True)
    parser.add_argument("-r", "--rows", type = int, default = 8)
    parser.add_argument("-l", "--length", type = int, default = 2048)
    parser.add_argument("-d", "--device", type = int, default = 0)
    parser.add_argument("--head-tokens", type = int, default = 256, help = "Compare only the first N tokens of the logits")
    main(parser.parse_args())
