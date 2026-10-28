"""
Cut a small, real-weight test model out of an unquantized HF checkpoint: a subset of decoder layers
and, for MoE models, the first N routed experts, with the embedding, final norm and head kept.

The result loads like the original (same architecture, real activation scales, real routing among
the kept experts) but fits a small GPU, so tools such as sc_measure can be run resident against
it as a reference for faster or streaming implementations. Its outputs are not meaningful text.

    python util/slice_model.py -i /models/big -o /models/big-slice -l 0,3 -e 128

Layer types are written out explicitly (layer_types), so keeping e.g. one linear-attention and one
full-attention layer stays valid where the original derives types from an interval. Every config
list with one entry per layer is re-indexed; 1-based layer id lists named with --one-based-ids are
remapped, dropping layers that are not kept. MTP layers are dropped unless --keep-mtp.
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
TOPK_KEYS = ("num_experts_per_tok", "moe_topk", "top_k")
LAYER_RE = re.compile(r"^(.*?\blayers\.)(\d+)(\..*)$")
EXPERT_IDX_RE = re.compile(r"\.experts\.(\d+)\.")


def text_config(cfg: dict) -> dict:
    return cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg


def layer_types_of(tc: dict) -> list[str] | None:
    n = tc["num_hidden_layers"]
    if isinstance(tc.get("layer_types"), list):
        return tc["layer_types"]
    if "full_attention_interval" in tc:
        k = tc["full_attention_interval"]
        return ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(n)]
    return None


def default_layers(tc: dict) -> list[int]:
    """The first layer of each distinct layer type, or the first two layers"""
    types = layer_types_of(tc)
    if not types:
        return [0, 1][:tc["num_hidden_layers"]]
    first = {}
    for i, t in enumerate(types):
        first.setdefault(t, i)
    return sorted(first.values())


def slice_config(cfg: dict, keep: list[int], experts: int | None, keep_mtp: bool, one_based: list[str]):
    tc = text_config(cfg)
    n = tc["num_hidden_layers"]
    types = layer_types_of(tc)
    remap = {old: new for new, old in enumerate(keep)}
    for key, val in list(tc.items()):
        if isinstance(val, list) and len(val) == n and key not in one_based:
            tc[key] = [val[i] for i in keep]
    if types is not None:
        tc["layer_types"] = [types[i] for i in keep]
    for key in one_based:
        if isinstance(tc.get(key), list):
            tc[key] = [remap[i - 1] + 1 for i in tc[key] if i - 1 in remap]
    tc["num_hidden_layers"] = len(keep)
    old_experts = None
    if experts is not None:
        for key in EXPERT_COUNT_KEYS:
            if key in tc:
                old_experts = tc[key]
                tc[key] = experts
        for key in TOPK_KEYS:
            if key in tc:
                tc[key] = min(tc[key], experts)
    if not keep_mtp:
        for key in list(tc):
            if key.startswith("mtp_num_hidden_layers") or key == "num_nextn_predict_layers":
                tc[key] = 0
    return remap, old_experts


def slice_tensor(name: str, t: torch.Tensor, experts: int | None, old_experts: int | None):
    """Keep the first `experts` routed experts: fused (E, ...) expert stacks, the router's rows and
    per-expert biases are cut along their expert dim; unfused per-expert tensors are dropped"""
    if experts is None or old_experts is None or experts >= old_experts:
        return t
    m = EXPERT_IDX_RE.search(name)
    if m:
        return t if int(m.group(1)) < experts else None
    routed = (".experts." in name or name.endswith(".experts") or ".mlp.gate." in name
              or "e_score_correction_bias" in name or "per_expert_scale" in name)
    if routed and "shared" not in name:
        for dim, size in enumerate(t.shape):
            if size == old_experts:
                return t.narrow(dim, 0, experts).contiguous()
    return t


def main(args):
    src, dst = args.input, args.output
    os.makedirs(dst, exist_ok = True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = text_config(cfg)
    keep = sorted(int(x) for x in args.layers.split(",")) if args.layers else default_layers(tc)
    assert all(0 <= i < tc["num_hidden_layers"] for i in keep), f"layers {keep} out of range"
    one_based = [k for k in args.one_based_ids.split(",") if k]
    remap, old_experts = slice_config(cfg, keep, args.experts, args.keep_mtp, one_based)

    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
    else:
        f = os.path.join(src, "model.safetensors")
        with safe_open(f, framework = "pt") as h:
            weight_map = {k: "model.safetensors" for k in h.keys()}

    by_file = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, []).append(name)

    shard, shard_bytes, shard_no, new_map, total = {}, 0, 0, {}, 0
    def flush():
        nonlocal shard, shard_bytes, shard_no
        if not shard:
            return
        shard_no += 1
        fname = f"model-{shard_no:05d}.safetensors"
        save_file(shard, os.path.join(dst, fname), metadata = {"format": "pt"})
        for k in shard:
            new_map[k] = fname
        shard, shard_bytes = {}, 0

    for fname in sorted(by_file):
        with safe_open(os.path.join(src, fname), framework = "pt") as h:
            for name in sorted(by_file[fname]):
                if not args.keep_mtp and (name.startswith("mtp.") or ".mtp." in name):
                    continue
                m = LAYER_RE.match(name)
                new_name = name
                if m:
                    old = int(m.group(2))
                    if old not in remap:
                        continue
                    new_name = f"{m.group(1)}{remap[old]}{m.group(3)}"
                t = slice_tensor(name, h.get_tensor(name), args.experts, old_experts)
                if t is None:
                    continue
                shard[new_name] = t
                shard_bytes += t.numel() * t.element_size()
                total += t.numel() * t.element_size()
                if shard_bytes >= args.shard_gb * 1024 ** 3:
                    flush()
    flush()

    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent = 2)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent = 2)
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f in ("model.safetensors.index.json", "config.json"):
            continue
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))
    info = dict(source = os.path.abspath(src), layers = keep, experts = args.experts,
                source_experts = old_experts, mtp = args.keep_mtp, bytes = total)
    json.dump(info, open(os.path.join(dst, "slice_info.json"), "w"), indent = 2)
    print(f" -- Sliced {len(keep)} layer(s) {keep}, experts {args.experts or 'all'}: "
          f"{total / 1024 ** 3:.2f} GiB in {shard_no} shard(s) -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[1].strip())
    parser.add_argument("-i", "--input", required = True, help = "Unquantized HF model directory")
    parser.add_argument("-o", "--output", required = True, help = "Output directory")
    parser.add_argument("-l", "--layers", default = None,
                        help = "Comma-separated 0-based layer indices to keep (default: first layer of each layer type)")
    parser.add_argument("-e", "--experts", type = int, default = None, help = "Keep the first N routed experts")
    parser.add_argument("--keep-mtp", action = "store_true", help = "Keep MTP layers and tensors")
    parser.add_argument("--one-based-ids", default = "ple_layer_ids",
                        help = "Config keys holding 1-based layer id lists to remap (default: ple_layer_ids)")
    parser.add_argument("--shard-gb", type = float, default = 4.0, help = "Output shard size in GiB")
    main(parser.parse_args())
