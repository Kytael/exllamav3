import sys, os
import mmap
import threading
import time
import weakref

import argparse
import json
import math
import zlib
from collections import defaultdict
from contextlib import contextmanager
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.measures import compute_kl_div
from exllamav3.util.tensor import g_tensor_cache
from exllamav3.modules.linear import Linear
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.mlp import GatedMLP
from exllamav3.modules.attn import Attention
from exllamav3.modules.sliding_attn import SlidingAttention
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.quant.fp16 import LinearFP16
from exllamav3.modules.quant.exl3_lib.quantize import finalize_capture_H, get_hadamard_dt, had_k
from datasets import load_dataset

"""
Single-model quantization sensitivity measurement by weight-space noise injection.

For every quantizable Linear (qmap set), perturbs the fp16 weights in place with seeded noise of
a given relative Frobenius norm, runs the rest of the model from a cached boundary state, and
records the KL divergence of the final logits against the clean reference.

Two noise models:

 - iid (default): isotropic Gaussian dW. Validated on qwen3.8-27b vs swap attribution of a real
   2bpw conversion: Spearman 0.61, median 1.95x KL overestimate with strong per-type structure
   (up to 5.8x on v_proj). The bias is almost entirely the missing LDLQ error shaping. Real
   quantization error is steered away from data-covariant input directions, so it produces
   1.5-7x less output error per unit weight error than iid noise.

 - shaped (--shaped): mimics the LDLQ error distribution. A capture pass over --h_rows extra
   calibration rows accumulates the same Hessians the quantizer uses (via Linear.capture_H and
   finalize_capture_H, so damping, sign flips, block Hadamard and the block-16 LDL are all
   byte-identical to conversion). Per LDLQ theory, the quantizer's weight error is
   dW_rot = L^-T eta with eta white and H_rot = L D L^T, so shaped noise is sampled as
   dW = P^T L^-T eta (P = block-Hadamard x sign flips), with per-output-channel scaling
   mimicking out_scales, then normalized to the target weight rfn. Falls back to iid where the
   capture is unusable (q_fallback).

Noise levels are either global (--rfn 0.29,0.145) or per-tensor, anchored to the measured error
of an actual quantized model (--rfn_ref rfn.json from sc_rfn_probe.py, --rfn_scale 1.0,0.5).
Measuring two levels an octave apart gives a per-tensor scaling exponent as a sanity check on
the quadratic law.

Output JSON feeds sc_optimize.py, which turns the sensitivities into a per-tensor bitrate
recipe. Results are written incrementally and the script resumes from a partial output file.

With --streaming, load one top-level module at a time and keep cached states in system RAM.
Perturbed states fan out at each noise site and share module loads. --max-sys budgets pending
states by splitting the work into passes, with results saved after each pass.
The largest module plus Hessian/noise workspace must still fit on the device.
Not validated on MoE models.
"""

# TODO: Validate the grouped MoE measurement real MoE models
# TODO: Streaming throughput: pending states go to pageable host memory after every module and
#       rows run one (1, L) forward at a time; pinned/asynchronous state buffers and batching the
#       pending rows into one forward per module would remove most of the transfer cost (tens of
#       TB of round trips on a 70B measurement)
# TODO: Rethink


class _LockedMemory(mmap.mmap):
    """Anonymous memory registered with the CUDA driver (page-locked) for as long as it lives.
    torch.frombuffer keeps it alive while any tensor views it; it is unregistered just before it is
    unmapped"""
    ptr = None

    def __del__(self):
        if self.ptr is not None:
            torch.cuda.cudart().cudaHostUnregister(self.ptr)


class PinnedPool:
    """Exact-size page-locked host buffers, handed out again by size once given back. torch's
    pinned allocator rounds every block up to a power of two, so a 671 MB state takes 1 GiB of
    locked RAM and a 1.02 GB row of reference logits takes 2 GiB: nearly half the RAM a pass could
    use. Buffers nobody gives back are freed as usual when their last view goes. Pageable (only
    slower to copy) if registering fails"""

    def __init__(self, enabled):
        self.enabled = enabled
        self.free = defaultdict(list)                # nbytes -> buffers given back
        self.live = weakref.WeakValueDictionary()    # data_ptr -> buffer

    def empty(self, shape, dtype):
        n = math.prod(shape) * dtype.itemsize
        if not self.enabled or n == 0:
            return torch.empty(shape, dtype = dtype)
        if self.free[n]:
            mem = self.free[n].pop()
        else:
            mem = _LockedMemory(-1, n)
            ptr = torch.frombuffer(mem, dtype = torch.uint8).data_ptr()
            try:
                cudart = torch.cuda.cudart()
                ok = cudart.cudaHostRegister(ptr, n, 0) == cudart.cudaError.success
            except Exception:
                ok = False
            if not ok:
                self.enabled = False
                return torch.empty(shape, dtype = dtype)
            mem.ptr = ptr
            self.live[ptr] = mem
        return torch.frombuffer(mem, dtype = torch.uint8).view(dtype).view(shape)

    def give(self, tensors):
        """Return whole buffers from empty() for reuse (anything else is ignored). The caller
        guarantees no unfinished copy still uses them, or that later copies into them are ordered
        after it on the same streams"""
        for t in tensors if isinstance(tensors, (list, tuple)) else ():
            mem = self.live.get(t.data_ptr()) if t.device.type == "cpu" else None
            if mem is not None and t.numel() * t.element_size() == len(mem):
                self.free[len(mem)].append(mem)


class ModuleRunner:
    """Run rows through one module at a time. Clone states because modules may mutate inputs,
    including modules that run on CPU."""

    def __init__(self, config, modules, device, streaming, pool):
        self.config = config
        self.modules = modules
        self.device = device
        self.streaming = streaming
        self.pool = pool
        self.ahead = None     # (idx, thread, result) of a module loading in the background
        self.wait_s = 0.0     # time spent waiting for module loads since the last take_wait()

    def take_wait(self):
        w, self.wait_s = self.wait_s, 0.0
        return w

    def _load(self, mod):
        defer = mod.can_defer_load()
        if defer:
            self.config.stc.begin_deferred_load()
        try:
            mod.load("cpu" if mod.caps.get("prefer_cpu") else self.device)
            if defer:
                self.config.stc.end_deferred_load()
        except BaseException:
            if defer:
                self.config.stc.abort_deferred_load()
            raise

    def prefetch(self, idx):
        """Streaming: start loading module idx in a background thread on a side stream, so its
        weights stream in while the current module computes (the loader reads with the GIL
        released). loaded(idx) then waits for it instead of loading. One module ahead at most"""
        if not self.streaming or self.ahead is not None or not (0 <= idx < len(self.modules)):
            return
        mod = self.modules[idx]
        if mod.caps.get("prefer_cpu") or self.device.type != "cuda":
            return
        result = {}
        def work():
            try:
                stream = torch.cuda.Stream(self.device)
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    self._load(mod)
                    result["event"] = torch.cuda.Event()
                    result["event"].record(stream)
            except BaseException as e:
                result["error"] = e
        thread = threading.Thread(target = work, daemon = True)
        thread.start()
        self.ahead = (idx, thread, result)

    def _join_ahead(self):
        """Wait for the background load; returns (idx, result) or None"""
        if self.ahead is None:
            return None
        idx, thread, result = self.ahead
        thread.join()
        self.ahead = None
        return idx, result

    @contextmanager
    def loaded(self, idx):
        mod = self.modules[idx]
        try:
            if self.streaming:
                t0 = time.perf_counter()
                done = self._join_ahead()
                if done is not None and done[0] == idx and "error" not in done[1]:
                    torch.cuda.current_stream(self.device).wait_event(done[1]["event"])
                else:
                    if done is not None:
                        # A prefetch of another module (or a failed one): drop it, load this one
                        if "error" not in done[1]:
                            self.modules[done[0]].unload()
                    self._load(mod)
                self.wait_s += time.perf_counter() - t0
            yield mod
        finally:
            if self.streaming:
                # Keep a prefetch running across the unload only once it has finished with the
                # loader, and let compute finish with this module's memory before freeing it
                ahead = self.ahead
                if ahead is not None:
                    ahead[1].join()
                torch.cuda.current_stream(self.device).synchronize()
                mod.unload()
                g_tensor_cache.drop_all()

    def rows(self, mod, states, capture=None, consume=None, ids=None, per_row=False, inspect=None):
        """states, ids: one entry per row group, (B, ...); ids are the rows' token ids, passed as
        params["input_ids"] (PLE layers read the token context from there; without it they run on
        a dummy all-zero history). per_row: forward each row of a group on its own (the LM head,
        whose full-vocab logits would not fit for a whole group); consume(r, x) then gets the
        global row index. Without consume, outputs come back as host tensors in page-locked
        memory: uploads and downloads run asynchronously, the downloads on a copy stream, so a
        group's transfer overlaps the next group's compute. inspect(n, x), if given, sees each
        item's device output before it is downloaded"""
        dev = getattr(mod, "device", None)
        on_gpu = dev is not None and torch.device(dev).type == "cuda"
        copy_stream = torch.cuda.Stream(self.device) if on_gpu and consume is None else None
        compute = torch.cuda.current_stream(self.device) if on_gpu else None
        up_stream = torch.cuda.Stream(self.device) if on_gpu else None

        # Work items in order: (group, row within it or None for the whole group, global row
        # index). States are fetched one group at a time: spilled groups are read from disk
        items, r = [], 0
        for g in range(len(states)):
            rows_g = states[g].shape[0] if not isinstance(states, SpilledStates) else states.meta[g][1][0]
            for b in (range(rows_g) if per_row else [None]):
                items.append((g, b, r))
                r += 1 if per_row else rows_g
        cached = [None, None]

        def item_state(n):
            g, b, _ = items[n]
            if cached[0] != g:
                cached[:] = [g, states[g]]
            return cached[1] if b is None else cached[1][b : b + 1]

        def item_ids(n):
            g, b, _ = items[n]
            if ids is None:
                return None
            return ids[g] if b is None else ids[g][b : b + 1]

        def upload(st):
            """Host states go up on their own stream, one item ahead of compute, so the copy
            overlaps the previous item's forward. A fresh device copy also protects the stored
            state from in-place edits; a module that runs on the CPU needs an explicit clone"""
            if on_gpu and st.device.type == "cpu" and st.is_floating_point():
                up_stream.wait_stream(compute)    # its memory may be recycled from compute's frees
                with torch.cuda.stream(up_stream):
                    x = st.to(self.device, non_blocking = True)
                ev = torch.cuda.Event()
                ev.record(up_stream)
                return x, ev
            return st.clone(), None

        outputs, in_flight = [], []
        nxt = upload(item_state(0)) if items else None
        for n, (_, _, r0) in enumerate(items):
            part_ids = item_ids(n)
            x, ev = nxt
            nxt = upload(item_state(n + 1)) if n + 1 < len(items) else None
            if ev is not None:
                compute.wait_event(ev)
                x.record_stream(compute)
            params = {} if capture is None else {"capture": capture}
            if part_ids is not None:
                params["input_ids"] = part_ids
            x = mod.prepare_for_device(x, params)
            x = mod.forward(x, params)
            if inspect is not None:
                inspect(n, x)
            if consume is not None:
                consume(r0, x)
            elif copy_stream is not None:
                out = self.pool.empty(x.shape, x.dtype)
                copy_stream.wait_stream(compute)
                with torch.cuda.stream(copy_stream):
                    out.copy_(x, non_blocking = True)
                in_flight.append(x)
                outputs.append(out)
            else:
                outputs.append(x.to("cpu", copy = True))
            del x, params
        if copy_stream is not None:
            copy_stream.synchronize()
        del in_flight
        return outputs


def estimate_resident_bytes(modules, rows, length, vocab_size, shaped, h_rows, stc):
    """Estimate resident VRAM for weights, cached states, logits and noise/Hessian workspace.
    Use checkpoint headers since unloaded modules may return None from weights_numel()."""
    def walk(mod):
        yield mod
        for child in getattr(mod, "modules", []):
            yield from walk(child)

    weights = 0
    for key, filename in stc.tensor_file_map.items():
        tensor = stc.file_headers[filename][key]
        start, end = tensor["data_offsets"]
        # FP8 weights expand to FP16; retain the larger size for FP32 tensors
        weights += max(2 * math.prod(tensor["shape"]), end - start)
    linear_weights = 0
    width = 0
    largest_weight = 0
    hessian_peak = 0
    for mod in modules:
        qmaps = {}
        for lin in walk(mod):
            if isinstance(lin, Linear):
                weight_bytes = 2 * lin.in_features * lin.out_features
                linear_weights += weight_bytes
                width = max(width, lin.in_features)
                largest_weight = max(largest_weight, weight_bytes)
                if lin.qmap is not None:
                    qmaps[lin.qmap] = max(qmaps.get(lin.qmap, 0), lin.in_features)
        if qmaps:
            hessian_peak = max(hessian_peak,
                               4 * sum(k * k for k in qmaps.values())
                               + 16 * max(qmaps.values()) ** 2)
    weights = max(weights, linear_weights)  # account for padded/repeated linear modules
    states = 2 * length * width * (rows * (len(modules) + 2) + (h_rows if shaped else 0))
    logits = 2 * length * vocab_size + 16 * min(length, 256) * vocab_size
    workspace = largest_weight + (hessian_peak if shaped else 0) + 3 * 1024 ** 3
    return math.ceil(1.15 * (weights + states + logits + workspace))


class SpilledStates:
    """One module's boundary states (a tensor per row group) kept in files under a spill directory
    instead of host memory. Streaming mode holds every module's boundary for the whole run (53
    modules x 0.67 GB on the harness model at -r 8 -l 2048), more than a 64 GB host can spare next
    to the pending experiment states; each is read back only when its module is perturbed. Reading
    yields fresh host tensors, so callers may not mutate shared state by accident"""

    def __init__(self, tensors, path):
        os.makedirs(path, exist_ok = True)
        self.meta = []
        for k, t in enumerate(tensors):
            t = t.contiguous()
            f = os.path.join(path, f"{k}.bin")
            t.view(torch.uint8).view(-1).numpy().tofile(f)
            self.meta.append((f, tuple(t.shape), t.dtype, t.numel() * t.element_size()))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, k):
        f, shape, dtype, nbytes = self.meta[k]
        return torch.from_file(f, shared = False, size = nbytes, dtype = torch.uint8).view(dtype).view(shape)

    def __iter__(self):
        return (self[k] for k in range(len(self)))

    def nbytes(self):
        return sum(m[3] for m in self.meta)


def states_nbytes(states):
    if isinstance(states, SpilledStates):
        return states.nbytes()
    return sum(t.numel() * t.element_size() for t in states)


def sysmem_budget(max_sys_gb):
    """Pending-state budget in bytes; default to half the RAM available after the reference pass"""
    if max_sys_gb is not None:
        return int(max_sys_gb * 1024 ** 3)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 // 2
    except OSError:
        pass
    return 8 * 1024 ** 3


def plan_fanout_passes(measurements_by_idx, state_bytes, budget):
    """Group modules by pending-state budget, counting control measurements too.
    Keep each module's measurements together, even if that module exceeds the budget."""
    passes, current, pending = [], [], 0
    for idx in sorted(measurements_by_idx):
        count = measurements_by_idx[idx]
        if current and (pending + count) * state_bytes > budget:
            passes.append(current)
            current, pending = [], 0
        current.append(idx)
        pending += count
    if current:
        passes.append(current)
    return passes

class Target:
    """One measurement target: a single quantizable Linear, or every routed-expert Linear of one
    MoE layer perturbed together. A layer's experts are measured (and later quantized) as one
    unit: sc_optimize then assigns them one bitrate, which keeps the fused MoE kernels on a
    single K instead of the mixed-K fallback, and it keeps the fan-out at a handful of targets
    per layer instead of hundreds."""

    def __init__(self, key, linears, qbits_key):
        self.key = key
        self.linears = linears
        self.qbits_key = qbits_key
        self.members = [lin.key for lin in linears]

    @property
    def grouped(self):
        return len(self.linears) > 1

    def weights_numel(self):
        return sum(lin.weights_numel() for lin in self.linears)


def collect_targets(mod):
    """Targets under one top-level module: the routed experts of each BlockSparseMLP as one
    grouped target (its routing gate and latent projections stay individual); the gate and up
    projections of each GatedMLP (dense MLPs and MoE shared experts alike) as one grouped target,
    since the fused gate/up GEMM and the shared-expert path of the MoE kernels need them at one
    bitrate; the q/k/v (+ full quantized gate) projections of each attention module and the
    qkv/z projections of each GatedDeltaNet, which the sliced mgemm bundles run as one launch at
    one K; every other quantizable Linear on its own."""
    targets = []
    grouped_ids = set()

    def group(m, name, linears):
        linears = [lin for lin in linears if lin is not None and lin.qmap is not None and id(lin) not in grouped_ids]
        if len(linears) > 1:
            targets.append(Target(f"{m.key}.{name}", linears, linears[0].qbits_key))
            grouped_ids.update(id(lin) for lin in linears)

    def walk(m):
        if isinstance(m, (Attention, SlidingAttention)):
            qkv = [m.q_proj, m.k_proj] + ([] if getattr(m, "use_k_as_v", False) else [m.v_proj])
            # A full quantized gate rides along in the bundle; an interleaved gate is part of
            # q_proj already and a headwise gate is fp16
            if getattr(m, "g_proj", None) is not None and getattr(m, "full_gate", False) \
                    and not getattr(m, "interleaved_gate", False):
                qkv.append(m.g_proj)
            group(m, "qkv", qkv)
        elif isinstance(m, GatedDeltaNet):
            group(m, "qkvz", [getattr(m, "qkv_proj", None), getattr(m, "z_proj", None)])
        if isinstance(m, BlockSparseMLP):
            experts = [lin for lin in (m.gates + m.ups + m.downs) if lin is not None and lin.qmap is not None]
            if experts:
                targets.append(Target(f"{m.key}.experts", experts, experts[0].qbits_key))
                grouped_ids.update(id(lin) for lin in experts)
        elif isinstance(m, GatedMLP):
            group(m, "gate_up", m.gates + m.ups)
        if isinstance(m, Linear) and m.qmap is not None and id(m) not in grouped_ids:
            targets.append(Target(m.key, [m], m.qbits_key))
        for ch in getattr(m, "modules", []):
            walk(ch)

    walk(mod)
    return targets


@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, eval_len):
    eval_tokens = tokenizer.encode(get_dataset_text({"dataset": "wiki2"}))
    num_tokens = eval_tokens.shape[-1]
    seqs = []
    for a in range(0, num_tokens - eval_len, eval_len):
        seqs.append(eval_tokens[:, a : a + eval_len])
        if len(seqs) >= rows:
            break
    assert len(seqs) >= rows, f"not enough calibration text for {rows} rows of {eval_len}"
    return torch.cat(seqs, dim = 0)


@torch.inference_mode()
def main(args):
    device = torch.device("cuda", args.device)
    torch.manual_seed(0)

    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(args.length)
    tokenizer = Tokenizer.from_config(config)
    vocab_size = tokenizer.actual_vocab_size

    model = Model.from_config(config)
    mode = args.load_mode
    streaming = mode == "streaming"
    if mode == "auto":
        required = estimate_resident_bytes(model.modules, args.rows, args.length,
                                           vocab_size, args.shaped, args.h_rows, config.stc)
        free, _ = torch.cuda.mem_get_info(device)
        streaming = required > free
        print(f" -- VRAM preflight: {free / 1024 ** 3:.1f} GiB free, "
              f"{required / 1024 ** 3:.1f} GiB estimated for resident measurement")
    print(f" -- {'Streaming' if streaming else 'Loading'} model: {args.model}")
    if not streaming:
        try:
            model.load(device = device)
        except torch.cuda.OutOfMemoryError:
            if mode != "auto":
                raise
            config.stc.abort_deferred_load()
            model.unload()
            g_tensor_cache.drop_all()
            # Release the OOM traceback before retrying so it cannot retain weight buffers
            streaming = True
        if streaming:
            torch.cuda.empty_cache()
            print(" -- Full-model load exceeded VRAM; switching to layer streaming")
    pinned = PinnedPool(device.type == "cuda")
    runner = ModuleRunner(config, model.modules, device, streaming, pinned)

    total_rows = args.rows + (args.h_rows if args.shaped else 0)
    if args.trace:
        # Packed self-sampled trace from sc_trace.py: in-distribution for the model,
        # and the same data the quantizer can calibrate on (convert.py --cal_data)
        from safetensors.torch import load_file
        data = load_file(args.trace)
        packed = data["input_ids"]
        lengths = data.get("lengths")     # per-row lengths (each row one example from position 0)
        assert packed.shape[0] >= total_rows, \
            f"trace has {packed.shape[0]} rows, need {total_rows} (rows + h_rows)"
        assert lengths is not None or packed.shape[1] >= args.length, \
            f"trace rows are {packed.shape[1]} tokens, need {args.length}"
        # Every row starts at its own position 0 (BOS, system prompt, whole turns) and is truncated
        # to --length; a row shorter than that keeps its own length rather than being padded or
        # continued into the next example
        ids = [packed[i : i + 1, : (args.length if lengths is None else min(args.length, int(lengths[i])))]
               for i in range(total_rows)]
        assert all(r.shape[1] > 0 for r in ids), "trace contains an empty row"
    else:
        ids = list(get_test_tokens(tokenizer, total_rows, args.length).split(1))
    rows = ids[:args.rows]

    def make_groups(id_rows, size):
        """Consecutive rows of equal length, at most `size` per group, as (B, length) id tensors.
        Every pass (reference, control, perturbed, capture) forwards the same groups, so shapes
        and kernels match between the reference and each measurement and the restart control
        stays exactly 0. Size 1 is the per-row behaviour"""
        groups = []
        for r in id_rows:
            if groups and len(groups[-1]) < size and groups[-1][-1].shape[1] == r.shape[1]:
                groups[-1].append(r)
            else:
                groups.append([r])
        return [g[0] if len(g) == 1 else torch.cat(g, dim = 0) for g in groups]

    row_groups = make_groups(rows, args.batch_rows)
    cap_states = make_groups(ids[args.rows:], args.batch_rows) if args.shaped else None
    cap_ids = cap_states

    mods = model.modules
    num_mods = len(mods)

    def to_host_pinned(x):
        """fp16 copy of x in page-locked host memory, so later uploads run asynchronously at full
        link speed (a row's reference logits are length x vocab x 2 bytes: 1 GB at 2,048 tokens
        and a 248k vocab, uploaded once per perturbed row). Pageable if pinning fails"""
        x = x.half()
        buf = pinned.empty(x.shape, torch.half)
        buf.copy_(x)
        return buf

    ref_dev = None     # reference logits copied to the GPU while there is room for them

    def device_room(need):
        """True if need bytes fit on the device with room left for the work"""
        free, total = torch.cuda.mem_get_info(device)
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return free - need >= max(8 << 30, total // 4)

    def stage_states(states):
        """Device copies of a module's boundary states if they fit, else the states as they are.
        Every experiment spawned at a module starts from the same input and is compared with the
        same clean output; staging them once saves a disk read (when spilled) and an upload per
        experiment"""
        if device.type != "cuda" or len(states) == 0:
            return states
        if not isinstance(states, SpilledStates) and not all(t.is_floating_point() for t in states):
            return states     # token ids: the embedding may run on the CPU
        if not device_room(states_nbytes(states)):
            return states
        return [t.to(device, non_blocking = True) for t in states]

    def stage_refs():
        """Copy every row's reference logits to the GPU if they fit beside the work (8 GB at
        8 rows of 2,048 tokens and a 248k vocab), so measurements stop re-uploading them once per
        experiment. Streaming stages them while the head is the only module loaded; resident mode
        once, if the model leaves room. Same data either way, so results are unchanged"""
        nonlocal ref_dev
        if ref_dev is not None or device.type != "cuda" or not ref_logits:
            return
        if not device_room(sum(t.numel() * t.element_size() for t in ref_logits)):
            return
        ref_dev = [t.to(device, non_blocking = True) for t in ref_logits]

    def unstage_refs():
        nonlocal ref_dev
        ref_dev = None

    def ref_at(r):
        return ref_logits[r] if ref_dev is None else ref_dev[r]

    def row_kld(r, x):
        ref = ref_at(r)
        kl_vocab = min(vocab_size, x.shape[-1], ref.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        ref2 = ref.reshape(-1, ref.shape[-1])
        # fp64 device sum, read once (bit-identical to adding the chunks' Python floats in order)
        kl_row = torch.zeros((), dtype = torch.float64, device = x.device)
        for a in range(0, x2.shape[0], 256):
            b = min(a + 256, x2.shape[0])
            kl_row += compute_kl_div(x2[a:b].float(), ref2[a:b].to(x.device, non_blocking = True), kl_vocab).sum().double()
        return kl_row.item() / x2.shape[0]

    @torch.inference_mode()
    def forward_rows(start_idx, states, collect_states = False):
        """
        Forward every row from module start_idx to the end, streaming the KL vs the reference
        per row. Returns (mean kld, per-row hidden state after module start_idx if requested).
        Resident mode only.
        """
        # KL sums stay on the device in fp64 and are read once per call, not once per 256-token
        # chunk: each fp32 chunk sum is widened and added in the same order as the Python floats
        # were, so the result is bit-identical
        kld_sum = None
        out_states = []
        r = 0
        for g, state in enumerate(states):
            params = {"input_ids": row_groups[g]}
            x = state.clone() if state.is_floating_point() else state
            # Every module but the head runs on the whole group
            for i in range(start_idx, num_mods - 1):
                mod = mods[i]
                x = mod.prepare_for_device(x, params)
                x = mod.forward(x, params)
                if collect_states and i == start_idx:
                    out_states.append(x.clone())
            for xb in head_rows(x, row_groups[g]):
                ref = ref_at(r)
                r += 1
                kl_vocab = min(vocab_size, xb.shape[-1], ref.shape[-1])
                # Chunk over tokens: full-vocab fp32 logits can be GBs
                x2 = xb.view(-1, xb.shape[-1])
                ref2 = ref.view(-1, ref.shape[-1])
                kl_row, n_row = torch.zeros((), dtype = torch.float64, device = xb.device), 0
                for a in range(0, x2.shape[0], 256):
                    b = min(a + 256, x2.shape[0])
                    kl_row += compute_kl_div(x2[a:b].float(), ref2[a:b].to(xb.device, non_blocking = True), kl_vocab).sum().double()
                    n_row += b - a
                kl_row /= n_row
                kld_sum = kl_row if kld_sum is None else kld_sum.to(kl_row.device) + kl_row
                del xb, x2
            del x
        return kld_sum.item() / r, out_states

    def head_rows(x, ids_group):
        """The last module (the LM head), one row at a time: a group's full-vocab logits would not
        fit (1 GB per row at 2,048 tokens and a 248k vocab). Yields each row's logits"""
        head = mods[num_mods - 1]
        for b in range(x.shape[0]):
            params = {"input_ids": ids_group[b : b + 1]}
            xb = head.prepare_for_device(x[b : b + 1], params)
            yield head.forward(xb, params)

    @torch.inference_mode()
    def ref_pass():
        """Full reference pass, caching the input state to every module and the final logits"""
        boundary = [[] for _ in range(num_mods)]
        logits = []
        if streaming:
            states = row_groups
            for i in range(num_mods):
                boundary[i] = states
                if args.spill_dir and i > 0:
                    # The in-memory tensors still feed this module; the boundary keeps only files
                    boundary[i] = SpilledStates(states, os.path.join(args.spill_dir, f"boundary-{i}"))
                with runner.loaded(i) as mod:
                    runner.prefetch(i + 1)
                    states = runner.rows(
                        mod, states, ids = row_groups,
                        consume=(lambda r, x: logits.append(to_host_pinned(x)))
                        if i == num_mods - 1 else None,
                        per_row = i == num_mods - 1,
                    )
            return boundary, logits
        for ids_group in row_groups:
            params = {"input_ids": ids_group}
            x = ids_group
            for i in range(num_mods - 1):
                mod = mods[i]
                x = mod.prepare_for_device(x, params)
                # Residual stream is mutated in place downstream; every cached state must be
                # cloned or all mid-stream experiments run on corrupted inputs
                boundary[i].append(x.clone() if x.is_floating_point() else x)
                x = mod.forward(x, params)
            boundary[num_mods - 1].append(x.clone() if x.is_floating_point() else x)
            for xb in head_rows(x, ids_group):
                logits.append(to_host_pinned(xb))
        return boundary, logits

    # Collect perturbation targets by top-level module: every quantizable Linear, with each MoE
    # layer's routed experts as one grouped target
    targets_by_idx = defaultdict(list)
    num_targets = 0
    num_grouped = 0
    for i in range(num_mods):
        for t in collect_targets(mods[i]):
            if not streaming:
                for lin in t.linears:
                    assert isinstance(lin.inner, LinearFP16), \
                        f"{lin.key}: expected unquantized (fp16) tensor, got {type(lin.inner).__name__}"
            targets_by_idx[i].append(t)
            num_targets += 1
            num_grouped += t.grouped
    if num_grouped:
        print(f" -- {num_grouped} grouped target(s): MoE routed experts per layer, gate/up of dense MLPs and "
              f"shared experts, attention q/k/v(/g), GDN qkv/z")

    expected_keys = {t.key for targets in targets_by_idx.values() for t in targets}

    # Per-target noise levels
    if args.rfn_ref:
        with open(args.rfn_ref, "r") as f:
            ref_data = json.load(f)
        ref_rfn = {r["key"]: r["rfn"] for r in ref_data["results"]}
        scales = [float(s) for s in args.rfn_scale.split(",")]
        missing = [m for tl in targets_by_idx.values() for t in tl for m in t.members if m not in ref_rfn]
        assert not missing, f"keys missing from {args.rfn_ref}: {missing[:5]}..."
        # Per-member anchors: each expert of a grouped target keeps its own measured error level
        member_levels = {key: [ref_rfn[key] * s for s in scales] for key in ref_rfn}
        num_levels = len(scales)
    else:
        rfns = [float(s) for s in args.rfn.split(",")]
        member_levels = defaultdict(lambda: rfns)
        num_levels = len(rfns)

    def target_rfn(target, li):
        """Nominal level of a target: its members' levels, numel-weighted rms for the record"""
        per = [member_levels[m][li] for m in target.members]
        if not target.grouped:
            return per[0], per
        w = [lin.weights_numel() for lin in target.linears]
        return math.sqrt(sum(wi * r * r for wi, r in zip(w, per)) / sum(w)), per

    # Resume from partial output
    results = []
    done = set()
    if args.out and os.path.exists(args.out):
        with open(args.out, "r") as f:
            prev = json.load(f)
        results = [r for r in prev["results"] if r["key"] in expected_keys]
        done = {r["key"] for r in results}
        print(f" -- Resuming: {len(done)} tensors already measured")

    def save():
        if not args.out:
            return
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            output = dict(
                model = args.model,
                rows = args.rows,
                length = args.length,
                draws = args.draws,
                mode = "shaped" if args.shaped else "iid",
                trace = args.trace,
                h_rows = args.h_rows if args.shaped else None,
                rfn_ref = args.rfn_ref,
                rfn_scale = args.rfn_scale if args.rfn_ref else None,
                rfn = None if args.rfn_ref else args.rfn,
                results = results,
            )
            json.dump(output, f, indent = 2)
        os.replace(tmp, args.out)

    if expected_keys <= done:
        save()
        model.unload()
        config.stc.close()
        print(" -- All tensors already measured")
        return

    save()
    print(" -- Reference pass")
    boundary, ref_logits = ref_pass()
    if not streaming:
        stage_refs()

    @torch.inference_mode()
    def perturb_iid(lin, rfn, seed):
        """
        Add iid Gaussian noise with ||n|| = rfn * ||W|| to the weights in place (fp32 math,
        chunked over input features). Returns (saved original weights, realized rfn, ||W||).
        """
        w = lin.inner.weight
        rows_per_chunk = max(1, 2 ** 24 // w.shape[1])
        w_sq = 0.0
        for a in range(0, w.shape[0], rows_per_chunk):
            w_sq += w[a : a + rows_per_chunk].float().square().sum().item()
        sigma = rfn * math.sqrt(w_sq / w.numel())
        gen = torch.Generator(device = w.device)
        gen.manual_seed(seed)
        saved = w.clone()
        err_sq = 0.0
        for a in range(0, w.shape[0], rows_per_chunk):
            b = min(a + rows_per_chunk, w.shape[0])
            n = torch.randn((b - a, w.shape[1]), generator = gen, device = w.device, dtype = torch.float)
            w[a:b] = (w[a:b].float() + n.mul_(sigma)).to(w.dtype)
            # Realized error after rounding to storage dtype
            err_sq += (w[a:b].float() - saved[a:b].float()).square().sum().item()
            del n
        return saved, (err_sq / w_sq) ** 0.5, w_sq ** 0.5

    @torch.inference_mode()
    def perturb_shaped(lin, rfn, seed, L, su):
        """
        Add LDLQ-shaped noise: dW = P^T L^-T eta (eta white, P the quantizer's sign-flip +
        block-Hadamard rotation), with per-output-channel scaling mimicking out_scales,
        normalized to ||dW|| = rfn * ||W||. L is finalize_capture_H's unit-block-lower factor
        (diagonal zeroed) of the rotated, regularized H; solve_triangular treats the unit
        diagonal implicitly. Noise up to NOISE_KEEP_BYTES is made once and kept between the norm
        and the update; larger noise (lm_head would be 5 GB) is made twice from a re-seeded
        generator. Returns (saved weights, err_sq, w_sq): fp64 device scalars, so a grouped target
        of a thousand linears reads them back once (the sums are the same fp64 sums of the same
        fp32 partials as Python-float accumulation, so the results are bit-identical).
        """
        w = lin.inner.weight                            # (k, n) = (in, out)
        k, n = w.shape
        assert L.shape[0] == k, f"{lin.key}: H dim {L.shape[0]} != in_features {k}"
        assert k % had_k == 0
        Lt = L.mT                                       # upper, unit diagonal implicit
        had = get_hadamard_dt(had_k, w.device, torch.float, 1.0 / math.sqrt(had_k))
        had_t = had.T.contiguous()
        su_col = su.view(k, 1).to(w.device)

        # Column norms: total weight norm + out_scales-like per-channel error weighting
        col_sq = torch.zeros(n, dtype = torch.float, device = w.device)
        rows_per_chunk = max(1, 2 ** 24 // n)
        for a in range(0, k, rows_per_chunk):
            col_sq += w[a : a + rows_per_chunk].float().square().sum(dim = 0)
        w_sq = col_sq.sum().double()
        col_scale = col_sq.sqrt()
        col_scale /= col_scale.square().mean().sqrt().clamp(min = 1e-20)
        col_scale.clamp_(min = 1e-4)

        cols_per_chunk = max(had_k, 2 ** 26 // k)
        keep = k * n * 4 <= NOISE_KEEP_BYTES

        def noise_chunks():
            gen = torch.Generator(device = w.device)
            gen.manual_seed(seed)
            for a in range(0, n, cols_per_chunk):
                b = min(a + cols_per_chunk, n)
                eta = torch.randn((k, b - a), generator = gen, device = w.device, dtype = torch.float)
                x = torch.linalg.solve_triangular(Lt, eta, upper = True, unitriangular = True)
                del eta
                x = (had_t @ x.view(k // had_k, had_k, b - a)).view(k, b - a)
                x *= su_col
                x *= col_scale[a:b].unsqueeze(0)
                yield a, b, x

        kept = []
        dw_sq = torch.zeros((), dtype = torch.float64, device = w.device)
        for a, b, x in noise_chunks():
            dw_sq += x.square().sum().double()
            if keep:
                kept.append((a, b, x))
            del x
        factor = rfn * torch.sqrt(w_sq / dw_sq)
        saved = w.clone()
        for a, b, x in (kept if keep else noise_chunks()):
            w[:, a:b] = (w[:, a:b].float() + x * factor).to(w.dtype)
            del x
        kept.clear()
        err_sq = torch.zeros((), dtype = torch.float64, device = w.device)
        for a in range(0, k, rows_per_chunk):
            b = min(a + rows_per_chunk, k)
            err_sq += (w[a:b].float() - saved[a:b].float()).square().sum().double()
        return saved, err_sq, w_sq

    NOISE_KEEP_BYTES = 1 << 28

    SAVED_ON_HOST_BYTES = 1 << 30      # grouped targets park their saved weights in host RAM past this

    @torch.inference_mode()
    def perturb_target(target, li, draw, shaping):
        """Perturb every member of a target at its own level (seeded per member, so single-Linear
        targets reproduce the ungrouped seeds). Returns (saved list, realized rfn over the whole
        target, ||W|| over the whole target, all members shaped)."""
        _, per = target_rfn(target, li)
        total_bytes = sum(lin.inner.weight.numel() * lin.inner.weight.element_size() for lin in target.linears)
        to_host = target.grouped and total_bytes > SAVED_ON_HOST_BYTES
        saved, sums, shaped_all = [], [], True
        for lin, rfn in zip(target.linears, per):
            seed = zlib.crc32(f"{lin.key}|{li}|{draw}".encode()) & 0x7fffffff
            shape_lin = shaping.get(lin.qmap) if args.shaped else None
            if shape_lin is not None:
                factors = tuple(t.to(lin.inner.weight.device) for t in shape_lin)
                sv, e_sq, l_sq = perturb_shaped(lin, rfn, seed, *factors)
                sums.append(torch.stack((e_sq, l_sq)))
                del factors
            else:
                shaped_all = False
                sv, rfn_actual, w_norm = perturb_iid(lin, rfn, seed)
                sums.append((rfn_actual, w_norm))
            if to_host:
                # Page-locked and asynchronous: the copy overlaps the next member's noise
                host = pinned.empty(sv.shape, sv.dtype)
                host.copy_(sv, non_blocking = True)
                sv = host
            saved.append(sv)
        # One readback for the whole target; then the same Python arithmetic as per-member floats
        dev = [i for i, x in enumerate(sums) if isinstance(x, torch.Tensor)]
        if dev:
            vals = torch.stack([sums[i] for i in dev]).tolist()
            for i, (e_sq, l_sq) in zip(dev, vals):
                sums[i] = ((e_sq / l_sq) ** 0.5, l_sq ** 0.5)
        err_sq, w_sq = 0.0, 0.0
        for rfn_actual, w_norm in sums:
            err_sq += (rfn_actual * w_norm) ** 2
            w_sq += w_norm ** 2
        if not target.grouped:
            return (saved, *sums[0], shaped_all)    # bit-identical to the ungrouped record
        return saved, (err_sq / w_sq) ** 0.5, w_sq ** 0.5, shaped_all

    @torch.inference_mode()
    def restore_target(target, saved):
        for lin, sv in zip(target.linears, saved):
            lin.inner.weight.copy_(sv, non_blocking = True)
        # Later copies into these buffers queue behind the restores on the same stream
        pinned.give(saved)
        saved.clear()

    @torch.inference_mode()
    def advance_capture(top_idx, capture):
        """Advance the capture rows through module top_idx, accumulating H per qmap if capture
        is a dict (shared across rows). Resident mode only."""
        nonlocal cap_states
        new_states = []
        for st, cap_row in zip(cap_states, cap_ids):
            if top_idx == num_mods - 1:
                # The head, one row at a time (see head_rows); its outputs are never needed
                for b in range(st.shape[0]):
                    params = {"input_ids": cap_row[b : b + 1]}
                    if capture is not None:
                        params["capture"] = capture
                    x = mods[top_idx].prepare_for_device(st[b : b + 1], params)
                    x = mods[top_idx].forward(x, params)
                    del x
                continue
            params = {"input_ids": cap_row}
            if capture is not None:
                params["capture"] = capture
            x = mods[top_idx].prepare_for_device(st, params)
            x = mods[top_idx].forward(x, params)
            new_states.append(x)
            del x
        cap_states = new_states

    print(f" -- {num_targets} target tensors, {args.draws} draw(s) at {num_levels} noise level(s), "
          f"{'shaped' if args.shaped else 'iid'} noise")

    control_kld = {}
    quant_args = {"sigma_reg": 0.025}

    def finalize_shaping(top_idx, capture):
        """Finalize per-qmap shaping factors, falling back to iid noise for unusable captures"""
        shaping = {}
        torch.manual_seed(zlib.crc32(f"su|{top_idx}".encode()) & 0x7fffffff)
        for qmap in list(capture):
            h_data = capture.pop(qmap)
            q_fallback, H, L, su, H_diag = finalize_capture_H(h_data, quant_args, False)
            if q_fallback or L is None:
                print(f" !! q_fallback for {qmap}, using iid noise")
                shaping[qmap] = None
            else:
                # On the device even when streaming: every member of a grouped target reads the
                # same factors, and they live only while this module's experiments spawn
                shaping[qmap] = (L, su)
            del H, H_diag, L, su, h_data
        return shaping

    def measurement_plan(target):
        """(level index, nominal rfn, draw) for every measurement on one target"""
        return [(li, target_rfn(target, li)[0], draw)
                for li in range(num_levels) for draw in range(args.draws)]

    def new_result(top_idx, target):
        res = dict(
            idx = top_idx,
            key = target.key,
            qbits_key = target.qbits_key,
            numel = target.weights_numel(),
            shaped = None,
            levels = [],
        )
        if target.grouped:
            # sc_optimize expands the group's bitrate to every member key in the recipe
            res["members"] = list(target.members)
        return res

    def injected_rfn(states, top_idx):
        """Injected error at the top-level module output, relative to the clean state"""
        if states is None or top_idx + 1 >= num_mods:
            return 0.0
        # fp64 device accumulators, read once (bit-identical to the per-row Python floats)
        inj_sq, ref_sq = None, None
        for st, clean in zip(states, boundary[top_idx + 1]):
            # Streaming keeps states in host memory: do the arithmetic on the GPU
            if st.device.type == "cpu":
                st = st.to(device, non_blocking = True)
            d = st.float() - clean.float().to(st.device)
            di = d.square().sum().double()
            ri = clean.float().square().sum().double().to(di.device)
            inj_sq = di if inj_sq is None else inj_sq.to(di.device) + di
            ref_sq = ri if ref_sq is None else ref_sq.to(di.device) + ri
            del d
        if inj_sq is None:
            return 0.0
        inj_sq, ref_sq = inj_sq.item(), ref_sq.item()
        return (inj_sq / ref_sq) ** 0.5 if ref_sq else 0.0

    def record(res, rfn, rfn_actual, draw, kld, inj_rfn, w_norm):
        res["levels"].append(dict(
            rfn = rfn,
            rfn_actual = rfn_actual,
            draw = draw,
            kld = kld,
            inj_rfn = inj_rfn,
        ))
        res["w_norm"] = w_norm
        print(f"    {res['key']:60} rfn {rfn_actual:.5f}   kld {kld:11.8f}   inj_rfn {inj_rfn:.6f}")

    def check_control(top_idx, kld):
        # No-perturbation control: restarting from the cached boundary must reproduce the
        # reference exactly
        control_kld[top_idx] = kld
        assert kld == 0.0, f"ctrl {kld} != 0 at module {top_idx}, restart machinery broken"

    if streaming:
        # Load each module once per pass, advancing pending states and spawning new perturbed states
        todo_by_idx = {idx: [t for t in targets_by_idx[idx] if t.key not in done]
                       for idx in sorted(targets_by_idx)}
        todo_by_idx = {idx: ts for idx, ts in todo_by_idx.items() if ts}
        measurements_by_idx = {idx: 1 + sum(len(measurement_plan(t)) for t in ts)
                              for idx, ts in todo_by_idx.items()}
        state_bytes = max((states_nbytes(states)
                           for states in boundary[1:]), default = 0)
        budget = sysmem_budget(args.max_sys)
        passes = plan_fanout_passes(measurements_by_idx, state_bytes, budget)
        peak = max((sum(measurements_by_idx[i] for i in p) for p in passes), default = 0)
        print(f" -- Fan-out: {sum(measurements_by_idx.values())} measurements over "
              f"{len(todo_by_idx)} module(s) in {len(passes)} pass(es); peak pending states "
              f"{peak * state_bytes / 1024 ** 3:.1f} GiB, budget {budget / 1024 ** 3:.1f} GiB")

        cap_idx = 0  # next module for capture rows
        for pass_no, members in enumerate(passes):
            # Capture rows stop where the next pass starts, so each module is captured once
            cap_stop = passes[pass_no + 1][0] - 1 if pass_no + 1 < len(passes) else members[-1]
            pending = []
            pass_results = []
            start = min(cap_idx, members[0]) if args.shaped else members[0]
            for idx in range(start, num_mods):
                last_mod = idx == num_mods - 1
                need_cap = args.shaped and cap_idx == idx and idx <= cap_stop
                spawn = todo_by_idx[idx] if idx in members else []
                assert not spawn or not args.shaped or need_cap
                if not (need_cap or pending or spawn):
                    continue
                print(f" -- [pass {pass_no + 1}/{len(passes)}] module {idx + 1}/{num_mods}: "
                      f"{len(pending)} pending, {len(spawn)} new tensor(s)")
                with runner.loaded(idx) as mod:
                    runner.prefetch(idx + 1)
                    t_mod = time.perf_counter()
                    phase = {}

                    def lap(name):
                        """Charge the time since the last lap to name; returns it"""
                        nonlocal t_mod
                        torch.cuda.synchronize(device)
                        t = time.perf_counter()
                        dt = t - t_mod
                        phase[name] = phase.get(name, 0.0) + dt
                        t_mod = t
                        return dt

                    def run(states, inspect = None):
                        """Forward rows through the loaded module: (states, None), or
                        (None, mean KL) on the last module"""
                        scores = []
                        out = runner.rows(
                            mod, states, ids = row_groups,
                            consume = (lambda r, x: scores.append(row_kld(r, x))) if last_mod else None,
                            per_row = last_mod, inspect = inspect,
                        )
                        return (None, sum(scores) / len(scores)) if last_mod else (out, None)

                    shaping = {}
                    if need_cap:
                        capture = {} if spawn else None
                        cap_states = runner.rows(mod, cap_states, capture = capture, ids = cap_ids,
                                                 consume = (lambda r, x: None) if last_mod else None,
                                                 per_row = last_mod)
                        cap_idx += 1
                        lap("capture")
                        if spawn:
                            shaping = finalize_shaping(idx, capture)
                        del capture
                        lap("finalize")

                    if last_mod:
                        stage_refs()

                    # Advance pending states through the unperturbed module; each experiment's old
                    # buffers then hold the next experiment's outputs
                    for exp in pending:
                        old = exp["states"]
                        exp["states"], exp["kld"] = run(old)
                        pinned.give(old)
                        del old
                    lap("advance")

                    spawn_in = clean_out = clean_sq = None
                    if spawn:
                        spawn_in = boundary[idx] if mod.caps.get("prefer_cpu") else stage_states(boundary[idx])
                        if not last_mod:
                            clean_out = stage_states(boundary[idx + 1])
                            # fp64 sum over groups in order, as injected_rfn adds them
                            for c in clean_out:
                                ri = c.float().square().sum().double().to(device)
                                clean_sq = ri if clean_sq is None else clean_sq + ri
                        states, kld = run(spawn_in)
                        pending.append(dict(control = idx, states = states, kld = kld))

                    def inj_hook(acc):
                        """Injected error from each group's device output, before its download:
                        the same fp64 sums as injected_rfn without uploading the states again"""
                        def hook(n, x):
                            d = x.float() - clean_out[n].float().to(x.device)
                            di = d.square().sum().double()
                            acc[0] = di if acc[0] is None else acc[0] + di
                        return hook

                    for target in spawn:
                        for lin in target.linears:
                            assert isinstance(lin.inner, LinearFP16), \
                                f"{lin.key}: expected unquantized (fp16) tensor, got {type(lin.inner).__name__}"
                        res = new_result(idx, target)
                        for li, rfn, draw in measurement_plan(target):
                            lap("spawn")
                            saved, rfn_actual, w_norm, shaped_all = perturb_target(target, li, draw, shaping)
                            res["shaped"] = shaped_all
                            t_p = lap("perturb")
                            acc = [None]
                            try:
                                states, kld = run(spawn_in, inspect = None if last_mod else inj_hook(acc))
                                t_f = lap("forward")
                                print(f"      spawn {target.key[-48:]:48} level {li}: "
                                      f"perturb {t_p:.2f} s, forward {t_f:.2f} s")
                            finally:
                                # Restore before unloading, which discards lin.inner
                                restore_target(target, saved)
                            lap("restore")
                            inj_rfn = 0.0
                            if acc[0] is not None:
                                inj_sq, ref_sq = acc[0].item(), clean_sq.item()
                                inj_rfn = (inj_sq / ref_sq) ** 0.5 if ref_sq else 0.0
                            pending.append(dict(
                                res = res, rfn = rfn, rfn_actual = rfn_actual, draw = draw,
                                w_norm = w_norm, inj_rfn = inj_rfn,
                                states = states, kld = kld,
                            ))
                            lap("inj")
                        pass_results.append(res)
                    for v in shaping.values():
                        del v
                    shaping.clear()
                    lap("spawn")
                    unstage_refs()
                    del spawn_in, clean_out, clean_sq
                print(f"    time: load wait {runner.take_wait():.2f} s, " +
                      ", ".join(f"{k} {v:.2f} s" for k, v in phase.items()))

            # Check controls and save the completed pass
            for exp in pending:
                if "control" in exp:
                    check_control(exp["control"], exp["kld"])
                else:
                    record(exp["res"], exp["rfn"], exp["rfn_actual"], exp["draw"], exp["kld"],
                           exp["inj_rfn"], exp["w_norm"])
            del pending
            results.extend(pass_results)
            save()
            torch.cuda.empty_cache()

    else:
        for top_idx in range(num_mods):
            if top_idx > max(targets_by_idx, default=-1):
                break
            todo = [t for t in targets_by_idx.get(top_idx, []) if t.key not in done]

            # Advance the capture rows through every module; accumulate H only where needed
            shaping = {}
            if args.shaped:
                capture = {} if todo else None
                advance_capture(top_idx, capture)
                if todo:
                    shaping = finalize_shaping(top_idx, capture)
                del capture

            if not todo:
                continue

            if top_idx not in control_kld:
                kld, _ = forward_rows(top_idx, boundary[top_idx])
                check_control(top_idx, kld)

            for target in todo:
                res = new_result(top_idx, target)
                for li, rfn, draw in measurement_plan(target):
                    saved, rfn_actual, w_norm, shaped_all = perturb_target(target, li, draw, shaping)
                    res["shaped"] = shaped_all
                    try:
                        kld, states = forward_rows(top_idx, boundary[top_idx],
                                                   collect_states = top_idx + 1 < num_mods)
                    finally:
                        restore_target(target, saved)
                    record(res, rfn, rfn_actual, draw, kld, injected_rfn(states, top_idx), w_norm)
                    del states

                results.append(res)
                save()

            for v in shaping.values():
                del v
            shaping.clear()
            torch.cuda.empty_cache()

    # Summary
    total = sum(r["levels"][0]["kld"] for r in results)
    print(f"\n -- Sum of per-tensor KLD at level 0: {total:.6f}")

    def layer_of(key):
        for part in key.split("."):
            if part.isdigit():
                return int(part)
        return None

    print("\n -- By layer (level 0):")
    by_layer = defaultdict(float)
    for r in results:
        l = layer_of(r["key"])
        by_layer["-" if l is None else l] += r["levels"][0]["kld"]
    for l, k in by_layer.items():
        bar = "#" * int(150 * k / max(total, 1e-12))
        print(f"      {str(l):10} {k:.6f}  ({100 * k / total:5.1f}%)  {bar}")

    print("\n -- Top contributors (level 0):")
    for r in sorted(results, key = lambda r: -r["levels"][0]["kld"])[:args.top]:
        k = r["levels"][0]["kld"]
        print(f"      {r['key']:60} kld {k:.6f}  ({100 * k / total:4.1f}%)")

    if num_levels > 1:
        alphas = []
        for r in results:
            l0, l1 = r["levels"][0], r["levels"][-1]
            if l0["kld"] > 0 and l1["kld"] > 0 and l0["rfn_actual"] != l1["rfn_actual"]:
                alphas.append(
                    math.log(l0["kld"] / l1["kld"]) / math.log(l0["rfn_actual"] / l1["rfn_actual"]))
        if alphas:
            alphas.sort()
            print(f"\n -- Scaling exponent (kld ~ rfn^a): "
                  f"min {alphas[0]:.2f}  median {alphas[len(alphas) // 2]:.2f}  max {alphas[-1]:.2f}")

    save()
    if args.out:
        print(f"\n -- Saved: {args.out}")
    model.unload()
    config.stc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-m", "--model", type = str, required = True, help = "Unquantized model directory")
    parser.add_argument("-r", "--rows", type = int, default = 10, help = "Number of eval rows, default: 10")
    parser.add_argument("-l", "--length", type = int, default = 1024, help = "Tokens per row, default: 1024")
    parser.add_argument("-d", "--device", type = int, default = 0, help = "CUDA device index")
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument("--load-mode", choices = ("auto", "resident", "streaming"),
                         default = "auto", help = "auto (default): check free VRAM before choosing full-model loading or streaming")
    loading.add_argument("--streaming", dest = "load_mode", action = "store_const", const = "streaming",
                         help = "Force one-module-at-a-time loading with states and shaping factors in system RAM")
    loading.add_argument("--no-streaming", dest = "load_mode", action = "store_const", const = "resident",
                         help = "Force full-model loading, bypassing the VRAM preflight")
    parser.add_argument("-ms", "--max-sys", dest = "max_sys", type = float, default = None,
                        help = "Streaming: system RAM for pending fan-out states, in GB (default: half of what is available)")
    parser.add_argument("-tr", "--trace", type = str, default = None, help = "Packed self-sampled trace (safetensors from sc_trace.py) to use as eval")
    parser.add_argument("-sh", "--shaped", action = "store_true", help = "LDLQ-shaped noise from captured Hessians (recommended; extra capture pass)")
    parser.add_argument("-hr", "--h_rows", type = int, default = 64, help = "Calibration rows for Hessian capture in shaped mode, default: 64")
    parser.add_argument("--spill-dir", dest = "spill_dir", type = str, default = None,
                        help = "Streaming: keep boundary states in files under this directory (a fast local disk) "
                               "instead of host memory, for hosts with less RAM than the boundaries need. Files are left in place")
    parser.add_argument("-br", "--batch-rows", dest = "batch_rows", type = int, default = 1,
                        help = "Forward up to N equal-length rows together (the head still runs per row). "
                               "Faster where per-row forwards are launch-bound; costs activation memory. Default: 1")
    parser.add_argument("-rfn", "--rfn", type = str, default = "0.29,0.145", help = "Comma-separated global noise levels (relative weight Frobenius norm)")
    parser.add_argument("-rr", "--rfn_ref", type = str, default = None, help = "Per-tensor noise anchors from sc_rfn_probe.py JSON (overrides --rfn)")
    parser.add_argument("-rs", "--rfn_scale", type = str, default = "1.0,0.5", help = "Scale factors applied to --rfn_ref anchors, default: 1.0,0.5")
    parser.add_argument("-dr", "--draws", type = int, default = 1, help = "Noise draws per level, default: 1")
    parser.add_argument("-t", "--top", type = int, default = 15, help = "Top contributors to print")
    parser.add_argument("-o", "--out", type = str, default = None, help = "Output file (JSON), resumes if present")
    main(parser.parse_args())
